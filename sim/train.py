import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing_extensions import Literal
from tensorboardX import SummaryWriter
from data import get_sim
from utils import dict_to_ls
import slune
import os

# Define the softmax regression model
class MultiSoftmaxRegression(nn.Module):
    def __init__(self, d, mt, my, theta=None):
        super(MultiSoftmaxRegression, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(d, my) for _ in range(mt)])
        self.my = my
        # If theta is given, use it to initialize the weights of the linear layers
        if theta is not None:
            for i, layer in enumerate(self.linear_layers):
                layer.weight.data = theta[i].clone().detach().float()

    def forward(self, x, t, log=False):
        outputs = torch.zeros(x.shape[0], self.my)
        for unique_t in t.unique():
            mask = (t == unique_t)
            x_subset = x[mask]
            if log:
                output_subset = torch.nn.functional.log_softmax(self.linear_layers[unique_t.item()](x_subset), dim=1)
            else:
                output_subset = torch.nn.functional.softmax(self.linear_layers[unique_t.item()](x_subset), dim=1)
            outputs[mask] = output_subset
        return outputs
    
    def predict(self, x, t):
        # Get the softmax probabilities
        probs = self.forward(x, t)
        # Get the class with the highest probability
        _, predictions = torch.max(probs, dim=1)
        return predictions

# Define the negative log likelihood loss
def negative_log_likelihood(model, x, t, y, *args):
    output = model(x, t)
    return -torch.log(output[torch.arange(len(output)), y] + 1e-20).mean()

# Define the bi-NCE loss
def bi_nce_loss(model, x, t, y, K):
    # Compute scores for 'real' samples
    out = model(x, t)
    real_scores = out[torch.arange(out.shape[0]), y] + 1e-20

    # Sample from p_{N} (noise distribution) and compute scores for 'fake'/'negative' samples
    # we use p_{N} ~ U(0, my-1) 
    y_noise = torch.randint(low=0, high=out.shape[1], size=(K, out.shape[0]))
    noise_scores = torch.stack([out[torch.arange(out.shape[0]), y_noise[i]] + 1e-20 for i in range(y_noise.shape[0])])
    log_noise_pdf = torch.log((1/out.shape[1]) * torch.ones_like(noise_scores))
    noise_scores = torch.exp(noise_scores - log_noise_pdf)
    sum_noise_scores = noise_scores.sum(dim=0)

    # Compute the loss
    return -torch.log(real_scores / sum_noise_scores).mean()


def train(**kwargs):
    """
    Train the model using the specified estimator.

    Args:
        n_train: int (number of training samples)
        n_test: int (number of testing samples)
        d: int (dimension of x)
        mx: int (size of feature space for x)
        mt: int (number of tasks)
        my: int (number of classes)
        model: Literal['multi_softmax'] (model to train)
        learning_rate: float (learning rate for optimizer)
        num_epochs: int (number of epochs to train for)
        batch_size: int (batch size for training, if None, use full dataset)
        est: Literal['mle', 'bi_nce'] (estimator to use)
        K: int (number of negative samples to use for bi_nce, ignored if est != 'bi_nce')
        patience: int (number of epochs to wait before early stopping)

    Returns:
        losses: list
        accuracies: list 
        true_theta: np.ndarray
        model: nn.Module

    """
    # Unpack the config
    n_train = kwargs['n_train']
    n_test = kwargs['n_test']
    d = kwargs['d']
    mx = kwargs['mx']
    mt = kwargs['mt']
    my = kwargs['my']
    model = kwargs['model']
    learning_rate = kwargs['learning_rate']
    num_epochs = kwargs['num_epochs']
    batch_size = kwargs['batch_size']
    est = kwargs['est']
    try: 
        K = kwargs['K']
    except KeyError:
        K = None
    patience = kwargs['patience']

    # Create save location using slune and tensorboard writer
    formatted_args = dict_to_ls(**kwargs)
    saver = slune.get_csv_saver(formatted_args, root_dir='results')
    path = os.path.dirname(saver.get_current_path())
    writer = SummaryWriter(path)
    
    # Generate / Load in the data
    train_dataset, test_dataset, true_theta, x_feature_space = get_sim(n_train, n_test, d, mx, mt, my)

    # Define the model and optimizer
    if model == "multi_softmax":
        model = MultiSoftmaxRegression(d, mt, my)
    else:
        raise ValueError("Invalid model")
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Define the loss function based on the estimator
    if est == "mle":
        loss_fun = negative_log_likelihood
    elif est == "bi_nce":
        assert K is not None
        loss_fun = bi_nce_loss
    else:
        raise ValueError("Invalid est(imator)")
    
    # Set-up for training
    losses = []
    accuracies = []
    dataloader = DataLoader(train_dataset, batch_size=batch_size or len(train_dataset), shuffle=True)
    best_loss = float('inf') # For early stopping
    patience_counter = 0 # For early stopping

    # Train the model
    cum_b = -1
    for epoch in range(num_epochs):
        for b, (x_batch, t_batch, y_batch) in enumerate(dataloader):
            cum_b += 1
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = loss_fun(model, x_batch, t_batch, y_batch, K)
            losses.append(loss.item())
            writer.add_scalar('Loss/train', loss.item(), cum_b)
            saver.log({'train_loss': loss.item()})

            # Calculate accuracy
            y_pred = model.predict(x_batch, t_batch)
            accuracy = (y_pred == y_batch).float().mean()
            accuracies.append(accuracy.item())
            writer.add_scalar('Accuracy/train', accuracy.item(), cum_b)
            saver.log({'train_accuracy': accuracy.item()})

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # If we have NaNs, stop training
            if torch.isnan(loss):
                print("NaNs encountered, stopping training", flush=True)
                return losses, accuracies, true_theta, model
        
        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', flush=True)

        # Test the model
        with torch.no_grad():
            x_test, t_test, y_test = test_dataset.tensors
            y_pred = model.predict(x_test, t_test)
            accuracy = (y_pred == y_test).float().mean()
            writer.add_scalar('Accuracy/test', accuracy.item(), cum_b)
            saver.log({'test_accuracy': accuracy.item()})
            if (epoch + 1) % 5 == 0:
                print(f'Accuracy of the model on the test set: {accuracy.item():.4f}', flush=True)

        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping", flush=True)
                break

    # Calculate the KL divergence between the true and estimated parameters
    with torch.no_grad():
        # Get all unique possible values of x 
        x = [torch.from_numpy(x_feature_space[i]) for i in range(len(x_feature_space))]
        t = torch.cat([torch.full(size=(x[i].shape[0],), fill_value=i) for i in range(mt)], dim=0).long()
        x = torch.cat(x, dim=0).float()
        # Get log probabilities for true parameters
        true_model = MultiSoftmaxRegression(d, mt, my, theta=torch.from_numpy(true_theta))
        true_log_p = true_model(x, t, log=True)
        # Get log probabilities for estimated parameters
        estimated_log_p = model(x, t, log=True)
        # Calculate the KL divergence between the true and estimated log probabilities
        kl_divergence = F.kl_div(true_log_p, estimated_log_p, reduction='batchmean', log_target=True)
    print(f"KL Divergence: {kl_divergence}", flush=True)
    saver.log({'kl_divergence': kl_divergence.item()})
    writer.add_scalar('KL Divergence', kl_divergence)

    saver.save_collated()
    return losses, accuracies, true_theta, model

# Main function used to execute a training example and visualize results, use --default_exp_name to specify which example to run
# Use: python train.py --default_exp_name=mle
    # To run the MLE example
# Use: python train.py --default_exp_name=bi_nce
    # To run the bi-NCE example
if __name__ == "__main__":
    #  Parse input from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--default_exp_name', type=str, help='Name of the example experiment you want to run, either "mle" or "bi_nce"', default="mle")
    args = parser.parse_args()
    config = {
        "n_train": 16000,
        "n_test": 1000,
        "d": 4,
        "mx": 100,
        "mt": 2,
        "my": 100,
        "model": "multi_softmax",
        "learning_rate": 0.01,
        "num_epochs": 5,
        "batch_size": 128,
        "est": args.default_exp_name,
        "K": 10,
        "patience": 10
    }
    # Train the model
    losses, accuracies, true_theta, model = train(**config)

    # Create save location using slune
    formatted_args = dict_to_ls(**config)
    saver = slune.get_csv_saver(formatted_args, root_dir='results')
    path = os.path.dirname(saver.get_current_path())
    # Save the true_theta and model to the save location using torch.save
    torch.save(true_theta, os.path.join(path, "true_theta.pt"))
    torch.save(model, os.path.join(path, "model.pt"))
    # Plot matrix of differences between true and estimated parameters
    for t in range(config['mt']):
        plt.figure()
        plt.imshow((true_theta[t] - model.linear_layers[t].weight.detach().numpy())**2)
        plt.colorbar()
        plt.savefig(os.path.join(path, f"theta_diff_task_{t}.png"))
        plt.close()

    # Plot the loss
    plt.figure()
    plt.plot(np.arange(len(losses)), np.array(losses))
    plt.savefig(os.path.join(path, "loss.png"))
    plt.close()

    # Plot the accuracy
    plt.figure()
    plt.plot(np.arange(len(accuracies)), np.array(accuracies))
    plt.savefig(os.path.join(path, "accuracy.png"))
    plt.close()

