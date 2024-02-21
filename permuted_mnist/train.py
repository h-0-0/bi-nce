from data import get_permuted_mnist
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from utils import dict_to_ls
import slune
import os
from model import MLP, bi_nce_loss, info_nce_loss, LinearClassifier

def train(**kwargs):
    """
    Train the model using the specified estimator.

    Args:
        benchmark: Literal['permuted_mnist'] (dataset to use)
        model: Literal['MLP'] (model to train)
        learning_rate: float (learning rate for optimizer)
        num_epochs: int (number of epochs to train for)
        batch_size: int (batch size for training, if None, use full dataset)
        est: Literal['bi_nce', 'info_nce'] (estimator to use)
        patience: int (number of epochs to wait before early stopping)
        net_structure: List[int] (structure of the network, first is number of layers then the rest are the hidden dimensions, behaviour depends on model)
        temperature: float (temperature for the estimator)
        t_encoding: Literal['one_hot', 'repeated'] (how to encode the task variable)

    Returns:
        losses: list
        accuracies: list 
        model: nn.Module

    """
    # Unpack the config
    benchmark = kwargs['benchmark']
    t_encoding = kwargs['t_encoding']
    model = kwargs['model']
    learning_rate = kwargs['learning_rate']
    num_epochs = kwargs['num_epochs']
    batch_size = kwargs['batch_size']
    est = kwargs['est']
    patience = kwargs['patience']
    net_structure = kwargs['net_structure']
    temperature = kwargs['temperature']

    # Create save location using slune and tensorboard writer
    formatted_args = dict_to_ls(**kwargs)
    saver = slune.get_csv_saver(formatted_args, root_dir='results')
    path = os.path.dirname(saver.get_current_path())
    writer = SummaryWriter(path)    

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Using {device} device ---", flush=True)
    print(f"--- Running config: {kwargs} ---", flush=True)
            
    # Generate / Load in the data
    if benchmark == "permuted_mnist":
        train_loader, test_loader = get_permuted_mnist(batch_size=batch_size, t_encoding=t_encoding)
    else:
        raise ValueError("Invalid benchmark")

    # Define the model and optimizer
    if t_encoding == "one_hot":
        if benchmark == "permuted_mnist":
            num_tasks = 10
            task_dim = 10
        else:
            raise ValueError("Invalid benchmark")
    elif t_encoding == "repeated":
        if benchmark == "permuted_mnist":
            num_tasks = 10
            task_dim = 200
        else:
            raise ValueError("Invalid benchmark")
    else:
        raise ValueError("Invalid t_encoding")
    
    if model == "MLP":
        num_layers, hidden_dim, output_dim = net_structure
        if est == "bi_nce":
            model = MLP(input_dim=784+task_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        elif est == "info_nce":
            model = MLP(input_dim=784, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
        else:
            raise ValueError("Invalid est(imator)")
    else:
        raise ValueError("Invalid model")
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Define the loss function based on the estimator
    if est == "bi_nce":
        loss_fun = bi_nce_loss
    elif est == "info_nce":
        loss_fun = info_nce_loss
    else:
        raise ValueError("Invalid est(imator)")
    
    # Set-up for training
    losses = []
    best_loss = float('inf') # For early stopping
    patience_counter = 0 # For early stopping

    # Train the model
    cum_b = -1
    for epoch in range(num_epochs):
        for b, (x_batch, _, t_batch) in enumerate(train_loader):
            cum_b += 1
            x_batch, t_batch = x_batch.to(device), t_batch.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss = loss_fun(model, x_batch, t_batch, temperature, device)
            losses.append(loss.item())
            writer.add_scalar('Loss/train', loss.item(), cum_b)
            saver.log({'train_loss': loss.item()})

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # If we have NaNs, stop training
            if torch.isnan(loss):
                print("NaNs encountered, stopping training", flush=True)
                return losses, model
        
        # Print progress
        if (epoch + 1) % 25 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', flush=True)


        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping", flush=True)
                break

    # Now that we have trained the model, we evaluate it by training a linear classifier on top of the frozen representations
    # Define the linear classifier
    linear_classifier = LinearClassifier(model.output_dim, 10).to(device)
    learning_rate = 0.1 * batch_size / 256
    optimizer = optim.SGD(linear_classifier.parameters(), lr=learning_rate)
    for param in model.parameters():
        param.requires_grad = False

    # Train the linear classifier
    cum_b = -1
    for epoch in range(50):
        for b, (x_batch, y_batch, t_batch) in enumerate(train_loader):
            cum_b += 1
            x_batch, y_batch, t_batch = x_batch.to(device), y_batch.to(device), t_batch.to(device)
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            x_batch_flat = x_batch.view(x_batch.size(0), -1)
            if est == "bi_nce":
                rep = model(x_batch_flat, t_batch)
            elif est == "info_nce":
                rep = model(x_batch_flat)
            else:
                raise ValueError("Invalid est(imator)")
            logits = linear_classifier(rep)
            loss = nn.CrossEntropyLoss()(logits, y_batch)
            writer.add_scalar('Eval/train_loss', loss.item(), cum_b)
            saver.log({'eval_train_loss': loss.item()})
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # If we have NaNs, stop training
            if torch.isnan(loss):
                print("NaNs encountered, stopping training", flush=True)
                return losses, model
            
            cum_b += 1
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}', flush=True)

    # Test the linear classifier
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x_test, y_test, t_test in test_loader:
            x_test, y_test, t_test = x_test.to(device), y_test.to(device), t_test.to(device)
            x_test_flat = x_test.view(x_test.size(0), -1)
            if est == "bi_nce":
                rep = model(x_test_flat, t_test)
            elif est == "info_nce":
                rep = model(x_test_flat)
            else:
                raise ValueError("Invalid est(imator)")
            logits = linear_classifier(rep)
            loss = nn.CrossEntropyLoss()(logits, y_test)
            total_loss += loss.item() * x_test.size(0)  # Multiply by batch size
            # Compute the accuracy
            _, predicted = torch.max(logits.data, 1)
            total_correct += (predicted == y_test).sum().item()
            total_samples += x_test.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    writer.add_scalar('Eval/test_loss', avg_loss, cum_b)
    saver.log({'eval_test_loss': avg_loss})
    writer.add_scalar('Eval/test_accuracy', accuracy, cum_b)
    saver.log({'eval_test_accuracy': accuracy})
    print(f'Accuracy of the network on the {total_samples} test images: {accuracy:.4f}', flush=True)

    saver.save_collated()
    return losses, model

# Main function used to execute a training example and visualize results, use --default_exp_name to specify which example to run
# Use: python train.py --default_exp_name=bi_nce
    # To run the bi-NCE example
# Use: python train.py --default_exp_name=info_nce
    # To run the info-NCE example
if __name__ == "__main__":
    #  Parse input from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--default_exp_name', type=str, help='Name of the example experiment you want to run, either "bi_nce" or "info_nce"', default="info_nce")
    args = parser.parse_args()
    config = {
        'benchmark': 'permuted_mnist',
        'model': 'MLP',
        'learning_rate': 1e-7,
        'num_epochs': 1,
        'batch_size': 512,
        'est': args.default_exp_name,
        'patience': 5,
        'net_structure': [1, 1024, 40],
        'temperature': 0.1,
        't_encoding': 'repeated',
    }
    # Train the model
    losses, model = train(**config)

    # Create save location using slune
    formatted_args = dict_to_ls(**config)
    saver = slune.get_csv_saver(formatted_args, root_dir='results')
    path = os.path.dirname(saver.get_current_path())
    # Save the model to the save location using torch.save
    torch.save(model, os.path.join(path, "model.pt"))

    # Plot the loss
    plt.figure()
    plt.plot(np.arange(len(losses)), np.array(losses))
    plt.savefig(os.path.join(path, "loss.png"))
    plt.close()

