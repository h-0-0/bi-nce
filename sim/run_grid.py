import argparse
import slune
from utils import dict_to_ls, ls_to_dict
import os
import matplotlib.pyplot as plt
from train import train
import numpy as np

if  __name__ == "__main__":
    # Parse input from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, help='Number of training examples', default=16000)
    parser.add_argument('--n_test', type=int, help='Number of testing examples', default=1000)
    parser.add_argument('--d', type=int, help='Dimension of x', default=4)
    parser.add_argument('--mx', type=int, help='Number of possible values for x', default=100)
    parser.add_argument('--mt', type=int, help='Number of tasks', default=2)
    parser.add_argument('--my', type=int, help='Number of possible values for y', default=100)
    parser.add_argument('--model', type=str, help='Model to use', default="multi_softmax")
    parser.add_argument('--est', type=str, help='Estimator to use', default="mle")
    parser.add_argument('--patience', type=int, help='Patience for early stopping', default=10)
    parser.add_argument('--K', type=int, help='Number of negative samples for task NCE', default=10)
    parser.add_argument('--num_epochs', type=int, help='Number of epochs to train for', default=5000)
    args = parser.parse_args()

    if args.est == 'mle':
        lrs = [1e-2]
    elif args.est == 'task_nce':
        lrs = [1e-1, 1e-2, 1e-3]
    else:
        raise ValueError(f"Estimator {args.est} not supported")
    to_search = {
        # These parameters are fixed
        "n_train": [args.n_train], 
        "n_test": [args.n_test],
        "d": [args.d],
        "mx": [args.mx],
        "mt": [args.mt],
        "my": [args.my],
        "model": [args.model],
        "est": [args.est],
        "patience": [args.patience],
        "K": [args.K],
        "num_epochs": [5000],

        # These are the parameters we want to search over
        "learning_rate": lrs,
        "batch_size": [64, 128, 512, args.n_train],
    }
    saver = slune.get_csv_saver(root_dir='results') 
    grid = slune.searchers.SearcherGrid(to_search, runs=1) 
    grid.check_existing_runs(saver) 
    # Perform grid search
    for args in grid: 
        print(args)
        # Train the model
        losses, accuracies, true_theta, model = train(**ls_to_dict(args))
        # Create save location using slune
        path = os.path.dirname(saver.get_path(args))
        # Plot matrix of differences between true and estimated parameters
        print(to_search['mt'])
        for t in range(int(to_search['mt'][0])):
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