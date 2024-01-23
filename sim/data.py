import numpy as np
from typing import Tuple
import torch
from torch.utils.data import TensorDataset

def get_xt(n_train: int, n_test: int, d: int, mx: int, mt: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Define mt gaussians in R^d, for each of the mt gaussians get mx samples.
    The mean of the i-th Gaussian is the vector of all i's.
    The covariance matrix is the identity matrix scaled by epsilon.
    Uniformly get n samples of t from [0, ..., mt-1],
    and get n samples of x where we get the i-th x is sampled from the set coming from the t[i]-th Gaussian.
    Return the samples of x and t.

    Args:
        n_train: int
        n_test: int
        d: int
        mx: int
        mt: int 

    Returns:
        train_x: np.array of shape (n_train, d)
        train_t: np.array of shape (n_train,)
        test_x: np.array of shape (n_test, d)
        test_t: np.array of shape (n_test,)
        samples: list of np.array of shape (mx, d)

    """
    epsilon = 0.1
    # Define the means of the mt Gaussians
    means = np.array([np.full(d, i) for i in range(mt)])

    # Define the covariance matrix
    s = np.eye(d) * epsilon

    # For each Gaussian, draw mx samples
    samples = [np.array([np.random.multivariate_normal(mean=mean, cov=s) for _ in range(mx)]) for mean in means]

    # Uniformly sample n values of t from [0, ..., mt-1]
    train_t = np.random.randint(0, mt, n_train)
    test_t = np.random.randint(0, mt, n_test)

    # For each t[i], sample an x from the set of samples that came from the t[i]-th Gaussian
    train_x = np.array([samples[t_i][np.random.randint(0, mx)] for t_i in train_t])
    test_x = np.array([samples[t_i][np.random.randint(0, mx)] for t_i in test_t])

    return train_x, train_t, test_x, test_t, samples


def get_theta(mt: int, my: int, d: int) -> np.ndarray:
    """
    Define mt * my * d matrices.
    Sample entries from N(0, 1).
    Return the matrices.

    Args:
        mt: int
        my: int
        d: int

    Returns:
        theta: np.array of shape (mt, my, d)

    """

    theta = np.random.normal(size=(mt, my, d))
    return theta

def get_y(train_x: np.ndarray, train_t: np.ndarray, test_x: np.ndarray, test_t: np.ndarray, theta: np.ndarray, n_train: int, n_test: int, my: int) -> np.ndarray:
    """
    y takes values in [0, ..., my-1].
    We get n samples of y from the conditional model:
        p(y|x,t; theta) = exp(x @ theta[y, t]) / sum_{y=0}^{my-1} exp(x @ theta[y, t])
    Return the samples of y.

    Args:
        train_x: np.array of shape (n_train, d)
        train_t: np.array of shape (n_train,)
        test_x: np.array of shape (n_test, d)
        test_t: np.array of shape (n_test,)
        theta: np.array of shape (mt, my, d)
        n_train: int
        n_test: int
        my: int

    Returns:
        y: np.array of shape (n,)

    """

    train_y = np.zeros(shape=(n_train))
    for i in range(n_train):
        p = np.exp(theta[train_t[i]] @ train_x[i])
        p_sum = np.sum(p)
        train_y[i] = np.random.choice(int(my), p = p / p_sum)

    test_y = np.zeros(shape=(n_test))
    for i in range(n_test):
        p = np.exp(theta[test_t[i]] @ test_x[i])
        p_sum = np.sum(p)
        test_y[i] = np.random.choice(int(my), p= p / p_sum)

    return train_y, test_y

def get_sim(n_train: int, n_test: int, d: int, mx: int, mt: int, my: int, seed: int=15022024) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Helper function that returns the samples of x, t, theta, y.

    Args:
        n_train: int (number of training samples)
        n_test: int (number of test samples)
        d: int (dimension of x)
        mx: int (size of feature space for x)
        mt: int (number of possible tasks)
        my: int (number of possible classes)

    Returns:
        train_dataset: TensorDataset
        test_dataset: TensorDataset
        theta: np.array of shape (mt, my, d) 
        x_feature_space: list of np.array of shape (mx, d)

    """
    # Try to load the data from a file with torch.load
    # If the file does not exist, generate the data and save it to a file
    prefix = "data/sim/n_train="+ str(n_train) + "_n_test=" + str(n_test) + "_d=" + str(d) + "_mx=" + str(mx) + "_mt=" + str(mt) + "_my=" + str(my) 
    try:
        train_dataset = torch.load(prefix + "_seed={}_train.pt".format(seed))
        test_dataset = torch.load(prefix + "_seed={}_test.pt".format(seed))
        theta = torch.load(prefix + "_seed={}_theta.pt".format(seed))
        x_feature_space = torch.load(prefix + "_seed={}_x_feature_space.pt".format(seed))
        return train_dataset, test_dataset, theta, x_feature_space
    except FileNotFoundError:
        # Set the seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        # First we get the training, testing data and the true theta
        train_x, train_t, test_x, test_t, x_feature_space = get_xt(n_train, n_test, d, mx, mt)
        theta = get_theta(mt, my, d)
        train_y, test_y = get_y(train_x, train_t, test_x, test_t, theta, n_train, n_test, my)
        # Turn training data into a torch tensor dataset
        train_x = torch.from_numpy(train_x).float()
        train_t = torch.from_numpy(train_t).long()
        train_y = torch.from_numpy(train_y).long()
        train_dataset = TensorDataset(train_x, train_t, train_y)
        # Now we turn it into a torch tensor dataset
        test_x = torch.from_numpy(test_x).float()
        test_t = torch.from_numpy(test_t).long()
        test_y = torch.from_numpy(test_y).long()
        test_dataset = TensorDataset(test_x, test_t, test_y)

        # Save the data to a file with torch.save, include the seed used to generate the data in the name
        import os
        if not os.path.exists('data/sim'):
            os.makedirs('data/sim')
        torch.save(train_dataset, prefix + "_seed={}_train.pt".format(seed))
        torch.save(test_dataset, prefix + "_seed={}_test.pt".format(seed))
        torch.save(theta, prefix + "_seed={}_theta.pt".format(seed))
        torch.save(x_feature_space, prefix + "_seed={}_x_feature_space.pt".format(seed))

        return train_dataset, test_dataset, theta, x_feature_space

def example_viz(x: torch.FloatTensor, t: torch.LongTensor, y: torch.LongTensor, theta: torch.FloatTensor, save_prefix: str=''):
    """Helper function to visualize the data.
    
    We assume that d=mt=my=2.
    
    Args:
        x: torch.FloatTensor of shape (n, 2)
        t: torch.LongTensor of shape (n,)
        y: torch.LongTensor of shape (n,)
        theta: torch.FloatTensor of shape (mt, my, 2)
        save_prefix: str
        
    """
    import os
    if not os.path.exists('viz'):
        os.makedirs('viz')
    save_prefix = 'viz/' + save_prefix + '_'
    import matplotlib.pyplot as plt
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=t)
    plt.savefig(save_prefix + 'x.png')
    plt.close()

    plt.figure()
    for i in range(mt):
        plt.imshow(theta[i], cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.savefig(save_prefix + 'theta_{}.png'.format(i))
        plt.close()

    plt.figure()
    fig, axs = plt.subplots(t.max().item() + 1)
    my = y.max().item() + 1  # Assuming y values start from 0
    colors = plt.cm.viridis(np.linspace(0, 1, t.max().item() + 1))  # Create a color map
    for i in range(t.max().item() + 1):
        y_ti = y[t == i]
        axs[i].hist(y_ti, bins=range(my+1), edgecolor='black', color=colors[i], density=True, alpha=0.5, label=f't={i}')
        axs[i].legend(loc='upper right')
    plt.savefig(save_prefix + 'y.png')
    plt.close()

# Main function used to create example dataset and then visualize it so one can easily validate correctness
if __name__ == "__main__":
    # Set our params
    n_train = 1000
    n_test = 100
    d = 2
    mx = 100
    mt = 2
    my = 2

    # Get the data
    train_dataset, test_dataset, theta, x_feature_space = get_sim(n_train, n_test, d, mx, mt, my)

    # Unpack the training data
    x, t, y = train_dataset.tensors
    # Plot the data
    example_viz(x, t, y, theta, "train")

    # Unpack the testing data
    x, t, y = test_dataset.tensors
    # Plot the data
    example_viz(x, t, y, theta, "test")