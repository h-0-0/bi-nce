import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from avalanche.benchmarks.classic import PermutedMNIST
import numpy as np

def get_one_permuted_mnist(rndm_state, normalize=True, data_root='./data'):
    """Returns the a Permuted MNIST dataset as a tuple of DataLoaders for training, validation, and testing.
    
    Args:
        normalize (bool): Whether or not to normalize the pixel values.
        data_root (str): The path to the directory where the Permuted MNIST dataset will be downloaded.
    """
    # Define the transformations to apply to the data

    rng_permute = np.random.RandomState(rndm_state)
    idx_permute = torch.from_numpy(rng_permute.permutation(784)).to(torch.int64)
    permute_transform = torchvision.transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28))

    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,)) if normalize else transforms.Lambda(lambda x: x),  # Normalize pixel values
        permute_transform
    ])

    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    return train_dataset, test_dataset
            

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, task_id):
        self.dataset = dataset
        self.task_id = task_id

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, self.task_id[index]

    def __len__(self):
        return len(self.dataset)

def get_permuted_mnist(batch_size: int=64, normalize=True, data_root='./data', t_encoding: str='one_hot'):
    """Returns the Permuted MNIST dataset as a tuple of DataLoaders for training and testing.
    
    Args:
        batch_size (int): The batch size to use for the DataLoaders.
        normalize (bool): Whether or not to normalize the pixel values.
        data_root (str): The path to the directory where the MNIST and FashionMNIST datasets will be downloaded.
        t_encoding (str): How to encode the task variable. Either 'one_hot' or 'repeated'.
    """
    permutation_rndm_states = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    train_datasets = []
    test_datasets = []
    for i, rndm_state in enumerate(permutation_rndm_states):
        if t_encoding == 'one_hot':
            train_task_ids =  F.one_hot(torch.ones(60000, dtype=torch.long)*i, num_classes=len(permutation_rndm_states))
            test_task_ids =  F.one_hot(torch.ones(10000, dtype=torch.long)*i, num_classes=len(permutation_rndm_states))
        elif t_encoding == 'repeated':
            n_repeat = 20
            train_task_ids =  F.one_hot(torch.ones(60000, dtype=torch.long)*i, num_classes=len(permutation_rndm_states)).repeat(1, n_repeat)
            test_task_ids =  F.one_hot(torch.ones(10000, dtype=torch.long)*i, num_classes=len(permutation_rndm_states)).repeat(1, n_repeat)

        train_dataset, test_dataset = get_one_permuted_mnist(rndm_state, normalize, data_root)

        train_datasets.append(CustomDataset(train_dataset, train_task_ids))
        test_datasets.append(CustomDataset(test_dataset, test_task_ids))

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Run this code to visualize the data
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Get the DataLoaders
    train_loader, test_loader = get_permuted_mnist(t_encoding='one_hot')

    # Get one batch of training images
    images, labels, task_ids = next(iter(train_loader))

    unique_task_ids = torch.unique(task_ids)

    for task_id in unique_task_ids:
        task_images = images[task_ids == task_id]

        # Create a grid from the images
        img_grid = torchvision.utils.make_grid(task_images[:16], nrow=4)

        # Convert the image grid to a numpy array and plot it
        img_grid_np = img_grid.numpy().transpose((1, 2, 0))
        plt.figure(figsize=(10, 10))
        plt.imshow(img_grid_np)
        plt.axis('off')

        # Save the plot
        import os
        if not os.path.exists('viz'):
            os.makedirs('viz')
        plt.savefig(f'viz/permuted_mnist_task_{task_id}.png')


