import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader, random_split

def get_mnist(batch_size: int=64, normalize=True, data_root='./data'):
    """Returns the MNIST dataset as a tuple of DataLoaders for training, validation, and testing.
    
    Args:
        batch_size (int): The batch size to use for the DataLoaders.
        normalize (bool): Whether or not to normalize the pixel values.
        data_root (str): The path to the directory where the MNIST dataset will be downloaded.
    """
    # Define the transformations to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,)) if normalize else transforms.Lambda(lambda x: x),  # Normalize pixel values
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)

    # Load the test set
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def get_fashion_mnist(batch_size: int=64, normalize=True, data_root='./data'):
    """Returns the FashionMNIST dataset as a tuple of DataLoaders for training and testing.
    
    Args:
        batch_size (int): The batch size to use for the DataLoaders.
        normalize (bool): Whether or not to normalize the pixel values.
        data_root (str): The path to the directory where the FashionMNIST dataset will be downloaded.
    """
    # Define the transformations to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.5,), (0.5,)) if normalize else transforms.Lambda(lambda x: x),  # Normalize pixel values
    ])

    # Load the FashionMNIST dataset
    train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)

    # Load the test set
    test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)

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

def get_bi_mnist(batch_size: int=64, normalize=True, data_root='./data'):
    """Returns a fused dataset of MNIST and FashionMNIST as a tuple of DataLoaders for training and testing.
    
    Args:
        batch_size (int): The batch size to use for the DataLoaders.
        normalize (bool): Whether or not to normalize the pixel values.
        data_root (str): The path to the directory where the MNIST and FashionMNIST datasets will be downloaded.
    """
    # Get the MNIST and FashionMNIST datasets
    mnist_train, mnist_test = get_mnist(batch_size, normalize, data_root)
    fashion_train, fashion_test = get_fashion_mnist(batch_size, normalize, data_root)

    # Modify y labels for fashion mnist by adding 10
    fashion_train.targets += 10
    fashion_test.targets += 10

    # Wrap the datasets in the custom dataset
    mnist_train = CustomDataset(mnist_train, torch.zeros(len(mnist_train), dtype=torch.long))
    mnist_test = CustomDataset(mnist_test, torch.zeros(len(mnist_test), dtype=torch.long))
    fashion_train = CustomDataset(fashion_train, torch.ones(len(fashion_train), dtype=torch.long))
    fashion_test = CustomDataset(fashion_test, torch.ones(len(fashion_test), dtype=torch.long))

    # Concatenate the datasets
    train_dataset = torch.utils.data.ConcatDataset([mnist_train, fashion_train])
    test_dataset = torch.utils.data.ConcatDataset([mnist_test, fashion_test])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

# Run this code to visualize the data
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Get the dataloaders
    train_loader, test_loader = get_bi_mnist()

    # Get one batch of training data
    dataiter = iter(train_loader)
    images, labels, task_ids = next(dataiter)

    # Separate the images based on task ID
    images_task0 = images[task_ids == 0]
    images_task1 = images[task_ids == 1]

    # Create a grid from the images
    img_grid_task0 = torchvision.utils.make_grid(images_task0)
    img_grid_task1 = torchvision.utils.make_grid(images_task1)

    # Create a figure with two subplots side by side
    fig, axs = plt.subplots(1, 2)

    # Show the images in the first subplot
    axs[0].imshow(img_grid_task0.numpy().transpose((1, 2, 0)))
    axs[0].set_title("Task 0")

    # Show the images in the second subplot
    axs[1].imshow(img_grid_task1.numpy().transpose((1, 2, 0)))
    axs[1].set_title("Task 1")

    # Add a title to the entire figure
    fig.suptitle('Batch from dataloader')

    # Adjust the spacing between the subplots
    plt.subplots_adjust(wspace=0.5)

    # Save the plot
    import os
    if not os.path.exists('viz'):
        os.makedirs('viz')
    plt.savefig("viz/batch_from_dataloader.png")
    plt.close()

    # Print the labels and task IDs
    print("Labels:", labels)
    print("Task IDs:", task_ids)



