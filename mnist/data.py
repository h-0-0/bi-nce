import torch
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F

def get_mnist(normalize=True, data_root='./data'):
    """Returns the MNIST dataset as a tuple of DataLoaders for training, validation, and testing.
    
    Args:
        normalize (bool): Whether or not to normalize the pixel values.
        data_root (str): The path to the directory where the MNIST dataset will be downloaded.
    """
    # Define the transformations to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1307,), (0.3081,)) if normalize else transforms.Lambda(lambda x: x),  # Normalize pixel values
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)

    # Load the test set
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def get_fashion_mnist(normalize=True, data_root='./data'):
    """Returns the FashionMNIST dataset as a tuple of DataLoaders for training and testing.
    
    Args:
        normalize (bool): Whether or not to normalize the pixel values.
        data_root (str): The path to the directory where the FashionMNIST dataset will be downloaded.
    """
    # Define the transformations to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.2860,), (0.3530,)) if normalize else transforms.Lambda(lambda x: x),  # Normalize pixel values
    ])

    # Load the FashionMNIST dataset
    train_dataset = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)

    # Load the test set
    test_dataset = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)

    return train_dataset, test_dataset

def get_kmnist(normalize=True, data_root='./data'):
    """Returns the KMNIST dataset as a tuple of DataLoaders for training and testing.
    
    Args:
        normalize (bool): Whether or not to normalize the pixel values.
        data_root (str): The path to the directory where the KMNIST dataset will be downloaded.
    """
    # Define the transformations to apply to the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize((0.1918,), (0.3483,)) if normalize else transforms.Lambda(lambda x: x),  # Normalize pixel values
    ])

    # Load the KMNIST dataset
    train_dataset = datasets.KMNIST(root=data_root, train=True, download=True, transform=transform)

    # Load the test set
    test_dataset = datasets.KMNIST(root=data_root, train=False, download=True, transform=transform)

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

def get_bi_mnist(batch_size: int=64, normalize=True, data_root='./data', t_encoding: str='one_hot'):
    """Returns a fused dataset of MNIST and FashionMNIST as a tuple of DataLoaders for training and testing.
    
    Args:
        batch_size (int): The batch size to use for the DataLoaders.
        normalize (bool): Whether or not to normalize the pixel values.
        data_root (str): The path to the directory where the MNIST and FashionMNIST datasets will be downloaded.
        t_encoding (str): How to encode the task variable. Either 'one_hot' or 'repeated'.
    """
    # Get the MNIST and FashionMNIST datasets
    mnist_train, mnist_test = get_mnist(normalize, data_root)
    fashion_train, fashion_test = get_fashion_mnist(normalize, data_root)

    # # Modify y labels for fashion mnist by adding 10
    # fashion_train.targets += 10
    # fashion_test.targets += 10

    # Encode the task variable based on value of t_encoding
    if t_encoding == "one_hot":
        mnist_train_task_ids = F.one_hot(torch.zeros(len(mnist_train), dtype=torch.long), num_classes=2)
        mnist_test_task_ids = F.one_hot(torch.zeros(len(mnist_test), dtype=torch.long), num_classes=2)
        fashion_train_task_ids = F.one_hot(torch.ones(len(fashion_train), dtype=torch.long), num_classes=2)
        fashion_test_task_ids = F.one_hot(torch.ones(len(fashion_test), dtype=torch.long), num_classes=2)
    elif t_encoding == "repeated":
        n_repeat = 250
        mnist_train_task_ids = F.one_hot(torch.zeros(len(mnist_train), dtype=torch.long), num_classes=2).repeat(1, n_repeat)
        mnist_test_task_ids = F.one_hot(torch.zeros(len(mnist_test), dtype=torch.long), num_classes=2).repeat(1, n_repeat)
        fashion_train_task_ids = F.one_hot(torch.ones(len(fashion_train), dtype=torch.long), num_classes=2).repeat(1, n_repeat)
        fashion_test_task_ids = F.one_hot(torch.ones(len(fashion_test), dtype=torch.long), num_classes=2).repeat(1, n_repeat)
    else:
        raise ValueError("t_encoding must be one of 'one_hot' or 'repeated' is {}" % t_encoding)
        
    # Wrap the datasets in the custom dataset
    mnist_train = CustomDataset(mnist_train, mnist_train_task_ids)
    mnist_test = CustomDataset(mnist_test, mnist_test_task_ids)
    fashion_train = CustomDataset(fashion_train, fashion_train_task_ids)
    fashion_test = CustomDataset(fashion_test, fashion_test_task_ids)

    # Concatenate the datasets
    train_dataset = torch.utils.data.ConcatDataset([mnist_train, fashion_train])
    test_dataset = torch.utils.data.ConcatDataset([mnist_test, fashion_test])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

def get_tri_mnist(batch_size: int=64, normalize=True, data_root='./data', t_encoding: str='one_hot'):
    """Returns a fused dataset of MNIST, FashionMNIST, and KMNIST as a tuple of DataLoaders for training and testing.
    
    Args:
        batch_size (int): The batch size to use for the DataLoaders.
        normalize (bool): Whether or not to normalize the pixel values.
        data_root (str): The path to the directory where the MNIST, FashionMNIST, and KMNIST datasets will be downloaded.
        t_encoding (str): How to encode the task variable. Either 'one_hot' or 'repeated'.
    """
    # Get the MNIST, FashionMNIST, and KMNIST datasets
    mnist_train, mnist_test = get_mnist(normalize, data_root)
    fashion_train, fashion_test = get_fashion_mnist(normalize, data_root)
    kmnist_train, kmnist_test = get_kmnist(normalize, data_root)

    # # Modify y labels for fashion mnist by adding 10
    # fashion_train.targets += 10
    # fashion_test.targets += 10

    # # Modify y labels for kmnist by adding 20
    # kmnist_train.targets += 20
    # kmnist_test.targets += 20

    # Encode the task variable based on value of t_encoding
    if t_encoding == "one_hot":
        mnist_train_task_ids = F.one_hot(torch.zeros(len(mnist_train), dtype=torch.long), num_classes=3)
        mnist_test_task_ids = F.one_hot(torch.zeros(len(mnist_test), dtype=torch.long), num_classes=3)
        fashion_train_task_ids = F.one_hot(torch.ones(len(fashion_train), dtype=torch.long), num_classes=3)
        fashion_test_task_ids = F.one_hot(torch.ones(len(fashion_test), dtype=torch.long), num_classes=3)
        kmnist_train_task_ids = F.one_hot(torch.ones(len(kmnist_train), dtype=torch.long)*2, num_classes=3)
        kmnist_test_task_ids = F.one_hot(torch.ones(len(kmnist_test), dtype=torch.long)*2, num_classes=3)
    elif t_encoding == "repeated":
        n_repeat = 200
        mnist_train_task_ids = F.one_hot(torch.zeros(len(mnist_train), dtype=torch.long), num_classes=3).repeat(1, n_repeat)
        mnist_test_task_ids = F.one_hot(torch.zeros(len(mnist_test), dtype=torch.long), num_classes=3).repeat(1, n_repeat)
        fashion_train_task_ids = F.one_hot(torch.ones(len(fashion_train), dtype=torch.long), num_classes=3).repeat(1, n_repeat)
        fashion_test_task_ids = F.one_hot(torch.ones(len(fashion_test), dtype=torch.long), num_classes=3).repeat(1, n_repeat)
        kmnist_train_task_ids = F.one_hot(torch.ones(len(kmnist_train), dtype=torch.long)*2, num_classes=3).repeat(1, n_repeat)
        kmnist_test_task_ids = F.one_hot(torch.ones(len(kmnist_test), dtype=torch.long)*2, num_classes=3).repeat(1, n_repeat)
    else:
        raise ValueError("t_encoding must be one of 'one_hot' or 'repeated' is {}" % t_encoding)
    
    # Wrap the datasets in the custom dataset
    mnist_train = CustomDataset(mnist_train, mnist_train_task_ids)
    mnist_test = CustomDataset(mnist_test, mnist_test_task_ids)
    fashion_train = CustomDataset(fashion_train, fashion_train_task_ids)
    fashion_test = CustomDataset(fashion_test, fashion_test_task_ids)
    kmnist_train = CustomDataset(kmnist_train, kmnist_train_task_ids)
    kmnist_test = CustomDataset(kmnist_test, kmnist_test_task_ids)

    # Concatenate the datasets
    train_dataset = torch.utils.data.ConcatDataset([mnist_train, fashion_train, kmnist_train])
    test_dataset = torch.utils.data.ConcatDataset([mnist_test, fashion_test, kmnist_test])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


# Run this code to visualize the data
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Get the dataloaders
    train_loader, test_loader = get_tri_mnist()

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



