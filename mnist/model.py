import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

def score_fun(y, temperature=0.1):
    """ Scoring function (cosine similarity)."""
    y_norm = F.normalize(y, dim=-1)
    similarity_matrix = torch.matmul(y_norm, y_norm.T) / temperature
    return similarity_matrix
    

def augment(x):
    """
    Returns composition of augmentations for self-supervised learning.
    """
    augmentations = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0), antialias=True),
        transforms.Lambda(lambda x: x.view(x.size(0), -1)),  # Flatten the image while keeping the first dimension
    ])
    return augmentations(x)

def info_nce_loss(model, x, t=None, temperature=0.1, device='cpu'):
    """ Info NCE loss."""
    # Augment each sample twice
    x1 = augment(x)
    x2 = augment(x)
    # Run all augmented samples through the model
    y1 = model(x1)
    y2 = model(x2)

    # Combine y1 and y2
    y = torch.cat([y1, y2], dim=0)
    
    # Compute the scores
    scores = score_fun(y, temperature)

    # Compute labels 
    labels = torch.cat([torch.arange(x.shape[0]) for i in range(2)], dim=0).to(device)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

    # Mask the diagonal elements
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    scores = scores[~mask].view(scores.shape[0], -1)

    # Select the positive and negative samples based on the labels
    positives = scores[labels.bool()].view(labels.shape[0], -1)
    negatives = scores[~labels.bool()].view(scores.shape[0], -1)

    # Concatenate the positives and negatives
    logits = torch.cat([positives, negatives], dim=1)

    # Compute the loss
    return nn.CrossEntropyLoss()(logits, labels)

def task_nce_loss(model, x, t, temperature=0.1, device='cpu'):
    """ Task NCE loss."""
    # Augment each sample twice
    x1 = augment(x)
    x2 = augment(x)
    # Run all augmented samples through the model
    y1 = model(x1, t)
    y2 = model(x2, t)

    # Combine y1 and y2
    y = torch.cat([y1, y2], dim=0)

    # Seperate y1 and y2 based on t 
    t_unique = torch.unique(t)
    twice_t = torch.cat([t, t], dim=0)
    y_t = [y[twice_t == task_id] for task_id in t_unique]
    
    # Compute the scores
    scores = [score_fun(y_t[i], temperature) for i in range(len(t_unique))]

    logits = []
    label_list = []
    for i, s in enumerate(scores):
        # Compute labels 
        labels = torch.arange(y_t[i].shape[0]).to(device)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Mask the diagonal elements
        mask = torch.eye(labels.shape[0], dtype=torch.bool)
        labels = labels[~mask].view(labels.shape[0], -1)
        s = s[~mask].view(s.shape[0], -1)

        # Select the positive and negative samples based on the labels
        positives = s[labels.bool()].view(labels.shape[0], -1)
        negatives = s[~labels.bool()].view(labels.shape[0], -1)

        # Concatenate the positives and negatives
        logits.append(torch.cat([positives, negatives], dim=1))
        label_list.append(labels)

    # Compute the loss
    loss = [nn.CrossEntropyLoss()(logits[i], label_list[i]) for i in range(len(t_unique))]
    return sum(loss)/len(t_unique)

class MLP(nn.Module):
    """ A multi layer perceptron for representation learning. """
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, num_layers=2, num_tasks=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_tasks = num_tasks
        
        # Define the layers
        self.layers = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dim)])
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x, t=None):
        if t is not None:
            # One-hot encode the label
            t = F.one_hot(t, num_classes=self.num_tasks).float()
            # Concatenate the one-hot encoded label
            x = torch.cat((x, t), dim=1)
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        return self.layers[-1](x)

class FunnelMLP(nn.Module):
    """ A multi layer perceptron with a funnel structure (decreasing layer size) for representation learning."""
    def __init__(self, input_size, hidden_sizes):
        super(FunnelMLP, self).__init__()
        layers = []
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LinearClassifier(nn.Module):
    """ Linear layer we will train as a classifier on top of the represertations from the MLP."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Define the layer
        self.layer = nn.Linear(self.input_dim, self.output_dim)
        
    def forward(self, x):
        return self.layer(x)