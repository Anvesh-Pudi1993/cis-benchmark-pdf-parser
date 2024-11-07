import numpy as np
import torch
import syft as sy
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.ensemble import IsolationForest
from syft.frameworks.torch.fl import utils

# PySyft setup
hook = sy.TorchHook(torch)

# Define clients
clients = [sy.VirtualWorker(hook, id=f"client_{i}") for i in range(3)]

# Transform and load CIFAR-10 dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)

# Split data for each client
datasets = utils.federated(train_loader, *clients)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(6*6*64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Initialize the model
model = CNN()

# Function to train model on each client
def train_on_client(data, model, optimizer):
    model.train()
    for images, labels in data:
        images, labels = images.to(model.conv1.weight.device), labels.to(model.fc2.weight.device)
        optimizer.zero_grad()
        output = model(images)
        loss = nn.functional.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
    return model

# Function to collect and analyze updates for potential poisoning
def detect_poisoning(updates):
    # Use an Isolation Forest to detect anomalies in parameter updates
    flatten_updates = [np.concatenate([param.cpu().detach().numpy().flatten() for param in model_update]) for model_update in updates]
    detector = IsolationForest(contamination=0.1)  # 10% contamination assumed
    is_poisoned = detector.fit_predict(flatten_updates)
    return any(is_poisoned == -1)

# Federated Training Loop
for epoch in range(5):
    optimizers = [optim.SGD(model.parameters(), lr=0.01) for _ in clients]
    updates = []
    
    for client_data, optimizer in zip(datasets.datasets.values(), optimizers):
        # Train model on client
        trained_model = train_on_client(client_data, model, optimizer)
        
        # Capture model parameters after training
        model_update = [param.clone() for param in trained_model.parameters()]
        updates.append(model_update)
    
    # Detect anomalies in updates to check for potential poisoning
    if detect_poisoning(updates):
        print(f"Potential data poisoning detected in epoch {epoch}!")
    else:
        print(f"No data poisoning detected in epoch {epoch}.")
    
    # Average updates and apply to global model
    utils.federated_avg(models=updates, model=model)
