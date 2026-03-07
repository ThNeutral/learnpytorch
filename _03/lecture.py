from typing import Any
import torch
import torchvision
import torchmetrics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm

def part1():
    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu" 

    BATCH_SIZE = 32

    def train_step(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                accuracy_fn,
                device: torch.device = device):
        train_loss, train_acc = 0, 0
        model.to(device)
        for batch, (X, y) in enumerate(data_loader):
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += accuracy_fn(y_pred, y)

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

        # Calculate loss and accuracy per epoch and print out what's happening
        train_loss /= len(data_loader)
        train_acc /= len(data_loader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

    def test_step(data_loader: torch.utils.data.DataLoader,
                model: torch.nn.Module,
                loss_fn: torch.nn.Module,
                accuracy_fn,
                device: torch.device = device):
        test_loss, test_acc = 0, 0
        model.to(device)
        model.eval() # put model in eval mode
        # Turn on inference context manager
        with torch.inference_mode(): 
            for X, y in data_loader:
                # Send data to GPU
                X, y = X.to(device), y.to(device)
                
                # 1. Forward pass
                test_pred = model(X)
                
                # 2. Calculate loss and accuracy
                test_loss += loss_fn(test_pred, y)
                test_acc += accuracy_fn(test_pred, y)
            
            # Adjust metrics and print out
            test_loss /= len(data_loader)
            test_acc /= len(data_loader)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

    class Model0(nn.Module):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)

            hidden_layer_width = 10

            self.layers = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    in_features=28 * 28,
                    out_features=hidden_layer_width,
                ),
                nn.ReLU(),
                nn.Linear(
                    in_features=hidden_layer_width,
                    out_features=10,
                ),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.layers(x)

    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
        target_transform=None,
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
        target_transform=None,
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    test_dataloader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    model_0 = Model0().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model_0.parameters(),
        lr=0.01
    )
    accuracy_fn = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(device)

    epochs = 3
    for epoch in tqdm(range(epochs)):
        print(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_dataloader, 
            model=model_0, 
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn
        )
        test_step(data_loader=test_dataloader,
            model=model_0,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn
        )

def execute():
    part1()