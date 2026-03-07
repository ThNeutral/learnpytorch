from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt 
import torch
from torch import nn
import torchmetrics 
from timeit import default_timer
from torchmetrics import ConfusionMatrix
from mlxtend import plotting

def get_dataloaders():
    train = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    
    train_dataloader = DataLoader(
        dataset=train,
        batch_size=32,
        shuffle=True
    )

    test = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )

    test_dataloader = DataLoader(
        dataset=test,
        batch_size=32,
        shuffle=True
    )

    return train_dataloader, test_dataloader

def train_step(
    model: nn.Module, 
    dataloader: DataLoader, 
    loss_fn: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    accuracy_fn: torchmetrics.Metric,
    device: torch.device
):
    model.train()
    train_loss, train_acc = 0.0, 0.0

    accuracy_fn = accuracy_fn.to(device)
    accuracy_fn.reset()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate accuracy
        accuracy_fn.update(y_pred, y)

    total_loss = train_loss / len(dataloader)
    total_acc = accuracy_fn.compute().item()
    
    return total_loss, total_acc


def test_step(
    model: nn.Module, 
    dataloader: DataLoader, 
    loss_fn: nn.Module, 
    accuracy_fn: torchmetrics.Metric,
    device: torch.device
):
    model.eval()
    test_loss = 0.0
    
    accuracy_fn = accuracy_fn.to(device)
    accuracy_fn.reset()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            # 3. Update accuracy state
            accuracy_fn.update(y_pred, y)

    total_loss = test_loss / len(dataloader)
    total_acc = accuracy_fn.compute().item()
    
    return total_loss, total_acc

def plot_confusion_matrix(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    # 1. Collect all predictions and labels across the entire dataset
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            logits = model(X)
            # Convert logits -> prediction labels
            preds = torch.argmax(logits, dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    # 2. Concatenate list of tensors into single tensors
    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)

    # 3. Setup ConfusionMatrix and update with full data
    confmat = ConfusionMatrix(num_classes=10, task='multiclass')
    confmat_tensor = confmat(preds=all_preds_tensor, target=all_labels_tensor)

    # 4. Plot using mlxtend
    fig, ax = plotting.plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        class_names=[str(i) for i in range(10)], # Labels 0-9 as strings
        figsize=(10, 7)
    )
    plt.show()

class TinyVGG(nn.Module):
    def __init__(
            self, 
            input_shape: int, 
            hidden_units: int, 
            output_shape: int
        ) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=hidden_units*7*7,
                out_features=output_shape
            )
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

def execute():
    train_dataloader, test_dataloader = get_dataloaders()
    loss_fn = nn.CrossEntropyLoss()
    accuracy_fn = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    EPOCHS = 5

    device_cpu = torch.device('cpu')
    model_cpu = TinyVGG(input_shape=1, hidden_units=10, output_shape=10).to(device_cpu)
    optimizer_cpu = torch.optim.Adam(params=model_cpu.parameters(), lr=0.01)

    print(f"\n[INFO] Starting training on {device_cpu}...")
    start_time_cpu = default_timer()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_step(model_cpu, train_dataloader, loss_fn, optimizer_cpu, accuracy_fn, device_cpu)
        test_loss, test_acc = test_step(model_cpu, test_dataloader, loss_fn, accuracy_fn, device_cpu)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    end_time_cpu = default_timer()
    total_cpu_time = end_time_cpu - start_time_cpu

    print("\n" + "="*30)
    print(f"CPU Total Training Time: {total_cpu_time:.3f} seconds")
    print("="*30)

    plot_confusion_matrix(model_cpu, test_dataloader, device_cpu)

    device_gpu = torch.device("cuda")
    model_gpu = TinyVGG(input_shape=1, hidden_units=10, output_shape=10).to(device_gpu)
    optimizer_gpu = torch.optim.Adam(params=model_gpu.parameters(), lr=0.01)

    print(f"\n[INFO] Starting training on {device_gpu}...")
    start_time_gpu = default_timer()
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_step(model_gpu, train_dataloader, loss_fn, optimizer_gpu, accuracy_fn, device_gpu)
        test_loss, test_acc = test_step(model_gpu, test_dataloader, loss_fn, accuracy_fn, device_gpu)
        print(f"Epoch: {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    end_time_gpu = default_timer()
    total_gpu_time = end_time_gpu - start_time_gpu

    print("\n" + "="*30)
    print(f"GPU Total Training Time: {total_gpu_time:.3f} seconds")
    print("="*30)

    plot_confusion_matrix(model_cpu, test_dataloader, device_cpu)