import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from _helpers import (
    plot_decision_boundary,
)
import matplotlib.pyplot as plt

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

EPOCHS = 1000
NUMBER_OF_SAMPLES = 1000
NOISE = 0.07
RANDOM_SEED = 42
SPLIT = 0.2
LEARNING_RATE = 0.1

class MoonPredictionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                in_features=2,
                out_features=8,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=8,
                out_features=8
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=8,
                out_features=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

def accuracy_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    if y_pred.size() != y_true.size():
        raise RuntimeError(f"Size mismatch in accuracy function. y_pred: {y_pred.size()}, y_true: {y_true.size()}")
    
    y_pred = y_pred.round()
    correct = y_pred == y_true

    return correct.float().mean().item()

def get_data():
    X, y = make_moons(
        n_samples=NUMBER_OF_SAMPLES,
        shuffle=True,
        noise=NOISE,
    )

    X = torch.from_numpy(X).type(torch.float).to(device)
    y = torch.from_numpy(y).type(torch.float).to(device)

    return train_test_split(
        X, y,
        test_size=SPLIT
    )

def execute():
    X_train,  X_test, y_train, y_test = get_data()
    print(y_train[:10])

    model = MoonPredictionModel().to(device)

    y_train = y_train.unsqueeze(dim=1)
    y_test = y_test.unsqueeze(dim=1)

    y_nonsense = model(X_train)
    print(y_nonsense[:5])

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(1, EPOCHS + 1):
        y_pred = model(X_train)
        
        loss = loss_fn(y_pred, y_train)
        acc = accuracy_fn(y_pred=y_pred, y_true=y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        with torch.inference_mode():
            y_pred_test = model(X_test)

            loss_test = loss_fn(y_pred_test, y_test)
            acc_test = accuracy_fn(y_pred=y_pred_test, y_true=y_test)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss {loss} | Accuracy {acc} | Test Loss {loss_test} | Test Accuracy {acc_test}")

    with torch.inference_mode():
        y_pred = model(X_test)

    print(X_test[:10])
    print(y_pred[:10])

    plt.figure(figsize=(10, 7))

    X_train = X_train.cpu()
    X_test = X_test.cpu()
    y_train = y_train.cpu()
    y_test = y_test.cpu()
    y_pred = y_pred.cpu()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Actual Labels (Ground Truth)
    ax1.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, alpha=0.5, label="Train")
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=20, edgecolors='black', label="Test Actual")
    ax1.set_title("Actual Data")
    ax1.legend()

    # Plot 2: Model Predictions
    ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=10, alpha=0.5, label="Train")
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=20, edgecolors='red', label="Test Predicted")
    ax2.set_title("Model Predictions")
    ax2.legend()

    plt.show()