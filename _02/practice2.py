import torch
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchmetrics import Accuracy

device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

N_CLASSES = 5
N1_HIDDEN = 32
N2_HIDDEN = 64
EPOCHS = 200

def make_spirals():
    N = 1000 # number of points per class
    D = 2 # dimensionality
    K = N_CLASSES # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    # lets visualize the data
    return X, y
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()

class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(
                in_features=2,
                out_features=N1_HIDDEN,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=N1_HIDDEN,
                out_features=N2_HIDDEN,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=N2_HIDDEN,
                out_features=N2_HIDDEN,
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=N2_HIDDEN,
                out_features=N_CLASSES,
            ),
        )

    def forward(self, x):
        return self.layers(x)

def execute():
    X, y = make_spirals()

    X = torch.from_numpy(X).type(torch.float).to(device)
    y = torch.from_numpy(y).type(torch.long).to(device)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        shuffle=True,
        test_size=0.2
    ) 

    model = Model().to(device)

    accuracy_fn = Accuracy(task='multiclass', num_classes=N_CLASSES).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=0.01
    )

    for epoch in range(1, EPOCHS + 1):
        y_pred_logits: torch.Tensor = model(X_train)
        # print(y_pred_logits[:5])
        y_pred = y_pred_logits.softmax(dim=1).argmax(dim=1).type(torch.float)
        # print(y_pred[:5])
        # print(y_train[:5])

        loss = loss_fn(y_pred_logits, y_train)
        acc = accuracy_fn(y_pred, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        with torch.inference_mode():
            y_pred_test_logits = model(X_test)
            y_pred_test = y_pred_test_logits.softmax(dim=1).argmax(dim=1).type(torch.float)

            loss_test = loss_fn(y_pred_test_logits, y_test)
            acc_test = accuracy_fn(y_pred_test, y_test)

        if epoch % 20 == 0:
            print(f"Epoch: {epoch} | Loss {loss} | Accuracy {acc} | Test Loss {loss_test} | Test Accuracy {acc_test}")

    with torch.inference_mode():
        y_pred_logits = model(X_test)
        y_pred = y_pred_logits.softmax(dim=1).argmax(dim=1)
        print(y_pred[:5])

    X_train = X_train.cpu()
    y_train = y_train.cpu()
    X_test = X_test.cpu()
    y_pred = y_pred.cpu()

    plt.figure()
    plt.scatter(
        x=X_train[:, 0], 
        y=X_train[:, 1],
        c=y_train,
        edgecolors='red'
    )
    plt.scatter(
        x=X_test[:, 0], 
        y=X_test[:, 1],
        c=y_pred,
        edgecolors='black'
    )
    plt.show()