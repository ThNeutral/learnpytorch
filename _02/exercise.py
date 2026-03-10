import torch
from torch import nn
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from _helpers import accuracy_fn, plot_decision_boundary

NUM_SAMPLES = 1000
RANDOM_STATE = 42
NOISE = 0.07
SPLIT = 0.8

EPOCHS = 1000
LEARNING_RATE = 0.1

def generate_data(n_samples: int, noise: float, random_state: float, split: float):
	X_data, y_data = make_moons(
		n_samples=n_samples,
		noise=noise,
		random_state=random_state
	)

	X_data = torch.from_numpy(X_data).type(torch.float)
	y_data = torch.from_numpy(y_data).type(torch.float)

	# plt.figure(figsize=(10, 7))
	# plt.scatter(X_data[:, 0], X_data[:, 1], c=y_data, cmap=plt.cm.RdYlBu)
	# plt.show()

	return train_test_split(
		X_data, y_data,
		random_state=random_state,
		train_size=split,
		shuffle=True
	)

class MoonPredictor(nn.Module):
    def __init__(self, in_features, out_features, hidden_units):
        super().__init__()
        
        self.layer1 = nn.Linear(in_features=in_features, 
                                 out_features=hidden_units)
        self.layer2 = nn.Linear(in_features=hidden_units, 
                                 out_features=hidden_units)
        self.layer3 = nn.Linear(in_features=hidden_units,
                                out_features=out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))

def execute():
	X_train, X_test, y_train, y_test = generate_data(
		n_samples=NUM_SAMPLES,
		noise=NOISE,
		random_state=RANDOM_STATE,
		split=SPLIT
	)

	model = MoonPredictor(
		in_features=2,
		out_features=1,
		hidden_units=10,
	)

	loss_fn = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.SGD(
		params=model.parameters(),
		lr=0.01
	)