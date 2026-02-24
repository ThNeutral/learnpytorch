import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from _helpers import accuracy_fn, plot_decision_boundary

NUM_SAMPLES = 1000
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42

def generate_data():
	X_blob, y_blob = make_blobs(
		n_samples=NUM_SAMPLES,
		n_features=NUM_FEATURES,
		centers=NUM_CLASSES,
		cluster_std=1.5,
		random_state=RANDOM_SEED
	)

	X_blob = torch.from_numpy(X_blob).type(torch.float)
	y_blob = torch.from_numpy(y_blob).type(torch.long)

	# plt.figure(figsize=(10, 7))
	# plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
	# plt.show()

	return train_test_split(
		X_blob, y_blob,
		test_size=0.2, 
		random_state=RANDOM_SEED
	)

class MultiClassClassifier(nn.Module):
	def __init__(self, input_features: int, output_features: int, hidden_units: int = 8):
		super().__init__()

		self.linear_layer_stack = nn.Sequential(
			nn.Linear(
				in_features=input_features,
				out_features=hidden_units
			),
			nn.Linear(
				in_features=hidden_units,
				out_features=hidden_units
			),
			nn.Linear(
				in_features=hidden_units,
				out_features=output_features
			)
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.linear_layer_stack(x)

def execute():
	X_blob_train, X_blob_test, y_blob_train, y_blob_test = generate_data()

	model = MultiClassClassifier(
		input_features=NUM_FEATURES,
		output_features=NUM_CLASSES,
		hidden_units=8
	)

	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.SGD(
		params=model.parameters(),
		lr=0.1
	)

	torch.manual_seed(RANDOM_SEED)

	epochs = 100

	for epoch in range(epochs):
		model.train()

		y_logits = model(X_blob_train)
		y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)

		loss = loss_fn(y_logits, y_blob_train)
		acc = accuracy_fn(
			y_true=y_blob_train,
			y_pred=y_pred
		)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		model.eval()
		with torch.inference_mode():
			test_logits = model(X_blob_test)
			test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)

			test_loss = loss_fn(test_logits, y_blob_test)
			test_acc = accuracy_fn(
				y_true=y_blob_test,
				y_pred=test_pred
			)

		if epoch % 10 == 0:
			print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test Loss: {test_loss:.5f}, Test Acc: {test_acc:.2f}%")

	with torch.inference_mode():
		y_pred = model(X_blob_test)

	plot_decision_boundary(model, X_blob_test, y_blob_test)
	plt.show()