import torch
from torch import nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

def execute():
	n_samples = 1000

	X, y = make_circles(
		n_samples=1000,
		noise=0.03,
		random_state=42
	)

	print(f"First 5 X features:\n{X[:5]}")
	print(f"\nFirst 5 y labels:\n{y[:5]}")

	# circles = pd.DataFrame({
	# 	'X1': X[:, 0],
	# 	'X2': X[:, 1],
	# 	'label': y
	# })

	# plt.scatter(
	# 	x=X[:, 0],
	# 	y=X[:, 1],
	# 	c=y,
	# 	cmap=plt.cm.RdYlBu
	# )
	# plt.show()

	X = torch.from_numpy(X).type(torch.float)
	y = torch.from_numpy(y).type(torch.float)

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	print(X_train)

	class CircleModelV0(nn.Module):
		def __init__(self):
			super().__init__()

			self.layer_1 = nn.Linear(
				in_features=2,
				out_features=5
			)
			
			self.layer_2 = nn.Linear(
				in_features=5,
				out_features=1
			)

		def forward(self, x: torch.Tensor):
			return self.layer_2(self.layer_1(x))
		
	model_0 = CircleModelV0()

	loss_fn = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.SGD(
		params=model_0.parameters(),
		lr=0.1
	)

	# Calculate accuracy (a classification metric)
	def accuracy_fn(y_true, y_pred):
		correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
		acc = (correct / len(y_pred)) * 100 
		return acc
	
	epochs = 200

	epoch_count = []
	train_loss_values = []
	test_loss_values = []

	for epoch in range(epochs):
		model_0.train()

		y_pred = model_0(X_train)

		loss = loss_fn(y_pred, y_test)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		model_0.eval()

		with torch.inference_mode():
			test_pred = model_0(X_test)

			test_loss = loss_fn(test_pred, y_test.type(torch.float))

			if epoch % 10 == 0:
				epoch_count.append(epoch)
				train_loss_values.append(loss.detach().numpy())
				test_loss_values.append(test_loss.detach().numpy())
				print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

	
