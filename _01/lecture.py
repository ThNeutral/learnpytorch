import torch
from torch import nn
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from pathlib import Path

def execute():
	weight = 0.7
	bias = 0.3

	start = 0
	end = 1
	step = 0.02

	X = torch.arange(start, end, step).unsqueeze(dim=1)
	y = weight * X + bias

	print(X[:10])
	print(y[:10])

	train_split = int(0.8 * len(X)) # 80% of data used for training set, 20% for testing 
	X_train, y_train = X[:train_split], y[:train_split]
	X_test, y_test = X[train_split:], y[train_split:]

	len(X_train), len(y_train), len(X_test), len(y_test)

	def plot_predictions(train_data=X_train, 
                     train_labels=y_train, 
                     test_data=X_test, 
                     test_labels=y_test, 
                     predictions=None):
		"""
		Plots training data, test data and compares predictions.
		"""
		plt.figure(figsize=(10, 7))

		# Plot training data in blue
		plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
		
		# Plot test data in green
		plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

		if predictions is not None:
			# Plot the predictions in red (predictions were made on the test data)
			plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

		# Show the legend
		plt.legend(prop={"size": 14})

		plt.show()

	# plot_predictions(predictions=y_preds)

	class LinearRegressionModel(nn.Module):
		def __init__(self):
			super().__init__()

			self.weights = nn.Parameter(
				torch.randn(1, dtype=torch.float),
				requires_grad=True
			)

			self.bias = nn.Parameter(
				torch.randn(1, dtype=torch.float),
				requires_grad=True
			)

		def forward(self, x: torch.Tensor):
			return self.weights * x + self.bias
		
	torch.manual_seed(42)

	model_00 = LinearRegressionModel()

	for param in model_00.parameters():
		print(param)

	with torch.inference_mode(): 
		y_preds = model_00(X_test)
	
	print(y_preds)

	# plot_predictions(predictions=y_preds)

	loss_fn = nn.L1Loss()
	optimizer = torch.optim.SGD(
		params=model_00.parameters(),
		lr=0.01
	)

	torch.manual_seed(42)

	epochs = 200

	train_loss_values = []
	test_loss_values = []
	epoch_count = []

	# Training
	for epoch in range(epochs):
		model_00.train()

		y_pred = model_00(X_train)

		loss = loss_fn(y_pred, y_train)

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		model_00.eval()

		with torch.inference_mode():
			test_pred = model_00(X_test)

			test_loss = loss_fn(test_pred, y_test.type(torch.float))

			if epoch % 10 == 0:
				epoch_count.append(epoch)
				train_loss_values.append(loss.detach().numpy())
				test_loss_values.append(test_loss.detach().numpy())
				print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

	# Inference
	model_00.eval()

	with torch.inference_mode():
		y_preds = model_00(X_test)
	
	# plot_predictions(predictions=y_preds)
	
	MODEL_PATH = Path("models")
	MODEL_PATH.mkdir(parents=True, exist_ok=True)

	MODEL_NAME = "01_linear_regressor_lecture.pth"
	MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

	print(f"Saving model to {MODEL_SAVE_PATH}")
	torch.save(
		obj=model_00.state_dict(),
		f=MODEL_SAVE_PATH
	)