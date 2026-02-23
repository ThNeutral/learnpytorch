import torch
import matplotlib.pyplot as plt
from pathlib import Path

def generate_data(
		weight: float, 
		bias: float,
		start: float,
		end: float,
		step: float,
		split: float
	) -> torch.Tensor:
	X = torch.arange(start, end, step).unsqueeze(dim=1)
	y = weight * X + bias

	split = int(len(X) * split)
	return X[:split], y[:split], X[split:], y[split:]

def plot_predictions(
		train_data: torch.Tensor, 
		train_labels: torch.Tensor, 
		test_data: torch.Tensor, 
        test_labels: torch.Tensor, 
        predictions: torch.Tensor
		):
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

class LinearRegressorModel(torch.nn.Module):
	def __init__(self):
		super().__init__()

		self.layer = torch.nn.Linear(
			in_features=1,
			out_features=1
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layer(x)
	
	def save_model(self, path: str):
		print(f"Saving model to {path}")
		torch.save(
			obj=self.state_dict(),
			f=path
		)
	
def train_step(
		model: LinearRegressorModel, 
		loss_fn: torch.nn.L1Loss,
		optimizer: torch.optim.SGD,
		X_train: torch.Tensor,
		y_train: torch.Tensor
	):
	model.train()

	y_pred = model(X_train)

	loss = loss_fn(y_pred, y_train)

	optimizer.zero_grad()

	loss.backward()

	optimizer.step()

	return y_pred, loss.detach().numpy()

def test_step(
		model: LinearRegressorModel,
		loss_fn: torch.nn.L1Loss,
		X_test: torch.Tensor,
		y_test: torch.Tensor,
):
	with torch.inference_mode():
		test_pred = model(X_test)

		test_loss = loss_fn(test_pred, y_test.type(torch.float))

		return test_loss.detach().numpy()
			

def execute():
	X_train, y_train, X_test, y_test = generate_data(
		weight=0.3,
		bias=0.9,
		start=0,
		end=2,
		step=0.01,
		split=0.8
	)
	print(X_train.size(), y_train.size(), X_test.size(), y_test.size())
	print(X_train[:10], y_train[:10])

	model = LinearRegressorModel()

	loss_fn = torch.nn.L1Loss()
	optimizer = torch.optim.SGD(
		lr=0.01,
		params=model.parameters()
	)

	epochs = 300

	train_loss_values = []
	test_loss_values = []
	epoch_count = []

	for epoch in range(epochs):
		y_pred, train_loss = train_step(model, loss_fn, optimizer, X_train, y_train)
		test_loss = test_step(model, loss_fn, X_test, y_test)
		if epoch % 20 == 0:
			epoch_count.append(epoch)
			train_loss_values.append(train_loss)
			test_loss_values.append(test_loss)
			print(f"Epoch: {epoch} | MAE Train Loss: {train_loss} | MAE Test Loss: {test_loss} ")

	model.eval()

	with torch.inference_mode():
		y_preds = model(X_test)

	# plot_predictions(
	# 	train_data=X_train.detach().numpy(),
	# 	train_labels=y_train.detach().numpy(),
	# 	test_data=X_test.detach().numpy(),
	# 	test_labels=y_test.detach().numpy(),
	# 	predictions=y_preds.detach().numpy()
	# )

	MODEL_PATH = Path("models")
	MODEL_PATH.mkdir(parents=True, exist_ok=True)

	MODEL_NAME = "01_linear_regressor_exercise.pth"
	MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

	model.save_model(MODEL_SAVE_PATH)