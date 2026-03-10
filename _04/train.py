import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchmetrics import Accuracy

def train_step(
	model: nn.Module,
	dataloader: DataLoader,
	optimizer: Optimizer,
	loss_fn: nn.Module,
	device: torch.device
):
	model.train()

	train_loss, train_acc = 0, 0

	for batch_idx, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)

		y_pred = model(X)

		loss = loss_fn(y_pred, y)
		train_loss += loss.item()

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
		train_acc += (y_pred_class == y).sum().item()/len(y_pred)

	train_loss = train_loss / len(dataloader)
	train_acc = train_acc / len(dataloader)
	return train_loss, train_acc

def test_step(
	model: nn.Module,
	dataloader: DataLoader,
	loss_fn: nn.Module,
	device: torch.device
):
	model.eval()

	test_loss, test_acc = 0, 0

	with torch.inference_mode():
		for batch_idx, (X, y) in enumerate(dataloader):
			X, y = X.to(device), y.to(device)

			y_pred = model(X)

			loss = loss_fn(y_pred, y)
			test_loss += loss.item()

			y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
			test_acc += (y_pred_class == y).sum().item()/len(y_pred)

	test_loss = test_loss / len(dataloader)
	test_acc = test_acc / len(dataloader)
	return test_loss, test_acc
