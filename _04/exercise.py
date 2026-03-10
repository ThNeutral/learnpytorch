import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam

from .dataloader import load_train_test_datasets
from .tinyvgg import TinyVGG
from .train import train_step, test_step

EPOCHS = 50

def test_transform():
	return transforms.Compose([
		transforms.Resize(64),
		transforms.ToTensor()
	])

def train_transform():
	return transforms.Compose([
		transforms.Resize(64),
		transforms.ToTensor()
	])

def execute():
	device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu'

	train_dataset, test_dataset = load_train_test_datasets(
		folder="data/pss",
		test_transform=test_transform(),
		train_transform=train_transform()
	)

	classes = train_dataset.classes

	train_dataloader, test_dataloader = DataLoader(train_dataset, batch_size=32), DataLoader(test_dataset, batch_size=32)

	model = TinyVGG(
		input_shape=3,
		hidden_units=10,
		output_shape=len(classes)
	)

	optim = Adam(
		params=model.parameters(),
		lr=0.01
	)
	loss_fn = nn.CrossEntropyLoss()

	for epoch in range(1, EPOCHS + 1):
		print(f"++++++++++ EPOCH {epoch} ++++++++++")

		train_loss, train_acc = train_step(
			model=model,
			optimizer=optim,
			dataloader=train_dataloader,
			loss_fn=loss_fn,
			device=device
		)
		print(f"Train | Loss: {train_loss} | Accuracy: {train_acc}")
		
		test_loss, test_acc = test_step(
			model=model,
			dataloader=train_dataloader,
			loss_fn=loss_fn,
			device=device
		)
		print(f"Test | Loss: {test_loss} | Accuracy: {test_acc}")

		print(f"++++++++++++++++++++++++++++++++")