import torch
from torch import nn
from torchsummary import summary

class TinyVGG(nn.Module):
	def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
		super().__init__()

		self.layer1 = nn.Sequential(
			nn.Conv2d(
				in_channels=input_shape,
				out_channels=hidden_units,
				kernel_size=3,
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=hidden_units,
				out_channels=hidden_units,
				kernel_size=3
			),
			nn.ReLU(),
			nn.MaxPool2d(
				kernel_size=2,
				stride=2
			)
		)

		self.layer2 = nn.Sequential(
			nn.Conv2d(
				in_channels=hidden_units,
				out_channels=hidden_units,
				kernel_size=3,
			),
			nn.ReLU(),
			nn.Conv2d(
				in_channels=hidden_units,
				out_channels=hidden_units,
				kernel_size=3
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
				in_features=hidden_units * 16 * 16,
				out_features=output_shape
			),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.classifier(self.layer2(self.layer1(x)))

def execute():
	model = TinyVGG(
		input_shape=3,
		hidden_units=10,
		output_shape=3
	)
