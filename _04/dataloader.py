from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import os
import torch

def load_classes(folder: str) -> tuple[list[str], dict[str, int]]:
	classes = sorted(entry.name for entry in os.scandir(folder) if entry.is_dir())
	classes_to_idx = {class_name: index for index, class_name in enumerate(classes)}

	return classes, classes_to_idx

class ImageDataset(Dataset):
	def __init__(self, folder: str, transform = None):
		self.paths = list(Path(folder).glob("*/*.jpg"))
		self.transform = transform
		self.classes, self.classes_to_idx = load_classes(folder)

	def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
		path = self.paths[index]
		image = Image.open(path)

		if self.transform is not None:
			image = self.transform(image)

		class_name = self.paths[index].parent.name
		class_idx = self.classes_to_idx[class_name]

		return image, class_idx
	
	def __len__(self) -> int:
		return len(self.paths)
	
def load_train_test_datasets(
		folder: str,
		test_transform = None,
		train_transform = None
	) -> tuple[ImageDataset, ImageDataset]:
	
	train_dataset = ImageDataset(f"{folder}/train", transform=train_transform)
	test_dataset = ImageDataset(f"{folder}/test", transform=test_transform)

	return train_dataset, test_dataset 
	
def execute():
	train_dataset, test_dataset = load_train_test_datasets("data/pss")
	print(len(train_dataset))
	print(len(test_dataset))