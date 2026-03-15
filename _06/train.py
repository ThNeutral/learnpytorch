import torch
from torch import nn
from torchvision import transforms
from torchsummary import summary
from matplotlib import pyplot as plt

from _05 import dataloaders, engine
from . import model

def execute():
    device = torch.accelerator.current_accelerator() or 'cpu'

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()
    ])

    train_dataloader, test_dataloader, class_names = dataloaders.create_dataloaders(
        train_dir="data/pizza_steak_sushi/train",
        test_dir="data/pizza_steak_sushi/test",
        transform=transform,
        batch_size=32,
        num_workers=0
    )

    model_0 = model.get_model(
        device=device,
        output_features=len(class_names)
    )

    summary(
        model=model_0, 
        input_size=(3, 224, 224)    
    )

    optimizer = torch.optim.Adam(
        lr=0.01,
        params=model_0.parameters()
    )
    loss_fn = nn.CrossEntropyLoss()

    EPOCHS = 35
    epochs = [i for i in range(EPOCHS)]
    result = engine.train(
        model=model_0,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=device
    )

    train_loss = result['train_loss']
    test_loss = result['test_loss']

    train_acc = result['train_acc']
    test_acc = result['test_acc']

    print(f"Train | Loss: {result['train_loss']} | Accuracy: {result['train_acc']}")
    print(f"Test | Loss: {result['test_loss']} | Accuracy: {result['test_acc']}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, label='Train Loss', marker='o', color='blue')
    ax1.plot(epochs, test_loss, label='Test Loss', marker='x', color='red', linestyle='--')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')

    ax2.plot(epochs, train_acc, label='Train Accuracy', marker='o', color='green')
    ax2.plot(epochs, test_acc, label='Test Accuracy', marker='x', color='orange', linestyle='--')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')

    plt.show()

if __name__ == "__main__":
    execute()
