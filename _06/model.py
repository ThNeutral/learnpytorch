import torch
from torch import nn
import torchvision

class TransferLearningModel():
    pass

def get_model(device: torch.device, output_features: int, seed: int = 42) -> nn.Module:
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    for param in model.features.parameters():
        param.requires_grad = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    classifier = nn.Sequential(
        nn.Dropout(
            p=0.2,
            inplace=True
        ),
        nn.Linear(
            in_features=1280,
            out_features=output_features
        )
    ).to(device)

    model.classifier = classifier

    return model

    