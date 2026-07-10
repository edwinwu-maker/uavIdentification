import torch.nn as nn
from torchvision.models import resnet18

NUM_CLASSES = 14


class DroneRFaResNet18(nn.Module):
    """ResNet-18 adapted for DroneRFa single-channel input (CPP / STFT).

    Input: (B, 1, H, W) — feature map from the selected RF channel.
    Output: (B, 14) — logits over 14 drone/background classes.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = resnet18(weights=None)
        # Replace first conv: 3-channel RGB → single-channel feature
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Replace final FC for our 14 classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class DroneRFaResNet18SmallStem(nn.Module):
    """ResNet-18 with a small stem for lower-resolution CPP/STFT feature maps."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = resnet18(weights=None)
        # CPP 小图保留更多早期空间细节：3x3/stride=1，并移除 maxpool。
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def build_model(model_name: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    if model_name == "resnet18":
        return DroneRFaResNet18(num_classes=num_classes)
    if model_name == "resnet18-small-stem":
        return DroneRFaResNet18SmallStem(num_classes=num_classes)
    raise ValueError(f"Unknown model: {model_name}")
