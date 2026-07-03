import torch.nn as nn
from torchvision.models import resnet18

NUM_CLASSES = 14


class DroneRFaResNet18(nn.Module):
    """ResNet-18 adapted for DroneRFa 2-channel input (CPP / STFT).

    Input: (B, 2, H, W) — 2-channel feature maps (RF0, RF1).
    Output: (B, 14) — logits over 14 drone/background classes.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = resnet18(weights=None)
        # Replace first conv: 3-channel RGB → 2-channel stft
        self.model.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Replace final FC for our 14 classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
