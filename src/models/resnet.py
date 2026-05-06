import torch.nn as nn
from torchvision.models import resnet18


class DroneRFaResNet18(nn.Module):
    """ResNet-18 adapted for DroneRFa spectrogram classification.

    Input: (B, 2, 1024, 1024) — 2-channel spectrograms (RF0, RF1).
    Output: (B, 25) — logits over 25 drone/background classes.
    """

    def __init__(self, num_classes: int = 25):
        super().__init__()
        self.model = resnet18(weights=None)
        # Replace first conv: 3-channel RGB → 2-channel spectrogram
        self.model.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Replace final FC for our 25 classes
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
