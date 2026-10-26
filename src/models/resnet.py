import torch.nn as nn
from torchvision.models import resnet18

from src.data.drone_rfa_io import RESEARCH_LABELS

NUM_CLASSES = len(RESEARCH_LABELS)


class DroneRFaResNet18(nn.Module):
    """ResNet-18 adapted for DroneRFa single-channel input (CPP / STFT).

    Input: (B, 1, H, W) — feature map from the selected RF channel.
    Output: (B, num_classes) — 默认 16 类：背景与 15 类无人机。
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.model = resnet18(weights=None)
        # Replace first conv: 3-channel RGB → single-channel feature
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # 按当前研究类别数设置分类层。
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
