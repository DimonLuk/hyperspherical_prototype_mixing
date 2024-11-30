import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock


class SmallResnet(nn.Module):
    def __init__(self):
        super().__init__()

        _network = [
            nn.Conv2d(3, 16, kernel_size=3, padding="same"),
            *[BasicBlock(16, 16) for _ in range(9)],
            BasicBlock(
                16,
                32,
                stride=2,
                downsample=lambda x: F.pad(
                    x[:, :, ::2, ::2], (0, 0, 0, 0, 32 // 4, 32 // 4), "constant", 0
                ),
            ),
            *[BasicBlock(32, 32) for _ in range(8)],
            BasicBlock(
                32,
                64,
                stride=2,
                downsample=lambda x: F.pad(
                    x[:, :, ::2, ::2], (0, 0, 0, 0, 64 // 4, 64 // 4), "constant", 0
                ),
            ),
            *[BasicBlock(64, 64) for _ in range(8)],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(1, -1),
        ]
        self.network = nn.Sequential(*_network)

    def forward(self, x):
        return self.network(x)
