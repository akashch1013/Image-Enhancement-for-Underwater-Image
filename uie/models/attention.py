"""Attention blocks used by the colour-compensation modules (paper Sec. 3.1, Fig. 3)."""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """CAB: GAP -> FC -> Sigmoid, producing a 1x1xC channel weight (SE-style, Hu et al. [21])."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        return self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """SAB: two 3x3 convs + Sigmoid, producing an HxWx1 spatial weight map."""

    def __init__(self, in_channels: int = 1, hidden: int = 16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
