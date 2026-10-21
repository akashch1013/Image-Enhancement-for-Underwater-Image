"""Attention-guided residual colour compensation (paper Sec. 3.1, Figs. 2-3).

Rnet (Eq. 1):  Irc = Ir + CAB(Ir) * SAB * (1 - Ir)
Gnet (Eq. 2):  Iga = Ig - CAB(Ig) * SAB
Bnet:          residual enhancement of the blue channel (Fig. 2, middle row)

A single-channel input gives CAB nothing to choose between, so each module first
lifts its channel to `width` feature maps with a 3x3 conv; CAB acts on those
features and a final 3x3 conv maps the compensation term back to one channel.
"""
import torch
import torch.nn as nn

from .attention import ChannelAttention, SpatialAttention


class _AttentionCompensation(nn.Module):
    """Shared body of Fig. 3(a)/(b): returns the compensation term CAB(I) * SAB [* (1 - I)]."""

    def __init__(self, width: int = 32, use_inverse: bool = False):
        super().__init__()
        self.use_inverse = use_inverse
        self.head = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.cab = ChannelAttention(width)
        self.fuse = nn.Conv2d(width, width, 1)  # the 1x1 conv after the channel product
        self.sab = SpatialAttention(1)
        self.tail = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        feat = feat * self.cab(feat)
        if self.use_inverse:
            # (1 - Ir) keeps already-bright red pixels from being over-assigned (Ancuti et al. [22])
            feat = feat * (1.0 - x)
        feat = self.fuse(feat) * self.sab(x)
        return self.tail(feat)


class RedCompensation(nn.Module):
    """Rnet: the red channel is attenuated, so a compensation term is added (Eq. 1)."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.comp = _AttentionCompensation(width, use_inverse=True)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return r + self.comp(r)


class GreenCompensation(nn.Module):
    """Gnet: the green channel is over-strengthened, so part of it is subtracted (Eq. 2)."""

    def __init__(self, width: int = 32):
        super().__init__()
        self.comp = _AttentionCompensation(width, use_inverse=False)

    def forward(self, g: torch.Tensor) -> torch.Tensor:
        return g - self.comp(g)


class _ResBlock(nn.Module):
    """conv - LReLU - conv - LReLU - conv + skip (Fig. 2, residual enhancement module)."""

    def __init__(self, width: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class BlueResidualEnhancement(nn.Module):
    """Brightness enhancement of the (well preserved) blue channel with stacked residual blocks."""

    def __init__(self, width: int = 32, n_blocks: int = 3):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.LeakyReLU(0.2, inplace=True))
        self.blocks = nn.Sequential(*[_ResBlock(width) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        return b + self.tail(self.blocks(self.head(b)))
