"""Contrast enhancement and scene deblurring: MSCNN + CLAHE + Gamma (paper Sec. 3.2, Fig. 4).

    ICE = CLAHE(I1),  IGC = Gamma(I1)
    ZCE = Top(ICE),   ZGC = Bottom(IGC)            # 3x3 -> 5x5 -> 7x7 conv refinement branches
    [WCE, WGC] = Sigmoid(Middle([I1, ICE, IGC]))   # multi-scale confidence branch, split in two
    IOUT = ZCE * WCE + ZGC * WGC                    # Eq. (3)
"""
import torch
import torch.nn as nn
from kornia.enhance import equalize_clahe


def clahe(x: torch.Tensor, clip_limit: float = 2.0, grid: int = 8) -> torch.Tensor:
    """Per-channel CLAHE of a Bx3xHxW image in [0,1].

    Treated as fixed pre-processing (like OpenCV CLAHE), so no gradient flows through it. Histograms
    and LUTs are always computed in fp32, also when the caller runs under mixed-precision autocast.
    """
    with torch.no_grad(), torch.autocast(x.device.type, enabled=False):
        return equalize_clahe(x.detach().float().clamp(0, 1), clip_limit=clip_limit, grid_size=(grid, grid))


def gamma_correct(x: torch.Tensor, gamma: float = 0.7) -> torch.Tensor:
    """Gamma correction balancing non-uniform illumination; gamma < 1 brightens dark regions."""
    return x.clamp(min=1e-6) ** gamma


def _conv_relu(cin: int, cout: int, k: int) -> list:
    return [nn.Conv2d(cin, cout, k, padding=k // 2), nn.ReLU(inplace=True)]


class _RefineBranch(nn.Sequential):
    """Top / bottom branch: 3x3, 5x5, 7x7 convs that remove casts and artefacts of CLAHE / Gamma."""

    def __init__(self, width: int):
        super().__init__(*_conv_relu(3, width, 3), *_conv_relu(width, width, 5), *_conv_relu(width, 3, 7))


class _ConfidenceBranch(nn.Sequential):
    """Middle branch: 3x3, 5x5, 7x7, 1x1, 3x3, 5x5, 7x7 convs, then a 3x3 conv to 6 maps and a Sigmoid."""

    def __init__(self, width: int):
        layers = _conv_relu(9, width, 3) + _conv_relu(width, width, 5) + _conv_relu(width, width, 7)
        layers += _conv_relu(width, width, 1)
        layers += _conv_relu(width, width, 3) + _conv_relu(width, width, 5) + _conv_relu(width, width, 7)
        layers += [nn.Conv2d(width, 6, 3, padding=1), nn.Sigmoid()]
        super().__init__(*layers)


class ContrastEnhancement(nn.Module):
    """The 'Ce' module of the ablation study."""

    def __init__(self, width: int = 32, gamma: float = 0.7, clip_limit: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.clip_limit = clip_limit
        self.top = _RefineBranch(width)
        self.middle = _ConfidenceBranch(width)
        self.bottom = _RefineBranch(width)

    def forward(self, i1: torch.Tensor, return_intermediate: bool = False):
        ice = clahe(i1, self.clip_limit).to(i1.dtype)
        igc = gamma_correct(i1, self.gamma)
        w_ce, w_gc = self.middle(torch.cat([i1, ice, igc], dim=1)).chunk(2, dim=1)
        z_ce, z_gc = self.top(ice), self.bottom(igc)
        out = z_ce * w_ce + z_gc * w_gc
        if return_intermediate:
            return out, dict(ice=ice, igc=igc, wce=w_ce, wgc=w_gc, zce=z_ce, zgc=z_gc)
        return out
