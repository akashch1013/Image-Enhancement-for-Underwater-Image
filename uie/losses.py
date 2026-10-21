"""Training objective (paper Sec. 3.3, Eq. 8):  L = lambda1 * (L1 + L_SSIM) + lambda2 * L_per."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19

_GRAY = (0.299, 0.587, 0.114)


def _gaussian_window(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


class SSIMLoss(nn.Module):
    """1 - mean SSIM on grayscale images, 11x11 Gaussian window (Eqs. 6-7).

    The paper states "c1 = 0.02, c2 = 0.03"; we read these as the usual K1/K2 constants,
    i.e. C1 = (K1 * L)^2, C2 = (K2 * L)^2 with dynamic range L = 1.
    """

    def __init__(self, window: int = 11, sigma: float = 1.5, k1: float = 0.02, k2: float = 0.03):
        super().__init__()
        self.register_buffer("window", _gaussian_window(window, sigma))
        self.register_buffer("gray", torch.tensor(_GRAY).view(1, 3, 1, 1))
        self.c1, self.c2 = k1 ** 2, k2 ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = (pred.float() * self.gray).sum(1, keepdim=True)
        y = (target.float() * self.gray).sum(1, keepdim=True)
        blur = lambda t: F.conv2d(t, self.window)  # noqa: E731  ('valid' filtering, like MATLAB)
        mu_x, mu_y = blur(x), blur(y)
        var_x = blur(x * x) - mu_x ** 2
        var_y = blur(y * y) - mu_y ** 2
        cov = blur(x * y) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + self.c1) * (2 * cov + self.c2)) / (
            (mu_x ** 2 + mu_y ** 2 + self.c1) * (var_x + var_y + self.c2))
        return 1.0 - ssim.mean()


class PerceptualLoss(nn.Module):
    """L1 distance between VGG-19 relu4_3 features of prediction and ground truth (Eq. 5)."""

    RELU4_3 = 25  # vgg19.features[:25] ends with the ReLU after conv4_3

    def __init__(self):
        super().__init__()
        self.features = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[: self.RELU4_3].eval()
        for p in self.features.parameters():
            p.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        f_pred = self.features((pred - self.mean) / self.std)
        with torch.no_grad():
            f_tgt = self.features((target - self.mean) / self.std)
        return F.l1_loss(f_pred.float(), f_tgt.float())


class CombinedLoss(nn.Module):
    def __init__(self, lambda1: float = 1.0, lambda2: float = 0.1):
        super().__init__()
        self.lambda1, self.lambda2 = lambda1, lambda2
        self.ssim = SSIMLoss()
        self.perceptual = PerceptualLoss() if lambda2 > 0 else None

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        l1 = F.l1_loss(pred.float(), target.float())
        l_ssim = self.ssim(pred, target)
        l_per = self.perceptual(pred, target) if self.perceptual is not None else pred.new_zeros(())
        total = self.lambda1 * (l1 + l_ssim) + self.lambda2 * l_per
        # parts stay as tensors so the training loop only syncs with the GPU when it logs
        return total, dict(l1=l1.detach(), ssim=l_ssim.detach(), per=l_per.detach())
