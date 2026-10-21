"""PCQI: Patch-based Contrast Quality Index (Wang et al. [38], IEEE SPL 2015).

Port of the authors' ``PCQI.m``: grayscale (0-255, rgb2gray weights), 11x11 Gaussian
window with sigma 1.5, 'valid' filtering, C=3, L=256. ``ref`` is the original
(degraded) image and ``enh`` the enhanced one; values above 1 indicate a contrast gain.
"""
import cv2
import numpy as np

_WIN = 11
_PAD = _WIN // 2


def _gray(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float64)
    return 0.2989 * x[..., 0] + 0.5870 * x[..., 1] + 0.1140 * x[..., 2]


def _filt(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    # correlation with a symmetric kernel == MATLAB filter2; crop the border to get 'valid'
    return cv2.filter2D(x, cv2.CV_64F, k, borderType=cv2.BORDER_REFLECT)[_PAD:-_PAD, _PAD:-_PAD]


def pcqi(ref: np.ndarray, enh: np.ndarray, C: float = 3.0, L: float = 256.0) -> float:
    """Mean PCQI between HxWx3 uint8 RGB images ``ref`` (input) and ``enh`` (enhanced)."""
    g = cv2.getGaussianKernel(_WIN, 1.5)
    k = g @ g.T
    k /= k.sum()

    x1, x2 = _gray(ref), _gray(enh)
    mu1, mu2 = _filt(x1, k), _filt(x2, k)
    s1_sq = np.maximum(_filt(x1 * x1, k) - mu1 * mu1, 0.0)
    s2_sq = np.maximum(_filt(x2 * x2, k) - mu2 * mu2, 0.0)
    s12 = _filt(x1 * x2, k) - mu1 * mu2

    m = (4.0 / np.pi) * np.arctan((s12 + C) / (s1_sq + C))
    m = m * ((s12 + C) / (np.sqrt(s1_sq) * np.sqrt(s2_sq) + C))
    m = m * np.exp(-np.abs(mu1 - mu2) / L)
    return float(m.mean())
