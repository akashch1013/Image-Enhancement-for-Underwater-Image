"""Full-reference metrics (paper Sec. 4.1.4): PSNR [40] and SSIM [41] on the luminance channel.

Luminance is the Y channel of ITU-R BT.601 YCbCr, computed like MATLAB's ``rgb2ycbcr``
(studio range 16-235, kept as float, no rounding). All inputs are HxWx3 uint8 RGB arrays.
"""
import numpy as np
from skimage.metrics import structural_similarity


def rgb_to_y(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float64)
    return 16.0 + (65.481 * x[..., 0] + 128.553 * x[..., 1] + 24.966 * x[..., 2]) / 255.0


def _psnr(a: np.ndarray, b: np.ndarray, data_range: float = 255.0) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(10.0 * np.log10(data_range ** 2 / mse))


def psnr_y(pred: np.ndarray, gt: np.ndarray) -> float:
    return _psnr(rgb_to_y(pred), rgb_to_y(gt))


def psnr_rgb(pred: np.ndarray, gt: np.ndarray) -> float:
    return _psnr(pred, gt)


def ssim_y(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(structural_similarity(
        rgb_to_y(pred), rgb_to_y(gt),
        gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255.0,
    ))
