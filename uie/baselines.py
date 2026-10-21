"""Classical (non-learned) reference methods for comparison.

Each baseline maps an HxWx3 uint8 RGB image to an HxWx3 uint8 RGB image. The CLAHE and
gamma baselines are the two hand-crafted operators the paper feeds into its MSCNN
(Sec. 3.2); ``ancuti_rc`` is the red-channel compensation of Ancuti et al. [22] that
motivates Eq. (1).
"""
import cv2
import numpy as np


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)


def clahe(img: np.ndarray, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    lab[..., 0] = op.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def gamma(img: np.ndarray, g: float = 0.7) -> np.ndarray:
    return _to_uint8((img.astype(np.float64) / 255.0) ** g)


def _grayworld(x: np.ndarray) -> np.ndarray:
    """Gray-world white balance on a float [0,1] image."""
    means = x.reshape(-1, 3).mean(0)
    return np.clip(x * (means.mean() / np.maximum(means, 1e-6)), 0.0, 1.0)


def grayworld(img: np.ndarray) -> np.ndarray:
    return _to_uint8(_grayworld(img.astype(np.float64) / 255.0))


def ancuti_rc(img: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Irc = Ir + alpha * (mean(Ig) - mean(Ir)) * (1 - Ir) * Ig, followed by gray-world."""
    x = img.astype(np.float64) / 255.0
    r, g = x[..., 0], x[..., 1]
    x[..., 0] = np.clip(r + alpha * (g.mean() - r.mean()) * (1.0 - r) * g, 0.0, 1.0)
    return _to_uint8(_grayworld(x))


BASELINES = {
    "clahe": clahe,
    "gamma": gamma,
    "grayworld": grayworld,
    "ancuti_rc": ancuti_rc,
}
