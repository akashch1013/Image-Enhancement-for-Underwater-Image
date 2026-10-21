"""UIQM: Underwater Image Quality Measure (Panetta et al. [34], IEEE JOE 2016).

UIQM = c1*UICM + c2*UISM + c3*UIConM with c1=0.0282, c2=0.2953, c3=3.5753 (the
original values; the Shi et al. text prints c3 as 3.5735, a typo). Conventions follow
the widely used Python port (e.g. FUnIE-GAN's ``uqim_utils.py``): alpha-trimmed
statistics with alpha=0.1, Sobel edge maps, 10x10 blocks. Block operations are
vectorised by cropping to a multiple of the block size and reshaping.
"""
import numpy as np
from scipy import ndimage

C1, C2, C3 = 0.0282, 0.2953, 3.5753
BLOCK = 10


def _trimmed_mean(x: np.ndarray, alpha_l: float = 0.1, alpha_r: float = 0.1) -> float:
    x = np.sort(x)
    k = x.size
    t_l, t_r = int(np.ceil(alpha_l * k)), int(np.floor(alpha_r * k))
    # same slice bounds as the reference port (it skips index t_l)
    return float(x[t_l + 1:k - t_r].sum() / (k - t_l - t_r))


def uicm(img: np.ndarray) -> float:
    """Colourfulness from opponent channels RG and YB."""
    x = img.astype(np.float64)
    rg = (x[..., 0] - x[..., 1]).ravel()
    yb = ((x[..., 0] + x[..., 1]) / 2.0 - x[..., 2]).ravel()
    mu_rg, mu_yb = _trimmed_mean(rg), _trimmed_mean(yb)
    s_rg, s_yb = np.mean((rg - mu_rg) ** 2), np.mean((yb - mu_yb) ** 2)
    return float(-0.0268 * np.hypot(mu_rg, mu_yb) + 0.1586 * np.sqrt(s_rg + s_yb))


def _blocks(x: np.ndarray, ws: int) -> np.ndarray:
    """Split HxW[xC] into (k2, k1, ws*ws[*C]) blocks, cropping the remainder."""
    k2, k1 = x.shape[0] // ws, x.shape[1] // ws
    x = x[:k2 * ws, :k1 * ws]
    x = x.reshape(k2, ws, k1, ws, *x.shape[2:]).swapaxes(1, 2)
    return x.reshape(k2, k1, -1)


def _sobel(x: np.ndarray) -> np.ndarray:
    mag = np.hypot(ndimage.sobel(x, 0), ndimage.sobel(x, 1))
    peak = mag.max()
    return mag * (255.0 / peak) if peak > 0 else mag


def _eme(x: np.ndarray, ws: int = BLOCK) -> float:
    b = _blocks(x, ws)
    mx, mn = b.max(-1), b.min(-1)
    valid = (mn > 0) & (mx > 0)
    logs = np.zeros_like(mx)
    logs[valid] = np.log(mx[valid] / mn[valid])
    return float(2.0 / (b.shape[0] * b.shape[1]) * logs.sum())


def uism(img: np.ndarray) -> float:
    """Sharpness: EME of Sobel edge maps weighted by the channel itself."""
    x = img.astype(np.float64)
    emes = [_eme(_sobel(x[..., c]) * x[..., c]) for c in range(3)]
    return float(0.299 * emes[0] + 0.587 * emes[1] + 0.114 * emes[2])


def uiconm(img: np.ndarray, ws: int = BLOCK) -> float:
    """Contrast: logAMEE over blocks spanning all three channels."""
    b = _blocks(img.astype(np.float64), ws)
    mx, mn = b.max(-1), b.min(-1)
    top, bot = mx - mn, mx + mn
    valid = (top > 0) & (bot > 0)
    ratio = np.zeros_like(top)
    ratio[valid] = top[valid] / bot[valid]
    terms = np.zeros_like(top)
    terms[valid] = ratio[valid] * np.log(ratio[valid])
    return float(-1.0 / (b.shape[0] * b.shape[1]) * terms.sum())


def uiqm(img: np.ndarray) -> float:
    """UIQM of an HxWx3 uint8 RGB image (higher is better)."""
    return float(C1 * uicm(img) + C2 * uism(img) + C3 * uiconm(img))
