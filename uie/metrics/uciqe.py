"""UCIQE: Underwater Colour Image Quality Evaluation (Yang & Sowmya [35], IEEE TIP 2015).

UCIQE = 0.4680*sigma_c + 0.2745*con_l + 0.2576*mu_s, computed MATLAB-style on the
uint8-encoded CIELab image scaled by 1/255 (L in [0,1], a/b offset by 128/255).
"""
import cv2
import numpy as np

C1, C2, C3 = 0.4680, 0.2745, 0.2576


def uciqe(img: np.ndarray) -> float:
    """UCIQE of an HxWx3 uint8 RGB image (higher is better)."""
    lab = cv2.cvtColor(np.ascontiguousarray(img, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64) / 255.0
    l, a, b = lab[..., 0].ravel(), lab[..., 1].ravel(), lab[..., 2].ravel()
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = chroma.std()

    l_sorted = np.sort(l)
    n = max(int(round(0.01 * l_sorted.size)), 1)
    con_l = l_sorted[-n:].mean() - l_sorted[:n].mean()

    nz = l > 0
    mu_s = (chroma[nz] / l[nz]).mean() if nz.any() else 0.0
    return float(C1 * sigma_c + C2 * con_l + C3 * mu_s)
