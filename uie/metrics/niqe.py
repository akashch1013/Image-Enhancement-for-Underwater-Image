"""NIQE: Natural Image Quality Evaluator (Mittal et al. 2013), lower is better.

Thin wrapper around ``pyiqa``'s NIQE. If pyiqa is not installed or its model
parameters cannot be fetched, ``available`` is False and every call returns NaN.
"""
import numpy as np


class NIQE:
    def __init__(self, device: str = "cpu"):
        self.device = device
        try:
            import pyiqa
            self.m = pyiqa.create_metric("niqe", device=device)
            self.available = True
        except Exception as exc:  # optional dependency
            print(f"[NIQE] unavailable ({exc}); scores will be NaN")
            self.m = None
            self.available = False

    def __call__(self, img: np.ndarray) -> float:
        """NIQE of an HxWx3 uint8 RGB image."""
        if not self.available:
            return float("nan")
        import torch

        t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div(255.0)
        with torch.no_grad():
            return float(self.m(t.to(self.device)).item())
