"""Paired image datasets laid out as <root>/<split>/{input,target}/<name>.png."""
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_rgb(path: str) -> np.ndarray:
    """Read an image as uint8 HxWx3 RGB."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_rgb(path: str, arr: np.ndarray) -> None:
    """Write a uint8 HxWx3 RGB array, creating parent directories."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


def list_images(folder: str) -> list:
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def list_pairs(root: str, split: str) -> list:
    """Return [(name, input_path, target_path)] for files present in both input/ and target/."""
    in_dir = os.path.join(root, split, "input")
    gt_dir = os.path.join(root, split, "target")
    targets = {os.path.splitext(f)[0]: f for f in list_images(gt_dir)} if os.path.isdir(gt_dir) else {}
    pairs = []
    for f in list_images(in_dir):
        stem = os.path.splitext(f)[0]
        if stem in targets:
            pairs.append((stem, os.path.join(in_dir, f), os.path.join(gt_dir, targets[stem])))
    return pairs


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HxWx3 -> float 3xHxW in [0, 1]."""
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div_(255.0)


class PairedFolder(Dataset):
    """Distorted/reference pairs; optional identical random flips for training."""

    def __init__(self, root: str, split: str, augment: bool = False, size: int = None):
        self.pairs = list_pairs(root, split)
        if not self.pairs:
            raise RuntimeError(f"no image pairs found under {os.path.join(root, split)}")
        self.augment = augment
        self.size = size

    def __len__(self) -> int:
        return len(self.pairs)

    def _read(self, path: str) -> np.ndarray:
        img = load_rgb(path)
        if self.size is not None and img.shape[:2] != (self.size, self.size):
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return img

    def __getitem__(self, idx: int) -> dict:
        name, in_path, gt_path = self.pairs[idx]
        inp, gt = self._read(in_path), self._read(gt_path)
        if self.augment:
            if random.random() < 0.5:
                inp, gt = inp[:, ::-1], gt[:, ::-1]
            if random.random() < 0.5:
                inp, gt = inp[::-1], gt[::-1]
        return {"input": to_tensor(inp), "target": to_tensor(gt), "name": name}
