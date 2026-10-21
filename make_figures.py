"""README figures: qualitative comparison grid, training curves and network intermediates.

    python make_figures.py --tag ugan_test --data data/ugan --split test \
        --methods input clahe grayworld ancuti_rc full --runs runs/full runs/wo_gnet \
        --ckpt full=runs/full/ckpt_final.pt
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uie.data import list_pairs, load_rgb

MAX_BYTES = 3 * 1024 * 1024


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-root", default="runs/outputs")
    p.add_argument("--tag", default="ugan_test")
    p.add_argument("--methods", nargs="+", default=["input", "clahe", "grayworld", "ancuti_rc", "full"])
    p.add_argument("--data", default="data/ugan")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--runs", nargs="*", default=[], help="run directories containing log.csv")
    p.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH",
                   help="checkpoint for the intermediates figure (first one is used)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="results/figures")
    return p.parse_args()


def pick(pairs, n):
    """Evenly spaced images, for variety across the split."""
    idx = np.linspace(0, len(pairs) - 1, min(n, len(pairs))).round().astype(int)
    return [pairs[i] for i in dict.fromkeys(idx.tolist())]


def save(fig, path, dpi=110):
    """Save as PNG, lowering dpi until the file fits the size budget."""
    for d in (dpi, 90, 75, 60):
        fig.savefig(path, dpi=d)
        if path.stat().st_size <= MAX_BYTES:
            break
    plt.close(fig)
    print(f"[figures] wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


def _na(ax):
    ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7, transform=ax.transAxes)


def comparison_grid(pairs, methods, pred_dir, out_path):
    has_ref = all(t is not None and Path(t).exists() for _, _, t in pairs)
    cols = list(methods) + (["reference"] if has_ref else [])
    fig, axes = plt.subplots(len(pairs), len(cols), figsize=(1.6 * len(cols), 1.6 * len(pairs) + 0.2),
                             squeeze=False)
    for i, (name, in_path, tgt_path) in enumerate(pairs):
        stem = Path(name).stem
        for j, col in enumerate(cols):
            ax = axes[i][j]
            ax.axis("off")
            if col == "reference":
                path = Path(tgt_path)
            else:
                path = pred_dir / col / f"{stem}.png"
                if col == "input" and not path.exists():
                    path = Path(in_path)
            if path.exists():
                ax.imshow(load_rgb(path))
            else:
                _na(ax)
            if i == 0:
                ax.set_title(col, fontsize=8)
    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
    save(fig, out_path)


def training_curves(run_dirs, out_path):
    logs = [(Path(r).name, Path(r) / "log.csv") for r in run_dirs if (Path(r) / "log.csv").exists()]
    if not logs:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for name, path in logs:
        df = pd.read_csv(path)
        w = max(1, len(df) // 100)
        axes[0].plot(df["iter"], df["loss"].rolling(w, min_periods=1).mean(), lw=1.2, label=name)
    axes[0].set_title("Total loss (smoothed)", fontsize=10)

    name, path = logs[0]
    df = pd.read_csv(path)
    w = max(1, len(df) // 100)
    for comp, label in (("l1", "L1"), ("ssim", "L_SSIM"), ("per", "L_per")):
        if comp in df:
            axes[1].plot(df["iter"], df[comp].rolling(w, min_periods=1).mean(), lw=1.2, label=label)
    axes[1].set_title(f"Loss components ({name})", fontsize=10)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("iteration")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, out_path, dpi=130)


def intermediates(ckpt, pairs, device, out_path):
    """Input, colour-corrected I1, the two traditional enhancements, the fusion weights and the output."""
    import torch
    from uie.models.network import load_model

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    model = load_model(ckpt, device)
    cols = [("input", None), ("I1 (colour corr.)", "i1"), ("I_CE (CLAHE)", "ice"), ("I_GC (gamma)", "igc"),
            ("W_CE (mean)", "wce"), ("W_GC (mean)", "wgc"), ("output", "out")]
    fig, axes = plt.subplots(len(pairs), len(cols), figsize=(1.7 * len(cols), 1.7 * len(pairs) + 0.2),
                             squeeze=False)
    for i, (_, in_path, _) in enumerate(pairs):
        raw = load_rgb(in_path)
        x = torch.from_numpy(raw).permute(2, 0, 1)[None].float().div(255).to(device)
        with torch.no_grad():
            inter = model(x, return_intermediate=True)
        for j, (title, key) in enumerate(cols):
            ax = axes[i][j]
            ax.axis("off")
            if i == 0:
                ax.set_title(title, fontsize=8)
            if key is None:
                ax.imshow(raw)
                continue
            t = inter.get(key)
            if t is None:
                _na(ax)
                continue
            t = t[0].float().clamp(0, 1).cpu()
            if key in ("wce", "wgc"):
                ax.imshow(t.mean(0).numpy(), cmap="viridis", vmin=0, vmax=1)
            else:
                ax.imshow(t.permute(1, 2, 0).numpy())
    fig.tight_layout(pad=0.2, w_pad=0.1, h_pad=0.1)
    save(fig, out_path)


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = pick(list_pairs(args.data, args.split), args.n)
    if pairs:
        comparison_grid(pairs, args.methods, Path(args.pred_root) / args.tag, out / f"comparisons_{args.tag}.png")
    training_curves(args.runs, out / "training_curves.png")
    if args.ckpt and pairs:
        _, path = args.ckpt[0].split("=", 1)
        intermediates(path, pairs[:3], args.device, out / f"intermediates_{args.tag}.png")


if __name__ == "__main__":
    main()
