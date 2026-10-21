"""Per-channel mean deviation to the ground truth (paper Fig. 1 for raw inputs, Fig. 11 for outputs).

deviation_c = mean(GT_c) - mean(img_c) on the 0-255 scale: positive means channel c is weaker
than in the reference (attenuated, e.g. red), negative means it is over-strong (e.g. green).

    python analyze_channels.py --data data/ugan --split test --tag ugan_test                  # Fig. 1
    python analyze_channels.py --data data/ugan --split test --tag ugan_test \
        --pred-dir runs/outputs/ugan_test/full --label full                                    # Fig. 11
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uie.data import list_pairs, load_rgb

CHANNELS = ("r", "g", "b")
COLORS = {"r": "tab:red", "g": "tab:green", "b": "tab:blue"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/ugan")
    p.add_argument("--split", default="test")
    p.add_argument("--tag", default="ugan_test")
    p.add_argument("--pred-dir", default=None, help="directory of predictions <stem>.png (omit for raw inputs)")
    p.add_argument("--label", default=None, help="name used in outputs (default: input / pred-dir name)")
    p.add_argument("--out", default="results")
    return p.parse_args()


def channel_deviation(data, split, pred_dir=None):
    rows = []
    for name, in_path, tgt_path in list_pairs(data, split):
        if tgt_path is None or not Path(tgt_path).exists():
            continue
        img_path = Path(pred_dir) / f"{Path(name).stem}.png" if pred_dir else Path(in_path)
        if not img_path.exists():
            continue
        # channel means do not depend on resolution, so prediction and GT need not share a size
        dev = load_rgb(tgt_path).reshape(-1, 3).mean(0) - load_rgb(img_path).reshape(-1, 3).mean(0)
        rows.append({"name": Path(name).stem, **{f"dev_{c}": float(d) for c, d in zip(CHANNELS, dev)}})
    return pd.DataFrame(rows)


def plot_curves(df, title, path):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    x = np.arange(len(df))
    for c in CHANNELS:
        col = df[f"dev_{c}"]
        ax.plot(x, col, color=COLORS[c], lw=1.1, label=f"{c.upper()} (mean {col.mean():+.2f})")
    ax.axhline(0, color="0.3", lw=0.8, ls="--")
    ax.set_xlabel("image index")
    ax.set_ylabel("mean(GT) − mean(image)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_summary(out, tag):
    """Grouped bars (mean ± std per channel) over every label analysed for this tag."""
    prefix = f"channel_dev_{tag}_"
    csvs = sorted((out / "metrics").glob(prefix + "*.csv"))
    if len(csvs) < 2:
        return
    data = {p.stem[len(prefix):]: pd.read_csv(p) for p in csvs}
    labels = sorted(data, key=lambda l: (l != "input", l))
    x, width = np.arange(len(labels)), 0.26
    fig, ax = plt.subplots(figsize=(max(4.5, 1.3 * len(labels)), 3.2))
    for j, c in enumerate(CHANNELS):
        means = [data[l][f"dev_{c}"].mean() for l in labels]
        stds = [data[l][f"dev_{c}"].std() for l in labels]
        ax.bar(x + (j - 1) * width, means, width, yerr=stds, color=COLORS[c], label=c.upper(), capsize=2)
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel("mean(GT) − mean(image)")
    ax.set_title(f"Channel deviation to ground truth ({tag})", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "figures" / f"channel_dev_{tag}_summary.png", dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    label = args.label or (Path(args.pred_dir).name if args.pred_dir else "input")
    out = Path(args.out)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    df = channel_deviation(args.data, args.split, args.pred_dir)
    if df.empty:
        raise SystemExit("no image/ground-truth pairs found")
    df.to_csv(out / "metrics" / f"channel_dev_{args.tag}_{label}.csv", index=False)
    plot_curves(df, f"Channel deviation to ground truth: {label} ({args.tag}, N={len(df)})",
                out / "figures" / f"channel_dev_{args.tag}_{label}.png")
    plot_summary(out, args.tag)
    print(f"[channels] {args.tag}/{label}: " +
          ", ".join(f"{c.upper()} {df[f'dev_{c}'].mean():+.2f}±{df[f'dev_{c}'].std():.2f}" for c in CHANNELS))


if __name__ == "__main__":
    main()
