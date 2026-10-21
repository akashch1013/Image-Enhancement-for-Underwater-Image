"""Summary charts for the CA4 report, drawn from results/metrics/*_summary.json.
usage: python report_charts.py <repo_dir>"""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(sys.argv[1])
MET, FIG = REPO / "results" / "metrics", REPO / "results" / "figures"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1f1f1e", "#5b5a55", "#e4e3de"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated, light mode)
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential blue
LABEL = {"input": "Raw input", "clahe": "CLAHE", "gamma": "Gamma", "grayworld": "Gray-world",
         "ancuti_rc": "Ancuti RC + WB", "full": "Proposed", "w_ce": "Full (w/ Ce)", "wo_gnet": "w/o Gnet",
         "wo_rnet": "w/o Rnet", "wo_rnet_gnet": "w/o Rnet+Gnet", "w_rnet_gnet": "w/ Rnet+Gnet (no Ce)"}
ORDER = ["input", "clahe", "gamma", "grayworld", "ancuti_rc", "full"]

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 10,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK,
                     "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(tag):
    f = MET / f"{tag}_summary.json"
    return json.loads(f.read_text()) if f.exists() else {}


def style(ax, title):
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def hbar_panels(series, metrics, titles, methods, out, decimals):
    """One panel per metric (never a dual axis); bars per method, one colour per series."""
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 0.42 * len(methods) + 1.2), sharey=True)
    axes = np.atleast_1d(axes)
    n = len(series)
    h = 0.8 / n
    y = np.arange(len(methods))[::-1]
    for ax, metric, title, d in zip(axes, metrics, titles, decimals):
        for i, (name, summ, color) in enumerate(series):
            vals = [summ.get(m, {}).get(metric, np.nan) for m in methods]
            pos = y + (n - 1 - 2 * i) * h / 2
            ax.barh(pos, vals, height=h * 0.92, color=color, edgecolor=SURFACE, linewidth=1.5, label=name)
            for p, v in zip(pos, vals):
                if np.isfinite(v):
                    ax.text(v, p, f" {v:.{d}f}", va="center", ha="left", fontsize=8, color=INK2)
        finite = [s[1].get(m, {}).get(metric) for s in series for m in methods]
        finite = [v for v in finite if v is not None and np.isfinite(v)]
        ax.set_xlim(0, max(finite) * 1.18 if finite else 1)
        style(ax, title)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABEL.get(m, m) for m in methods])
    if n > 1:
        axes[0].legend(loc="lower center", bbox_to_anchor=(0.5 * len(metrics), 1.12), ncol=n, frameon=False,
                       fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIG / out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ugan, uieb, c60, abl, lam = (load(t) for t in ("ugan_test", "uieb_test", "uieb_c60", "ugan_ablation", "ugan_lambda"))
    hbar_panels([("UGAN test (synthetic)", ugan, BLUE), ("UIEB test (real)", uieb, ORANGE)],
                ["psnr_rgb", "ssim"], ["PSNR-RGB (dB) ↑", "SSIM ↑"], [m for m in ORDER if m in ugan], "report_metrics_bar.png", [2, 3])
    if c60:
        hbar_panels([("UIEB-Challenging-60", c60, BLUE)], ["uiqm", "uciqe"], ["UIQM ↑", "UCIQE ↑"],
                    [m for m in ORDER if m in c60], "report_noref_bar.png", [3, 3])
    if abl:
        order = [m for m in ["wo_rnet_gnet", "wo_rnet", "wo_gnet", "w_rnet_gnet", "w_ce"] if m in abl]
        hbar_panels([("Ablation", abl, BLUE)], ["psnr_rgb", "ssim"], ["PSNR-RGB (dB) ↑", "SSIM ↑"], order,
                    "report_ablation_bar.png", [2, 3])
    if lam:
        grid = {}
        for k, v in lam.items():
            m = re.match(r"l1_([\d.]+)_l2_([\d.]+)", k)
            if m:
                grid[(float(m[1]), float(m[2]))] = v["psnr_rgb"]
        l1s, l2s = sorted({a for a, _ in grid}), sorted({b for _, b in grid})
        z = np.array([[grid.get((a, b), np.nan) for b in l2s] for a in l1s])
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
        im = ax.imshow(z, cmap=cmap)
        for i in range(len(l1s)):
            for j in range(len(l2s)):
                if np.isfinite(z[i, j]):
                    dark = (z[i, j] - np.nanmin(z)) / max(np.nanmax(z) - np.nanmin(z), 1e-9) > 0.55
                    ax.text(j, i, f"{z[i, j]:.2f}", ha="center", va="center", fontsize=10,
                            color="#ffffff" if dark else INK)
        ax.set_xticks(range(len(l2s)), [f"λ2 = {b:g}" for b in l2s])
        ax.set_yticks(range(len(l1s)), [f"λ1 = {a:g}" for a in l1s])
        ax.set_title("PSNR-RGB (dB) on UGAN-100 by loss weights", loc="left", color=INK, fontsize=11, fontweight="bold")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, colors=INK2)
        fig.tight_layout()
        fig.savefig(FIG / "report_lambda_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote report_lambda_heatmap.png")


if __name__ == "__main__":
    main()
