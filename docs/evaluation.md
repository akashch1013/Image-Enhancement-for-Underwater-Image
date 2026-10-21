# Evaluation & metrics

## Metrics

| Metric | Type | Better | Notes |
|---|---|---|---|
| PSNR-RGB | full reference | ↑ | all three channels; matches the paper's reported values |
| PSNR-Y | full reference | ↑ | luminance (BT.601), which is what the paper's text describes |
| SSIM | full reference | ↑ | luminance, Gaussian window σ = 1.5 |
| UIQM | no reference | ↑ | colourfulness + sharpness + contrast, weights 0.0282 / 0.2953 / 3.5753 |
| UCIQE | no reference | ↑ | chroma, saturation and contrast in CIELab, weights 0.4680 / 0.2745 / 0.2576 |
| PCQI | input → output | ↑ | patch-based contrast quality relative to the raw input (1.0 = unchanged) |
| NIQE | no reference | ↓ | via `pyiqa`; skipped if it isn't installed |

!!! note "Absolute UIQM / UCIQE values"
    Implementations of UIQM and UCIQE normalise differently, so absolute values differ between
    papers. Compare the ranking of methods, not the absolute numbers. Ours follow the common
    MATLAB-style ports; see the [metrics API](api/metrics.md).

## Classical baselines

| Name | Algorithm |
|---|---|
| `clahe` | CLAHE on the L channel of CIELab (clip 2.0, 8×8 tiles) |
| `gamma` | \(I^{0.7}\) |
| `grayworld` | gray-world white balance |
| `ancuti_rc` | red-channel compensation \(I_{rc} = I_r + (\bar I_g - \bar I_r)(1-I_r)I_g\) followed by gray-world (Ancuti et al., 2018) |

## Scripts

```bash
# baselines and checkpoints on one split; results merge into the same tag across calls
python evaluate.py --data data/ugan --split test --tag ugan_test \
    --methods input clahe gamma grayworld ancuti_rc --ckpt full=runs/full/ckpt_final.pt

# paper Figs. 1 and 11: per-image mean(GT) - mean(image) for R, G, B
python analyze_channels.py --data data/ugan --split test --tag ugan_test --label input
python analyze_channels.py --data data/ugan --split test --tag ugan_test \
    --pred-dir runs/outputs/ugan_test/full --label full

# comparison grids, training curves, intermediate maps
python make_figures.py --tag ugan_test --data data/ugan --methods input clahe grayworld ancuti_rc full \
    --runs runs/full --ckpt full=runs/full/ckpt_final.pt

# everything, for all runs in runs/
DEVICE=cuda ABL_ITERS=4000 SWEEP_ITERS=2000 bash scripts/evaluate_all.sh
```

Outputs:

- `results/metrics/<tag>_per_image.csv` and `<tag>_summary.json`;
- the Markdown table `results/tables/<tag>.md`;
- figures in `results/figures/` and sample outputs in `results/samples/<tag>/<method>/`.
