# UIENetHybrid — Underwater Image Enhancement

A PyTorch implementation of

> Z. Shi, Y. Wang, Z. Zhou, W. Ren, **"Integrating deep learning and traditional image enhancement
> techniques for underwater image enhancement"**, *IET Image Processing* 16 (13): 3471–3484, 2022.
> [doi:10.1049/ipr2.12544](https://doi.org/10.1049/ipr2.12544)

Underwater photos lose red light first and gain a green-blue cast. The method corrects each colour
channel separately (red is boosted, green is reduced, blue is refined) using attention-guided residual
modules. It then improves contrast with a multi-scale CNN that fuses the results of two traditional
techniques, **CLAHE** and **Gamma correction**.

![Output of every stage](assets/generated/intermediates_ugan_test.png)

## Headline results

Results on the UGAN test set (100 synthetic images). The paper trained for 50 epochs; this run trained for 30.7 epochs (25 min on one NVIDIA L40S).

| | Paper | This implementation |
|---|---|---|
| Raw input PSNR / SSIM | 16.85 dB / 0.692 | 16.77 dB / 0.854 |
| **Proposed PSNR / SSIM** | 25.67 dB / 0.891 | **23.63 dB / 0.884** |
| Gain over input | +8.82 dB | +6.86 dB |

The full tables, ablation and loss-weight study are on the [Results](results.md) page. What the
experiments confirm, and where they differ from the paper, is on [Reproduction notes](reproduction.md).

## What is in the repository

<div class="grid cards" markdown>

- **[Method](method.md)**: the architecture, equations and implementation choices
- **[Getting started](getting-started.md)**: install, prepare the datasets, enhance an image
- **[Training](training.md)**: `train.py`, the single-machine pipeline and the parallel Modal pipeline
- **[Evaluation & metrics](evaluation.md)**: PSNR, SSIM, UIQM, UCIQE, PCQI, NIQE and the evaluation scripts
- **[CA4 report](report.md)**: the result-analysis report as PDF and Word
- **[API reference](api/models.md)**: generated from the source code

</div>

## Quick start

```bash
pip install -r requirements.txt
python prepare_data.py --download --raw raw --out data       # UGAN + UIEB
python train.py --data data/ugan --run-name full             # paper settings
python evaluate.py --data data/ugan --split test --tag ugan_test --ckpt full=runs/full/ckpt_final.pt
```

A trained checkpoint (1.3 MB) is included at `results/checkpoints/uienet_full.pt`.
