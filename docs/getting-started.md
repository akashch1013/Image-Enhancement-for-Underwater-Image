# Getting started

## Install

```bash
git clone https://github.com/akashch1013/UIENetHybrid.git
cd UIENetHybrid
pip install -r requirements.txt        # add `pyiqa` for the NIQE metric
```

Python 3.10+ and PyTorch 2.1+ are required. A GPU is optional, since `train.py` falls back to the
CPU, but strongly recommended for training.

## Prepare the datasets

```bash
python prepare_data.py --download --raw raw --out data
```

This fetches the datasets from the Hugging Face Hub and writes 256×256 PNGs:

```
data/ugan/{train,test}/{input,target}/   UGAN: 6128 CycleGAN pairs, split 4000 train / 100 test (seed 0)
data/uieb/{train,test}/{input,target}/   UIEB raw / reference pairs
data/uieb_c60/test/input/                UIEB-Challenging-60 (no references; needs --uieb-dir)
data/README_data.json                    counts and sources
```

If you have the official UIEB release (folders `raw/`, `reference/` and `challenging/`, plus a
`split_seed_42.json`), pass `--uieb-dir /path/to/UIEB`. That gives the official test-90 split and
the Challenging-60 set the paper evaluates on.

| Dataset | Used for | Source |
|---|---|---|
| UGAN (Fabbri et al., ICRA 2018) | training, synthetic test | `iamvastava/underwater_image_enhancement` (Underwater ImageNet, all 6128 pairs) |
| UIEB test-90 | real images with references | official UIEB release, or `Hikari0608/UIEB` on the Hub |
| UIEB Challenging-60 | real images, no-reference metrics | official UIEB release |

## Enhance an image with the released checkpoint

```python
import torch
from uie.models import load_model
from uie.data import load_rgb, save_rgb, to_tensor

model = load_model("results/checkpoints/uienet_full.pt")           # 317,697 parameters
x = to_tensor(load_rgb("my_underwater.png"))[None]                   # 1x3xHxW in [0, 1]
with torch.no_grad():
    y = model(x)[0].permute(1, 2, 0).mul(255).round().byte().numpy()
save_rgb("enhanced.png", y)
```

`model(x, return_intermediate=True)` also returns `i1`, `ice`, `igc`, `wce` and `wgc`: the output of every
stage, as shown on the [home page](index.md).

## Sanity check

```bash
python train.py --smoke
```

This runs 20 iterations on random tensors and checks:

- output shape and value range;
- the Eq. (1) fixed point: a saturated red channel passes Rnet unchanged;
- the parameter count;
- the checkpoint save/load round-trip.
