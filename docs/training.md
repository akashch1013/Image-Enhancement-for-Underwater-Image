# Training

## `train.py`

Defaults follow the paper (§4.1.2): Adam with lr \(2\times10^{-4}\), batch 8, 256×256 inputs, 50 epochs, λ1 = 1, λ2 = 0.1.

```bash
python train.py --data data/ugan --run-name full
python train.py --data data/ugan --run-name wo_gnet --no-gnet --max-iters 4000
python train.py --data data/ugan --run-name sweep --lambda1 10 --lambda2 1 --max-iters 2000
```

| Option | Meaning |
|---|---|
| `--epochs`, `--max-iters`, `--time-budget-min` | Training stops at whichever limit is reached first |
| `--save-at-iters K ...` | Extra checkpoints, used to compare ablations at equal iterations |
| `--no-rnet`, `--no-gnet`, `--no-ce` | The Table-4 ablation variants |
| `--lambda1`, `--lambda2` | Loss weights (Table 5) |
| `--crop N` | Random N×N crops (the CPU fallback uses 128) |
| `--amp` | fp16 mixed precision; **off by default**, see [Reproduction notes](reproduction.md) |
| `--threads` | CPU threads; defaults to the cgroup CPU quota, not the host core count |
| `--smoke` | 20-iteration sanity run with built-in checks |

Each run writes to `runs/<run-name>/`:

- `log.csv`: iteration, epoch, loss parts, it/s;
- `ckpt_final.pt` and any `ckpt_iter<K>.pt`;
- `summary.json` and `args.json`.

## Single-machine pipeline

```bash
MAIN_MIN=25 ABL_ITERS=1500 SWEEP_ITERS=0 bash scripts/run_all.sh
```

The script:

- trains the full model, then the four ablations (each budget is an environment variable);
- evaluates the classical baselines on the CPU while the model trains;
- then runs [`scripts/evaluate_all.sh`](evaluation.md).

Without a GPU it switches to 128×128 crops and trains the ablations alongside the main run.

## Parallel pipeline on Modal

```bash
MODAL_ENVIRONMENT=gpu modal run --detach modal_app.py
MODAL_ENVIRONMENT=gpu modal volume get underwater shi2022/results.tgz .
```

`modal_app.py`:

- prepares the data once and stores it on a volume;
- launches **13 training jobs in parallel**: the full model on an L40S, plus the 4 ablations and the 8-point λ sweep on A10Gs;
- then runs one evaluation job.

This is how the published results were produced:

| Run | GPU | Iterations | Time |
|---|---|---|---|
| Full model | L40S | 15,364 (30.7 epochs) | 25.0 min |
| Ablations (4) | A10G | 4,000 each | 8.7–14.9 min |
| λ sweep (8) | A10G | 2,000 each | about 7.5 min |

![Training curves](assets/generated/training_curves.png)
