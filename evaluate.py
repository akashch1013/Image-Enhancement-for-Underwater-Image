"""Evaluate enhancement methods on one data split (paper Sec. 4.1.4).

Methods are the raw input, the classical baselines in `uie.baselines` and trained
checkpoints (`--ckpt NAME=PATH`). Predictions are written to <pred-root>/<tag>/<method>/,
metrics to results/metrics/<tag>_{per_image.csv,summary.json} and a Markdown table to
results/tables/<tag>.md. Existing results for the same tag are merged, so baselines and
models can be evaluated in separate invocations.

    python evaluate.py --data data/ugan --split test --tag ugan_test \
        --methods input clahe gamma grayworld ancuti_rc --ckpt full=runs/full/ckpt_final.pt
"""
import argparse
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from uie import metrics as M
from uie.data import list_pairs, load_rgb, save_rgb

# metric -> (table header, higher is better, decimals); dict order is the table column order
METRIC_INFO = {
    "psnr": ("PSNR (dB)", True, 2),
    "psnr_rgb": ("PSNR-RGB (dB)", True, 2),
    "ssim": ("SSIM", True, 3),
    "uiqm": ("UIQM", True, 3),
    "uciqe": ("UCIQE", True, 3),
    "pcqi": ("PCQI", True, 3),
    "niqe": ("NIQE", False, 3),
    "ms_per_img": ("ms/img", False, 1),
}
REF_METRICS = {"psnr", "psnr_rgb", "ssim"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/ugan")
    p.add_argument("--split", default="test")
    p.add_argument("--tag", default="ugan_test")
    p.add_argument("--methods", nargs="*", default=["input", "clahe", "gamma", "grayworld", "ancuti_rc"])
    p.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "uiqm", "uciqe", "pcqi", "niqe"])
    p.add_argument("--out", default="results")
    p.add_argument("--pred-root", default="runs/outputs")
    p.add_argument("--save-samples", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def _has_target(tgt_path):
    return tgt_path is not None and Path(tgt_path).exists()


def _match(img, shape):
    """Resize `img` to `shape` (H, W, 3) if needed, so full-reference metrics line up."""
    if img.shape == shape:
        return img
    return np.asarray(Image.fromarray(img).resize((shape[1], shape[0]), Image.BICUBIC))


def _worker_init():
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass


def _eval_one(task):
    """Produce (input/baseline) or load (model) one prediction and compute its CPU metrics."""
    method, kind, name, in_path, tgt_path, pred_path, metrics = task
    raw = load_rgb(in_path)
    if kind == "model":
        pred = load_rgb(pred_path)
    else:
        if kind == "baseline":
            from uie.baselines import BASELINES
            pred = BASELINES[method](raw)
        else:
            pred = raw
        save_rgb(pred_path, pred)

    row = {"method": method, "name": name}
    if _has_target(tgt_path) and REF_METRICS & metrics:
        gt = load_rgb(tgt_path)
        pred_gt = _match(pred, gt.shape)
        if "psnr" in metrics:
            row["psnr"] = float(M.psnr_y(pred_gt, gt))
            row["psnr_rgb"] = float(M.psnr_rgb(pred_gt, gt))
        if "ssim" in metrics:
            row["ssim"] = float(M.ssim_y(pred_gt, gt))
    if "uiqm" in metrics:
        row["uiqm"] = float(M.uiqm(pred))
    if "uciqe" in metrics:
        row["uciqe"] = float(M.uciqe(pred))
    if "pcqi" in metrics:
        row["pcqi"] = float(M.pcqi(_match(raw, pred.shape), pred))
    return row


def _parallel_map(fn, tasks, workers):
    if workers <= 1 or len(tasks) < 2:
        return [fn(t) for t in tasks]
    # spawn: the parent may already hold a CUDA context, which must not be forked
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_worker_init) as ex:
        return list(ex.map(fn, tasks, chunksize=max(1, len(tasks) // (workers * 4))))


def resolve_device(name):
    import torch
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("[evaluate] CUDA unavailable, falling back to CPU")
        return "cpu"
    return name


def run_model(ckpt, pairs, pred_dir, device, batch_size):
    """Enhance every input with a checkpoint, save PNGs and return mean inference ms/img."""
    import torch
    from uie.models.network import load_model

    model = load_model(ckpt, device)
    pred_dir.mkdir(parents=True, exist_ok=True)

    def forward(imgs):
        x = torch.from_numpy(np.stack(imgs)).to(device).permute(0, 3, 1, 2).float().div_(255)
        with torch.no_grad():
            y = model(x)
        return y.clamp(0, 1).mul(255).round().byte().permute(0, 2, 3, 1).cpu().numpy()

    # group consecutive same-size images into batches
    batches, cur = [], []
    for name, in_path, _ in pairs:
        img = load_rgb(in_path)
        if cur and (len(cur) == batch_size or img.shape != cur[0][1].shape):
            batches.append(cur)
            cur = []
        cur.append((Path(name).stem, img))
    if cur:
        batches.append(cur)

    forward([img for _, img in batches[0]])  # warm-up: cuDNN autotune and lazy init stay untimed
    elapsed = 0.0
    for batch in batches:
        t0 = time.perf_counter()
        out = forward([img for _, img in batch])  # .cpu() synchronises with the device
        elapsed += time.perf_counter() - t0
        for (stem, _), pred in zip(batch, out):
            save_rgb(pred_dir / f"{stem}.png", pred)
    return 1000.0 * elapsed / len(pairs)


def _clean(v):
    return None if v is None or not np.isfinite(v) else round(float(v), 4)


def write_table(summary, path, tag):
    cols = [m for m in METRIC_INFO if any(s.get(m) is not None for s in summary.values())]
    best = {}
    for c in cols:
        vals = [s[c] for s in summary.values() if s.get(c) is not None]
        best[c] = max(vals) if METRIC_INFO[c][1] else min(vals)

    def fmt(c, v):
        if v is None:
            return "–"
        s = f"{v:.{METRIC_INFO[c][2]}f}"
        return f"**{s}**" if s == f"{best[c]:.{METRIC_INFO[c][2]}f}" else s

    head = ["Method"] + [f"{METRIC_INFO[c][0]} {'↑' if METRIC_INFO[c][1] else '↓'}" for c in cols] + ["N"]
    lines = [f"Results on `{tag}` (mean over images; best per column in bold).", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for method, s in summary.items():
        lines.append("| " + " | ".join([method] + [fmt(c, s.get(c)) for c in cols] + [str(s.get("n", ""))]) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    args = parse_args()
    pairs = list_pairs(args.data, args.split)
    if not pairs:
        raise SystemExit(f"no images found in {args.data}/{args.split}")
    metrics = set(args.metrics)
    if "psnr" in metrics:
        metrics.add("psnr_rgb")
    if not any(_has_target(t) for _, _, t in pairs):
        metrics -= REF_METRICS

    out, pred_root = Path(args.out), Path(args.pred_root) / args.tag
    methods = [(m, "input" if m == "input" else "baseline", None) for m in args.methods]
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        methods.append((name, "model", path))

    device = resolve_device(args.device) if args.ckpt or "niqe" in metrics else "cpu"
    ms_per_img = {}
    for name, kind, ckpt in methods:
        if kind == "model":
            print(f"[evaluate] {args.tag}: running {name} ({ckpt}) on {device}")
            ms_per_img[name] = run_model(ckpt, pairs, pred_root / name, device, args.batch_size)

    for name, _, _ in methods:
        (pred_root / name).mkdir(parents=True, exist_ok=True)
    tasks = [(m, kind, Path(n).stem, in_p, tgt_p, str(pred_root / m / f"{Path(n).stem}.png"), metrics)
             for m, kind, _ in methods for n, in_p, tgt_p in pairs]
    print(f"[evaluate] {args.tag}: {len(tasks)} predictions x metrics {sorted(metrics)}")
    rows = _parallel_map(_eval_one, tasks, args.workers)

    if "niqe" in metrics:
        try:
            niqe = M.NIQE(device)
            for row, task in zip(rows, tasks):
                row["niqe"] = float(niqe(load_rgb(task[5])))
        except Exception as e:  # pyiqa missing or its weights unavailable
            print(f"[evaluate] NIQE skipped: {e}")

    # samples for the README
    for m, _, _ in methods:
        sample_dir = out / "samples" / args.tag / m
        sample_dir.mkdir(parents=True, exist_ok=True)
        for n, _, _ in pairs[: args.save_samples]:
            stem = Path(n).stem
            Image.fromarray(load_rgb(pred_root / m / f"{stem}.png")).save(sample_dir / f"{stem}.jpg", quality=90)

    # merge with earlier invocations for the same tag
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    csv_path = out / "metrics" / f"{args.tag}_per_image.csv"
    df = df_new
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        df = pd.concat([old[~old["method"].isin(df_new["method"].unique())], df_new], ignore_index=True)
    df.to_csv(csv_path, index=False)

    json_path = out / "metrics" / f"{args.tag}_summary.json"
    summary = json.loads(json_path.read_text()) if json_path.exists() else {}
    for m, group in df_new.groupby("method", sort=False):
        s = {c: _clean(group[c].replace([np.inf, -np.inf], np.nan).mean())
             for c in METRIC_INFO if c in group}
        if m in ms_per_img:
            s["ms_per_img"] = _clean(ms_per_img[m])
        s["n"] = int(len(group))
        summary[m] = s
    json_path.write_text(json.dumps(summary, indent=2))

    table_path = out / "tables" / f"{args.tag}.md"
    write_table(summary, table_path, args.tag)
    print(table_path.read_text())


if __name__ == "__main__":
    main()
