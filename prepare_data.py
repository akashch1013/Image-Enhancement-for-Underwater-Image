"""Build the training / evaluation sets used in this reproduction.

Layout produced (all RGB PNG, resized to --size x --size):
    <out>/ugan/{train,test}/{input,target}/<name>.png   synthetic UGAN pairs (paper Sec. 4.1.1)
    <out>/uieb/{train,test}/{input,target}/<name>.png   real UIEB raw/reference pairs
    <out>/uieb_c60/test/input/<name>.png                UIEB-Challenging-60 (no reference; needs --uieb-dir)
    <out>/README_data.json                              counts and sources

Sources (Hugging Face Hub, fetched only with --download):
    UGAN (Fabbri et al. [37]): datasets/iamvastava/underwater_image_enhancement
        -> "underwater_imagenet (1).zip", trainA = distorted, trainB = clean, 6128 pairs
    UIEB (Li et al. [24]):     datasets/Hikari0608/UIEB -> parquet, columns raw/gt, 800 train + 90 val
    or, with --uieb-dir, the official UIEB release laid out as raw/, reference/, challenging/ plus a
    split_seed_42.json ({train, val, test, challenging} stem lists): train+val -> train, test -> test.
"""
import argparse
import glob
import json
import os
import random
import zipfile
from multiprocessing import Pool

import cv2
import numpy as np

UGAN_REPO = "iamvastava/underwater_image_enhancement"
UIEB_REPO = "Hikari0608/UIEB"
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

SIZE = 256  # set from --size before the worker pool starts


def download(raw: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(UGAN_REPO, repo_type="dataset", local_dir=os.path.join(raw, "ugan"))
    snapshot_download(UIEB_REPO, repo_type="dataset", local_dir=os.path.join(raw, "uieb"))


def _decode_resize(buf: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise ValueError("could not decode image")
    if img.shape[:2] != (SIZE, SIZE):
        img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return img


def _write_pair(job) -> str:
    """job = (name, input_bytes, target_bytes, out_split_dir)."""
    name, in_buf, gt_buf, out_dir = job
    for sub, buf in (("input", in_buf), ("target", gt_buf)):
        if buf is None:  # no-reference split
            continue
        cv2.imwrite(os.path.join(out_dir, sub, name + ".png"), _decode_resize(buf))
    return name


def _init_worker(size: int) -> None:
    global SIZE
    SIZE = size
    cv2.setNumThreads(1)


def _make_dirs(split_dir: str) -> None:
    for sub in ("input", "target"):
        os.makedirs(os.path.join(split_dir, sub), exist_ok=True)


def _mean_red_minus_blue(bufs) -> float:
    vals = []
    for b in bufs:
        img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR).astype(np.float32)
        vals.append(img[..., 2].mean() - img[..., 0].mean())
    return float(np.mean(vals))


def find_ugan_pairs(raw: str):
    """Locate trainA/trainB in the UGAN zip (or an extracted folder); return (zip|None, [(stem, a, b)])."""
    zips = glob.glob(os.path.join(raw, "ugan", "*.zip"))
    if zips:
        zf = zipfile.ZipFile(zips[0])
        names = [n for n in zf.namelist() if n.lower().endswith(IMG_EXTS)]
        read = zf.read
    else:
        zf = None
        names = [os.path.relpath(p, raw) for p in glob.glob(os.path.join(raw, "ugan", "**", "*"), recursive=True)
                 if p.lower().endswith(IMG_EXTS)]
        read = lambda n: open(os.path.join(raw, n), "rb").read()  # noqa: E731

    by_dir = {}
    for n in names:
        parts = n.replace("\\", "/").split("/")
        by_dir.setdefault(parts[-2], {})[os.path.splitext(parts[-1])[0]] = n
    if "trainA" not in by_dir or "trainB" not in by_dir:
        raise RuntimeError(f"trainA/trainB not found in UGAN source; folders: {sorted(by_dir)}")
    a, b = by_dir["trainA"], by_dir["trainB"]
    stems = sorted(set(a) & set(b))

    # EUVP/UGAN convention is trainA = distorted, trainB = clean; verify with the paper's Fig. 1
    # prior (distorted images have a strongly attenuated red channel) and swap if needed.
    probe = stems[:40]
    if _mean_red_minus_blue(read(a[s]) for s in probe) > _mean_red_minus_blue(read(b[s]) for s in probe):
        a, b = b, a
    return zf, read, [(s, a[s], b[s]) for s in stems]


def build_ugan(args, pool) -> dict:
    _, read, pairs = find_ugan_pairs(args.raw)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_test = min(args.n_test, len(pairs) // 10 or 1)
    n_train = min(args.n_train, len(pairs) - n_test)
    splits = {"test": pairs[:n_test], "train": pairs[n_test:n_test + n_train]}
    counts = {"available_pairs": len(pairs)}
    for split, items in splits.items():
        out_dir = os.path.join(args.out, "ugan", split)
        _make_dirs(out_dir)
        # read in the parent (zip handles are not shared across processes), encode in workers
        jobs = ((s, read(pa), read(pb), out_dir) for s, pa, pb in items)
        counts[split] = sum(1 for _ in pool.imap_unordered(_write_pair, jobs, chunksize=16))
        print(f"ugan/{split}: {counts[split]}", flush=True)
    return counts


def build_uieb(args, pool) -> dict:
    import pyarrow.parquet as pq

    counts = {}
    for split, pattern in (("train", "train-*.parquet"), ("test", "val-*.parquet")):
        files = sorted(glob.glob(os.path.join(args.raw, "uieb", "data", pattern)))
        if not files:
            print(f"uieb/{split}: no parquet files found, skipping", flush=True)
            continue
        out_dir = os.path.join(args.out, "uieb", split)
        _make_dirs(out_dir)
        n = 0
        for f in files:
            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=32, columns=["raw", "gt"]):
                rows = batch.to_pylist()
                jobs = []
                for r in rows:
                    stem = os.path.splitext(os.path.basename(r["raw"].get("path") or f"img{n}"))[0]
                    stem = stem.strip("()").replace(" ", "_") or f"img{n}"
                    jobs.append((stem, r["raw"]["bytes"], r["gt"]["bytes"], out_dir))
                n += sum(1 for _ in pool.imap_unordered(_write_pair, jobs))
        counts[split] = n
        print(f"uieb/{split}: {n}", flush=True)
    return counts


def build_uieb_folder(args, pool) -> dict:
    """Official UIEB folders (raw/, reference/, challenging/) split by split_seed_42.json."""
    root = args.uieb_dir
    split = json.load(open(os.path.join(root, "split_seed_42.json")))
    read = lambda sub, stem: open(os.path.join(root, sub, stem + ".png"), "rb").read()  # noqa: E731
    plan = {("uieb", "train"): split["train"] + split.get("val", []), ("uieb", "test"): split["test"],
            ("uieb_c60", "test"): split["challenging"]}
    counts = {}
    for (name, sp), stems in plan.items():
        out_dir = os.path.join(args.out, name, sp)
        os.makedirs(os.path.join(out_dir, "input"), exist_ok=True)
        if name == "uieb":
            os.makedirs(os.path.join(out_dir, "target"), exist_ok=True)
            jobs = ((s, read("raw", s), read("reference", s), out_dir) for s in stems)
        else:
            jobs = ((s, read("challenging", s), None, out_dir) for s in stems)
        counts[f"{name}/{sp}"] = sum(1 for _ in pool.imap_unordered(_write_pair, jobs, chunksize=8))
        print(f"{name}/{sp}: {counts[f'{name}/{sp}']}", flush=True)
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default="/workspace/raw", help="where the raw downloads live")
    p.add_argument("--out", default="data")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--download", action="store_true", help="fetch the raw datasets from the HF Hub first")
    p.add_argument("--skip-uieb", action="store_true")
    p.add_argument("--uieb-dir", default=None, help="official UIEB folders instead of the HF parquet")
    args = p.parse_args()

    if args.download:
        download(args.raw)
    os.makedirs(args.out, exist_ok=True)
    info = {"size": args.size, "seed": args.seed,
            "sources": {"ugan": f"hf://datasets/{UGAN_REPO}", "uieb": f"hf://datasets/{UIEB_REPO}"}}
    with Pool(args.workers, initializer=_init_worker, initargs=(args.size,)) as pool:
        info["ugan"] = build_ugan(args, pool)
        if args.uieb_dir:
            info["uieb"] = build_uieb_folder(args, pool)
            info["sources"]["uieb"] = "official UIEB release (raw / reference / challenging)"
        elif not args.skip_uieb:
            info["uieb"] = build_uieb(args, pool)
    with open(os.path.join(args.out, "README_data.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
