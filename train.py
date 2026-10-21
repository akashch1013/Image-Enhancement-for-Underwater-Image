"""Train UIENet (Shi et al. 2022) on paired underwater images.

Paper settings (Sec. 4.1.2): Adam, lr 2e-4, batch 8, 256x256 inputs, 50 epochs, lambda1 = 1, lambda2 = 0.1.
Training stops at whichever comes first: --epochs, --max-iters or --time-budget-min.

    python train.py --data data/ugan --run-name full
    python train.py --data data/ugan --run-name wo_gnet --no-gnet --max-iters 2000
    python train.py --smoke            # 20-iteration sanity run with built-in checks
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from uie.losses import CombinedLoss
from uie.models import UIENet, count_parameters, load_model, save_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/ugan")
    p.add_argument("--run-name", default="full")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--max-iters", type=int, default=0, help="0 = no limit")
    p.add_argument("--time-budget-min", type=float, default=0, help="0 = no limit")
    p.add_argument("--save-at-iters", type=int, nargs="*", default=[], help="extra checkpoints")
    p.add_argument("--lambda1", type=float, default=1.0)
    p.add_argument("--lambda2", type=float, default=0.1)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.7)
    p.add_argument("--no-rnet", action="store_true")
    p.add_argument("--no-gnet", action="store_true")
    p.add_argument("--no-ce", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--crop", type=int, default=0, help="random square crop for training (0 = full 256x256)")
    p.add_argument("--threads", type=int, default=0, help="CPU threads (0 = from the cgroup CPU quota)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def cpu_quota() -> int:
    """Number of CPUs granted to this container (os.cpu_count() reports the whole host on RunPod)."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


def random_crop(x: torch.Tensor, y: torch.Tensor, size: int):
    h, w = x.shape[-2:]
    if not size or size >= min(h, w):
        return x, y
    top, left = torch.randint(0, h - size + 1, (1,)).item(), torch.randint(0, w - size + 1, (1,)).item()
    return x[..., top:top + size, left:left + size], y[..., top:top + size, left:left + size]


def build_model(args) -> UIENet:
    return UIENet(width=args.width, gamma=args.gamma, use_rnet=not args.no_rnet,
                  use_gnet=not args.no_gnet, use_ce=not args.no_ce)


def random_batches(batch_size: int, size: int = 256):
    """Stand-in data for --smoke when no dataset has been prepared yet."""
    while True:
        target = torch.rand(batch_size, 3, size, size)
        yield dict(input=target * torch.tensor([0.4, 1.1, 0.9]).view(1, 3, 1, 1), target=target)


def sanity_checks(model: UIENet, device) -> None:
    """Checks from the plan: shapes/range, Eq. 1 fixed point, parameter count."""
    x = torch.rand(2, 3, 64, 64, device=device)
    with torch.no_grad():
        out = model(x, return_intermediate=True)
        assert out["out"].shape == x.shape and 0 <= out["out"].min() and out["out"].max() <= 1
        if model.config["use_rnet"]:
            ones = torch.ones(1, 1, 32, 32, device=device)
            # (1 - Ir) = 0 on a saturated red channel, so Rnet must return it unchanged
            assert torch.allclose(model.rnet(ones), ones), "Rnet should be identity when Ir = 1"
    print(f"[smoke] sanity checks passed; parameters = {count_parameters(model):,}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    else:
        torch.set_num_threads(args.threads or cpu_quota())
    if args.smoke:
        args.max_iters, args.log_every, args.run_name = 20, 5, args.run_name + "_smoke"

    run_dir = Path(args.out_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    model = build_model(args).to(device)
    print(f"device={device} threads={torch.get_num_threads()} amp={use_amp} params={count_parameters(model):,} config={model.config}")
    if args.smoke:
        sanity_checks(model, device)

    if Path(args.data, "train").is_dir():
        from uie.data import PairedFolder
        dataset = PairedFolder(args.data, "train", augment=True)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                            num_workers=args.workers, pin_memory=device.type == "cuda",
                            persistent_workers=args.workers > 0)
        iters_per_epoch = len(loader)
        print(f"train pairs={len(dataset)} iters/epoch={iters_per_epoch}")
    elif args.smoke:
        loader, iters_per_epoch = None, args.max_iters
        print("[smoke] no dataset found, using random tensors")
    else:
        raise FileNotFoundError(f"{args.data}/train not found; run prepare_data.py first")

    criterion = CombinedLoss(args.lambda1, args.lambda2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    max_iters = args.epochs * iters_per_epoch
    if args.max_iters:
        max_iters = min(max_iters, args.max_iters)
    budget_s = args.time_budget_min * 60 if args.time_budget_min else float("inf")
    save_at = set(args.save_at_iters)

    log_f = open(run_dir / "log.csv", "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["iter", "epoch", "loss", "l1", "ssim", "per", "lr", "elapsed_s", "it_per_s"])

    it, epoch, t0, stop = 0, 0, time.time(), False
    t_last, it_last = t0, 0
    model.train()
    while not stop:
        epoch += 1
        batches = loader if loader is not None else (b for _, b in zip(range(iters_per_epoch), random_batches(args.batch_size)))
        for batch in batches:
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            x, y = random_crop(x, y, args.crop)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(x)
                loss, parts = criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            it += 1

            if it % args.log_every == 0 or it == 1:
                now = time.time()
                speed = (it - it_last) / max(now - t_last, 1e-9)
                t_last, it_last = now, it
                row = [it, epoch, loss.item(), parts["l1"].item(), parts["ssim"].item(), parts["per"].item(),
                       args.lr, round(now - t0, 1), round(speed, 2)]
                log.writerow(row)
                log_f.flush()
                print(f"it {it}/{max_iters} ep {epoch} loss {row[2]:.4f} (l1 {row[3]:.4f} ssim {row[4]:.4f} "
                      f"per {row[5]:.4f}) {speed:.2f} it/s {now - t0:.0f}s", flush=True)
            if it in save_at:
                save_checkpoint(run_dir / f"ckpt_iter{it}.pt", model, iter=it, epoch=epoch, args=vars(args))
            if it >= max_iters or time.time() - t0 >= budget_s:
                stop = True
                break
        if epoch >= args.epochs:
            stop = True
    log_f.close()

    elapsed = time.time() - t0
    final = run_dir / "ckpt_final.pt"
    save_checkpoint(final, model, iter=it, epoch=epoch, args=vars(args))
    summary = dict(run=args.run_name, iters=it, epochs_completed=round(it / iters_per_epoch, 2),
                   elapsed_min=round(elapsed / 60, 2), params=count_parameters(model),
                   final_loss=round(loss.item(), 5), device=str(device), config=model.config,
                   lambda1=args.lambda1, lambda2=args.lambda2)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))

    if args.smoke:
        reloaded = load_model(final, device)
        x = torch.rand(1, 3, 64, 64, device=device)
        model.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), reloaded(x), atol=1e-5), "checkpoint round-trip mismatch"
        print("[smoke] checkpoint round-trip OK")


if __name__ == "__main__":
    if Path("/workspace").is_dir():  # RunPod: keep the VGG-19 weights on the persistent volume
        os.environ.setdefault("TORCH_HOME", "/workspace/torch_cache")
    main()
