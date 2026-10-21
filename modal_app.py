"""Run the whole reproduction on Modal: data preparation, then the full model, the Table-4 ablations
and the Table-5 lambda sweep trained *in parallel* (one GPU container each), then evaluation + figures.

    MODAL_ENVIRONMENT=gpu modal run --detach modal_app.py
    MODAL_ENVIRONMENT=gpu modal volume get underwater shi2022/results.tgz .

Expects the `underwater` volume to hold the official UIEB release under data/UIEB
(raw/, reference/, challenging/, split_seed_42.json); UGAN is fetched from the HF Hub.
Everything this app writes goes to shi2022/ on that volume.
"""
import os
import shutil
import subprocess

import modal

REPO, VOL, WORK = "/root/repo", "/vol", "/vol/shi2022"
MAIN_GPU, AUX_GPU = "L40S", "A10G"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", "kornia", "opencv-python-headless", "scikit-image",
                 "scipy", "pandas", "pyarrow", "matplotlib", "tqdm", "huggingface_hub")
    .run_commands("pip install pyiqa 'torch==2.8.0' 'torchvision==0.23.0' || true")  # optional, for NIQE
    .run_commands("python -c \"import torchvision; torchvision.models.vgg19(weights='IMAGENET1K_V1')\"")
    .add_local_dir(".", REPO, ignore=["Resources", "data", "runs", "results", ".git", "**/__pycache__"])
)
vol = modal.Volume.from_name("underwater")
app = modal.App("shi2022-uienet", image=image)

ABLATIONS = {"wo_gnet": "--no-gnet", "wo_rnet": "--no-rnet", "wo_rnet_gnet": "--no-rnet --no-gnet",
             "w_rnet_gnet": "--no-ce"}
LAMBDAS = (0.1, 1, 10)


def sh(cmd: str) -> None:
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=REPO)


def stage_data() -> None:
    """Unpack the prepared dataset onto the container's local disk and link it into the repo."""
    if not os.path.isdir("/tmp/data"):
        sh(f"tar -xf {WORK}/data.tar -C /tmp")
    if not os.path.exists(f"{REPO}/data"):
        os.symlink("/tmp/data", f"{REPO}/data")


@app.function(cpu=16, memory=32768, timeout=3600, volumes={VOL: vol})
def prepare() -> str:
    if os.path.exists(f"{WORK}/data.tar"):
        return "data already prepared"
    os.makedirs(WORK, exist_ok=True)
    sh("python -c \"from huggingface_hub import snapshot_download as s; "
       "s('iamvastava/underwater_image_enhancement', repo_type='dataset', local_dir='/tmp/raw/ugan')\"")
    sh(f"python prepare_data.py --raw /tmp/raw --out /tmp/data --uieb-dir {VOL}/data/UIEB --workers 16")
    sh(f"tar -cf {WORK}/data.tar -C /tmp data")
    vol.commit()
    return open("/tmp/data/README_data.json").read()


def _train(name: str, args: str) -> str:
    stage_data()
    sh(f"python train.py --data data/ugan --run-name {name} --out-dir runs --workers 6 {args}")
    dst = f"{WORK}/runs/{name}"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(f"{REPO}/runs/{name}", dst)
    vol.commit()
    return open(f"{dst}/summary.json").read()


@app.function(gpu=MAIN_GPU, cpu=8, memory=32768, timeout=2 * 3600, volumes={VOL: vol})
def train_main(name: str, args: str) -> str:
    return _train(name, args)


@app.function(gpu=AUX_GPU, cpu=8, memory=32768, timeout=2 * 3600, volumes={VOL: vol})
def train_aux(name: str, args: str) -> str:
    return _train(name, args)


@app.function(gpu=AUX_GPU, cpu=16, memory=32768, timeout=3600, volumes={VOL: vol})
def evaluate(abl_iters: int, sweep_iters: int) -> str:
    vol.reload()
    stage_data()
    shutil.copytree(f"{WORK}/runs", f"{REPO}/runs", dirs_exist_ok=True)
    sh(f"DEVICE=cuda ABL_ITERS={abl_iters} SWEEP_ITERS={sweep_iters} EVAL_WORKERS=16 bash scripts/evaluate_all.sh")
    shutil.rmtree(f"{WORK}/results", ignore_errors=True)
    shutil.copytree(f"{REPO}/results", f"{WORK}/results")
    sh(f"tar -czf {WORK}/results.tgz -C {REPO} results")
    vol.commit()
    return "\n".join(open(f"{REPO}/results/tables/{t}").read() for t in sorted(os.listdir(f"{REPO}/results/tables")))


@app.function(timeout=4 * 3600)
def pipeline(main_min: float, abl_iters: int, abl_min: float, sweep_iters: int, sweep_min: float) -> str:
    print(prepare.remote(), flush=True)
    main = train_main.spawn("full", f"--time-budget-min {main_min} --save-at-iters {sweep_iters} {abl_iters}")
    jobs = [(n, f"--max-iters {abl_iters} --time-budget-min {abl_min} {f}") for n, f in ABLATIONS.items()]
    jobs += [(f"sweep_l1_{a}_l2_{b}", f"--lambda1 {a} --lambda2 {b} --max-iters {sweep_iters} "
              f"--time-budget-min {sweep_min}") for a in LAMBDAS for b in LAMBDAS if (a, b) != (1, 0.1)]
    for res in train_aux.starmap(jobs, return_exceptions=True):
        print(res, flush=True)
    print(main.get(), flush=True)
    return evaluate.remote(abl_iters, sweep_iters)


@app.local_entrypoint()
def run(main_min: float = 25, abl_iters: int = 4000, abl_min: float = 15, sweep_iters: int = 2000,
        sweep_min: float = 10):
    print(pipeline.remote(main_min, abl_iters, abl_min, sweep_iters, sweep_min))
