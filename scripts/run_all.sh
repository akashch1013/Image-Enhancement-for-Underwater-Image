#!/usr/bin/env bash
# Time-budgeted experiment pipeline: train the full model + Table-4 ablations, evaluate everything,
# reproduce the channel-deviation analysis (Figs. 1 & 11) and draw the figures.
# All knobs are environment variables, e.g.  MAIN_MIN=30 ABL_ITERS=3000 bash scripts/run_all.sh
set -uo pipefail
cd "$(dirname "$0")/.."

if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE=cuda
    : "${MAIN_MIN:=20}" "${ABL_ITERS:=1500}" "${ABL_MIN:=4}" "${CROP:=0}"
else
    # CPU fallback: 128x128 crops, and ablations train concurrently with the main run
    DEVICE=cpu
    : "${MAIN_MIN:=15}" "${ABL_ITERS:=200}" "${ABL_MIN:=7}" "${CROP:=128}"
    : "${MAIN_THREADS:=10}" "${ABL_THREADS:=8}"
fi
: "${BS:=8}" "${WORKERS:=4}" "${SWEEP_ITERS:=0}" "${EVAL_WORKERS:=16}"
LOG=runs/logs
mkdir -p "$LOG" results
echo "device=$DEVICE MAIN_MIN=$MAIN_MIN ABL_ITERS=$ABL_ITERS CROP=$CROP BS=$BS" | tee "$LOG/pipeline.log"

TRAIN=(python train.py --data data/ugan --batch-size "$BS" --crop "$CROP" --workers "$WORKERS")
BASELINES=(input clahe gamma grayworld ancuti_rc)
ABLATIONS=("wo_gnet --no-gnet" "wo_rnet --no-rnet" "wo_rnet_gnet --no-rnet --no-gnet" "w_rnet_gnet --no-ce")
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/pipeline.log"; }

# --- CPU side job: classical baselines + raw-input channel statistics (paper Fig. 1) -------------
(
    for d in ugan uieb; do
        python evaluate.py --data data/$d --split test --tag ${d}_test --methods "${BASELINES[@]}" \
            --device cpu --workers 2
        python analyze_channels.py --data data/$d --split test --tag ${d}_test --label input
    done
) > "$LOG/baselines.log" 2>&1 &
BASE_PID=$!

run_ablations() {
    for v in "${ABLATIONS[@]}"; do
        set -- $v; name=$1; shift
        stamp "ablation $name"
        "${TRAIN[@]}" --run-name "$name" --max-iters "$ABL_ITERS" --time-budget-min "$ABL_MIN" \
            ${ABL_THREADS:+--threads $ABL_THREADS} "$@" > "$LOG/train_$name.log" 2>&1
    done
}

stamp "training full model"
if [ "$DEVICE" = cuda ]; then
    "${TRAIN[@]}" --run-name full --time-budget-min "$MAIN_MIN" --save-at-iters "$ABL_ITERS" \
        > "$LOG/train_full.log" 2>&1
    run_ablations
else
    "${TRAIN[@]}" --run-name full --time-budget-min "$MAIN_MIN" --save-at-iters "$ABL_ITERS" \
        --threads "$MAIN_THREADS" > "$LOG/train_full.log" 2>&1 &
    FULL_PID=$!
    run_ablations
    wait $FULL_PID
fi
stamp "training done"

# --- optional Table-5 lambda sweep (same iteration count as the ablations) ----------------------
if [ "$SWEEP_ITERS" -gt 0 ]; then
    for l1 in 0.1 1 10; do for l2 in 0.1 1 10; do
        stamp "sweep l1=$l1 l2=$l2"
        "${TRAIN[@]}" --run-name "sweep_l1_${l1}_l2_${l2}" --lambda1 $l1 --lambda2 $l2 \
            --max-iters "$SWEEP_ITERS" > "$LOG/train_sweep_${l1}_${l2}.log" 2>&1
    done; done
fi

wait $BASE_PID
stamp "evaluating"
for d in ugan uieb; do
    python evaluate.py --data data/$d --split test --tag ${d}_test --methods \
        --ckpt full=runs/full/ckpt_final.pt --device "$DEVICE" --workers "$EVAL_WORKERS"
    python analyze_channels.py --data data/$d --split test --tag ${d}_test \
        --pred-dir runs/outputs/${d}_test/full --label full
done

# Ablation table (Table 4): every variant compared at the same number of iterations
ABL_CKPTS=(--ckpt "w_ce=runs/full/ckpt_iter${ABL_ITERS}.pt")
for v in "${ABLATIONS[@]}"; do name=${v%% *}; ABL_CKPTS+=(--ckpt "$name=runs/$name/ckpt_final.pt"); done
python evaluate.py --data data/ugan --split test --tag ugan_ablation --methods "${ABL_CKPTS[@]}" \
    --metrics psnr ssim uiqm uciqe --device "$DEVICE" --workers "$EVAL_WORKERS" --save-samples 4

stamp "figures"
RUNS=(runs/full); for v in "${ABLATIONS[@]}"; do RUNS+=(runs/${v%% *}); done
for d in ugan uieb; do
    python make_figures.py --tag ${d}_test --data data/$d --split test \
        --methods input clahe grayworld ancuti_rc full --runs "${RUNS[@]}" \
        --ckpt full=runs/full/ckpt_final.pt --device "$DEVICE"
done
python make_figures.py --tag ugan_ablation --data data/ugan --split test \
    --methods wo_rnet_gnet wo_rnet wo_gnet w_rnet_gnet w_ce --n 4 --device "$DEVICE"

# --- collect training artefacts that are small enough to commit ---------------------------------
for r in "${RUNS[@]}"; do
    mkdir -p results/runs/$(basename $r)
    cp $r/log.csv $r/summary.json $r/args.json results/runs/$(basename $r)/ 2>/dev/null
done
mkdir -p results/checkpoints && cp runs/full/ckpt_final.pt results/checkpoints/uienet_full.pt
cp "$LOG"/*.log results/runs/ 2>/dev/null
stamp "pipeline finished"
