#!/usr/bin/env bash
# Single-machine, time-budgeted pipeline: train the full model + Table-4 ablations (+ optional Table-5
# lambda sweep), then run scripts/evaluate_all.sh. All knobs are environment variables, e.g.
#   MAIN_MIN=30 ABL_ITERS=3000 bash scripts/run_all.sh
# (modal_app.py runs the same experiments in parallel on Modal GPUs.)
set -uo pipefail
cd "$(dirname "$0")/.."

if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    DEVICE=cuda
    : "${MAIN_MIN:=20}" "${ABL_ITERS:=1500}" "${ABL_MIN:=4}" "${CROP:=0}"
else
    # CPU fallback: 128x128 crops, and ablations train concurrently with the main run
    DEVICE=cpu
    : "${MAIN_MIN:=15}" "${ABL_ITERS:=200}" "${ABL_MIN:=7}" "${CROP:=128}"
    : "${MAIN_THREADS:=8}" "${ABL_THREADS:=7}"
fi
: "${BS:=8}" "${WORKERS:=4}" "${SWEEP_ITERS:=0}" "${EVAL_WORKERS:=16}"
LOG=runs/logs
mkdir -p "$LOG" results
echo "device=$DEVICE MAIN_MIN=$MAIN_MIN ABL_ITERS=$ABL_ITERS CROP=$CROP BS=$BS" | tee "$LOG/pipeline.log"

TRAIN=(python train.py --data data/ugan --batch-size "$BS" --crop "$CROP" --workers "$WORKERS")
ABLATIONS=("wo_gnet --no-gnet" "wo_rnet --no-rnet" "wo_rnet_gnet --no-rnet --no-gnet" "w_rnet_gnet --no-ce")
stamp() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG/pipeline.log"; }

# classical baselines + raw-input channel statistics run on the CPU while the model trains
(
    for d in ugan uieb uieb_c60; do
        [ -d data/$d/test/input ] || continue
        t=$([ $d = uieb_c60 ] && echo uieb_c60 || echo ${d}_test)
        python evaluate.py --data data/$d --split test --tag $t --methods input clahe gamma grayworld ancuti_rc \
            --device cpu --workers 2
        [ $d != uieb_c60 ] && python analyze_channels.py --data data/$d --split test --tag $t --label input
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

SAVE_AT=("$ABL_ITERS"); [ "$SWEEP_ITERS" -gt 0 ] && SAVE_AT+=("$SWEEP_ITERS")
stamp "training full model"
if [ "$DEVICE" = cuda ]; then
    "${TRAIN[@]}" --run-name full --time-budget-min "$MAIN_MIN" --save-at-iters "${SAVE_AT[@]}" \
        > "$LOG/train_full.log" 2>&1
    run_ablations
else
    "${TRAIN[@]}" --run-name full --time-budget-min "$MAIN_MIN" --save-at-iters "${SAVE_AT[@]}" \
        --threads "$MAIN_THREADS" > "$LOG/train_full.log" 2>&1 &
    FULL_PID=$!
    run_ablations
    wait $FULL_PID
fi

if [ "$SWEEP_ITERS" -gt 0 ]; then
    for l1 in 0.1 1 10; do for l2 in 0.1 1 10; do
        [ "$l1" = 1 ] && [ "$l2" = 0.1 ] && continue   # that is the full model
        stamp "sweep l1=$l1 l2=$l2"
        "${TRAIN[@]}" --run-name "sweep_l1_${l1}_l2_${l2}" --lambda1 $l1 --lambda2 $l2 \
            --max-iters "$SWEEP_ITERS" > "$LOG/train_sweep_${l1}_${l2}.log" 2>&1
    done; done
fi
stamp "training done"

wait $BASE_PID
stamp "evaluating"
DEVICE=$DEVICE ABL_ITERS=$ABL_ITERS SWEEP_ITERS=$SWEEP_ITERS EVAL_WORKERS=$EVAL_WORKERS RUN_BASELINES=0 \
    bash scripts/evaluate_all.sh > "$LOG/evaluate.log" 2>&1
cp "$LOG"/*.log results/runs/ 2>/dev/null
stamp "pipeline finished"
