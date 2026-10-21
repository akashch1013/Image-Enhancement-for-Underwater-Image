#!/usr/bin/env bash
# Evaluate trained runs in runs/ and write everything that gets committed to results/.
#   ugan_test      synthetic UGAN test-100 (paper Table 1)
#   uieb_test      real UIEB test-90 with references (Table 3 + no-reference metrics)
#   uieb_c60       UIEB-Challenging-60, no references (Table 2), when data/uieb_c60 exists
#   ugan_ablation  Table 4 variants, all at ABL_ITERS iterations
#   ugan_lambda    Table 5 lambda sweep at SWEEP_ITERS iterations (when SWEEP_ITERS > 0)
set -uo pipefail
cd "$(dirname "$0")/.."
: "${DEVICE:=cuda}" "${ABL_ITERS:=200}" "${SWEEP_ITERS:=0}" "${EVAL_WORKERS:=16}" "${RUN_BASELINES:=1}"
BASELINES=(input clahe gamma grayworld ancuti_rc)
ABLATIONS=(wo_gnet wo_rnet wo_rnet_gnet w_rnet_gnet)
SETS=(ugan uieb)
[ -d data/uieb_c60/test/input ] && SETS+=(uieb_c60)
tag() { [ "$1" = uieb_c60 ] && echo uieb_c60 || echo "$1_test"; }

for d in "${SETS[@]}"; do
    if [ "$RUN_BASELINES" = 1 ]; then
        python evaluate.py --data data/$d --split test --tag $(tag $d) --methods "${BASELINES[@]}" \
            --device cpu --workers "$EVAL_WORKERS"
        [ "$d" != uieb_c60 ] && python analyze_channels.py --data data/$d --split test --tag $(tag $d) --label input
    fi
    python evaluate.py --data data/$d --split test --tag $(tag $d) --methods \
        --ckpt full=runs/full/ckpt_final.pt --device "$DEVICE" --workers "$EVAL_WORKERS"
    [ "$d" != uieb_c60 ] && python analyze_channels.py --data data/$d --split test --tag $(tag $d) \
        --pred-dir runs/outputs/$(tag $d)/full --label full
done

# Table 4: every variant compared at the same number of iterations
CK=(--ckpt "w_ce=runs/full/ckpt_iter${ABL_ITERS}.pt")
for a in "${ABLATIONS[@]}"; do [ -f runs/$a/ckpt_final.pt ] && CK+=(--ckpt "$a=runs/$a/ckpt_final.pt"); done
python evaluate.py --data data/ugan --split test --tag ugan_ablation --methods "${CK[@]}" \
    --metrics psnr ssim uiqm uciqe --device "$DEVICE" --workers "$EVAL_WORKERS" --save-samples 4

# Table 5: lambda1 / lambda2 grid; (1, 0.1) is the full model's checkpoint at SWEEP_ITERS
if [ "$SWEEP_ITERS" -gt 0 ]; then
    CK=(--ckpt "l1_1_l2_0.1=runs/full/ckpt_iter${SWEEP_ITERS}.pt")
    for r in runs/sweep_*; do [ -f $r/ckpt_final.pt ] && CK+=(--ckpt "$(basename $r | sed 's/^sweep_//')=$r/ckpt_final.pt"); done
    python evaluate.py --data data/ugan --split test --tag ugan_lambda --methods "${CK[@]}" \
        --metrics psnr ssim --device "$DEVICE" --workers "$EVAL_WORKERS" --save-samples 0
fi

RUNS=(runs/full); for a in "${ABLATIONS[@]}"; do [ -d runs/$a ] && RUNS+=(runs/$a); done
for d in "${SETS[@]}"; do
    python make_figures.py --tag $(tag $d) --data data/$d --split test \
        --methods input clahe grayworld ancuti_rc full --runs "${RUNS[@]}" \
        --ckpt full=runs/full/ckpt_final.pt --device "$DEVICE"
done
python make_figures.py --tag ugan_ablation --data data/ugan --split test \
    --methods wo_rnet_gnet wo_rnet wo_gnet w_rnet_gnet w_ce --n 4 --device "$DEVICE"

# collect the small training artefacts
for r in runs/full runs/wo_* runs/w_* runs/sweep_*; do
    [ -d "$r" ] || continue
    mkdir -p results/runs/$(basename $r)
    cp $r/log.csv $r/summary.json $r/args.json results/runs/$(basename $r)/ 2>/dev/null
done
mkdir -p results/checkpoints && cp runs/full/ckpt_final.pt results/checkpoints/uienet_full.pt
