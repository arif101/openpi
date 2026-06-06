#!/bin/bash
# ONE-COMMAND training-free gradient-steering sweep — run on a fresh GPU box to get the first
# LIBERO-PRO TASK-axis number for our binding direction. NO training. ~1-2 GPU-hours total.
#
# PREREQUISITES on the GPU box (standard openpi setup, per the reference harness):
#   - openpi repo cloned at $REPO (git pull for pi05_guided.py + sample_actions_guided)
#   - .venv with openpi + JAX-CUDA + transformers + LIBERO (third_party/libero)
#   - HF token exported for gated PaliGemma:  export HF_TOKEN=...   (rotate the leaked one)
#   - data/libero_pro/bddl_files/ present. If absent, restore from either:
#       * local backup:  scp -r ~/openpi-box-backup/2026-06-05-motor/libero_pro_bddls  <box>:$REPO/data/libero_pro/bddl_files
#       * or the LIBERO-PRO release (arXiv 2510.03827).
#   - pi05_libero checkpoint auto-downloads from GCS on first run.
#
# Usage:  bash motor_distill/run_guided_sweep.sh
set -e
REPO=${REPO:-/root/openpi}
cd "$REPO"
export PYTHONPATH=third_party/libero MUJOCO_GL=egl
mkdir -p logs
SEEDS="7 42 123"
WEIGHTS="0.0 0.05 0.1 0.2 0.4"   # guide_w=0.0 MUST reproduce stock pi0.5 TASK baseline (sanity gate)

for w in $WEIGHTS; do
  for s in $SEEDS; do
    log="logs/guided_w${w}_s${s}.log"
    echo "=== guide_w=$w seed=$s -> $log ==="
    .venv/bin/python motor_distill/pi05_guided.py --seed "$s" --n 10 --guide-w "$w" > "$log" 2>&1 || echo "FAILED w=$w s=$s"
    grep "OFFICIAL:" "$log" | tail -1
  done
done

echo ""
echo "================ SWEEP SUMMARY (official place-T) ================"
for w in $WEIGHTS; do
  vals=$(for s in $SEEDS; do grep -oE "OFFICIAL: [0-9.]+" "logs/guided_w${w}_s${s}.log" 2>/dev/null | grep -oE "[0-9.]+$"; done | tr '\n' ' ')
  echo "guide_w=$w :  seeds[$SEEDS] = $vals"
done
echo "================================================================="
echo "READ guide_w=0.0 FIRST: it must match stock pi0.5 TASK-axis (the sanity gate)."
echo "Then the best guide_w vs CAG 21.7% / VLS (+13% LIBERO-PRO) is the training-free result."
echo "GUIDED_SWEEP_DONE"
