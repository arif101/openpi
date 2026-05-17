#!/usr/bin/env bash
# Phase A matrix v2 — gated grip-stability cost.
#
# Same 6 cells as the prior matrices. Difference vs matrix 3 (grip):
#   --grip-radius 0.05   only count grip-cost when EE was already within 5cm
#                        of the tracked object. Fixes the approach-phase
#                        jitter that drove matrix 3 to -13.3pp pooled.
#
# Output dir distinct so we keep the negative-result matrix 3 around
# for the writeup ("ungated grip-cost fails; gated form recovers").

set -euo pipefail

OUT_DIR="data/contact_mpc/reason_v3_mppi_grip_v2"
ROLLOUT_DIR="data/contact_mpc/refined_rollouts_libero10_grip_v2"
TRACE_DIR="data/contact_mpc/failure_traces_libero10_grip_v2"
mkdir -p "$OUT_DIR" "$ROLLOUT_DIR" "$TRACE_DIR"

export PATH="$HOME/.local/bin:$PATH"
PYBIN="uv run python3 -u"
export PYTHONPATH="src:third_party/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

COMMON_ARGS=(
  --mppi-K 16 --mppi-sigma 0.1 --mppi-lambda 0.1
  --log-refined-rollouts "$ROLLOUT_DIR"
  --log-failure-traces "$TRACE_DIR"
  --mppi-early-exit-calls 6
  --mppi-trust-threshold 0.0
  --w-grip 20.0
  --grip-radius 0.05
  --output-dir "$OUT_DIR"
)

run_cell() {
  local task="$1" pert="$2" seed="$3" trials="$4"
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] task=$task pert=${pert}cm seed=$seed N=$trials (GRIP-v2 r=5cm)"
  echo "=================================================================="
  $PYBIN scripts/run_reason_v3_mppi.py \
    --task-suite libero_10 --task-idx "$task" \
    --num-trials "$trials" --perturbation-cm "$pert" --seed "$seed" \
    "${COMMON_ARGS[@]}"
}

for SEED in 7 21 42; do
  run_cell 3 5.0 "$SEED" 10
done
run_cell 5 5.0 7 10
run_cell 7 5.0  7 5
run_cell 7 10.0 7 5

echo
echo "[$(date +%H:%M:%S)] Grip-v2 matrix complete."
echo "  4-way compare: gate-off / gate-on / grip-v1 / grip-v2"
