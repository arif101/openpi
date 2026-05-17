#!/usr/bin/env bash
# Same 6-cell matrix as run_phase_a_matrix.sh and run_phase_a_matrix_trust.sh
# but with --w-grip 20 ON and --mppi-trust-threshold 0.0 (gate disabled).
#
# Tests the diagnostic finding that 71% of failures are FAILED MANIPULATION
# (EE reaches object, gripper actuates, but object lost in carry). The grip
# term penalizes candidates where ||EE - tracked_obj|| grows during the
# 5-step rollout (object slipping away from EE).
#
# Output dir distinct from the other two matrices so we get a clean 3-way
# ablation: gate-off | gate-on | grip-on.

set -euo pipefail

OUT_DIR="data/contact_mpc/reason_v3_mppi_grip"
ROLLOUT_DIR="data/contact_mpc/refined_rollouts_libero10_grip"
TRACE_DIR="data/contact_mpc/failure_traces_libero10_grip"
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
  --output-dir "$OUT_DIR"
)

run_cell() {
  local task="$1" pert="$2" seed="$3" trials="$4"
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] task=$task pert=${pert}cm seed=$seed N=$trials (GRIP 20)"
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
echo "[$(date +%H:%M:%S)] Grip-stability matrix complete."
echo "  Three-way compare:"
echo "    gate-off:  data/contact_mpc/reason_v3_mppi/"
echo "    gate-on:   data/contact_mpc/reason_v3_mppi_trust/"
echo "    grip-on:   data/contact_mpc/reason_v3_mppi_grip/"
