#!/usr/bin/env bash
# Same matrix as run_phase_a_matrix.sh, but with --mppi-trust-threshold 0.03
# active. Tests the hypothesis that the −20pp regressions on task 3 seed=7
# and task 7 10cm seed=7 are noise-from-averaging artifacts that the gate
# suppresses, while the +10/+30pp rescues survive.
#
# Writes JSONs into a separate output dir so the gate-off baseline JSONs
# stay intact for direct comparison.
#
# Estimated wall time: same as no-gate matrix (~1.5h on RTX 6000 Ada).
# Actually slightly faster because gate-rejected calls skip the action
# substitution step (negligible).

set -euo pipefail

OUT_DIR="data/contact_mpc/reason_v3_mppi_trust"
ROLLOUT_DIR="data/contact_mpc/refined_rollouts_libero10_trust"
TRACE_DIR="data/contact_mpc/failure_traces_libero10_trust"
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
  --mppi-trust-threshold 0.03
  --output-dir "$OUT_DIR"
)

run_cell() {
  local task="$1" pert="$2" seed="$3" trials="$4"
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] task=$task pert=${pert}cm seed=$seed N=$trials (TRUST 0.03)"
  echo "=================================================================="
  $PYBIN scripts/run_reason_v3_mppi.py \
    --task-suite libero_10 --task-idx "$task" \
    --num-trials "$trials" --perturbation-cm "$pert" --seed "$seed" \
    "${COMMON_ARGS[@]}"
}

# Task 3: same 3 seeds × N=10 as the no-gate matrix
for SEED in 7 21 42; do
  run_cell 3 5.0 "$SEED" 10
done

# Task 5: single-object validator
run_cell 5 5.0 7 10

# Task 7: documented multi-object negative case
run_cell 7 5.0  7 5
run_cell 7 10.0 7 5

echo
echo "=================================================================="
echo "[$(date +%H:%M:%S)] Trust-gate matrix complete."
echo "  Compare: data/contact_mpc/reason_v3_mppi/*.json (gate off)"
echo "           data/contact_mpc/reason_v3_mppi_trust/*.json (gate 0.03)"
echo "=================================================================="
