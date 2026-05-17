#!/usr/bin/env bash
# Phase A publishable matrix — runs all required experiments in one queue.
#
# Goal: enough evidence for the headline result "physics-grounded MPPI on
# Pi0.5 gives +Xpp lift on LIBERO-PRO across seeds and tasks." Per the lit
# audit, the publishable bar is N≥10 trials × ≥3 seeds × ≥2 task suites.
#
# This script targets:
#   - Task 3 ("put bowl in drawer + close"): the strong-signal task. Three
#     seeds × N=10. This is the headline.
#   - Task 5 ("place book in caddy"): second single-object validator at one
#     seed × N=10 to claim task-generalization within single-object.
#   - Task 7 ("put both soup + cream cheese in basket"): multi-object
#     validator at 5cm and 10cm — already shows 0pp lift; we re-run with
#     early-exit on so wall time drops. Documented negative case in paper.
#
# Estimated wall time on the GPU box we used (~80s/MPPI episode at task 3):
#   - Task 3 × 3 seeds × 10 trials × 2 modes (baseline + MPPI) ≈ 1.3 hours
#   - Task 5 × 1 seed × 10 trials × 2 modes                   ≈ 0.5 hours
#   - Task 7 × 5cm × 1 seed × 5 × 2                            ≈ 0.25 hours
#   - Task 7 × 10cm × 1 seed × 5 × 2 (with early-exit)         ≈ 0.15 hours
#   Total: ~2.5 hours of GPU. Cheap; do it overnight.
#
# Run from /openpi on the GPU box:
#   bash scripts/run_phase_a_matrix.sh 2>&1 | tee data/contact_mpc/phase_a_matrix.log

set -euo pipefail

ROLLOUT_DIR="data/contact_mpc/refined_rollouts_libero10"
TRACE_DIR="data/contact_mpc/failure_traces_libero10"
mkdir -p "$ROLLOUT_DIR" "$TRACE_DIR"

PYBIN="uv run python3 -u"
export PYTHONPATH="src:third_party/libero"

# Common args used by every run
COMMON_ARGS=(
  --mppi-K 16 --mppi-sigma 0.1 --mppi-lambda 0.1
  --log-refined-rollouts "$ROLLOUT_DIR"
  --log-failure-traces "$TRACE_DIR"
  --mppi-early-exit-calls 6   # avoid burning compute on stuck trials
)

run_cell() {
  local task="$1" pert="$2" seed="$3" trials="$4"
  echo "=================================================================="
  echo "[$(date +%H:%M:%S)] task=$task pert=${pert}cm seed=$seed N=$trials"
  echo "=================================================================="
  $PYBIN scripts/run_reason_v3_mppi.py \
    --task-suite libero_10 --task-idx "$task" \
    --num-trials "$trials" --perturbation-cm "$pert" --seed "$seed" \
    "${COMMON_ARGS[@]}"
}

# -------------- Task 3: the headline, 3 seeds × N=10 --------------
for SEED in 7 21 42; do
  run_cell 3 5.0 "$SEED" 10
done

# -------------- Task 5: second single-object validator --------------
run_cell 5 5.0 7 10

# -------------- Task 7: documented multi-object negative case --------------
run_cell 7 5.0  7 5    # saturated baseline → expected ≈0pp
run_cell 7 10.0 7 5    # rescuable-failure boundary → expected ≈0pp, early-exit triggers

echo
echo "=================================================================="
echo "[$(date +%H:%M:%S)] Phase A matrix complete."
echo "Result JSONs:    data/contact_mpc/reason_v3_mppi/*.json"
echo "Refined rollouts: $ROLLOUT_DIR (Phase B training data)"
echo "Failure traces:   $TRACE_DIR (diagnostic data)"
echo "=================================================================="
