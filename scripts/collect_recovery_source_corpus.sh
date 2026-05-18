#!/usr/bin/env bash
# Source corpus for the backward-reachability data generator.
#
# Need ~200 successful baseline traces with init_sim_state at 5cm
# perturbation across multiple seeds for single-object tasks. Expected
# yield from generator: ~5 valid pairs per trace = ~1000 pairs total,
# enough to train a small recovery adapter.

set -euo pipefail
OUT_DIR="data/contact_mpc/recovery_source_traces"
mkdir -p "$OUT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="src:third_party/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
PYBIN="uv run python3 -u"

# 2 tasks × 5 seeds × 10 trials per cell × 1 perturbation = 100 baseline rollouts
# Baseline success rate at 5cm task 3 is ~60-90% across seeds → expect ~75 successful
# (Task 5 success rate ~70%, expect ~35 more) → ~110 successful baseline traces.
for TASK in 3 5; do
  for SEED in 7 21 42 99 137; do
    echo "=== task=$TASK seed=$SEED ==="
    $PYBIN scripts/run_reason_v3_mppi.py \
      --task-suite libero_10 --task-idx "$TASK" \
      --num-trials 10 --perturbation-cm 5.0 --seed "$SEED" \
      --mppi-K 16 --mppi-sigma 0.1 --mppi-lambda 0.1 \
      --log-physics-traces "$OUT_DIR" \
      --physics-trace-image-stride 10
  done
done

# Quick stats
N_OK=$(ls "$OUT_DIR" | grep "PHYS_OK_.*_baseline_" | wc -l)
N_FAIL=$(ls "$OUT_DIR" | grep "PHYS_FAIL_.*_baseline_" | wc -l)
echo
echo "Source corpus done."
echo "  Successful baseline traces: $N_OK"
echo "  Failed baseline traces:     $N_FAIL"
echo "  Total: $(ls $OUT_DIR | wc -l)"
echo "  Storage: $(du -sh $OUT_DIR | awk '{print $1}')"
