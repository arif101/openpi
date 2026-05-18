#!/usr/bin/env bash
# Small focused recollection for the recovery-regime diagnostic.
#
# Single perturbation (5cm), single mode (baseline only — MPPI replay
# drifts too much), 4 trials × 3 seeds × 2 tasks = 24 baseline rollouts.
# Each task gets ~12 episodes; ~5-6 will fail at 5cm based on prior data,
# giving us a real failure pool with reproducible init_sim_state.
#
# Wall time: ~15-20 min on RTX 6000 Ada.

set -euo pipefail
OUT_DIR="data/contact_mpc/physvla_traces_diag"
mkdir -p "$OUT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="src:third_party/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
PYBIN="uv run python3 -u"

for TASK in 3 7; do
  for SEED in 7 21 42; do
    echo "=== task=$TASK seed=$SEED pert=5.0cm ==="
    $PYBIN scripts/run_reason_v3_mppi.py \
      --task-suite libero_10 --task-idx "$TASK" \
      --num-trials 4 --perturbation-cm 5.0 --seed "$SEED" \
      --mppi-K 16 --mppi-sigma 0.1 --mppi-lambda 0.1 \
      --log-physics-traces "$OUT_DIR" \
      --physics-trace-image-stride 10
  done
done

echo
echo "Diagnostic corpus done. Files: $(ls $OUT_DIR | wc -l), size: $(du -sh $OUT_DIR | awk '{print $1}')"
