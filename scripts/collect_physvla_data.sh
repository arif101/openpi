#!/usr/bin/env bash
# PhysVLA Week 1: collect physics-rich rollout data for training the
# latent dynamics model + physics aux heads.
#
# We deliberately roll out Pi0.5 *without* MPPI — we want the model to
# learn dynamics from any policy, not specifically the MPPI-refined one.
# Diverse seeds × diverse perturbations × multiple tasks gives the
# encoder/dynamics broad coverage.
#
# Storage estimate:
#   Per step: ~3kB physics state + (1/5) × 150kB images = ~33kB/step
#   Per episode (~300 steps avg): ~10MB
#   Across 200 episodes: ~2GB
#
# Wall-time estimate: 200 episodes × ~10s/episode (no MPPI) = ~35 min
# on RTX 6000 Ada.
#
# Run from /openpi:
#   bash scripts/collect_physvla_data.sh

set -euo pipefail

OUT_DIR="data/contact_mpc/physvla_traces"
mkdir -p "$OUT_DIR"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="src:third_party/libero"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
PYBIN="uv run python3 -u"

# Phase A used 3 seeds × 6 cells. Here we expand: multiple tasks, multiple
# seeds, multiple perturbation magnitudes. Pi0.5 only (no MPPI), so
# diversity comes from initial state perturbations + seed variation.
TASKS=(0 3 5 7 9)   # mix of single-object + multi-object + single-stage
SEEDS=(7 21 42 99 137)
PERTS=(0.0 2.5 5.0 7.5 10.0)

# Tag each cell so we can resume / parallelize later.
for TASK in "${TASKS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    for PERT in "${PERTS[@]}"; do
      echo "============================================================"
      echo "[$(date +%H:%M:%S)] task=$TASK seed=$SEED pert=${PERT}cm"
      echo "============================================================"
      # 2 trials per cell × 25 cells × 5 perturbations = ~250 episodes
      $PYBIN scripts/run_reason_v3_mppi.py \
        --task-suite libero_10 --task-idx "$TASK" \
        --num-trials 2 --perturbation-cm "$PERT" --seed "$SEED" \
        --mppi-K 16 --mppi-sigma 0.1 --mppi-lambda 0.1 \
        --log-physics-traces "$OUT_DIR" \
        --physics-trace-image-stride 5
    done
  done
done

echo
echo "[$(date +%H:%M:%S)] PhysVLA data collection done."
echo "Files:  $(ls $OUT_DIR | wc -l) trace npz's"
echo "Size:   $(du -sh $OUT_DIR | awk '{print $1}')"
