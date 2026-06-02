#!/bin/bash
# DMP grasp test across tasks x perturbations (the re-targetability proof).
#   nohup env N=30 bash motor_distill/run_dmp_grasp.sh > /root/dmpg.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-/root/egl_nvidia.json}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/motor_distill:$PWD/third_party/libero:${PYTHONPATH:-}
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl
N=${N:-30}; TASKS=${TASKS:-3 9}; PERTS=${PERTS:-0 5 10}
mkdir -p logs/dmpg
for T in $TASKS; do for P in $PERTS; do
  ( echo N | uv run --no-sync python motor_distill/eval_dmp_grasp.py \
      --task-idx "$T" --pert-dir data/keystone/pert${P} --ref-dir data/keystone/pert0 --n "$N" \
      > "logs/dmpg/dmpg_t${T}_p${P}.log" 2>&1 ) &
done; done
wait
echo "DMPG_DONE"
