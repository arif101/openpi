#!/bin/bash
# Grasp-only servo test across tasks x perturbations.
#   nohup env V=plain N=40 bash motor_distill/run_grasp.sh > /root/grasp.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-/root/egl_nvidia.json}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/motor_distill:$PWD/third_party/libero:${PYTHONPATH:-}
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl
V=${V:-plain}; N=${N:-40}; TASKS=${TASKS:-3 9}; PERTS=${PERTS:-0 5 10}
mkdir -p logs/grasp
for T in $TASKS; do for P in $PERTS; do
  ( echo N | uv run --no-sync python motor_distill/eval_grasp.py \
      --ckpt data/keystone/ckpt/head_${V}.pt --task-idx "$T" \
      --pert-dir data/keystone/pert${P} --ref-dir data/keystone/pert0 \
      --n "$N" --replan 2 \
      > "logs/grasp/grasp_${V}_t${T}_p${P}.log" 2>&1 ) &
done; done
wait
echo "GRASP_DONE"
