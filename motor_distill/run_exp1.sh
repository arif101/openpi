#!/bin/bash
# Exp 1: 3-arm OOD grasp. {ndp, plain} x pert {0,5,10}; arm A (Pi0.5-lift) reported per job.
#   nohup env T=3 N=30 bash motor_distill/run_exp1.sh > /root/exp1.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-/root/egl_nvidia.json}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/motor_distill:$PWD/third_party/libero:${PYTHONPATH:-}
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl
T=${T:-3}; N=${N:-30}
mkdir -p logs/exp1
for KIND in ndp plain; do for P in 0 5 10; do
  ( echo N | uv run --no-sync python motor_distill/eval_heads.py \
      --ckpt data/keystone/heads/head_${KIND}_t${T}.pt --task-idx "$T" \
      --pert-dir data/keystone/pert${P} --n "$N" \
      > "logs/exp1/exp1_${KIND}_t${T}_p${P}.log" 2>&1 ) &
done; done
wait
echo "EXP1_DONE"
