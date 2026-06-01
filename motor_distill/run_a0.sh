#!/bin/bash
# Launch all A0 keystone runs in parallel: 4 tasks x {plain, equiv}.
# Run from the repo root on the GPU box (under setsid so it survives ssh close):
#   setsid bash motor_distill/run_a0.sh 30 > /root/a0_all.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-/root/egl_nvidia.json}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/motor_distill:$PWD/third_party/libero:$PYTHONPATH
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl
N=${1:-30}
mkdir -p logs/a0
for V in plain equiv; do
  for T in 0 3 6 9; do
    ( echo N | uv run --no-sync python motor_distill/eval_a0.py \
        --ckpt data/keystone/ckpt/head_${V}.pt --task-idx "$T" \
        --trace-dir data/keystone/pert0 --n "$N" --replan 8 \
        > "logs/a0/a0_${V}_t${T}.log" 2>&1 ) &
  done
done
wait
echo "ALL_A0_DONE"
