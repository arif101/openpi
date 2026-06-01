#!/bin/bash
# Cheap closed-loop precision knob sweep: replan + sample-steps on one task/head.
#   nohup env T=3 V=plain N=30 bash motor_distill/run_knob.sh > /root/knob.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PATH=$HOME/.local/bin:$PATH
export __EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-/root/egl_nvidia.json}
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONPATH=$PWD/motor_distill:$PWD/third_party/libero:${PYTHONPATH:-}
export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYOPENGL_PLATFORM=egl
T=${T:-3}; V=${V:-plain}; N=${N:-30}
mkdir -p logs/knob
for cfg in "1 10" "2 10" "4 10" "2 30"; do
  set -- $cfg; R=$1; S=$2
  ( echo N | uv run --no-sync python motor_distill/eval_a0.py \
      --ckpt data/keystone/ckpt/head_${V}.pt --task-idx "$T" \
      --trace-dir data/keystone/pert0 --n "$N" --replan "$R" --sample-steps "$S" --g-mode step \
      > "logs/knob/knob_t${T}_${V}_r${R}_s${S}.log" 2>&1 ) &
done
wait
echo "KNOB_DONE"
