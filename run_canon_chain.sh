#!/bin/bash
# SE(2)-canonicalization probe: does removing the goal-bearing DOF flatten the position-shift curve?
cd /root/openpi
export PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl
PY=.venv/bin/python
STDB=/root/LIBERO-PRO/libero/libero/bddl_files/libero_object
STDI=/root/LIBERO-PRO/libero/libero/init_files/libero_object

echo "=== [1] canonicalize pi0.5 distillation data ==="
$PY motor_distill/make_canon.py --src data/motor_demos --out data/motor_demos_canon 2>&1 | grep -vi deprecat | tail -3

echo "=== [2] train WristMotor on CANONICALIZED data -> motor_head_canon.pt ==="
$PY motor_distill/train_wristcam_motor.py --data data/motor_demos_canon --out data/motor_head_canon.pt --epochs 60 2>&1 | grep -vi deprecat | tail -6

echo "=== [3] CANON eval posshift (d=0 must be ~0.73 or convention bug; flat => bearing was the leak) ==="
$PY motor_distill/eval_canon_posshift.py --head data/motor_head_canon.pt --canon 1 --bddl-dir $STDB --init-dir $STDI \
    --n 10 --trials 3 --offsets 0,0.06,0.12,0.16 --init-start 30 --hide 0 2>&1 | grep -E "cm :|cm  success|delta|CANON motor"

echo "CANONCHAIN_EXIT=0"
