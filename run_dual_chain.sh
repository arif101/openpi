#!/bin/bash
# Dual-goal / NO-STATE-MACHINE motor pipeline on libero_object (known-working pick-place skill).
# Validates that removing the held-based state machine does NOT regress the 0.533 swap result.
set -e
cd /root/openpi
export PYTHONPATH=/root/LIBERO-PRO:motor_distill
export MUJOCO_GL=egl
PY=.venv/bin/python
BD=/root/LIBERO-Pro-data/bddl_files
ID=/root/LIBERO-Pro-data/init_files

echo "=== COLLECT std (object_object) ==="
$PY motor_distill/collect_motor_data_hide.py --bddl-dir $BD/libero_object_object --init-dir $ID/libero_object_object \
    --out data/dual_std --container basket --trials 8 --seeds 0,1

echo "=== COLLECT swap (object_swap) ==="
$PY motor_distill/collect_motor_data_hide.py --bddl-dir $BD/libero_object_swap --init-dir $ID/libero_object_swap \
    --out data/dual_swap --container basket --trials 8 --seeds 0,1

echo "=== CANON ==="
$PY motor_distill/make_canon.py --src data/dual_std  --out data/dual_std_canon
$PY motor_distill/make_canon.py --src data/dual_swap --out data/dual_swap_canon

echo "=== TRAIN (dual-goal, no held) ==="
$PY motor_distill/train_wristcam_motor.py --data data/dual_std_canon,data/dual_swap_canon \
    --out data/motor_head_dual_canon.pt --epochs 60

echo "=== ORACLE-ISOLATION GATE (motor alone, oracle binder + oracle loc) on object_swap held-out inits ==="
$PY motor_distill/eval_e2e_dual.py --head data/motor_head_dual_canon.pt \
    --bddl-dir $BD/libero_object_swap --init-dir $ID/libero_object_swap --container basket \
    --binder oracle --loc oracle --n 10 --trials 3 --init-start 20

echo "=== HONEST E2E (dino binder + rayplane loc) on object_swap ==="
$PY motor_distill/eval_e2e_dual.py --head data/motor_head_dual_canon.pt \
    --bddl-dir $BD/libero_object_swap --init-dir $ID/libero_object_swap --container basket \
    --binder dino --loc rayplane --proto-dir $BD/libero_object_swap --proto-init-dir $ID/libero_object_swap \
    --proto-inits 30,32,34 --n 10 --trials 3 --init-start 20
echo "DUALCHAIN_EXIT=0"
