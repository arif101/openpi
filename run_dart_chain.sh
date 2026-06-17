#!/bin/bash
# PHASE-2 goal-correcting build: DART correction-data (diverse positions) -> retrain -> posshift curve vs pi0.5 ref.
cd /root/openpi
export PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl
PY=.venv/bin/python
STDB=/root/LIBERO-PRO/libero/libero/bddl_files/libero_object
STDI=/root/LIBERO-PRO/libero/libero/init_files/libero_object

echo "=== [1] collect DART-posdiv correction data (pi0.5+hide+perturb+displace) ==="
$PY motor_distill/collect_dart_posdiv.py --bddl-dir $STDB --init-dir $STDI \
    --out data/motor_demos_dart --trials 10 --seeds 0,1 --max-disp 0.10 --dart-noise 0.02 --dart-prob 0.35 2>&1 | grep -vi deprecat
echo "dart demos: $(ls data/motor_demos_dart 2>/dev/null | wc -l)"

echo "=== [2] train goal-correcting motor on standard + DART-correction -> motor_head_dart.pt ==="
$PY motor_distill/train_wristcam_motor.py --data data/motor_demos,data/motor_demos_dart \
    --out data/motor_head_dart.pt --epochs 60 2>&1 | grep -vi deprecat | tail -8

echo "=== [3] posshift  NEW dart head  hide=1 (deployment cond) ==="
$PY motor_distill/eval_posshift.py --head data/motor_head_dart.pt --bddl-dir $STDB --init-dir $STDI \
    --n 10 --trials 3 --offsets 0,0.06,0.12,0.16 --init-start 30 --surface-goal 0 --hide 1 2>&1 | grep -E "cm  success|cm :|delta"

echo "=== [4] posshift  NEW dart head  hide=0 ==="
$PY motor_distill/eval_posshift.py --head data/motor_head_dart.pt --bddl-dir $STDB --init-dir $STDI \
    --n 10 --trials 3 --offsets 0,0.06,0.12,0.16 --init-start 30 --surface-goal 0 --hide 0 2>&1 | grep -E "cm  success|cm :|delta"

echo "=== [5] posshift  OLD head  hide=1 (consistent baseline vs pi0.5) ==="
$PY motor_distill/eval_posshift.py --head data/motor_head.pt --bddl-dir $STDB --init-dir $STDI \
    --n 10 --trials 3 --offsets 0,0.06,0.12,0.16 --init-start 30 --surface-goal 0 --hide 1 2>&1 | grep -E "cm  success|cm :|delta"

echo "DARTCHAIN_EXIT=0"
