#!/bin/bash
# Phase 2 chain: wait for position-diverse collection -> retrain wrist motor on standard+swap ->
# eval OLD vs NEW under IDENTICAL deployment conditions (hide+surface-goal) on HELD-OUT swap inits.
cd /root/openpi
export PYTHONPATH=/root/LIBERO-PRO:motor_distill
export MUJOCO_GL=egl
PY=.venv/bin/python
SWAP_B=/root/LIBERO-Pro-data/bddl_files/libero_object_swap
SWAP_I=/root/LIBERO-Pro-data/init_files/libero_object_swap
STD_B=/root/LIBERO-PRO/libero/libero/bddl_files/libero_object
STD_I=/root/LIBERO-PRO/libero/libero/init_files/libero_object

echo "=== [1] waiting for collection (COLLECTHIDE_EXIT) ==="
while ! grep -q COLLECTHIDE_EXIT=0 logs/collect_swap.log 2>/dev/null; do sleep 30; done
echo "collection done; raw swap dir (may contain stale files):"; ls data/motor_demos_swap 2>/dev/null | wc -l

echo "=== [1b] stage ONLY this-run swap demos (newer than 20:05 marker; stale dag_* are <=19:42) -> clean dir ==="
mkdir -p data/motor_demos_swap_clean
find data/motor_demos_swap -name "*.npz" -newermt "2026-06-07 20:05" -exec cp {} data/motor_demos_swap_clean/ \;
echo "clean swap demos:"; ls data/motor_demos_swap_clean | wc -l
echo "stale (excluded):"; ls data/motor_demos_swap | grep -c "^dag_" || true

echo "=== [2] train wrist motor on standard+CLEAN-swap -> motor_head_div.pt ==="
$PY motor_distill/train_wristcam_motor.py --data data/motor_demos,data/motor_demos_swap_clean \
    --out data/motor_head_div.pt --epochs 60 2>&1 | grep -vi deprecat

echo "=== [3A] NEW head (div)  swap  held-out(20+)  hide+surface ==="
$PY motor_distill/eval_wristcam_motor.py --head data/motor_head_div.pt --bddl-dir $SWAP_B --init-dir $SWAP_I \
    --container basket --n 10 --trials 3 --init-start 20 --surface-goal 1 --hide 1 2>&1 | grep -vi deprecat

echo "=== [3B] OLD head        swap  held-out(20+)  hide+surface  (fair baseline) ==="
$PY motor_distill/eval_wristcam_motor.py --head data/motor_head.pt --bddl-dir $SWAP_B --init-dir $SWAP_I \
    --container basket --n 10 --trials 3 --init-start 20 --surface-goal 1 --hide 1 2>&1 | grep -vi deprecat

echo "=== [3C] OLD head        swap  held-out(20+)  surface (NO hide, reproduce ~50%) ==="
$PY motor_distill/eval_wristcam_motor.py --head data/motor_head.pt --bddl-dir $SWAP_B --init-dir $SWAP_I \
    --container basket --n 10 --trials 3 --init-start 20 --surface-goal 1 --hide 0 2>&1 | grep -vi deprecat

echo "=== [4] NEW head (div)   STANDARD held-out(20+) surface (regression check) ==="
$PY motor_distill/eval_wristcam_motor.py --head data/motor_head_div.pt --bddl-dir $STD_B --init-dir $STD_I \
    --container basket --n 10 --trials 3 --init-start 20 --surface-goal 1 --hide 0 2>&1 | grep -vi deprecat

echo "CHAIN_EXIT=0"
