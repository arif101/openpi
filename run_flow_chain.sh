#!/bin/bash
# Precision-preserving distillation test: flow-matching head vs L1-regression head, same pi0.5 data.
cd /root/openpi
export PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl
PY=.venv/bin/python
STDB=/root/LIBERO-PRO/libero/libero/bddl_files/libero_object
STDI=/root/LIBERO-PRO/libero/libero/init_files/libero_object
SWB=/root/LIBERO-Pro-data/bddl_files/libero_object_swap
SWI=/root/LIBERO-Pro-data/init_files/libero_object_swap

echo "=== [1] train FLOW-MATCHING motor on pi0.5 standard demos (same data as L1 head) ==="
$PY motor_distill/train_flowmotor.py --data data/motor_demos --out data/flowmotor.pt --epochs 120 2>&1 | grep -vi deprecat | tail -12

echo "=== [2] FLOW  standard posshift (vs L1: 0.73/0.63/0.30/0.33) ==="
$PY motor_distill/eval_flowmotor.py --head data/flowmotor.pt --bddl-dir $STDB --init-dir $STDI \
    --n 10 --trials 3 --offsets 0,0.06,0.12,0.16 --init-start 30 --surface-goal 0 --hide 0 2>&1 | grep -E "cm :|cm  success"

echo "=== [3] FLOW  swap hide0  (vs L1 0.50) ==="
$PY motor_distill/eval_flowmotor.py --head data/flowmotor.pt --bddl-dir $SWB --init-dir $SWI \
    --n 10 --trials 3 --offsets 0 --init-start 20 --surface-goal 0 --hide 0 2>&1 | grep -E "cm  success"

echo "=== [4] FLOW  swap hide1  (vs L1 0.567) ==="
$PY motor_distill/eval_flowmotor.py --head data/flowmotor.pt --bddl-dir $SWB --init-dir $SWI \
    --n 10 --trials 3 --offsets 0 --init-start 20 --surface-goal 0 --hide 1 2>&1 | grep -E "cm  success"

echo "FLOWCHAIN_EXIT=0"
