#!/bin/bash
# Isolate the canon scheme for the NO-STATE-MACHINE dual-goal motor. Reuses already-collected dual_std/dual_swap data.
# episode mode already measured = 0.267 oracle-isolation swap. Test perstep_obj and gripkey.
set -e
cd /root/openpi
export PYTHONPATH=/root/LIBERO-PRO:motor_distill
export MUJOCO_GL=egl
PY=.venv/bin/python
BD=/root/LIBERO-Pro-data/bddl_files
ID=/root/LIBERO-Pro-data/init_files

for MODE in perstep_obj gripkey; do
  echo "########## MODE=$MODE ##########"
  echo "=== CANON ($MODE) ==="
  $PY motor_distill/make_canon.py --src data/dual_std  --out data/dual_std_$MODE  --mode $MODE
  $PY motor_distill/make_canon.py --src data/dual_swap --out data/dual_swap_$MODE --mode $MODE
  echo "=== TRAIN ($MODE) ==="
  $PY motor_distill/train_wristcam_motor.py --data data/dual_std_$MODE,data/dual_swap_$MODE \
      --out data/motor_head_dual_$MODE.pt --epochs 60
  echo "=== ORACLE-ISOLATION ($MODE) on object_swap ==="
  $PY motor_distill/eval_e2e_dual.py --head data/motor_head_dual_$MODE.pt \
      --bddl-dir $BD/libero_object_swap --init-dir $ID/libero_object_swap --container basket \
      --binder oracle --loc oracle --canon-mode $MODE --n 10 --trials 3 --init-start 20
done
echo "CANONABL_EXIT=0"
