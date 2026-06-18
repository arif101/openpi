#!/bin/bash
# Honest unified depth multi-instance head: train on DISJOINT inits 0-19, eval on 20+ (no overlap).
cd /root/openpi
EV="PYTHONPATH=/root/LIBERO-PRO:motor_distill MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/root/egl_nvidia.json"
BS=/root/LIBERO-Pro-data/bddl_files; IS=/root/LIBERO-Pro-data/init_files

echo "=== [1/4] wait for object depth collect ==="
while ! grep -q GROUNDDEPTH_EXIT gdepth_obj.log 2>/dev/null; do sleep 15; done
echo "obj collect done"

echo "=== [2/4] recollect spatial @ 20 inits (train 0-19, disjoint from eval 20+) ==="
eval $EV python3 motor_distill/collect_ground_depth.py --bddl-dir $BS/libero_spatial_swap --init-dir $IS/libero_spatial_swap --out data/ground_depth_spatial20 --res 448 --n 10 --n-inits 20

echo "=== [3/4] train UNIFIED inst head on object+spatial (disjoint train inits) ==="
PYTHONPATH=motor_distill /root/piv/bin/python motor_distill/train_inst_head.py --data data/ground_depth_obj,data/ground_depth_spatial20 --out data/inst_head_all.pt --epochs 80

echo "=== [4/4] eval SPATIAL DISJOINT (init-start 20, inits 20-22 NOT in train 0-19) ==="
eval $EV python3 motor_distill/eval_physics_place.py --bddl-dir $BS/libero_spatial_swap --init-dir $IS/libero_spatial_swap \
  --head data/motor_head_BEST.pt --reflex 1 --clear 0.14 --lang-plan 1 --container plate --loc depthflip --cont-depth 1 \
  --binder learned --ground-head data/ground_head_os.pt --inst-head data/inst_head_all.pt --proto-inits 42,44,46 \
  --trials 3 --n 10 --init-start 20
echo "UHEAD_CHAIN_EXIT=0"
