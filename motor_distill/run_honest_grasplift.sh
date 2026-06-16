#!/bin/bash
cd /root/openpi
export PYTHONPATH=/root/LIBERO-PRO:/root/openpi/motor_distill MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/root/egl_nvidia.json
BD=/root/LIBERO-Pro-data/bddl_files; ID=/root/LIBERO-Pro-data/init_files
for AX in swap object; do
  echo "###### grasp-lift=0.08 binder=dino loc=depthflip suite=libero_object_$AX ######"
  python3 motor_distill/eval_physics_place.py --head data/motor_head_coadapt.pt \
    --bddl-dir $BD/libero_object_$AX --init-dir $ID/libero_object_$AX \
    --container basket --reflex 0 --binder dino --loc depthflip --z-cont-center 0.035 --rim-h 0.075 --clear 0.14 --grasp-lift 0.08 \
    --n 10 --trials 5 --init-start 20 \
    --proto-dir $BD/libero_object_$AX --proto-init-dir $ID/libero_object_$AX --proto-inits 30,32,34
done
echo "ALLDONE"
