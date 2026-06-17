# LIBERO-PRO — Setup & Run Guide (Factored Policy)

End-to-end guide to (1) stand up the **LIBERO-PRO** benchmark, (2) set up the **π0.5 teacher**, and
(3) run our **factored, fully-learned policy** pipeline (collect → canonicalize → distill → evaluate).

> **What this repo is.** A *factored* manipulation policy for LIBERO-PRO: an open-vocab **binder**
> (frozen DINOv2/SAM) + a masked-depth **localizer** + a small **wrist-cam visual-servo motor**
> (~3M params) that consumes a *relative* 3D goal and emits a 10×7 delta-EE action chunk, with a
> **learned place** distilled from offline teachers (π0.5 and an analytic grounded-physics place).
> The deployed policy contains **no π0.5 and no analytic geometry** — those are *offline teachers only*.

---

## 0. TL;DR

```bash
# (A) System env: LIBERO-PRO + robosuite/mujoco (numpy 1.26.4)         [section 1]
# (B) π0.5 teacher in a SEPARATE venv, served over websocket :8000     [section 2]   (optional: only for data collection)
# (C) Pipeline:
#     collect teacher rollouts  ->  make_canon  ->  train_wristcam_motor  ->  eval_physics_place
#                                                                              [sections 3-4]
```

The benchmark axes we target: each LIBERO-PRO suite (`object`, `spatial`, `goal`, `10`) is perturbed
along 4 axes encoded in the directory suffix: `_lan` (instruction paraphrase), `_object`
(object-attribute variant), `_swap` (**position perturbation** — the hard one), `_task`. The bar is
**VLS overall 36.81%** (training-free VLM-reward steering on frozen π0.5); π0.5 alone is 23.69%.

---

## 1. System environment (LIBERO-PRO + simulator)

> Tested on Ubuntu + NVIDIA A100 (CUDA 12, driver ≥ 535). `MUJOCO_GL=egl` for headless rendering.

### 1.1 Clone the LIBERO-PRO **fork** (not stock LIBERO)
Stock LIBERO raises `KeyError: 'bigger_alphabet_soup'` etc. — the PRO fork has the scaled objects.

```bash
git clone https://github.com/Zxy-MLlab/LIBERO-PRO /root/LIBERO-PRO
```

### 1.2 Python deps (pin numpy 1.26.4 — numpy 2.x breaks gym/robosuite)
```bash
pip install "numpy==1.26.4" \
    robosuite==1.4.1 mujoco \
    future matplotlib opencv-python-headless imageio imageio-ffmpeg h5py \
    transformers "huggingface_hub" \
    openpi-client \
    segment-anything
# LIBERO package (first import prompts to write a config; answer "N"):
printf "N\n" | python3 -c "import libero.libero"   # run with PYTHONPATH=/root/LIBERO-PRO
```

### 1.3 Headless EGL rendering
```bash
export MUJOCO_GL=egl
# Point EGL at the NVIDIA ICD (create if missing):
cat > /root/egl_nvidia.json <<'JSON'
{ "file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" } }
JSON
export __EGL_VENDOR_LIBRARY_FILENAMES=/root/egl_nvidia.json
```

### 1.4 Benchmark data (BDDL task files + pruned init states)
Layout we assume (HF: `zhouxueyang/LIBERO-Pro`):
```
/root/LIBERO-Pro-data/
  bddl_files/libero_{object,spatial,goal,10}_{lan,object,swap,task}/*.bddl
  init_files/libero_{object,spatial,goal,10}_{lan,object,swap,task}/*.pruned_init   # torch.load(..., weights_only=False)
```

### 1.5 Standard env for every run in this repo
```bash
export PYTHONPATH=/root/LIBERO-PRO:motor_distill
export MUJOCO_GL=egl __EGL_VENDOR_LIBRARY_FILENAMES=/root/egl_nvidia.json
```

### 1.6 Smoke test
```bash
python3 motor_distill/diag_node_localizer.py   # masked-depth localizer sanity (median <3cm)
```

---

## 2. π0.5 teacher (offline only — needed for *data collection*, not for eval)

π0.5 (a 3B JAX VLA) is used **offline** to generate distillation data. To avoid the
`jax` ↔ `numpy 1.26 / robosuite` dependency clash, run it in a **separate venv** as a **websocket
policy server**; the collector (system python, with robosuite/LIBERO) queries it via `openpi_client`.

```bash
# Separate venv with the FULL openpi (jax + pi0.5), NOT the system env:
python3 -m venv /root/piv
git clone --depth 1 https://github.com/Physical-Intelligence/openpi /root/openpi-full
/root/piv/bin/pip install -e /root/openpi-full            # pulls jax[cuda12], flax, lerobot, ...

# openpi pins a specific lerobot commit (PyPI lerobot 0.3.x breaks `lerobot.common.*`):
/root/piv/bin/pip install "lerobot @ git+https://github.com/huggingface/lerobot@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
/root/piv/bin/pip install "numpy==1.26.4"                 # lerobot may bump numpy 2.x; pin it back

# Serve pi05_libero on :8000 (auto-downloads gs://openpi-assets/checkpoints/pi05_libero on first run).
# IMPORTANT: unset PYTHONPATH so the server uses only its own venv.
cd /root/openpi-full && unset PYTHONPATH && /root/piv/bin/python scripts/serve_policy.py --env LIBERO
```

Verify the round-trip from **system** python:
```python
from openpi_client import websocket_client_policy as wcp
import numpy as np
c = wcp.WebsocketClientPolicy(host="localhost", port=8000)
obs = {"observation/state": np.random.rand(8).astype(np.float32),
       "observation/image": np.random.randint(256,size=(224,224,3),dtype=np.uint8),
       "observation/wrist_image": np.random.randint(256,size=(224,224,3),dtype=np.uint8),
       "prompt": "pick up the bowl and place it on the plate"}
print(c.infer(obs)["actions"].shape)   # -> (10, 7)
```

---

## 3. The pipeline (collect → canonicalize → distill → evaluate)

All commands assume the env from §1.5 and (for collection) the π0.5 server from §2.

### 3.1 Collect distillation data
Two teachers; both log `(wrist, obj_rel, cont_rel, proprio, 10x7 chunk)` per **successful** episode.

**(a) π0.5 teacher (grasp + general manipulation), via websocket:**
```bash
python3 motor_distill/collect_motor_data_ws.py \
  --bddl-dir /root/LIBERO-Pro-data/bddl_files/libero_spatial_swap \
  --init-dir /root/LIBERO-Pro-data/init_files/libero_spatial_swap \
  --container plate --hide 1 --goal-mode depthflip --trials 5 --seeds 0,1 \
  --out data/motor_demos_bowl
# Goal suite: --container auto  (parses each task's (:goal (On X Y)); skips articulated Open/Turnon)
```

**(b) Analytic grounded-physics place (precise release) — "analytic-DAgger":**
```bash
python3 motor_distill/eval_physics_place.py \
  --head data/<motor>.pt --reflex 0 --obj-aware-release 1 \
  --bddl-dir /root/LIBERO-Pro-data/bddl_files/libero_object_swap \
  --init-dir /root/LIBERO-Pro-data/init_files/libero_object_swap \
  --container basket --binder oracle --loc oracle --clear 0.14 \
  --n 10 --trials 4 --logdir data/dag_place     # --logdir logs successful grasp+analytic-place chunks
```

### 3.2 Canonicalize (SE(2) frame normalization — removes the in-plane bearing DOF)
```bash
python3 motor_distill/make_canon.py --src data/dag_place --out data/dag_place_canon --mode perstep_obj
```
> The eval applies the **same** `perstep_obj` canon at inference, so train==deploy frames.

### 3.3 Distill the motor (L1 regression + multi-factor wrist augmentation)
`--data` accepts **comma-separated** dirs; mix teachers/suites freely.
```bash
python3 motor_distill/train_wristcam_motor.py \
  --data data/dag_place_canon,data/dag_place2_canon,data/dag_goal_canon,data/coadapt_std_canon,data/coadapt_swap_canon,data/motor_demos_bowl_canon \
  --out data/motor_head.pt --epochs 60 --aug 1 --grip-w 3.0 --phase-w 1.0
```
Key flags: `--aug` multi-factor wrist augmentation (validated robustness lever); `--grip-w` weights the
gripper channel (sharper release); `--phase-w` weights the learned grasp/place phase classifier.

### 3.4 Evaluate (official `_check_success()` metric)
```bash
python3 motor_distill/eval_physics_place.py \
  --head data/motor_head.pt --reflex 1 \                 # reflex=1 = pure learned motor (no analytic place)
  --bddl-dir /root/LIBERO-Pro-data/bddl_files/libero_object_swap \
  --init-dir /root/LIBERO-Pro-data/init_files/libero_object_swap \
  --container basket --binder dino --loc depthflip \     # honest perception
  --z-cont-center 0.035 --rim-h 0.075 --clear 0.14 --n 10 --trials 3 --init-start 20 \
  --proto-dir <bddl-dir> --proto-init-dir <init-dir> --proto-inits 30,32,34   # DINOv2 prototype bank
```

#### `eval_physics_place.py` flag reference
| flag | meaning |
|---|---|
| `--reflex` | `1` = pure learned motor (grasp+place); `0` = learned grasp + analytic grounded-physics place |
| `--binder` | `oracle` (sim body_pos) · `dino` (DINOv2 proto-bank match) · `sam` (SAM regions, no body_pos = fully honest 2D attention) |
| `--loc` | `oracle` · `depthflip` (masked-depth unproject, ~1.7cm) · `nodeloc` · `rayplane` |
| `--container` | `basket` · `plate` · `auto` (goal suite: route per-task from `(:goal (On X Y))`) |
| `--flow PATH` | use a flow-matching head (`train_flowmotor.py`) instead of L1 (A/B) |
| `--logdir DIR` | log successful trajectories as distillation npz (analytic-DAgger) |
| `--obj-aware-release` | release so the OBJECT (EE+grasp-offset), not the EE, lands at rim+clear |

---

## 4. End-to-end example (one suite, fully learned)
```bash
# 1) collect analytic-place demos on object_swap (basket)
python3 motor_distill/eval_physics_place.py --head data/motor_head_aug.pt --reflex 0 --obj-aware-release 1 \
  --bddl-dir $BDDL/libero_object_swap --init-dir $INIT/libero_object_swap --container basket \
  --binder oracle --loc oracle --clear 0.14 --n 10 --trials 4 --logdir data/dag_obj
# 2) canon + 3) train
python3 motor_distill/make_canon.py --src data/dag_obj --out data/dag_obj_canon --mode perstep_obj
python3 motor_distill/train_wristcam_motor.py --data data/dag_obj_canon,data/coadapt_std_canon --out data/motor.pt --aug 1 --grip-w 3.0
# 4) eval (pure learned, honest binder)
python3 motor_distill/eval_physics_place.py --head data/motor.pt --reflex 1 \
  --bddl-dir $BDDL/libero_object_swap --init-dir $INIT/libero_object_swap --container basket \
  --binder dino --loc depthflip --clear 0.14 --n 10 --trials 3 --init-start 20 \
  --proto-dir $BDDL/libero_object_swap --proto-init-dir $INIT/libero_object_swap --proto-inits 30,32,34
```

---

## 5. Gotchas (hard-won)
- **`parse_bddl` multi-instance.** Spatial/goal BDDLs declare same-type instances on one line
  (`akita_black_bowl_1 akita_black_bowl_2 - akita_black_bowl`). Capture **all** `\w+_\d+` tokens or the
  target silently falls back to the wrong body (was a real bug, fixed).
- **lerobot pin.** openpi needs the pinned commit `0cf8648…`; PyPI lerobot 0.3.x lacks `lerobot.common.*`.
  Re-pin `numpy==1.26.4` afterward.
- **`unset PYTHONPATH`** before serving π0.5 (otherwise the system PYTHONPATH can shadow venv imports).
- **`--container auto`** skips articulated (`Open`/`Turnon`) and push (`main_table_*`) goal tasks — those
  need separate skill primitives, not pick-place.
- **Convention.** Score with the env's official `_check_success()`. Wrist image is fed `[::-1,::-1]`
  (matches the deployed preprocessing). `.pruned_init` files load with `torch.load(..., weights_only=False)`.

---

## 6. Repo map (`motor_distill/`)
| file | role |
|---|---|
| `cf_harness.py` | BDDL parsing, body resolution, `build_obs`, proprio helpers |
| `node_localizer.py` | masked-depth → metric 3D localizer (the "where") |
| `train_wristcam_motor.py` | the **motor**: wrist-CNN + goal-MLP + L1 head; `--aug`, `--grip-w`, phase head |
| `train_flowmotor.py` / `eval_flowmotor.py` | flow-matching action head (A/B; underfits our unimodal data) |
| `canon.py` / `make_canon.py` | SE(2) canonicalization (apply at train; the eval re-applies at inference) |
| `collect_motor_data_ws.py` | π0.5-teacher data collector (websocket); `--container auto` goal routing |
| `collect_motor_data_hide.py` | in-process π0.5 collector (hide distractors → clean scene) |
| `eval_servo_posshift.py` | π0-free analytic closed-loop OSC servo (ceiling + teacher) |
| `eval_physics_place.py` | **main eval harness**: binder + localizer + motor + grounded-physics place; `--logdir` analytic-DAgger; `--container auto`; `--flow` |

See `RESEARCH_SYNTHESIS.md` for the method narrative and `PAPER_v2.md` for the writeup.
