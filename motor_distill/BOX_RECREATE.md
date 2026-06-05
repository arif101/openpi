# GPU box: what to save + how to recreate

## Saved off the box (in ~/openpi-box-backup/, NOT regenerable cheaply)
- `runs/*.pkl` — **trained heads** (the actual results). Key ones:
  - `reach_head.pkl` (281K) — distilled goal-conditioned reach/grasp motor (67% reach / 47% grasp).
  - `motor_head.pkl` (282K) — unified phase-conditioned motor (reach+place).
  - others (bind_head, siglip_head, box_head, cf_bind_*) — superseded/negative-result binders; kept for completeness.
- `data/reach/` (0.9M, 76 demos) + `data/place/` (1.1M, 40 demos) — π0.5-collected demo corpora (GPU-time to make).
- `data/libero_pro/` (0.8M) — the perturbed LIBERO-PRO bddl/init files (also re-pullable from HF zhouxueyang/LIBERO-Pro).

## NOT saved (regenerable or superseded)
- `data/bind8` (2.1G), `data/cf_train` (54M), `data/siglipd`, `data/box` — superseded by the OWLv2 binder; skip.
- π0.5 / OWLv2 / SigLIP / GroundingDINO checkpoints — auto re-download from gs://openpi-assets and HF (just cached).
- The venv — recreate via the steps below.
- All CODE — in git (committed).

## Recreate the environment on a fresh GPU (~30–60 min)
1. Clone the repo; `uv venv .venv` (on LOCAL disk, not NFS — stale-handle issues).
2. `uv pip install -e .` then `uv pip install robosuite==1.4.1 "bddl==1.0.1" "gym==0.25.2" cloudpickle easydict thop transformers`.
3. LIBERO at `third_party/libero` (submodule); run with `PYTHONPATH=/root/openpi/third_party/libero`.
4. System deps for rendering: `apt install pkg-config ffmpeg libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libswscale-dev libswresample-dev`; EGL: write `/root/egl_nvidia.json` ICD, `export MUJOCO_GL=egl`.
5. HF token with gated PaliGemma access (`huggingface-cli login`) — π0.5 uses gemma.
6. Pull LIBERO-PRO bddls into `data/libero_pro/` (HF zhouxueyang/LIBERO-Pro) OR restore from backup.
7. Restore `runs/` and `data/{reach,place}` from backup if continuing the distilled-motor work.

## Run commands (sanity)
- Headline eval: `PYTHONPATH=third_party/libero MUJOCO_GL=egl .venv/bin/python motor_distill/eval_e2e_owl.py --n 12 --thr 0.01 --seed 7`
- Diagnostic: `... motor_distill/cf_bind_diag.py --bddl-dir data/libero_pro/bddl_files/libero_object_task`
