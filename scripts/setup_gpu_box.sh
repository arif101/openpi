#!/usr/bin/env bash
#
# One-shot setup for a fresh GPU box: pull data from HF and rebuild the
# Phase 4 artifacts (world model + value function) needed by the demo
# pipeline.
#
# Prerequisites on this box BEFORE running:
#   - Repo cloned at /workspace/openpi, checked out on waypoint-conditioning
#   - `uv sync` completed successfully
#   - FFmpeg dev headers installed (see README)
#   - GPU visible: `uv run python3 -c "import jax; print(jax.devices())"`
#
# Run with nohup so it survives disconnects:
#   nohup bash scripts/setup_gpu_box.sh > setup.log 2>&1 &
#   tail -f setup.log
#
# Idempotent: re-running skips steps whose outputs already exist.
# Total wall time on A40: ~1-2 hrs (mostly world-model training).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/openpi}"
cd "$REPO_ROOT"

HF_REPO="arif101/libero90_vlm_features"
ARTIFACTS_HF_REPO="${ARTIFACTS_HF_REPO:-arif101/openpi-mcts-artifacts}"
ARTIFACTS_TARBALL="${ARTIFACTS_TARBALL:-gpu_artifacts.tar.gz}"
ROLLOUTS_DIR="data/contact_mpc/rollouts"
FEATURES_DIR="data/contact_mpc/features"
WM_DIR="data/contact_mpc/world_model"
WM_TARGET_DIR="data/contact_mpc/mpc_results"
VF_DIR="data/contact_mpc/value_function"

mkdir -p "$ROLLOUTS_DIR" "$FEATURES_DIR" "$WM_DIR" "$WM_TARGET_DIR" "$VF_DIR"

log() { echo "[$(date '+%F %T')] $*"; }
step() { log ""; log "===== $* ====="; }

# -------------------------------------------------------------------------
step "0a. Initialize git submodules (LIBERO, aloha)"
#
# `git clone` without --recurse-submodules leaves third_party/libero empty,
# which causes `ModuleNotFoundError: No module named 'libero'` at eval time.
# Idempotent — already-initialized submodules are no-ops.

if [[ -f .gitmodules ]]; then
    git submodule update --init --recursive
    log "Submodules initialized."
fi

# -------------------------------------------------------------------------
step "0b. Restore prior artifacts from HF (if available)"
#
# Pulls the gpu_artifacts.tar.gz that save_gpu_artifacts.sh uploaded to
# $ARTIFACTS_HF_REPO. Untars in place so that the trained WM + V(h)
# checkpoints + symlinks land at their canonical paths and the steps
# below short-circuit ("[skip] X already trained").
#
# Disable by setting ARTIFACTS_HF_REPO=- (any value where the dataset
# doesn't exist will also just log a warning and continue from scratch).

if [[ -f "$WM_DIR/world_model_H10_medium.pt" && -f "$VF_DIR/value_function.pt" ]]; then
    log "[skip] WM + V(h) already on disk — not pulling tarball"
elif [[ "$ARTIFACTS_HF_REPO" == "-" ]]; then
    log "[skip] ARTIFACTS_HF_REPO=-, training from scratch"
else
    log "Pulling $ARTIFACTS_TARBALL from HF dataset $ARTIFACTS_HF_REPO..."
    uv run python3 - <<PY || log "  WARN: tarball pull failed; will train from scratch"
import shutil, pathlib, sys
from huggingface_hub import hf_hub_download
try:
    src = hf_hub_download(
        repo_id="$ARTIFACTS_HF_REPO",
        filename="$ARTIFACTS_TARBALL",
        repo_type="dataset",
    )
    dst = pathlib.Path("$ARTIFACTS_TARBALL")
    shutil.copy(src, dst)
    print(f"Pulled to {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
except Exception as e:
    print(f"Failed: {e}", file=sys.stderr)
    sys.exit(1)
PY
    if [[ -f "$ARTIFACTS_TARBALL" ]]; then
        log "Extracting $ARTIFACTS_TARBALL..."
        tar xzf "$ARTIFACTS_TARBALL"
        log "Extracted. WM + V(h) + experiment results restored."
    fi
fi

# -------------------------------------------------------------------------
step "1. Verify environment"

uv run python3 - <<'PY'
import sys
import jax
import torch
print(f"Python: {sys.version.split()[0]}")
print(f"JAX devices: {jax.devices()}")
print(f"Torch CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  {torch.cuda.get_device_name(0)}")
PY

# -------------------------------------------------------------------------
step "1b. Install LIBERO runtime deps (under-declared by LIBERO's setup.py)"

# These are required by scripts that import libero.libero.envs.
# The `libero` dep group in pyproject.toml enumerates them.
uv sync --group libero

# Container-level GL libraries for headless MuJoCo rendering. Safe to run
# even if already installed. If apt-get is unavailable (non-Debian base),
# ignore this block — you'll need to install libegl1 / libosmesa6 manually.
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq libegl1 libegl1-mesa libgles2 libgl1 libosmesa6 >/dev/null
    log "Installed headless GL libraries (EGL + OSMesa)"
fi

# -------------------------------------------------------------------------
step "2. Pull rollouts + features from HF"

ROLLOUTS_NPZ="$ROLLOUTS_DIR/rollouts_libero_90.npz"
FEATURES_NPZ="$FEATURES_DIR/libero90_features_H10.npz"
PROBE_PKL="$FEATURES_DIR/best_probe.pkl"

uv run python3 - <<PY
import os, shutil, pathlib
from huggingface_hub import hf_hub_download

targets = [
    ("rollouts_libero_90.npz", "$ROLLOUTS_NPZ"),
    ("libero90_features_H10.npz", "$FEATURES_NPZ"),
    ("best_probe.pkl", "$PROBE_PKL"),
]

for fname, dst in targets:
    dst = pathlib.Path(dst)
    if dst.exists():
        print(f"[skip] {dst} ({dst.stat().st_size / 1e6:.1f} MB) already exists")
        continue
    print(f"[pull] {fname} from $HF_REPO")
    src = hf_hub_download(repo_id="$HF_REPO", filename=fname, repo_type="dataset")
    shutil.copy(src, dst)
    print(f"  -> {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
PY

# -------------------------------------------------------------------------
step "3. Train world model (medium / H=10)"

WM_CKPT="$WM_DIR/world_model_H10_medium.pt"
WM_CFG="$WM_DIR/world_model_config_H10_medium.pt"

if [[ -f "$WM_CKPT" && -f "$WM_CFG" ]]; then
    log "[skip] world model already trained at $WM_CKPT"
else
    log "Training (est. 30-45 min on A40)..."
    PYTHONPATH="$REPO_ROOT/src" uv run python3 scripts/run_train_world_model.py \
        --features-path "$FEATURES_NPZ" \
        --probe-path "$PROBE_PKL" \
        --output-dir "$WM_DIR" \
        --skip-sweep \
        --num-epochs 100 \
        --batch-size 64
fi

# Symlink to the canonical path expected by downstream scripts.
TARGET_WM="$WM_TARGET_DIR/world_model.pt"
TARGET_CFG="$WM_TARGET_DIR/world_model_config.pt"
ln -sf "$(realpath "$WM_CKPT")" "$TARGET_WM"
ln -sf "$(realpath "$WM_CFG")" "$TARGET_CFG"
log "Linked: $TARGET_WM -> $WM_CKPT"

# -------------------------------------------------------------------------
step "4. Train value function"

VF_CKPT="$VF_DIR/value_function.pt"
VF_CFG="$VF_DIR/value_function_config.pt"

if [[ -f "$VF_CKPT" && -f "$VF_CFG" ]]; then
    log "[skip] value function already trained at $VF_CKPT"
else
    log "Training (est. 20-40 min on A40)..."
    PYTHONPATH="$REPO_ROOT/src" uv run python3 scripts/run_train_value_function.py \
        --hf-repo "$HF_REPO" \
        --output-dir "$VF_DIR" \
        --n-pairs 50000 \
        --num-epochs 100 \
        --batch-size 256
fi

# -------------------------------------------------------------------------
step "4b. Train action-conditional Q(h, a) value function"
#
# This is the Sprint 1 "better VF" experiment: action-conditional scorer
# with feature-space perturbation augmentation. Replaces V(h) for the
# LIBERO-90 robustness eval. Skip if checkpoint already exists.

QHA_DIR="data/contact_mpc/q_function_libero90"
QHA_CKPT="$QHA_DIR/q_function.pt"
QHA_CFG="$QHA_DIR/q_function_config.pt"
mkdir -p "$QHA_DIR"

if [[ -f "$QHA_CKPT" && -f "$QHA_CFG" ]]; then
    log "[skip] Q(h, a) already trained at $QHA_CKPT"
else
    log "Training Q(h, a) (est. 30-45 min on A40)..."
    PYTHONPATH="$REPO_ROOT/src" uv run python3 -u scripts/run_train_q_function.py \
        --rollouts-files "$HF_REPO:rollouts_libero_90.npz" \
        --output-dir "$QHA_DIR" \
        --perturb-noise-std-fraction 0.1 \
        --num-epochs 100 \
        --batch-size 256
fi

# -------------------------------------------------------------------------
step "5. Run unit tests"

PYTHONPATH="$REPO_ROOT/src" uv run python3 -m pytest \
    src/openpi/contact_mpc/planner/mcts_test.py \
    src/openpi/contact_mpc/value_function/pairwise_dataset_test.py \
    --no-header --noconftest -q

# -------------------------------------------------------------------------
step "6. Summary"

log "Artifacts ready:"
ls -lh "$ROLLOUTS_NPZ" "$FEATURES_NPZ" "$PROBE_PKL" 2>/dev/null || true
ls -lh "$WM_CKPT" "$WM_CFG" 2>/dev/null || true
ls -lh "$VF_CKPT" "$VF_CFG" 2>/dev/null || true
ls -lh "$QHA_CKPT" "$QHA_CFG" 2>/dev/null || true
ls -lh "$TARGET_WM" "$TARGET_CFG" 2>/dev/null || true

log ""
log "Done. Next: run the LIBERO-PRO eval gate."
log ""
log "Reproduce the Phase 5 baseline (V(h) + MCTS, seed=7, 5cm):"
log "  PYTHONPATH=src:third_party/libero MUJOCO_GL=egl uv run python3 -u \\"
log "    scripts/run_libero_pro_mcts.py \\"
log "    --world-model $TARGET_WM --value-function $VF_CKPT \\"
log "    --value-function-type v --task-suite libero_10 \\"
log "    --num-trials 5 --perturbation-cm 5.0 --seed 7 \\"
log "    --mcts-width-k 4 --mcts-max-depth 1 --mcts-num-simulations 32 \\"
log "    --output-dir data/contact_mpc/baseline_check"
log ""
log "Run Q(h, a) + MCTS on LIBERO-90 (the decision gate):"
log "  ... same command but --value-function $QHA_CKPT --value-function-type qha"
log "      --task-suite libero_90"
