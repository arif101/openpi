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
ROLLOUTS_DIR="data/contact_mpc/rollouts"
FEATURES_DIR="data/contact_mpc/features"
WM_DIR="data/contact_mpc/world_model"
WM_TARGET_DIR="data/contact_mpc/mpc_results"
VF_DIR="data/contact_mpc/value_function"

mkdir -p "$ROLLOUTS_DIR" "$FEATURES_DIR" "$WM_DIR" "$WM_TARGET_DIR" "$VF_DIR"

log() { echo "[$(date '+%F %T')] $*"; }
step() { log ""; log "===== $* ====="; }

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
step "5. Run unit tests"

PYTHONPATH="$REPO_ROOT/src" uv run python3 -m pytest \
    src/openpi/contact_mpc/attribution/ \
    src/openpi/contact_mpc/eval/ \
    --no-header --noconftest -q

# -------------------------------------------------------------------------
step "6. Summary"

log "Artifacts ready:"
ls -lh "$ROLLOUTS_NPZ" "$FEATURES_NPZ" "$PROBE_PKL" 2>/dev/null || true
ls -lh "$WM_CKPT" "$WM_CFG" 2>/dev/null || true
ls -lh "$VF_CKPT" "$VF_CFG" 2>/dev/null || true
ls -lh "$TARGET_WM" "$TARGET_CFG" 2>/dev/null || true

log ""
log "Done. Next: set ANTHROPIC_API_KEY and run the Stage A smoke test from demo/README.md."
