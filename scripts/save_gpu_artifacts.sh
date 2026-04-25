#!/usr/bin/env bash
#
# Save expensive-to-recreate artifacts before spinning down a GPU box.
#
# Tarballs trained checkpoints, experiment results, and diagnostic outputs
# into a single file ready to scp/upload. Skips data that's already on HF
# or GCS (rollouts, features, Pi0.5 weights). Skips frames (recreatable
# from rollouts via replay_failure_frames.py).
#
# Total bundle size: typically <100 MB — easy to download or push to HF.
#
# Usage:
#   bash scripts/save_gpu_artifacts.sh                     # default: ./gpu_artifacts.tar.gz
#   bash scripts/save_gpu_artifacts.sh /tmp/backup.tar.gz  # custom path
#   PUSH_TO_HF=1 HF_REPO=arif101/openpi-mcts-artifacts \
#     bash scripts/save_gpu_artifacts.sh                   # also push to HF dataset
#
# To restore on a new box:
#   tar xzf gpu_artifacts.tar.gz -C /workspace/openpi
#   bash scripts/setup_gpu_box.sh                          # rebuilds the rest

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/openpi}"
cd "$REPO_ROOT"

OUT="${1:-gpu_artifacts.tar.gz}"
PUSH_TO_HF="${PUSH_TO_HF:-0}"
HF_REPO="${HF_REPO:-}"

log() { echo "[$(date '+%F %T')] $*"; }

# Build inventory
log "===== Inventory ====="

# World-model variants (baseline + regularizer combinations)
WM_DIR="data/contact_mpc/world_model"
WM_FILES=()
for f in "$WM_DIR"/world_model_H*.pt "$WM_DIR"/world_model_config_H*.pt; do
    [[ -f "$f" ]] && WM_FILES+=("$f")
done
log "World-model checkpoints: ${#WM_FILES[@]} files"
for f in "${WM_FILES[@]}"; do
    log "  $(du -h "$f" | cut -f1)  $f"
done

# Symlinked canonical paths (so users don't have to re-symlink later)
SYMLINKS=()
for sl in data/contact_mpc/mpc_results/world_model.pt \
          data/contact_mpc/mpc_results/world_model_config.pt \
          data/contact_mpc/mpc_results/world_model_vic.pt \
          data/contact_mpc/mpc_results/world_model_config_vic.pt \
          data/contact_mpc/mpc_results/world_model_nce_vic.pt \
          data/contact_mpc/mpc_results/world_model_config_nce_vic.pt \
          data/contact_mpc/mpc_results/world_model_ncel2_vic.pt \
          data/contact_mpc/mpc_results/world_model_config_ncel2_vic.pt; do
    [[ -L "$sl" ]] && SYMLINKS+=("$sl")
done
log "Symlinks: ${#SYMLINKS[@]} (recreated by including target dir)"

# Value function
VF_FILES=()
for f in data/contact_mpc/value_function/value_function.pt \
         data/contact_mpc/value_function/value_function_config.pt \
         data/contact_mpc/value_function/value_function_metrics.npz; do
    [[ -f "$f" ]] && VF_FILES+=("$f")
done
log "Value function: ${#VF_FILES[@]} files"
for f in "${VF_FILES[@]}"; do
    log "  $(du -h "$f" | cut -f1)  $f"
done

# Experiment A result JSONs
EXP_DIRS=()
for d in data/contact_mpc/experiment_a_*/ \
         data/contact_mpc/libero_pro/ \
         data/contact_mpc/libero_pro_mcts/ \
         data/contact_mpc/mpc_results/; do
    [[ -d "$d" ]] && EXP_DIRS+=("$d")
done
log "Experiment result dirs: ${#EXP_DIRS[@]}"
for d in "${EXP_DIRS[@]}"; do
    SZ=$(du -sh "$d" 2>/dev/null | cut -f1)
    log "  $SZ  $d"
done

# Diagnostic reports + plots
DIAG_DIRS=()
for d in data/contact_mpc/diagnostics*/; do
    [[ -d "$d" ]] && DIAG_DIRS+=("$d")
done
log "Diagnostic dirs: ${#DIAG_DIRS[@]}"
for d in "${DIAG_DIRS[@]}"; do
    SZ=$(du -sh "$d" 2>/dev/null | cut -f1)
    log "  $SZ  $d"
done

# Logs (the experiment_a_*.log files in repo root)
LOG_FILES=()
for f in experiment_a*.log smoke_mcts.log replay_full.log replay_with_obs.log \
         replay_smoke.log diag_vic.log setup.log libero_pro_mcts*.log \
         train_*.log; do
    [[ -f "$f" ]] && LOG_FILES+=("$f")
done
log "Run logs: ${#LOG_FILES[@]} files"

# Build the tarball
log ""
log "===== Building tarball: $OUT ====="

# Construct file list, only including paths that exist
TAR_PATHS=()
[[ ${#WM_FILES[@]} -gt 0 ]] && TAR_PATHS+=("${WM_FILES[@]}")
[[ ${#VF_FILES[@]} -gt 0 ]] && TAR_PATHS+=("${VF_FILES[@]}")
[[ ${#EXP_DIRS[@]} -gt 0 ]] && TAR_PATHS+=("${EXP_DIRS[@]}")
[[ ${#DIAG_DIRS[@]} -gt 0 ]] && TAR_PATHS+=("${DIAG_DIRS[@]}")
[[ ${#LOG_FILES[@]} -gt 0 ]] && TAR_PATHS+=("${LOG_FILES[@]}")

if [[ ${#TAR_PATHS[@]} -eq 0 ]]; then
    log "ERROR: nothing to save"
    exit 1
fi

# --dereference resolves symlinks so the tarball is self-contained.
# --ignore-failed-read tolerates partial trees if any sub-path is missing.
tar --dereference --ignore-failed-read -czf "$OUT" "${TAR_PATHS[@]}"

log "Tarball created:"
log "  $(du -h "$OUT" | cut -f1)  $OUT"
log ""
log "Manifest (top-level entries):"
# Use { ... || true; } to swallow the SIGPIPE that tar gets when head closes
# its stdin after reading 30 lines. Without this, set -euo pipefail kills
# the script before the push block runs.
{ tar tzf "$OUT" 2>/dev/null || true; } | head -30
TOTAL_ENTRIES=$(tar tzf "$OUT" 2>/dev/null | wc -l)
log "  ... ($TOTAL_ENTRIES total entries)"

# Optionally push to HF
if [[ "$PUSH_TO_HF" == "1" ]]; then
    if [[ -z "$HF_REPO" ]]; then
        log "ERROR: PUSH_TO_HF=1 but HF_REPO is empty"
        exit 1
    fi
    log ""
    log "===== Pushing to HF dataset $HF_REPO ====="
    uv run python3 - <<PY
from huggingface_hub import HfApi, create_repo
api = HfApi()
try:
    create_repo("$HF_REPO", repo_type="dataset", exist_ok=True, private=False)
except Exception as e:
    print(f"create_repo: {e}")
api.upload_file(
    path_or_fileobj="$OUT",
    path_in_repo="$(basename "$OUT")",
    repo_id="$HF_REPO",
    repo_type="dataset",
)
print(f"Pushed $OUT to https://huggingface.co/datasets/$HF_REPO")
PY
fi

log ""
log "===== Done ====="
log ""
log "To download to your laptop:"
log "  scp -P <runpod-port> root@<runpod-ip>:$REPO_ROOT/$OUT ."
log ""
log "To restore on a new GPU box:"
log "  cd /workspace/openpi"
log "  tar xzf $(basename "$OUT")"
log "  bash scripts/setup_gpu_box.sh   # rebuilds env, re-downloads HF data"
log ""
log "What's NOT in this tarball (deliberately):"
log "  - rollouts_libero_90.npz       (re-pull from arif101/libero90_vlm_features)"
log "  - libero90_features_H10.npz    (same HF dataset)"
log "  - Pi0.5 weights                (gs://openpi-assets/checkpoints/pi05_libero)"
log "  - data/contact_mpc/frames/     (recreate via replay_failure_frames.py)"
log "  - .venv, __pycache__           (re-installed by setup_gpu_box.sh)"
