#!/usr/bin/env bash
#
# One-shot bootstrap for a brand-new GPU box. Handles every step from
# zero-installed-system to ready-to-eval:
#
#   1. Install `uv` if missing (Python package manager)
#   2. Install apt deps for av==11.0.0 build + headless MuJoCo rendering
#   3. Clone the repo with submodules if not already present
#   4. Checkout the working branch
#   5. Init submodules (in case repo was cloned without --recurse-submodules)
#   6. Run `uv sync` to install Python deps from the lock file
#   7. Run scripts/setup_gpu_box.sh (pulls HF data + tarball, trains Q(h, a))
#
# After this finishes, the box is ready to run LIBERO-PRO MCTS evals or the
# hierarchy validation script.
#
# Usage (on a fresh Ubuntu/Debian root shell):
#
#   curl -LsSf https://raw.githubusercontent.com/arif101/openpi/waypoint-conditioning/scripts/bootstrap_gpu_box.sh | bash
#
# Or if you've already cloned the repo:
#
#   bash scripts/bootstrap_gpu_box.sh
#
# Env vars (all optional):
#
#   REPO_ROOT               Where to clone (default: /openpi)
#   REPO_URL                Git URL  (default: https://github.com/arif101/openpi.git)
#   BRANCH                  Branch   (default: waypoint-conditioning)
#   ARTIFACTS_HF_REPO       HF dataset for tarball pull (passed to setup_gpu_box.sh,
#                           default: arif101/openpi-mcts-artifacts; "-" to skip)
#   SKIP_APT                Set to "1" to skip apt-get steps (e.g., already done)
#   SKIP_SETUP              Set to "1" to stop after `uv sync` (skip setup_gpu_box.sh)

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/openpi}"
REPO_URL="${REPO_URL:-https://github.com/arif101/openpi.git}"
BRANCH="${BRANCH:-waypoint-conditioning}"
SKIP_APT="${SKIP_APT:-0}"
SKIP_SETUP="${SKIP_SETUP:-0}"

log() { echo "[bootstrap] [$(date '+%F %T')] $*"; }
step() { log ""; log "===== $* ====="; }

# -----------------------------------------------------------------------------
step "1. Install uv if missing"

if command -v uv >/dev/null 2>&1; then
    log "uv already installed: $(uv --version)"
else
    log "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source uv into the current shell's PATH
    if [[ -f "$HOME/.local/bin/env" ]]; then
        # shellcheck source=/dev/null
        source "$HOME/.local/bin/env"
    elif [[ -f "$HOME/.cargo/env" ]]; then
        # shellcheck source=/dev/null
        source "$HOME/.cargo/env"
    else
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
    log "uv installed: $(uv --version)"
fi

# -----------------------------------------------------------------------------
step "2. Install apt dependencies (ffmpeg dev headers + headless GL libs)"

if [[ "$SKIP_APT" == "1" ]]; then
    log "[skip] SKIP_APT=1"
elif ! command -v apt-get >/dev/null 2>&1; then
    log "[skip] apt-get not available (non-Debian system); install equivalents manually:"
    log "       pkg-config ffmpeg libav*-dev libegl1 libegl1-mesa libgles2 libgl1 libosmesa6"
else
    log "Running apt-get update + install..."
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq \
        pkg-config ffmpeg \
        libavformat-dev libavcodec-dev libavdevice-dev \
        libavutil-dev libswscale-dev libswresample-dev libavfilter-dev \
        libegl1 libegl1-mesa libgles2 libgl1 libosmesa6 \
        git ca-certificates >/dev/null
    log "apt deps installed."
fi

# -----------------------------------------------------------------------------
step "3. Clone repository (if not already at \$REPO_ROOT)"

if [[ -d "$REPO_ROOT/.git" ]]; then
    log "Repo already at $REPO_ROOT — skipping clone."
else
    if [[ -e "$REPO_ROOT" ]]; then
        log "ERROR: $REPO_ROOT exists but is not a git repo. Move/delete it and re-run."
        exit 1
    fi
    log "Cloning $REPO_URL → $REPO_ROOT (with submodules)..."
    git clone --recurse-submodules "$REPO_URL" "$REPO_ROOT"
fi

cd "$REPO_ROOT"

# -----------------------------------------------------------------------------
step "4. Checkout branch + sync"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    log "Switching from '$CURRENT_BRANCH' to '$BRANCH'..."
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
else
    log "Already on $BRANCH; pulling latest..."
    git pull --ff-only origin "$BRANCH" || log "  (pull failed; continuing with local HEAD)"
fi

# -----------------------------------------------------------------------------
step "5. Initialize / update submodules"

git submodule update --init --recursive
log "Submodules initialized."

# -----------------------------------------------------------------------------
step "6. uv sync (install Python deps from lock file)"

uv sync
log "uv sync complete."

# -----------------------------------------------------------------------------
step "7. Run setup_gpu_box.sh (HF data pull + tarball restore + Q(h,a) training)"

if [[ "$SKIP_SETUP" == "1" ]]; then
    log "[skip] SKIP_SETUP=1 — stopping before setup_gpu_box.sh"
    log ""
    log "To finish setup later:"
    log "  REPO_ROOT=$REPO_ROOT bash scripts/setup_gpu_box.sh"
    exit 0
fi

REPO_ROOT="$REPO_ROOT" bash scripts/setup_gpu_box.sh

log ""
log "===== Bootstrap complete ====="
log ""
log "Next: run the LIBERO-PRO MCTS eval gate or hierarchy validation."
log "See setup_gpu_box.sh's final summary or scripts/validate_hierarchy_prompt.py."
