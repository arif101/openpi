#!/usr/bin/env bash
#
# Smoke-test MCTS over a chosen world-model variant on LIBERO-PRO.
#
# Runs 1 task × 2 trials with verbose MCTS diagnostics so you can see
# whether search is discriminating between candidates or just running
# over noise. Use BEFORE the full 3-hour Experiment A run.
#
# Usage:
#   bash scripts/smoke_test_mcts.sh                # default: ncel2_vic variant
#   bash scripts/smoke_test_mcts.sh ncel2_vic      # explicit variant
#   bash scripts/smoke_test_mcts.sh nce_vic        # legacy cosine variant
#   bash scripts/smoke_test_mcts.sh ""             # baseline WM (no regularizers)
#
# Environment overrides:
#   K=4 SIMS=16 DEPTH=1 TRIALS=2 TASKS=1 bash scripts/smoke_test_mcts.sh
#
# Run with nohup if you want survival:
#   nohup bash scripts/smoke_test_mcts.sh ncel2_vic > smoke_mcts.log 2>&1 &
#   tail -f smoke_mcts.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/openpi}"
cd "$REPO_ROOT"

VARIANT="${1:-ncel2_vic}"
SIZE="${SIZE:-medium}"
HORIZON="${HORIZON:-10}"
K="${K:-4}"
SIMS="${SIMS:-16}"
DEPTH="${DEPTH:-1}"
TRIALS="${TRIALS:-2}"
TASKS="${TASKS:-1}"
PERTURB_CM="${PERTURB_CM:-5.0}"
TASK_SUITE="${TASK_SUITE:-libero_10}"

if [[ -n "$VARIANT" ]]; then
    SUFFIX="_${VARIANT}"
else
    SUFFIX=""
fi

SOURCE_CKPT="data/contact_mpc/world_model/world_model_H${HORIZON}_${SIZE}${SUFFIX}.pt"
SOURCE_CFG="data/contact_mpc/world_model/world_model_config_H${HORIZON}_${SIZE}${SUFFIX}.pt"
LINK_CKPT="data/contact_mpc/mpc_results/world_model.pt"
LINK_CFG="data/contact_mpc/mpc_results/world_model_config.pt"

log() { echo "[$(date '+%F %T')] $*"; }

# 1. Verify the requested checkpoint exists
if [[ ! -f "$SOURCE_CKPT" || ! -f "$SOURCE_CFG" ]]; then
    log "ERROR: checkpoint for variant '${VARIANT}' not found at $SOURCE_CKPT"
    log "Train it first with scripts/run_train_world_model.py"
    exit 1
fi

# 2. Symlink to canonical paths used by the eval script
mkdir -p "$(dirname "$LINK_CKPT")"
ln -sf "$(realpath "$SOURCE_CKPT")" "$LINK_CKPT"
ln -sf "$(realpath "$SOURCE_CFG")"  "$LINK_CFG"
log "Variant: '${VARIANT}'  (suffix='${SUFFIX}')"
log "Symlinked to canonical paths:"
log "  $LINK_CKPT -> $SOURCE_CKPT"
log "  $LINK_CFG  -> $SOURCE_CFG"

# 3. Run smoke test with verbose MCTS diagnostics
log ""
log "===== Smoke test: ${TASKS} task × ${TRIALS} trials, K=${K}, sims=${SIMS}, depth=${DEPTH} ====="
log ""

MUJOCO_GL=egl PYTHONPATH=/workspace/openpi/src:/workspace/openpi/third_party/libero \
/workspace/openpi/.venv/bin/python -u scripts/run_libero_pro_mcts.py \
    --world-model "$LINK_CKPT" \
    --world-model-config "$LINK_CFG" \
    --value-function data/contact_mpc/value_function/value_function.pt \
    --value-function-config data/contact_mpc/value_function/value_function_config.pt \
    --task-suite "$TASK_SUITE" \
    --num-trials "$TRIALS" \
    --max-tasks "$TASKS" \
    --perturbation-cm "$PERTURB_CM" \
    --mcts-num-simulations "$SIMS" \
    --mcts-width-k "$K" \
    --mcts-max-depth "$DEPTH" \
    --verbose-mcts

log ""
log "===== Diagnostic interpretation ====="
log ""
log "Look at the [MCTS t=...] lines in the log above. Each line shows:"
log "  visits=[v0,v1,v2,v3]  → how many simulations went down each child"
log "  Q_spread              → range of Q-values across children"
log "  top_visit_frac        → fraction of visits on the most-visited child"
log "  chosen_idx            → which child was finally selected"
log ""
log "GOOD signs (search is discriminating):"
log "  - top_visit_frac > 0.5  (one child clearly preferred)"
log "  - Q_spread > 0.05       (children get meaningfully different scores)"
log "  - chosen_idx varies across decisions (search is responding to context)"
log ""
log "BAD signs (search is over noise):"
log "  - top_visit_frac ~ 0.25  (visits split evenly across 4 children)"
log "  - Q_spread < 0.001       (all candidates score essentially identically)"
log "  - chosen_idx always 0    (search is degenerate — falling back to first child)"
log ""
log "FINAL RESULTS line above shows baseline % vs MCTS %. Even with bad MCTS"
log "diagnostics, a positive delta means search is doing something useful."
