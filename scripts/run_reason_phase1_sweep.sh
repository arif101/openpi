#!/usr/bin/env bash
#
# REASON-VLA Phase 1 sweep — the kill-or-pass experiment.
#
# Runs iterative refinement at N ∈ {0, 1, 2, 3, 5, 10} on LIBERO-10 5cm seed=7.
# If success(N=5) - success(N=0) < +3pp, the iterative-refinement thesis dies.
# Otherwise, the architectural premise is validated → proceed to Phase 2.
#
# Usage:
#   bash scripts/run_reason_phase1_sweep.sh
#
# Override defaults via env vars:
#   TASK_SUITE=libero_90 PERTURBATION_CM=5.0 SEED=7 bash scripts/run_reason_phase1_sweep.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/openpi}"
cd "$REPO_ROOT"

WORLD_MODEL="${WORLD_MODEL:-data/contact_mpc/mpc_results/world_model.pt}"
VALUE_FUNCTION="${VALUE_FUNCTION:-data/contact_mpc/value_function/value_function.pt}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
PERTURBATION_CM="${PERTURBATION_CM:-5.0}"
SEED="${SEED:-7}"
NUM_TRIALS="${NUM_TRIALS:-5}"
MAX_TASKS="${MAX_TASKS:-}"
REFINEMENT_LR="${REFINEMENT_LR:-0.05}"
REFINEMENT_CLIP="${REFINEMENT_CLIP:-1.0}"
REFINEMENT_TRIGGER="${REFINEMENT_TRIGGER:-always}"
OUTPUT_DIR="${OUTPUT_DIR:-data/contact_mpc/reason_phase1}"
N_VALUES="${N_VALUES:-0 1 2 3 5 10}"

MAX_TASKS_ARG=""
if [[ -n "$MAX_TASKS" ]]; then
    MAX_TASKS_ARG="--max-tasks $MAX_TASKS"
fi

log() { echo "[$(date '+%F %T')] $*"; }

mkdir -p "$OUTPUT_DIR"
SUMMARY="$OUTPUT_DIR/sweep_summary.txt"
: > "$SUMMARY"

log "REASON-VLA Phase 1 sweep" | tee -a "$SUMMARY"
log "  task_suite=$TASK_SUITE perturbation=${PERTURBATION_CM}cm seed=$SEED" | tee -a "$SUMMARY"
log "  N values: $N_VALUES" | tee -a "$SUMMARY"
log "  refinement_lr=$REFINEMENT_LR clip=$REFINEMENT_CLIP trigger=$REFINEMENT_TRIGGER" | tee -a "$SUMMARY"
log "  output_dir=$OUTPUT_DIR" | tee -a "$SUMMARY"

for N in $N_VALUES; do
    log "" | tee -a "$SUMMARY"
    log "===== N=$N =====" | tee -a "$SUMMARY"
    PYTHONPATH=src:third_party/libero uv run python3 -u scripts/run_reason_phase1.py \
        --world-model "$WORLD_MODEL" \
        --value-function "$VALUE_FUNCTION" \
        --task-suite "$TASK_SUITE" \
        --num-trials "$NUM_TRIALS" \
        --perturbation-cm "$PERTURBATION_CM" \
        --seed "$SEED" \
        $MAX_TASKS_ARG \
        --num-refinement-iterations "$N" \
        --refinement-lr "$REFINEMENT_LR" \
        --refinement-clip "$REFINEMENT_CLIP" \
        --refinement-trigger "$REFINEMENT_TRIGGER" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$OUTPUT_DIR/run_N${N}.log"
done

log "" | tee -a "$SUMMARY"
log "===== SWEEP COMPLETE =====" | tee -a "$SUMMARY"
log "" | tee -a "$SUMMARY"
log "Decision rule: kill REASON-VLA if success(N=5) - success(N=0) < +3pp" | tee -a "$SUMMARY"
log "" | tee -a "$SUMMARY"
log "Per-N success rates (extract from JSON files):" | tee -a "$SUMMARY"
for N in $N_VALUES; do
    JSON="$OUTPUT_DIR/reason_phase1_${TASK_SUITE}_${PERTURBATION_CM}cm_seed${SEED}_N${N}.json"
    if [[ -f "$JSON" ]]; then
        RATE=$(python3 -c "import json; d=json.loads(open('$JSON').read()); print(f\"{d['result']['success_rate']:.1f}%\")" 2>/dev/null || echo "ERR")
        log "  N=$N: $RATE" | tee -a "$SUMMARY"
    else
        log "  N=$N: <missing>" | tee -a "$SUMMARY"
    fi
done
