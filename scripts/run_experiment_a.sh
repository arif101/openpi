#!/usr/bin/env bash
#
# Experiment A — full LIBERO-PRO MCTS evaluation.
#
# Compares vanilla Pi0.5 vs Pi0.5 + MCTS over our latent world model on
# LIBERO-10 with 5cm object perturbation. ~2-3 hours on A40.
#
# Default config: 10 tasks × 5 trials × 2 modes = 100 episodes,
# K=4 candidates, 32 MCTS simulations per decision, depth=1,
# contact-triggered (~18% of decisions).
#
# Usage:
#   bash scripts/run_experiment_a.sh                    # default: ncel2_vic
#   bash scripts/run_experiment_a.sh ncel2_vic          # explicit variant
#   bash scripts/run_experiment_a.sh nce_vic            # legacy cosine variant
#   bash scripts/run_experiment_a.sh ""                 # baseline WM (no regularizers)
#
# Recommended: run with nohup so it survives disconnects.
#   nohup bash scripts/run_experiment_a.sh ncel2_vic > experiment_a.log 2>&1 &
#   echo $! > experiment_a.pid
#   tail -f experiment_a.log
#
# Environment overrides:
#   K=4 SIMS=32 DEPTH=1 TRIALS=5 TASKS=10 \
#   PERTURB_CM=5.0 TASK_SUITE=libero_10 \
#   bash scripts/run_experiment_a.sh ncel2_vic
#
# Check progress mid-run:
#   ps -p $(cat experiment_a.pid) && echo running || echo done
#   tail -100 experiment_a.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/openpi}"
cd "$REPO_ROOT"

VARIANT="${1:-ncel2_vic}"
SIZE="${SIZE:-medium}"
HORIZON="${HORIZON:-10}"
K="${K:-4}"
SIMS="${SIMS:-32}"
DEPTH="${DEPTH:-1}"
TRIALS="${TRIALS:-5}"
TASKS="${TASKS:-10}"
PERTURB_CM="${PERTURB_CM:-5.0}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
SEED="${SEED:-7}"

if [[ -n "$VARIANT" ]]; then
    SUFFIX="_${VARIANT}"
else
    SUFFIX=""
fi

SOURCE_CKPT="data/contact_mpc/world_model/world_model_H${HORIZON}_${SIZE}${SUFFIX}.pt"
SOURCE_CFG="data/contact_mpc/world_model/world_model_config_H${HORIZON}_${SIZE}${SUFFIX}.pt"
LINK_CKPT="data/contact_mpc/mpc_results/world_model.pt"
LINK_CFG="data/contact_mpc/mpc_results/world_model_config.pt"
OUT_DIR="data/contact_mpc/experiment_a${SUFFIX}"
OUT_JSON="${OUT_DIR}/libero_pro_mcts_${TASK_SUITE}_${PERTURB_CM}cm.json"

log() { echo "[$(date '+%F %T')] $*"; }

# ----- 1. Prereq checks -----

if [[ ! -f "$SOURCE_CKPT" || ! -f "$SOURCE_CFG" ]]; then
    log "ERROR: world model checkpoint not found at $SOURCE_CKPT"
    log "Train it first via scripts/run_train_world_model.py"
    exit 1
fi

VF_CKPT="data/contact_mpc/value_function/value_function.pt"
VF_CFG="data/contact_mpc/value_function/value_function_config.pt"
if [[ ! -f "$VF_CKPT" || ! -f "$VF_CFG" ]]; then
    log "ERROR: value function not found at $VF_CKPT"
    log "Train it via scripts/run_train_value_function.py (or rerun setup_gpu_box.sh)"
    exit 1
fi

# ----- 2. Symlink to canonical paths -----

mkdir -p "$(dirname "$LINK_CKPT")" "$OUT_DIR"
ln -sf "$(realpath "$SOURCE_CKPT")" "$LINK_CKPT"
ln -sf "$(realpath "$SOURCE_CFG")"  "$LINK_CFG"

# ----- 3. Print run config -----

log "===== Experiment A — full LIBERO-PRO MCTS evaluation ====="
log ""
log "Variant:           '${VARIANT}'  (suffix='${SUFFIX}')"
log "World model:       $SOURCE_CKPT"
log "Value function:    $VF_CKPT"
log "Task suite:        $TASK_SUITE"
log "Tasks × Trials:    $TASKS × $TRIALS = $((TASKS * TRIALS)) episodes per mode"
log "Perturbation:      ${PERTURB_CM} cm"
log "MCTS:              K=$K, sims=$SIMS, depth=$DEPTH (contact-triggered)"
log "Seed:              $SEED"
log "Output:            $OUT_JSON"
log ""

# ----- 4. Run -----

START=$(date +%s)

MUJOCO_GL=egl PYTHONPATH=/workspace/openpi/src:/workspace/openpi/third_party/libero \
/workspace/openpi/.venv/bin/python -u scripts/run_libero_pro_mcts.py \
    --world-model "$LINK_CKPT" \
    --world-model-config "$LINK_CFG" \
    --value-function "$VF_CKPT" \
    --value-function-config "$VF_CFG" \
    --task-suite "$TASK_SUITE" \
    --num-trials "$TRIALS" \
    --max-tasks "$TASKS" \
    --perturbation-cm "$PERTURB_CM" \
    --mcts-num-simulations "$SIMS" \
    --mcts-width-k "$K" \
    --mcts-max-depth "$DEPTH" \
    --seed "$SEED" \
    --output-dir "$OUT_DIR"

END=$(date +%s)
ELAPSED_MIN=$(( (END - START) / 60 ))

log ""
log "===== Run finished in ${ELAPSED_MIN} min ====="

# ----- 5. Headline summary -----

if [[ -f "$OUT_JSON" ]]; then
    log ""
    log "===== Headline ====="
    PYTHONPATH=src uv run python3 - <<PY
import json
d = json.load(open("$OUT_JSON"))
b = d["baseline"]
m = d["mcts"]

print(f"Variant:    ${VARIANT}")
print(f"Suite:      {d['args']['task_suite']}, perturbation {d['args']['perturbation_cm']}cm")
print(f"MCTS cfg:   sims={d['args']['mcts_num_simulations']}, K={d['args']['mcts_width_k']}, depth={d['args']['mcts_max_depth']}")
print()
print(f"Baseline:   {b['success_rate']:.1f}%   ({b['total_successes']}/{b['total_episodes']})")
print(f"MCTS:       {m['success_rate']:.1f}%   ({m['total_successes']}/{m['total_episodes']})")
print(f"Delta:      {m['success_rate'] - b['success_rate']:+.1f} pp")
print()
print(f"MCTS calls: {m['total_mcts_calls']}")
print()
print("Per-task changes (where MCTS differed from baseline):")
print(f"  {'Task':>4}  {'Baseline':>10}  {'MCTS':>10}  {'Δ':>6}  Description")
print(f"  {'----':>4}  {'--------':>10}  {'----':>10}  {'-':>6}  ----")
for tid in sorted(b["per_task"].keys(), key=int):
    bp = b["per_task"][tid]
    mp = m["per_task"][tid]
    if abs(bp["rate"] - mp["rate"]) > 1e-9:
        delta_pp = (mp["rate"] - bp["rate"]) * 100
        sign = "+" if delta_pp >= 0 else ""
        print(f"  {tid:>4}  {bp['rate']*100:>9.0f}%  {mp['rate']*100:>9.0f}%  {sign}{delta_pp:>5.1f}  {bp['task'][:60]}")
PY
fi

log ""
log "Full results JSON: $OUT_JSON"
