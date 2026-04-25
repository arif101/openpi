#!/usr/bin/env bash
#
# Re-diagnose the VICReg-trained world model after retraining.
#
# Symlinks the new _vic checkpoint into the canonical mpc_results path
# (so downstream scripts can find it under a stable name), then runs the
# diagnostic that compares predicted vs real feature distributions.
#
# Usage (run with nohup for survival across disconnects):
#   nohup bash scripts/diagnose_vic_checkpoint.sh > diag_vic.log 2>&1 &
#   tail -f diag_vic.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/openpi}"
cd "$REPO_ROOT"

SOURCE_CKPT="data/contact_mpc/world_model/world_model_H10_medium_vic.pt"
SOURCE_CFG="data/contact_mpc/world_model/world_model_config_H10_medium_vic.pt"
LINK_CKPT="data/contact_mpc/mpc_results/world_model_vic.pt"
LINK_CFG="data/contact_mpc/mpc_results/world_model_config_vic.pt"
OUT_DIR="data/contact_mpc/diagnostics_vic"

log() { echo "[$(date '+%F %T')] $*"; }

# 1. Sanity-check the new checkpoint exists
if [[ ! -f "$SOURCE_CKPT" || ! -f "$SOURCE_CFG" ]]; then
    log "ERROR: VIC checkpoint not found at $SOURCE_CKPT (or _config). Run training first:"
    log "  PYTHONPATH=src uv run python3 scripts/run_train_world_model.py \\"
    log "    --features-path data/contact_mpc/features/libero90_features_H10.npz \\"
    log "    --probe-path data/contact_mpc/features/best_probe.pkl \\"
    log "    --output-dir data/contact_mpc/world_model --skip-sweep \\"
    log "    --use-vicreg --vicreg-match-real-std"
    exit 1
fi

# 2. Symlink to canonical paths
mkdir -p "$(dirname "$LINK_CKPT")"
ln -sf "$(realpath "$SOURCE_CKPT")" "$LINK_CKPT"
ln -sf "$(realpath "$SOURCE_CFG")"  "$LINK_CFG"
log "Linked: $LINK_CKPT -> $SOURCE_CKPT"
log "Linked: $LINK_CFG  -> $SOURCE_CFG"

# 3. Run the diagnostic
log ""
log "===== Running diagnostic on VIC checkpoint ====="
PYTHONPATH=src uv run python3 scripts/diagnose_world_model.py \
    --world-model "$LINK_CKPT" \
    --world-model-config "$LINK_CFG" \
    --value-function data/contact_mpc/value_function/value_function.pt \
    --value-function-config data/contact_mpc/value_function/value_function_config.pt \
    --features data/contact_mpc/features/libero90_features_H10.npz \
    --rollouts data/contact_mpc/rollouts/rollouts_libero_90.npz \
    --output-dir "$OUT_DIR"

log ""
log "===== Side-by-side: baseline vs VIC ====="
log "Look for these key numbers in the report above and below:"
log "  Per-dim variance ratio: baseline 0.099 -> VIC ?"
log "  Fraction collapsed:     baseline 99%   -> VIC ?"
log "  VF score std (pred):    baseline 0.154 -> VIC ?"
log "  VF score gap (real):    baseline 0.65  (success-failure)"
log "  VF score gap (pred):    baseline ~0    -> VIC ?"

# 4. Pretty-print the report JSONs side-by-side if both exist
BASELINE_REPORT="data/contact_mpc/diagnostics/wm_diagnostic_report.json"
VIC_REPORT="$OUT_DIR/wm_diagnostic_report.json"

if [[ -f "$BASELINE_REPORT" && -f "$VIC_REPORT" ]]; then
    log ""
    log "===== Baseline vs VIC, key fields ====="
    PYTHONPATH=src uv run python3 - <<PY
import json, sys
b = json.load(open("$BASELINE_REPORT"))["diagnostics"]
v = json.load(open("$VIC_REPORT"))["diagnostics"]

def fetch(d, *keys):
    for k in keys:
        d = d.get(k, {}) if isinstance(d, dict) else None
    return d

rows = [
    ("Per-dim variance ratio (mean)",
     b["variance_per_dim"]["per_dim_variance_ratio_mean"],
     v["variance_per_dim"]["per_dim_variance_ratio_mean"]),
    ("Per-dim variance ratio (median)",
     b["variance_per_dim"]["per_dim_variance_ratio_median"],
     v["variance_per_dim"]["per_dim_variance_ratio_median"]),
    ("Fraction collapsed dims (<0.5)",
     b["variance_per_dim"]["fraction_collapsed_dims"],
     v["variance_per_dim"]["fraction_collapsed_dims"]),
    ("Manifold drift ratio",
     b["manifold_drift"]["drift_ratio"],
     v["manifold_drift"]["drift_ratio"]),
    ("Cosine(pred, real)",
     b["cosine_similarity"]["cosine_pred_vs_real_target_mean"],
     v["cosine_similarity"]["cosine_pred_vs_real_target_mean"]),
    ("Spearman(error, variance)",
     b["error_structure"]["spearman_error_vs_variance"],
     v["error_structure"]["spearman_error_vs_variance"]),
]
if "vf_score_shift" in b and "vf_score_shift" in v:
    rows += [
        ("VF(real) std",
         b["vf_score_shift"]["vf_real_std"],
         v["vf_score_shift"]["vf_real_std"]),
        ("VF(pred) std",
         b["vf_score_shift"]["vf_pred_std"],
         v["vf_score_shift"]["vf_pred_std"]),
        ("VF success-failure gap (real)",
         b["vf_score_shift"].get("vf_real_success_failure_gap"),
         v["vf_score_shift"].get("vf_real_success_failure_gap")),
        ("VF success-failure gap (pred)",
         b["vf_score_shift"].get("vf_pred_success_failure_gap"),
         v["vf_score_shift"].get("vf_pred_success_failure_gap")),
    ]

print(f"{'Metric':<40} {'Baseline':>12} {'VIC':>12}  {'Δ':>10}")
print("-" * 80)
for name, b_val, v_val in rows:
    if b_val is None or v_val is None:
        print(f"{name:<40} {'-':>12} {'-':>12}  {'-':>10}")
    else:
        delta = v_val - b_val
        print(f"{name:<40} {b_val:>12.4f} {v_val:>12.4f}  {delta:>+10.4f}")
PY
fi

log ""
log "Done. Report at $OUT_DIR/wm_diagnostic_report.json"
log "Plots at  $OUT_DIR/wm_diagnostic_plots.png"
