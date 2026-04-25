#!/usr/bin/env bash
#
# Diagnose a world-model variant against the baseline.
#
# Usage:
#   bash scripts/diagnose_vic_checkpoint.sh                  # default: nce_vic (best so far)
#   bash scripts/diagnose_vic_checkpoint.sh vic              # VICReg-only variant
#   bash scripts/diagnose_vic_checkpoint.sh nce              # InfoNCE-only variant
#   bash scripts/diagnose_vic_checkpoint.sh nce_vic          # both regularizers
#   bash scripts/diagnose_vic_checkpoint.sh ""               # baseline (no regularizers)
#
# The variant string maps directly to the suffix in train.py's checkpoint
# naming: world_model_H{H}_{size}_{variant}.pt. Run with nohup if you
# want survival across disconnects:
#   nohup bash scripts/diagnose_vic_checkpoint.sh nce_vic > diag.log 2>&1 &
#   tail -f diag.log

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/workspace/openpi}"
cd "$REPO_ROOT"

VARIANT="${1:-nce_vic}"
SIZE="${SIZE:-medium}"
HORIZON="${HORIZON:-10}"

if [[ -n "$VARIANT" ]]; then
    SUFFIX="_${VARIANT}"
else
    SUFFIX=""
fi

SOURCE_CKPT="data/contact_mpc/world_model/world_model_H${HORIZON}_${SIZE}${SUFFIX}.pt"
SOURCE_CFG="data/contact_mpc/world_model/world_model_config_H${HORIZON}_${SIZE}${SUFFIX}.pt"
LINK_CKPT="data/contact_mpc/mpc_results/world_model${SUFFIX}.pt"
LINK_CFG="data/contact_mpc/mpc_results/world_model_config${SUFFIX}.pt"
OUT_DIR="data/contact_mpc/diagnostics${SUFFIX}"
BASELINE_REPORT="data/contact_mpc/diagnostics/wm_diagnostic_report.json"

log() { echo "[$(date '+%F %T')] $*"; }

log "Variant: '${VARIANT}'  (suffix='${SUFFIX}')"
log "Source checkpoint: $SOURCE_CKPT"

# 1. Sanity-check the new checkpoint exists. Print a useful retraining
#    command on miss based on the requested variant.
if [[ ! -f "$SOURCE_CKPT" || ! -f "$SOURCE_CFG" ]]; then
    log "ERROR: checkpoint not found at $SOURCE_CKPT"
    log ""
    log "To train the '${VARIANT}' variant, run:"
    log ""
    case "$VARIANT" in
        "")
            log "  PYTHONPATH=src uv run python3 scripts/run_train_world_model.py \\"
            log "    --features-path data/contact_mpc/features/libero90_features_H10.npz \\"
            log "    --probe-path data/contact_mpc/features/best_probe.pkl \\"
            log "    --output-dir data/contact_mpc/world_model --skip-sweep"
            ;;
        "vic")
            log "  PYTHONPATH=src uv run python3 scripts/run_train_world_model.py \\"
            log "    --features-path data/contact_mpc/features/libero90_features_H10.npz \\"
            log "    --probe-path data/contact_mpc/features/best_probe.pkl \\"
            log "    --output-dir data/contact_mpc/world_model --skip-sweep \\"
            log "    --use-vicreg --vicreg-match-real-std"
            ;;
        "nce")
            log "  PYTHONPATH=src uv run python3 scripts/run_train_world_model.py \\"
            log "    --features-path data/contact_mpc/features/libero90_features_H10.npz \\"
            log "    --probe-path data/contact_mpc/features/best_probe.pkl \\"
            log "    --output-dir data/contact_mpc/world_model --skip-sweep \\"
            log "    --use-infonce --infonce-weight 0.1 --infonce-temperature 0.1"
            ;;
        "nce_vic")
            log "  PYTHONPATH=src uv run python3 scripts/run_train_world_model.py \\"
            log "    --features-path data/contact_mpc/features/libero90_features_H10.npz \\"
            log "    --probe-path data/contact_mpc/features/best_probe.pkl \\"
            log "    --output-dir data/contact_mpc/world_model --skip-sweep \\"
            log "    --use-vicreg --vicreg-match-real-std \\"
            log "    --use-infonce --infonce-weight 0.1 --infonce-temperature 0.1"
            ;;
        *)
            log "  (Unknown variant '${VARIANT}'. Common choices: '', 'vic', 'nce', 'nce_vic'.)"
            ;;
    esac
    exit 1
fi

# 2. Symlink to canonical paths for downstream scripts
mkdir -p "$(dirname "$LINK_CKPT")"
ln -sf "$(realpath "$SOURCE_CKPT")" "$LINK_CKPT"
ln -sf "$(realpath "$SOURCE_CFG")"  "$LINK_CFG"
log "Linked: $LINK_CKPT -> $SOURCE_CKPT"
log "Linked: $LINK_CFG  -> $SOURCE_CFG"

# 3. Run the diagnostic
log ""
log "===== Running diagnostic on '${VARIANT}' checkpoint ====="
PYTHONPATH=src uv run python3 scripts/diagnose_world_model.py \
    --world-model "$LINK_CKPT" \
    --world-model-config "$LINK_CFG" \
    --value-function data/contact_mpc/value_function/value_function.pt \
    --value-function-config data/contact_mpc/value_function/value_function_config.pt \
    --features data/contact_mpc/features/libero90_features_H10.npz \
    --rollouts data/contact_mpc/rollouts/rollouts_libero_90.npz \
    --output-dir "$OUT_DIR"

VARIANT_REPORT="$OUT_DIR/wm_diagnostic_report.json"

# 4. Side-by-side comparison vs baseline if both exist (and the variant
#    isn't itself the baseline).
if [[ -n "$VARIANT" && -f "$BASELINE_REPORT" && -f "$VARIANT_REPORT" ]]; then
    log ""
    log "===== Baseline vs '${VARIANT}', key fields ====="
    PYTHONPATH=src uv run python3 - <<PY
import json
b = json.load(open("$BASELINE_REPORT"))["diagnostics"]
v = json.load(open("$VARIANT_REPORT"))["diagnostics"]

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

print(f"{'Metric':<40} {'Baseline':>12} {'$VARIANT':>12}  {'Δ':>10}")
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
log "Done. Report at $VARIANT_REPORT"
log "Plots at  $OUT_DIR/wm_diagnostic_plots.png"
