"""Phase 2b robust analysis with bootstrap CIs, baselines, and compute savings.

Adds rigor to the gate evaluation:
  1. Bootstrap 95% CI on TPR and FPR per task and aggregate.
  2. **Sentinel STAC proxy baseline** (chunk-to-chunk cosine on commanded action
     directions): the closest published competitor. Compare AUROC.
  3. **Length-only control baseline**: predict failure from trace length alone.
     Tests whether ε-gate adds info beyond "this rollout is taking too long."
  4. **Compute-savings calculation**: if we aborted at gate-fire, how many
     wasted steps did we save?
  5. ε-cluster prediction (mechanism stratum) per failure.

Usage:
    python scripts/phase2b_robust_analysis.py \\
        --new-traces-dir /workspace/traces_phase2b_pert \\
        --calib-summary data/contact_mpc/prediction_error_analysis/summary.json \\
        --output /workspace/logs/phase2b_robust.json
"""
from __future__ import annotations
import argparse
import json
import pathlib
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ACTION_HORIZON = 10
LOW_EXEC_THRESHOLD = 0.01
LOOKAHEAD_K = 5
WINDOW_CHUNKS = 5
WINDOW_STEPS = WINDOW_CHUNKS * ACTION_HORIZON
GATE_THRESHOLD = 0.160
N_SUSTAINED_STEPS = 5 * ACTION_HORIZON

EPS_FEATURES = [
    "corr_x", "corr_y", "corr_z", "mean_cos_align", "median_exec_ratio",
    "frac_low_exec_ratio", "max_consecutive_low_exec",
    "persistent_dir_after_high_eps",
]

RNG = np.random.default_rng(0)


def gate_first_fire(action, ee_pos):
    T = action.shape[0]
    if T <= LOOKAHEAD_K + WINDOW_STEPS:
        return None
    cmd_norm = np.linalg.norm(action[:T - LOOKAHEAD_K, :3], axis=1) + 1e-9
    real_norm = np.linalg.norm(ee_pos[LOOKAHEAD_K:] - ee_pos[:T - LOOKAHEAD_K], axis=1)
    low_exec = (real_norm / cmd_norm < LOW_EXEC_THRESHOLD).astype(np.float32)
    low_exec_padded = np.concatenate([low_exec, np.zeros(LOOKAHEAD_K, dtype=np.float32)])
    consec = 0
    for t in range(WINDOW_STEPS, T - LOOKAHEAD_K):
        windowed = low_exec_padded[t - WINDOW_STEPS:t].mean()
        if windowed > GATE_THRESHOLD:
            consec += 1
            if consec >= N_SUSTAINED_STEPS:
                return t
        else:
            consec = 0
    return None


def stac_score(action):
    """Sentinel STAC proxy: 1 - mean(cos similarity between adjacent
    non-overlapping action chunks' commanded directions). Higher = more erratic.
    """
    T = action.shape[0]
    chunk_starts = list(range(0, T, ACTION_HORIZON))
    dirs = []
    for c0 in chunk_starts:
        c1 = min(c0 + ACTION_HORIZON, T)
        if c1 - c0 < 2:
            continue
        v = action[c0:c1, :3].mean(axis=0)
        n = np.linalg.norm(v)
        if n > 1e-6:
            dirs.append(v / n)
    if len(dirs) < 2:
        return None
    dirs = np.array(dirs)
    cos_adj = np.einsum("ij,ij->i", dirs[:-1], dirs[1:])
    return float(1.0 - cos_adj.mean())


def eps_features_for_trace(action, ee_pos):
    T = action.shape[0]
    if T <= 2:
        return None
    cmd = action[:-1, :3]
    realized = np.diff(ee_pos, axis=0)
    cmd_norm = np.linalg.norm(cmd, axis=1) + 1e-9
    real_norm = np.linalg.norm(realized, axis=1) + 1e-9
    per_axis = {}
    for ax, name in enumerate(["x", "y", "z"]):
        if np.std(cmd[:, ax]) > 1e-6 and np.std(realized[:, ax]) > 1e-6:
            per_axis[name] = float(np.corrcoef(cmd[:, ax], realized[:, ax])[0, 1])
        else:
            per_axis[name] = float("nan")
    cmd_unit = cmd / cmd_norm[:, None]
    real_unit = realized / real_norm[:, None]
    cos_align = np.einsum("ij,ij->i", cmd_unit, real_unit)
    exec_ratio = real_norm / cmd_norm
    chunk_starts = list(range(0, T - 1, ACTION_HORIZON))
    chunk_mean_eps, chunk_cmd_dirs = [], []
    for c0 in chunk_starts:
        c1 = min(c0 + ACTION_HORIZON, T - 1)
        if c1 - c0 < 2:
            continue
        chunk_mean_eps.append(float(np.mean(1.0 - cos_align[c0:c1])))
        v = cmd[c0:c1].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-9)
        chunk_cmd_dirs.append(v)
    chunk_mean_eps = np.array(chunk_mean_eps)
    chunk_cmd_dirs = np.array(chunk_cmd_dirs)
    persistent_dir_after_high_eps = 0
    if len(chunk_cmd_dirs) > 1:
        for k in range(len(chunk_cmd_dirs) - 1):
            if float(np.dot(chunk_cmd_dirs[k], chunk_cmd_dirs[k + 1])) > 0.9:
                persistent_dir_after_high_eps += int(chunk_mean_eps[k] > 0.5)

    def _max_run(mask):
        if mask.size == 0:
            return 0
        best = cur = 0
        for b in mask:
            if b:
                cur += 1; best = max(best, cur)
            else:
                cur = 0
        return best

    return {
        "corr_x": per_axis["x"], "corr_y": per_axis["y"], "corr_z": per_axis["z"],
        "mean_cos_align": float(np.mean(cos_align)),
        "median_exec_ratio": float(np.median(exec_ratio)),
        "frac_low_exec_ratio": float(np.mean(exec_ratio < 0.01)),
        "max_consecutive_low_exec": int(_max_run(exec_ratio < 0.01)),
        "persistent_dir_after_high_eps": int(persistent_dir_after_high_eps),
    }


def bootstrap_tpr_fpr(rows, n_boot=2000):
    """Bootstrap 95% CI on TPR (gate-fires-on-fail) and FPR (gate-fires-on-ok)."""
    tprs, fprs = [], []
    n = len(rows)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        sample = [rows[i] for i in idx]
        n_fail = sum(1 for r in sample if not r["success"])
        n_ok = sum(1 for r in sample if r["success"])
        if n_fail == 0 or n_ok == 0:
            continue
        fire_fail = sum(1 for r in sample if r["fire_step"] is not None and not r["success"])
        fire_ok = sum(1 for r in sample if r["fire_step"] is not None and r["success"])
        tprs.append(fire_fail / n_fail)
        fprs.append(fire_ok / n_ok)
    return (float(np.percentile(tprs, 2.5)), float(np.percentile(tprs, 97.5)),
            float(np.percentile(fprs, 2.5)), float(np.percentile(fprs, 97.5)))


def auroc_safe(y_true, y_score):
    """AUROC, robust to score nan / single-class."""
    y_score = np.array(y_score, dtype=float)
    valid = ~np.isnan(y_score)
    if valid.sum() < 4 or len(set(np.array(y_true)[valid])) < 2:
        return float("nan")
    return float(roc_auc_score(np.array(y_true)[valid], y_score[valid]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-traces-dir", required=True)
    ap.add_argument("--calib-summary", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    # Fit ε-cluster model on calibration corpus (task-3 failures)
    calib = json.loads(pathlib.Path(args.calib_summary).read_text())
    calib_fails = [r for r in calib["rows"] if r["label"] != "PHYS_OK"]
    X_calib = np.array([[r[f] for f in EPS_FEATURES] for r in calib_fails])
    for j in range(X_calib.shape[1]):
        col = X_calib[:, j]
        if np.any(np.isnan(col)):
            X_calib[np.isnan(col), j] = np.nanmedian(col)
    scaler = StandardScaler().fit(X_calib)
    kmeans = KMeans(n_clusters=2, n_init=20, random_state=0).fit(scaler.transform(X_calib))
    cluster_means = [X_calib[kmeans.labels_ == c, EPS_FEATURES.index("frac_low_exec_ratio")].mean()
                     for c in range(2)]
    c1_cluster = int(np.argmax(cluster_means))

    # Process new traces — compute all signals + features
    traces = []
    for p in sorted(pathlib.Path(args.new_traces_dir).glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        T = int(d["action"].shape[0])
        success = bool(d["success"])
        task_id = int(d["task_id"])
        action = d["action"]
        ee_pos = d["ee_pos"]
        fire = gate_first_fire(action, ee_pos)
        stac = stac_score(action)
        feats = eps_features_for_trace(action, ee_pos)
        cluster = None
        if not success and feats is not None:
            x = np.array([[feats[f] for f in EPS_FEATURES]])
            for j in range(x.shape[1]):
                if np.isnan(x[0, j]):
                    x[0, j] = np.nanmedian(X_calib[:, j])
            xs = scaler.transform(x)
            cluster_raw = int(kmeans.predict(xs)[0])
            cluster = "C1" if cluster_raw == c1_cluster else "C0"
        traces.append({"trace": p.name, "task_id": task_id, "T": T, "success": success,
                       "fire_step": fire, "stac_score": stac, "cluster": cluster,
                       "features": feats})

    # Per-task: TPR/FPR + bootstrap CIs + baselines
    by_task = defaultdict(list)
    for tr in traces:
        by_task[tr["task_id"]].append(tr)

    print(f"\n{'task':>4s}  {'OK':>3s}  {'F':>3s}  {'C0':>3s}  {'C1':>3s}  "
          f"{'TPR (95%CI)':>16s}  {'FPR (95%CI)':>16s}  "
          f"{'STAC_AUROC':>10s}  {'len_AUROC':>9s}  {'gate_AUROC':>10s}  {'med_fire':>9s}")
    summary = {}
    aggregate_rows = []
    for tid in sorted(by_task):
        rows = by_task[tid]
        aggregate_rows.extend(rows)
        n_ok = sum(1 for r in rows if r["success"])
        n_fail = sum(1 for r in rows if not r["success"])
        n_c0 = sum(1 for r in rows if r["cluster"] == "C0")
        n_c1 = sum(1 for r in rows if r["cluster"] == "C1")
        fire_fail = sum(1 for r in rows if r["fire_step"] is not None and not r["success"])
        fire_ok = sum(1 for r in rows if r["fire_step"] is not None and r["success"])
        tpr = fire_fail / max(1, n_fail)
        fpr = fire_ok / max(1, n_ok)

        # Bootstrap CIs
        tpr_lo, tpr_hi, fpr_lo, fpr_hi = bootstrap_tpr_fpr(rows, n_boot=args.n_boot)

        # AUROC baselines on this task
        y = np.array([0 if r["success"] else 1 for r in rows])
        # Gate as binary fire indicator (1=fired)
        gate_score = np.array([1.0 if r["fire_step"] is not None else 0.0 for r in rows])
        gate_auc = auroc_safe(y, gate_score)
        # STAC: higher = more erratic = predicted failure
        stac_arr = np.array([r["stac_score"] if r["stac_score"] is not None else np.nan for r in rows])
        stac_auc = auroc_safe(y, stac_arr)
        # Length: longer trace = more likely failure (timeout)
        len_arr = np.array([r["T"] for r in rows], dtype=float)
        len_auc = auroc_safe(y, len_arr)

        fires = [r["fire_step"] for r in rows if r["fire_step"] is not None]
        med = int(np.median(fires)) if fires else None
        print(f"{tid:>4d}  {n_ok:>3d}  {n_fail:>3d}  {n_c0:>3d}  {n_c1:>3d}  "
              f"  {tpr:.2f} [{tpr_lo:.2f},{tpr_hi:.2f}]".ljust(20) +
              f"  {fpr:.2f} [{fpr_lo:.2f},{fpr_hi:.2f}]".ljust(20) +
              f"  {stac_auc:>10.2f}  {len_auc:>9.2f}  {gate_auc:>10.2f}  {str(med):>9s}")
        summary[str(tid)] = dict(
            n_ok=n_ok, n_fail=n_fail, n_c0=n_c0, n_c1=n_c1,
            tpr=tpr, fpr=fpr,
            tpr_ci_lo=tpr_lo, tpr_ci_hi=tpr_hi,
            fpr_ci_lo=fpr_lo, fpr_ci_hi=fpr_hi,
            gate_auroc=gate_auc, stac_auroc=stac_auc, len_auroc=len_auc,
            med_fire_step=med,
        )

    # Aggregate across all tasks
    rows_a = aggregate_rows
    n_ok = sum(1 for r in rows_a if r["success"])
    n_fail = sum(1 for r in rows_a if not r["success"])
    n_c0 = sum(1 for r in rows_a if r["cluster"] == "C0")
    n_c1 = sum(1 for r in rows_a if r["cluster"] == "C1")
    fire_fail = sum(1 for r in rows_a if r["fire_step"] is not None and not r["success"])
    fire_ok = sum(1 for r in rows_a if r["fire_step"] is not None and r["success"])
    tpr = fire_fail / max(1, n_fail)
    fpr = fire_ok / max(1, n_ok)
    tpr_lo, tpr_hi, fpr_lo, fpr_hi = bootstrap_tpr_fpr(rows_a, n_boot=args.n_boot)
    y = np.array([0 if r["success"] else 1 for r in rows_a])
    gate_auc = auroc_safe(y, [1.0 if r["fire_step"] is not None else 0.0 for r in rows_a])
    stac_auc = auroc_safe(y, [r["stac_score"] if r["stac_score"] is not None else np.nan for r in rows_a])
    len_auc = auroc_safe(y, [r["T"] for r in rows_a])
    print(f"\nAGG   {n_ok:>3d}  {n_fail:>3d}  {n_c0:>3d}  {n_c1:>3d}  "
          f"  {tpr:.2f} [{tpr_lo:.2f},{tpr_hi:.2f}]  {fpr:.2f} [{fpr_lo:.2f},{fpr_hi:.2f}]  "
          f"{stac_auc:>10.2f}  {len_auc:>9.2f}  {gate_auc:>10.2f}")

    # Compute savings: if aborted at fire_step on fires, how many steps saved?
    saved_steps = 0
    fires_count = 0
    for r in rows_a:
        if r["fire_step"] is not None:
            saved_steps += (r["T"] - r["fire_step"])
            fires_count += 1
    total_steps = sum(r["T"] for r in rows_a)
    print(f"\n=== Compute savings if aborted at gate-fire ===")
    print(f"  fires: {fires_count}/{len(rows_a)}")
    print(f"  total trace steps (no abort): {total_steps}")
    print(f"  saved steps (with abort):     {saved_steps} ({100*saved_steps/total_steps:.1f}%)")

    summary["aggregate"] = dict(
        n_ok=n_ok, n_fail=n_fail, n_c0=n_c0, n_c1=n_c1,
        tpr=tpr, fpr=fpr,
        tpr_ci_lo=tpr_lo, tpr_ci_hi=tpr_hi,
        fpr_ci_lo=fpr_lo, fpr_ci_hi=fpr_hi,
        gate_auroc=gate_auc, stac_auroc=stac_auc, len_auroc=len_auc,
        saved_steps=saved_steps, total_steps=total_steps,
        savings_pct=100*saved_steps/total_steps if total_steps else 0,
    )
    summary["traces"] = traces

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
