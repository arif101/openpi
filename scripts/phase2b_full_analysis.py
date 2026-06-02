"""Phase 2b full analysis — gate eval + ε-cluster prediction on new task traces.

Runs three analyses:
  1. Gate eval: fixed 5-param gate, TPR/FPR per task (success/fail labels)
  2. ε-cluster prediction: apply the K=2 KMeans model fit on task-3 corpus
     to the new task 0/6/9 failure traces. Categorizes each failure as C0
     (generation-style) or C1 (execution-stuck).
  3. Joint analysis: does the gate fire on the right cluster? Is the cluster
     distribution similar across tasks?

The ε-cluster model is re-fit on the task-3 features (we have them locally)
and applied to the new task features (loaded from GPU traces).

Usage:
    python scripts/phase2b_full_analysis.py \\
        --new-traces-dir data/contact_mpc/phase2b_gpu_results/traces_phase2b_pert \\
        --calib-summary data/contact_mpc/prediction_error_analysis/summary.json \\
        --output data/contact_mpc/phase2b_full_analysis.json
"""
from __future__ import annotations
import argparse
import json
import pathlib
from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
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


def gate_first_fire(action, ee_pos):
    """Streaming gate: locked five-parameter recipe."""
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


def eps_features_for_trace(action, ee_pos):
    """Compute the 8 ε-features (same as analyze_prediction_error.py)."""
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
            same_dir = float(np.dot(chunk_cmd_dirs[k], chunk_cmd_dirs[k + 1]))
            if same_dir > 0.9:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-traces-dir", required=True)
    ap.add_argument("--calib-summary", required=True,
                    help="data/contact_mpc/prediction_error_analysis/summary.json")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    # 1. Fit ε-cluster model on calibration corpus (task-3 failures)
    calib = json.loads(pathlib.Path(args.calib_summary).read_text())
    calib_fails = [r for r in calib["rows"] if r["label"] != "PHYS_OK"]
    X_calib = np.array([[r[f] for f in EPS_FEATURES] for r in calib_fails])
    for j in range(X_calib.shape[1]):
        col = X_calib[:, j]
        if np.any(np.isnan(col)):
            X_calib[np.isnan(col), j] = np.nanmedian(col)
    scaler = StandardScaler().fit(X_calib)
    Xs_calib = scaler.transform(X_calib)
    kmeans = KMeans(n_clusters=2, n_init=20, random_state=0).fit(Xs_calib)
    # Identify which cluster is C1 (execution-stuck) — higher frac_low_exec_ratio
    cluster_means = [X_calib[kmeans.labels_ == c, EPS_FEATURES.index("frac_low_exec_ratio")].mean()
                     for c in range(2)]
    c1_cluster = int(np.argmax(cluster_means))
    print(f"calib-fit: cluster {c1_cluster} = C1 (frac_low_exec mean {cluster_means[c1_cluster]:.3f})")
    print(f"           cluster {1 - c1_cluster} = C0 (frac_low_exec mean {cluster_means[1 - c1_cluster]:.3f})")

    # 2. Process new traces
    traces = []
    for p in sorted(pathlib.Path(args.new_traces_dir).glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        T = int(d["action"].shape[0])
        success = bool(d["success"])
        task_id = int(d["task_id"])
        fire = gate_first_fire(d["action"], d["ee_pos"])
        feats = eps_features_for_trace(d["action"], d["ee_pos"])
        cluster = None
        if not success and feats is not None:
            # Predict cluster
            x = np.array([[feats[f] for f in EPS_FEATURES]])
            for j in range(x.shape[1]):
                if np.isnan(x[0, j]):
                    x[0, j] = np.nanmedian(X_calib[:, j])
            xs = scaler.transform(x)
            cluster_raw = int(kmeans.predict(xs)[0])
            cluster = "C1" if cluster_raw == c1_cluster else "C0"
        traces.append({"trace": p.name, "task_id": task_id, "T": T, "success": success,
                       "fire_step": fire, "cluster": cluster, "features": feats})

    # 3. Per-task summary
    by_task = defaultdict(list)
    for tr in traces:
        by_task[tr["task_id"]].append(tr)

    print(f"\n{'task':>4s}  {'OK':>3s}  {'FAIL':>4s}  {'C0':>3s}  {'C1':>3s}  "
          f"{'TPR':>5s}  {'FPR':>5s}  {'TPR_C1':>7s}  {'TPR_C0':>7s}  {'med_fire':>9s}")
    summary = {}
    for tid in sorted(by_task):
        rows = by_task[tid]
        n_ok = sum(1 for r in rows if r["success"])
        n_fail = sum(1 for r in rows if not r["success"])
        n_c0 = sum(1 for r in rows if r["cluster"] == "C0")
        n_c1 = sum(1 for r in rows if r["cluster"] == "C1")
        fire_fail = sum(1 for r in rows if r["fire_step"] is not None and not r["success"])
        fire_ok = sum(1 for r in rows if r["fire_step"] is not None and r["success"])
        fire_c1 = sum(1 for r in rows if r["fire_step"] is not None and r["cluster"] == "C1")
        fire_c0 = sum(1 for r in rows if r["fire_step"] is not None and r["cluster"] == "C0")
        tpr = fire_fail / n_fail if n_fail > 0 else float("nan")
        fpr = fire_ok / n_ok if n_ok > 0 else float("nan")
        tpr_c1 = fire_c1 / n_c1 if n_c1 > 0 else float("nan")
        tpr_c0 = fire_c0 / n_c0 if n_c0 > 0 else float("nan")
        fires = [r["fire_step"] for r in rows if r["fire_step"] is not None]
        med = int(np.median(fires)) if fires else None
        print(f"{tid:>4d}  {n_ok:>3d}  {n_fail:>4d}  {n_c0:>3d}  {n_c1:>3d}  "
              f"{tpr:>5.2f}  {fpr:>5.2f}  {tpr_c1:>7.2f}  {tpr_c0:>7.2f}  {str(med):>9s}")
        summary[str(tid)] = dict(n_ok=n_ok, n_fail=n_fail, n_c0=n_c0, n_c1=n_c1,
                                  tpr=tpr, fpr=fpr, tpr_c1=tpr_c1, tpr_c0=tpr_c0,
                                  med_fire_step=med, rows=rows)

    # Aggregate
    all_rows = traces
    n_ok_a = sum(1 for r in all_rows if r["success"])
    n_fail_a = sum(1 for r in all_rows if not r["success"])
    n_c0_a = sum(1 for r in all_rows if r["cluster"] == "C0")
    n_c1_a = sum(1 for r in all_rows if r["cluster"] == "C1")
    fire_fail_a = sum(1 for r in all_rows if r["fire_step"] is not None and not r["success"])
    fire_ok_a = sum(1 for r in all_rows if r["fire_step"] is not None and r["success"])
    fire_c1_a = sum(1 for r in all_rows if r["fire_step"] is not None and r["cluster"] == "C1")
    fire_c0_a = sum(1 for r in all_rows if r["fire_step"] is not None and r["cluster"] == "C0")
    print(f"\nAGG   {n_ok_a:>3d}  {n_fail_a:>4d}  {n_c0_a:>3d}  {n_c1_a:>3d}  "
          f"{fire_fail_a/max(1,n_fail_a):>5.2f}  {fire_ok_a/max(1,n_ok_a):>5.2f}  "
          f"{fire_c1_a/max(1,n_c1_a):>7.2f}  {fire_c0_a/max(1,n_c0_a):>7.2f}")
    summary["aggregate"] = dict(n_ok=n_ok_a, n_fail=n_fail_a, n_c0=n_c0_a, n_c1=n_c1_a,
                                 tpr=fire_fail_a/max(1,n_fail_a),
                                 fpr=fire_ok_a/max(1,n_ok_a),
                                 tpr_c1=fire_c1_a/max(1,n_c1_a),
                                 tpr_c0=fire_c0_a/max(1,n_c0_a))

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
