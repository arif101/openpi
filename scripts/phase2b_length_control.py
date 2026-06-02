"""Length-controlled gate evaluation.

Tests whether the gate adds predictive info OVER trace-length alone, via
joint logistic regression. If gate coefficient is significant after
controlling for length → gate isn't just a length-detector.

Also computes: AT each step t, conditional on the rollout being still
alive, does gate-fire predict eventual failure better than chance? This
isolates the gate's value from the trivial "long-rollouts-fail" confound.
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np
import scipy.stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ACTION_HORIZON = 10
LOW_EXEC_THRESHOLD = 0.01
LOOKAHEAD_K = 5
WINDOW_STEPS = 50
GATE_THRESHOLD = 0.160
N_SUSTAINED_STEPS = 50


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces-dir", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = []
    for p in sorted(pathlib.Path(args.traces_dir).glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        fire = gate_first_fire(d["action"], d["ee_pos"])
        rows.append({"trace": p.name, "T": int(d["action"].shape[0]),
                     "success": bool(d["success"]), "fire_step": fire,
                     "task_id": int(d["task_id"])})

    print(f"N = {len(rows)} traces ({sum(1 for r in rows if r['success'])} OK / {sum(1 for r in rows if not r['success'])} FAIL)")

    # 1. Univariate AUROC for each predictor
    y = np.array([0 if r["success"] else 1 for r in rows])
    lengths = np.array([r["T"] for r in rows], dtype=float)
    gate_bin = np.array([1.0 if r["fire_step"] is not None else 0.0 for r in rows])
    gate_step = np.array([r["fire_step"] if r["fire_step"] is not None else r["T"] + 10 for r in rows], dtype=float)

    print("\n=== Univariate AUROC ===")
    if len(set(y)) > 1:
        print(f"  length:            {roc_auc_score(y, lengths):.3f}")
        print(f"  gate (binary):     {roc_auc_score(y, gate_bin):.3f}")
        # Earlier fire = more confident failure → invert sign
        try:
            gate_step_auc = roc_auc_score(y, -gate_step)
            print(f"  gate (fire_step):  {gate_step_auc:.3f} (earlier fire ⇒ failure)")
        except Exception:
            pass

    # 2. Joint logistic regression: y = β1 * length + β2 * gate + ε
    print("\n=== Joint logistic regression (does gate add info beyond length?) ===")
    if len(set(y)) > 1:
        X = np.column_stack([lengths / 500.0, gate_bin])  # rescale length to [0,1]ish
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        print(f"  coefficients (length_scaled, gate_bin): {clf.coef_[0]}")
        print(f"  intercept: {clf.intercept_[0]:.3f}")
        # Significance via likelihood ratio test (length-only vs length+gate)
        ll_full = clf.score(X, y)
        clf_len = LogisticRegression(max_iter=1000).fit(lengths.reshape(-1, 1) / 500.0, y)
        ll_len = clf_len.score(lengths.reshape(-1, 1) / 500.0, y)
        print(f"  accuracy (length-only): {ll_len:.3f}")
        print(f"  accuracy (length+gate): {ll_full:.3f}")
        # McNemar-like: pairs where prediction differs
        p_len = clf_len.predict(lengths.reshape(-1, 1) / 500.0)
        p_full = clf.predict(X)
        gain = sum((p_full == y) & (p_len != y)) - sum((p_full != y) & (p_len == y))
        print(f"  net additional correct predictions from adding gate: {gain}")

    # 3. Time-conditional analysis: for traces still alive at step t, P(failure)?
    print("\n=== Time-conditional gate-fire vs eventual outcome ===")
    print("  At each step t in {100, 150, 200, 260, 300, 400, 500}:")
    print("  - Among traces alive at step t, what's P(failure | gate fired by t) vs P(failure | gate didn't fire by t)?")
    for t in [100, 150, 200, 260, 300, 400, 500]:
        # Only traces with T >= t (alive at step t)
        alive = [r for r in rows if r["T"] >= t]
        fired_by_t = [r for r in alive if r["fire_step"] is not None and r["fire_step"] <= t]
        not_fired_by_t = [r for r in alive if r["fire_step"] is None or r["fire_step"] > t]
        if not alive:
            continue
        p_fail_fired = sum(1 for r in fired_by_t if not r["success"]) / max(1, len(fired_by_t))
        p_fail_notfired = sum(1 for r in not_fired_by_t if not r["success"]) / max(1, len(not_fired_by_t))
        # Chi-square test on contingency table
        table = np.array([[sum(1 for r in fired_by_t if not r["success"]),
                           sum(1 for r in fired_by_t if r["success"])],
                          [sum(1 for r in not_fired_by_t if not r["success"]),
                           sum(1 for r in not_fired_by_t if r["success"])]])
        if table.sum() > 0 and (table > 0).any():
            try:
                _, pvalue = scipy.stats.fisher_exact(table)
            except Exception:
                pvalue = None
        else:
            pvalue = None
        print(f"    t={t:>3d}: alive={len(alive):>3d}, fired={len(fired_by_t):>3d} "
              f"(P_fail={p_fail_fired:.2f}), not-fired={len(not_fired_by_t):>3d} "
              f"(P_fail={p_fail_notfired:.2f})  fisher_p={pvalue}")

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps({"rows": rows}, indent=2, default=str))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
