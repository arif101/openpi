"""Validate the reachability field on the data we actually care about.

R²=0.5 on random samples doesn't necessarily mean the field knows that
Pi0.5's "command +y, no motion" stuck pattern is infeasible. Test
directly: at every step of every trace, ask the field to predict the
realized EE delta over the next K=20 OSC steps. Compare to ground truth
(ee[t+K] - ee[t]).

Three failure-relevant tests:

  1. Per-step correlation across all 14 IN_TRUE_CLOSE_FALSE traces +
     a matched sample of 14 PHYS_OK traces.

  2. Inflection-zone test: at the stuck-inflection step of each failing
     trace, what does the field predict for the actual command? It should
     predict small motion. If it predicts large motion (matching the
     command), the field has missed the infeasibility entirely.

  3. AUROC: across the union of (failure-trace stuck-zone steps,
     success-trace moving-zone steps), how well does ||predicted_realized||
     separate "barely moved" from "actually moved"?
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn


class ReachabilityField(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(14, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, q, a):
        return self.net(torch.cat([q, a], dim=-1))


def find_inflection(action: np.ndarray, ee: np.ndarray, window: int = 30) -> int:
    T = action.shape[0]
    for t in range(window, T - window):
        win = slice(t, t + window)
        if np.mean(action[win, 1]) > 0.4 and np.max(np.abs(np.diff(ee[win, 1]))) < 0.002:
            return t
    return max(T - 100, 0)


def predict_for_trace(model, d, device, K=20, stride=5, max_t=None) -> dict:
    """Per-step predictions across a trace. Returns (T_used, pred, actual) arrays."""
    qpos = d["qpos"]  # (T, n_q)
    ee = d["ee_pos"]   # (T, 3)
    act = d["action"]  # (T, 7)
    T = act.shape[0]
    if max_t is not None: T = min(T, max_t)
    times = list(range(0, T - K, stride))
    q_in = torch.tensor(qpos[times, :7], dtype=torch.float32, device=device)
    a_in = torch.tensor(act[times], dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(q_in, a_in).cpu().numpy()
    actual = ee[[t + K for t in times]] - ee[times]
    return {"times": np.asarray(times), "pred": pred, "actual": actual.astype(np.float32)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strata-json", required=True)
    ap.add_argument("--traces-dir", required=True)
    ap.add_argument("--field-pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=20)
    args = ap.parse_args()

    strata = json.loads(pathlib.Path(args.strata_json).read_text())
    itcf_names = [n for n, r in strata.items() if r.get("label") == "IN_TRUE_CLOSE_FALSE"]
    print(f"N IN_TRUE_CLOSE_FALSE: {len(itcf_names)}")
    traces_dir = pathlib.Path(args.traces_dir)
    ok_paths = sorted(traces_dir.glob("PHYS_OK_libero_10_task3_*.npz"))[:len(itcf_names)]
    print(f"N matched PHYS_OK:    {len(ok_paths)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ReachabilityField().to(device)
    model.load_state_dict(torch.load(args.field_pt, map_location=device))
    model.eval()

    # Per-trace predictions
    pred_all, actual_all, label_all, trace_all, time_all = [], [], [], [], []
    for n in itcf_names:
        d = np.load(traces_dir / n, allow_pickle=True)
        r = predict_for_trace(model, d, device, K=args.K)
        pred_all.append(r["pred"]); actual_all.append(r["actual"])
        label_all.append(np.full(len(r["pred"]), "FAIL", dtype=object))
        trace_all.append([n] * len(r["pred"]))
        time_all.append(r["times"])
    for p in ok_paths:
        d = np.load(p, allow_pickle=True)
        r = predict_for_trace(model, d, device, K=args.K)
        pred_all.append(r["pred"]); actual_all.append(r["actual"])
        label_all.append(np.full(len(r["pred"]), "OK", dtype=object))
        trace_all.append([p.name] * len(r["pred"]))
        time_all.append(r["times"])

    pred = np.concatenate(pred_all, axis=0)
    actual = np.concatenate(actual_all, axis=0)
    labels = np.concatenate(label_all, axis=0)
    pred_norm = np.linalg.norm(pred, axis=1)
    actual_norm = np.linalg.norm(actual, axis=1)
    print(f"Total prediction points: {len(pred)}")

    # Correlation per axis
    print("\nCorrelation predicted vs. actual (per axis, across ALL points):")
    for ax, name in enumerate("xyz"):
        c = float(np.corrcoef(pred[:, ax], actual[:, ax])[0, 1])
        print(f"  {name}: r={c:+.3f}")
    # R² across all points
    ss_res = ((pred - actual) ** 2).sum(axis=0)
    ss_tot = ((actual - actual.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1 - ss_res / np.maximum(ss_tot, 1e-9)
    print(f"  R² xyz: ({r2[0]:+.3f}, {r2[1]:+.3f}, {r2[2]:+.3f})  mean R²={r2.mean():+.3f}")

    # Inflection-zone test on failure traces
    print("\n=== Inflection-zone test (14 IN_TRUE_CLOSE_FALSE traces) ===")
    print(f"{'trace':<60} {'cmd_dy':>7} {'actual_xyz':>30} {'pred_xyz':>30} {'pred_match?':>14}")
    n_correct_low = 0
    for n in itcf_names:
        d = np.load(traces_dir / n, allow_pickle=True)
        T = d["action"].shape[0]
        infl = find_inflection(d["action"], d["ee_pos"])
        if infl + args.K >= T: continue
        q_in = torch.tensor(d["qpos"][infl, :7][None], dtype=torch.float32, device=device)
        a_in = torch.tensor(d["action"][infl][None], dtype=torch.float32, device=device)
        with torch.no_grad():
            p = model(q_in, a_in).cpu().numpy()[0]
        actual = d["ee_pos"][infl + args.K] - d["ee_pos"][infl]
        cmd_dy = float(d["action"][infl, 1])
        # If predicted_norm < 5mm, field correctly says "infeasible"
        pred_n = float(np.linalg.norm(p))
        actual_n = float(np.linalg.norm(actual))
        is_low = pred_n < 0.01
        actual_low = actual_n < 0.01
        # Correct if field's prediction matches the actual outcome direction
        match = "yes" if (is_low == actual_low) else "no"
        if is_low and actual_low: n_correct_low += 1
        print(f"  {n[:58]:<60} {cmd_dy:>+6.2f}  ({actual[0]:+.3f},{actual[1]:+.3f},{actual[2]:+.3f}) ({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) {match:>10}")

    print(f"\nInflection-zone: field correctly identified low realized motion in {n_correct_low}/{len(itcf_names)} cases")

    # AUROC: can pred_norm separate moved vs. not-moved samples?
    moved = actual_norm > 0.02  # 2cm threshold
    if 0 < moved.sum() < len(moved):
        # Simple AUROC computation
        from numpy import argsort
        order = argsort(pred_norm)  # ascending: smaller pred_norm should mean "less likely to move"
        y = moved[order].astype(int)
        # AUROC = mean rank of positives / n_neg
        n_pos = y.sum(); n_neg = (1 - y).sum()
        ranks = np.arange(1, len(y) + 1)
        sum_rank_pos = ranks[y == 1].sum()
        auroc = (sum_rank_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
        print(f"\nAUROC (predicted_norm separating moved>2cm vs not): {auroc:.3f}")
        print(f"  n_moved>2cm: {n_pos}  n_static: {n_neg}")

    out = {
        "n_points": int(len(pred)),
        "r2_x": float(r2[0]), "r2_y": float(r2[1]), "r2_z": float(r2[2]),
        "r2_mean": float(r2.mean()),
        "n_correct_low_inflection": int(n_correct_low),
        "n_inflection_traces": len(itcf_names),
    }
    if 0 < moved.sum() < len(moved):
        out["auroc"] = float(auroc)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
