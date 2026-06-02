"""Paired baseline-vs-intervention analysis with bootstrap CIs.

Reads two trace directories (baseline, intervention) collected at the SAME
perturbation seed. Computes trial-by-trial transitions and the four core
quantities for paper:

  recovery_rate = P(intervention=OK | baseline=FAIL)
  damage_rate   = P(intervention=FAIL | baseline=OK)
  net_effect    = success_rate(intervention) - success_rate(baseline)
  break_even_p  = damage_rate / (recovery_rate + damage_rate)

with bootstrap 95% CIs on each.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import re
from collections import Counter
import numpy as np

RNG = np.random.default_rng(0)


def parse_outcomes(traces_dir):
    out = {}
    for p in pathlib.Path(traces_dir).glob("*.npz"):
        m = re.match(r"PHYS_(OK|FAIL)_libero10_task(\d+)_trial(\d+)_ts.*\.npz", p.name)
        if not m:
            continue
        success = m.group(1) == "OK"
        task = int(m.group(2))
        trial = int(m.group(3))
        out[(task, trial)] = {"success": success, "trace": p.name}
    return out


def transition_table(baseline, intervention):
    pairs = []
    for key in sorted(set(baseline) & set(intervention)):
        b = baseline[key]["success"]
        i = intervention[key]["success"]
        pairs.append((key, b, i))
    return pairs


def bootstrap_rates(pairs, n_boot=5000):
    """Returns 95% CIs on recovery_rate, damage_rate, net_effect, break_even."""
    rec_rates, dam_rates, net_effects, break_evens = [], [], [], []
    n = len(pairs)
    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        sample = [pairs[i] for i in idx]
        n_b_fail = sum(1 for k, b, i in sample if not b)
        n_b_ok = sum(1 for k, b, i in sample if b)
        n_rec = sum(1 for k, b, i in sample if not b and i)
        n_dam = sum(1 for k, b, i in sample if b and not i)
        rec = n_rec / max(1, n_b_fail)
        dam = n_dam / max(1, n_b_ok)
        net = (sum(1 for k, b, i in sample if i) - sum(1 for k, b, i in sample if b)) / max(1, n)
        be = dam / max(rec + dam, 1e-9)
        rec_rates.append(rec)
        dam_rates.append(dam)
        net_effects.append(net)
        break_evens.append(be)
    def ci(arr):
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    return {
        "recovery_rate": ci(rec_rates),
        "damage_rate": ci(dam_rates),
        "net_effect": ci(net_effects),
        "break_even_p_fail": ci(break_evens),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", required=True)
    ap.add_argument("--intervention-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    baseline = parse_outcomes(args.baseline_dir)
    intervention = parse_outcomes(args.intervention_dir)
    pairs = transition_table(baseline, intervention)
    print(f"N shared trials: {len(pairs)}")

    # Point estimates
    n_b_fail = sum(1 for k, b, i in pairs if not b)
    n_b_ok = sum(1 for k, b, i in pairs if b)
    n_rec = sum(1 for k, b, i in pairs if not b and i)
    n_dam = sum(1 for k, b, i in pairs if b and not i)
    n_i_fail = sum(1 for k, b, i in pairs if not i)
    n_i_ok = sum(1 for k, b, i in pairs if i)

    base_rate = n_b_fail / len(pairs)
    inter_rate = n_i_fail / len(pairs)
    rec_pt = n_rec / max(1, n_b_fail)
    dam_pt = n_dam / max(1, n_b_ok)
    net_pt = (n_i_ok - n_b_ok) / len(pairs)
    be_pt = dam_pt / max(rec_pt + dam_pt, 1e-9)

    cis = bootstrap_rates(pairs, n_boot=args.n_boot)

    print(f"\n=== Aggregate (N={len(pairs)}) ===")
    print(f"  baseline success rate:        {n_b_ok}/{len(pairs)} = {100*(1-base_rate):.1f}%")
    print(f"  intervention success rate:    {n_i_ok}/{len(pairs)} = {100*(1-inter_rate):.1f}%")
    print(f"  recovery rate (P(I=OK|B=F)):  {n_rec}/{n_b_fail} = {rec_pt:.2f}  [95%CI {cis['recovery_rate'][0]:.2f}, {cis['recovery_rate'][1]:.2f}]")
    print(f"  damage rate (P(I=F|B=OK)):    {n_dam}/{n_b_ok} = {dam_pt:.2f}  [95%CI {cis['damage_rate'][0]:.2f}, {cis['damage_rate'][1]:.2f}]")
    print(f"  net effect:                    {net_pt:+.3f}  [95%CI {cis['net_effect'][0]:+.3f}, {cis['net_effect'][1]:+.3f}]")
    print(f"  break-even P(fail):            {be_pt:.3f}  [95%CI {cis['break_even_p_fail'][0]:.3f}, {cis['break_even_p_fail'][1]:.3f}]")

    print(f"\n=== Transition counts ===")
    cnts = Counter()
    for k, b, i in pairs:
        if b and i: cnts["OK→OK"] += 1
        elif not b and i: cnts["FAIL→OK"] += 1
        elif b and not i: cnts["OK→FAIL"] += 1
        else: cnts["FAIL→FAIL"] += 1
    for k in ["OK→OK", "FAIL→OK", "OK→FAIL", "FAIL→FAIL"]:
        print(f"  {k:>12s}: {cnts[k]}")

    print(f"\n=== Per-task ===")
    print(f"  {'task':>5s}  {'B_OK':>4s}  {'B_F':>3s}  {'I_OK':>4s}  {'I_F':>3s}  {'rec':>5s}  {'dam':>5s}  {'net':>6s}")
    by_task = {}
    for k, b, i in pairs:
        tid = k[0]
        by_task.setdefault(tid, []).append((b, i))
    for tid in sorted(by_task):
        p = by_task[tid]
        b_ok = sum(1 for b, i in p if b)
        b_f = sum(1 for b, i in p if not b)
        i_ok = sum(1 for b, i in p if i)
        i_f = sum(1 for b, i in p if not i)
        rec = sum(1 for b, i in p if not b and i) / max(1, b_f)
        dam = sum(1 for b, i in p if b and not i) / max(1, b_ok)
        net = (i_ok - b_ok) / len(p)
        print(f"  {tid:>5d}  {b_ok:>4d}  {b_f:>3d}  {i_ok:>4d}  {i_f:>3d}  {rec:>5.2f}  {dam:>5.2f}  {net:>+6.2f}")

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output).write_text(json.dumps({
        "n": len(pairs),
        "base_success_rate": 1 - base_rate,
        "intervention_success_rate": 1 - inter_rate,
        "recovery_rate_point": rec_pt,
        "damage_rate_point": dam_pt,
        "net_effect_point": net_pt,
        "break_even_point": be_pt,
        "cis": cis,
        "transitions": dict(cnts),
        "by_task": {str(tid): {
            "B_OK": sum(1 for b, i in by_task[tid] if b),
            "B_F": sum(1 for b, i in by_task[tid] if not b),
            "I_OK": sum(1 for b, i in by_task[tid] if i),
            "I_F": sum(1 for b, i in by_task[tid] if not i),
            "rec": sum(1 for b, i in by_task[tid] if not b and i) / max(1, sum(1 for b, i in by_task[tid] if not b)),
            "dam": sum(1 for b, i in by_task[tid] if b and not i) / max(1, sum(1 for b, i in by_task[tid] if b)),
            "net": (sum(1 for b, i in by_task[tid] if i) - sum(1 for b, i in by_task[tid] if b)) / len(by_task[tid]),
        } for tid in by_task},
    }, indent=2))
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
