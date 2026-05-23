---
name: IN_FALSE failure diagnostic — NO_APPROACH dominates (2026-05-23)
description: Applied the cmd-vs-realized signature methodology to the 12 IN_FALSE task-3 failure traces (bowl trajectory instead of EE trajectory). 83% are NO_APPROACH — Pi0.5 never gets EE within 8cm of bowl, gripper never commanded closed near bowl. Different failure mode than IN_TRUE; needs different reward component.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
## Result

Applied diagnostic to 12 task-3 traces labeled `IN_FALSE_CLOSE_FALSE` (11) or
`IN_FALSE_CLOSE_TRUE` (1). For each, computed bowl-trajectory metrics:
`ee_to_bowl_min` (closest EE-bowl approach), `bowl_motion` (max bowl
displacement from init), `bowl_max_y` (forward progress), `grip_close_steps_near_bowl`.

| Sub-mode | Count | % | Signature |
|---|---|---|---|
| `NO_APPROACH` | 10 | 83% | `ee_to_bowl_min` 8–27cm, `grip_close_near_bowl_steps = 0` for all 10 |
| `OTHER` | 1 | 8% | Got close, partial grasp, bowl drifted back |
| `PLACEMENT_MISS` | 1 | 8% | Bowl reached drawer area, missed AABB |

**Pi0.5 never even attempts a grasp in 83% of IN_FALSE traces.** Gripper command stays at −1 (open) throughout the entire approach phase. Bowl stays near its initial position.

## Why this matters

This is a structurally different failure mode than the IN_TRUE_CLOSE_FALSE pattern we diagnosed previously:

- **IN_TRUE_CLOSE_FALSE** (14 traces, 53% of failures): Pi0.5 generates correct command (saturated `+y` drawer-close motion), EE doesn't realize it (kinematic null-space drift). Pi0.5 *tries* and the embodiment *can't*.

- **IN_FALSE / NO_APPROACH** (10 traces, 38% of failures): Pi0.5 *doesn't try*. EE never gets close enough to attempt a grasp; gripper never commands closed in the vicinity of the bowl. The bowl stays where it started.

| Stratum | Failure character | Required intervention |
|---|---|---|
| IN_TRUE_CLOSE_FALSE | Execution-stuck (cmd correct, EE doesn't move) | Tracking-error reward penalty |
| IN_FALSE / NO_APPROACH | Approach-not-initiated (EE never near bowl, no grasp attempt) | EE-to-bowl distance progress reward |
| IN_FALSE / PLACEMENT_MISS | Bowl reached drawer area, missed AABB | Goal-region shaping reward |

## Sub-pattern within NO_APPROACH

The 10 NO_APPROACH traces split:
- 5 with `ee_to_bowl_min` in 8–12cm range (close, but never closed gripper — possible perception near-miss)
- 5 with `ee_to_bowl_min > 14cm` (way off — Pi0.5 doesn't even target the bowl)

Worth a deeper look if we pursue NO_APPROACH-targeted intervention. Different sub-sub-modes may need different signals.

## What this changes

**The single-bet "feasibility-as-reward" GRPO experiment as I scoped it would only target the 14 IN_TRUE traces (54% of failures).** The dominant IN_FALSE mode (NO_APPROACH, 38%) is invisible to a tracking-error reward — Pi0.5 isn't commanding sustained-but-unrealized motion in those traces; it's just not commanding approach motion at all.

The honest experiment design:
- **A:** sparse task success only (baseline)
- **B:** sparse + tracking-error penalty (targets IN_TRUE)
- **C:** sparse + tracking-error penalty + EE-to-bowl progress reward (targets IN_TRUE AND NO_APPROACH)

The A→B→C ablation attributes per-component lift to per-stratum failure reduction. That's the per-component contribution story for a paper-shaped writeup.

## How to apply

- The diagnostic apparatus generalizes beyond IN_TRUE. Apply it to other failure strata as they appear.
- The hypothesis "dense supervision matters" survives, but the formulation "single tracking-error penalty solves it" is too narrow. Multiple targeted dense terms, validated per-stratum.
- Before committing to A→B→C, do one more diagnostic: split the 10 NO_APPROACH traces by `ee_to_bowl_min` (8–12cm vs >14cm) and check whether they're really sub-sub-modes requiring further decomposition. ~30 min of analysis.

## Files

- Diagnostic script: `/tmp/in_false_diagnostic.py` on local macOS
- Output: `/tmp/openpi-data/in_false_diagnostic.json` (12 entries with per-trace metrics)
- Source corpus: HF dataset `arif101/openpi-contact-mpc-2026-05-19`, folder `recovery_source_traces/`
