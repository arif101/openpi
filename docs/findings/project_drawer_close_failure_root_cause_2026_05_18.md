---
name: Drawer-close failure root cause — EE stuck, not Pi0.5 generation (2026-05-18)
description: Across all 14 IN_TRUE_CLOSE_FALSE task-3 traces, Pi0.5 commands sustained +y motion (drawer-close direction) but EE doesn't move in 11/14. This is execution failure, not action-generation failure. MPPI cannot help by design.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
## Result

Inspected the last 100 steps of every IN_TRUE_CLOSE_FALSE task-3 trace
(N=14). Computed:
- `cmd_dy` = mean commanded EE +y velocity in those steps
- `ee_dy` = actual EE +y displacement
- `drawer_dq` = white_cabinet_1_bottom_level qpos change

| Statistic | Value |
|---|---|
| mean `cmd_dy` | **+0.62 ± 0.17** (saturated near +0.7 in most) |
| mean `ee_dy` | +0.025 m ± 0.049 |
| traces with `cmd_dy > 0.2` AND `|ee_dy| < 2cm` | **11 / 14 (79%)** |
| traces with `|cmd_dy| < 0.1` (Pi0.5 not trying) | **0 / 14 (0%)** |
| traces with both EE motion AND drawer close >10cm | 2 / 14 (14%) — still failed BDDL |

## What this implies

**Hypothesis (1) is ruled out:** Pi0.5 always generates drawer-close commands.
Hypothesis was that base policy didn't sample drawer-close action chunks.
False — it samples them in 100% of failing traces.

**The actual failure mode is EE-stuck-against-geometry.** Pi0.5 issues
saturated +y commands for hundreds of steps; EE doesn't move forward;
controller is fighting an external constraint (cabinet exterior, wine
rack, table edge, joint limits, or controller singularity). The
position-delta input space cannot escape this state because the
delta is correct but unrealizable.

**MPPI cannot help this failure mode by design:**
- Candidates are Gaussian perturbations `nominal + ε, σ=0.1` around a
  prior that is already correct (+y).
- Noise just adds wiggle to a command physics is refusing.
- No σ value can construct the structurally-different maneuver needed
  (retract, reposition, re-approach, push) — that requires a multi-stage
  plan that's beyond MPPI's H=10 horizon and Gaussian sampling structure.

## How to apply

1. **Stop investigating MPPI cost tuning / horizon / candidate diversity
   for the IN_TRUE_CLOSE_FALSE stratum.** The bottleneck is upstream of
   MPPI's design envelope.

2. **The right fix is a stuck-detector + scripted reposition primitive.**
   Trigger condition: `mean(action_dy) over last 30 steps > 0.3` AND
   `|EE_y displacement over last 30 steps| < 1cm`. Action when triggered:
   open gripper, retract EE 5cm in -y, lift 3cm in +z, then re-approach
   the drawer face at the right height (read from drawer site_z), then
   resume pushing.

3. **This redefines task #59 (failure detector).** The signature isn't
   "low task progress" or "low velocity" — it's "high command magnitude
   sustained, low realized motion." Cleanly observable, cheap.

4. **This redefines the recovery question.** No longer "given a mid-failure
   state, can a planner recover?" (which required compromised replay).
   Now: "given a sustained-stuck-detection event mid-rollout, does the
   scripted reposition primitive change the outcome?" This is testable
   inline, no state restore needed — the primitive activates during the
   live rollout when the detector fires.

5. **Phase B (LoRA distillation) becomes less compelling for this failure
   mode.** Distilling Pi0.5 on "good" trajectories won't fix execution
   failure that depends on geometry collision avoidance which Pi0.5 isn't
   computing. The right fix is structurally below the policy level.

## Files

- Analysis script (one-shot): `/tmp/post_place_pattern.py` (not committed)
- v2 BDDL stratification: `data/contact_mpc/failure_strata_bddl_v2.json`
- Inspected traces: 14 IN_TRUE_CLOSE_FALSE traces in `data/contact_mpc/recovery_source_traces`
- Sample trace inspected first: `PHYS_FAIL_libero_10_task3_seed7_pert5.0cm_mppi_ts1779131018854.npz`
- v2 re-stratifier: `scripts/restratify_failures_bddl.py` (commit 85829d0)
