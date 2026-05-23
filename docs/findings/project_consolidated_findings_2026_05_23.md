---
name: Consolidated findings, 2026-04-25 → 2026-05-23
description: Single-document synthesis of what the openpi project has learned across four weeks of experiments. Diagnostic apparatus, validated and falsified hypotheses, strategic position. Reading this should make every prior memory file optional, not required.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
# What we have learned

## TL;DR

1. **Inference-time deliberation hits a structural ceiling.** MPPI on Pi0.5 gives 0pp average lift on the task we measured at the perturbation we care about (task 3, 5cm, N=40 per mode, paired seeds). The earlier seed-specific +20pp didn't survive controlled measurement. Q(h,a) ≈ V(h) ±1pp on LIBERO-90 confirmed the same ceiling for value-function search.

2. **Failure modes decompose into structurally different sub-strata.** On our reference task we now have two diagnosed modes:
   - **IN_TRUE_CLOSE_FALSE (14 traces, 54%):** Pi0.5 commands the right thing (saturated +y) for 300+ steps; EE doesn't move (OSC kinematic null-space drift). *Execution failure under correct intent.*
   - **IN_FALSE / NO_APPROACH (10 traces, 38%):** Pi0.5 never gets EE within 8cm of the bowl; gripper never commanded closed in vicinity of bowl. *Approach-not-initiated.*

   These need different interventions. No single reward term targets both.

3. **MPPI's design envelope cannot fix the IN_TRUE class.** Candidates are Gaussian σ=0.1 noise around a correct prior; the noise can't construct multi-stage maneuvers; the cost function has no drawer-qpos term anyway. Two-out-of-three structural reasons it fails to help.

4. **The cheap "embodiment-feasibility field" path is also dead.** A learned f(q,a) → realized_motion model passes R²=0.51 on its own training distribution but R²=0.10 on the deployment trace distribution. Reason: training data excluded scene objects; deployment failures depend on scene contact. The "cheap and unlimited" promise of the field was cheap precisely because it abstracted away the variable that drives the failure mode.

5. **A diagnostic apparatus that gives per-failure-mode visibility is the unique artifact we built.** BDDL atomic-predicate stratification via state-jump (no action replay), with the cmd-vs-realized signature. The signature generalizes: we applied it to EE trajectory for IN_TRUE, then to bowl trajectory for IN_FALSE, and both decomposed into clean sub-modes. NVIDIA's Cosmos-RL doesn't have this; it optimizes aggregate success. This is the defensible contribution.

6. **The strategic position that survives, sharpened:** **dense supervisory signal design for VLA training, with multiple per-stratum-targeted reward components, demonstrated via stratified ablation.** Not one universal feasibility signal — a *decomposed* signal with per-component contribution attributable to per-stratum lift. Reward / loss / curriculum design, validated per-stratum.

---

## The diagnostic apparatus

Three methodology pieces that we'll use again and that aren't free elsewhere:

### 1. BDDL atomic re-stratification via direct state-jump

The original stratifier used Euclidean proxies (`obj_to_goal_cm < 10`) for what should be a BDDL predicate evaluation. We replaced it with direct evaluation of each goal atomic against the recorded final sim state, reached via `set_state_from_flattened([t, qpos[-1], qvel[-1]])` — **no action replay**.

This matters because action replay drifts: across 26 task-3 PHYS_FAIL traces, replay drift averaged 9.6 cm, median 4.7 cm, max 39.5 cm. State-jump has zero drift by construction.

Output: per-trace label by which atomic was False (`IN_TRUE_CLOSE_FALSE`, `IN_FALSE_CLOSE_TRUE`, etc.) plus per-region AABB-slack diagnostics. The labels match the corpus's on-the-fly `success` flag in every case after the fix; the v1 (replay-based) version disagreed on one trace and was misleading on five others.

Scripts: `scripts/restratify_failures_bddl.py`, `scripts/test_field_on_failure_traces.py`.

### 2. The execution-failure signature

For any trace with a high-confidence stratum label, we can ask: **over a sliding window of K steps, what's the ratio of (mean cmd magnitude) to (realized EE motion magnitude)?** Across 14 IN_TRUE_CLOSE_FALSE traces:

| Stat | Value |
|---|---|
| mean cmd_dy (last 100 steps) | +0.62 (saturated near +0.7) |
| mean EE-y displacement | +2.5 cm |
| joint motion range (mean) | 0.43 rad |
| EE Cartesian displacement (mean) | 2.0 cm |
| joint-motion / EE-motion ratio | ~20× |
| traces commanding but not realizing | 11/14 |
| traces where Pi0.5 didn't try | 0/14 |

That 20:1 joint-to-EE motion ratio is the **null-space drift signature**: controller is actuating, joints reconfigure, EE doesn't progress. Diagnostic in <100 lines of analysis, generalizes to any future Franka+OSC failure.

### 3. State-jump validation as a general principle

Whenever a diagnostic needs to "restore" to some moment in a trace, we now know: **don't replay actions, jump state**. The OSC controller's internal state (integrator, filters, last-target pose) isn't captured in `sim.get_state().flatten()`, so action-replay diverges on contact-rich trajectories. State-jump bypasses the controller's history entirely.

This invalidated several earlier diagnostics (real-P3 v2's 0/5 recovery, dead-state check at restore_frac=0.5, the controlled diagnostic from sprint 1 task #53) — those should be treated as inconclusive, not as findings.

---

## Empirical findings, validated

### MPPI doesn't lift on the failure mode it was supposed to amplify

Paired-seed measurement: baseline Pi0.5 = 27/40, MPPI = 27/40 (task 3, 5cm, seeds {7, 21, 42, 99}). Per-stratum breakdown: IN_TRUE_CLOSE_FALSE failure rate is **identical between baseline and MPPI** (7/40 vs 7/40). MPPI doesn't touch the drawer-close failure mode at all.

The earlier "+20pp" memory was seed-specific. Per-seed numbers: seed 42 mppi −3 failures (helps); seed 99 mppi +2 failures (hurts). Average zero. Variance high.

### Pi0.5 generates the right action; the controller can't execute it

100% of IN_TRUE_CLOSE_FALSE failing traces show Pi0.5 commanding sustained +y motion at saturation (>+0.5) for 300+ steps. 0% show Pi0.5 giving up or commanding the wrong direction. The policy has the right idea; the embodiment doesn't realize it.

### Drawer-close in successful trajectories works by bowl-mediation, not direct pushing

Mean EE position at the moment of drawer closing in 53 PHYS_OK traces: (+0.031, +0.154, +0.939). That's **inside the open drawer cavity, behind the bowl** — not at the drawer face. Successful trajectories close the drawer by pushing the bowl, which pushes the drawer's back wall. Failing trajectories retreat to EE_y ≈ 0.04 (10cm behind the bowl), losing the contact path.

This is task-3-specific but informative: the right action sequence isn't obvious from the task description.

### The IN_FALSE stratum decomposes into NO_APPROACH (dominant) + small tail

Applied the cmd-vs-realized methodology — mirrored from EE trajectory analysis to bowl trajectory analysis — across the 12 IN_FALSE task-3 traces. Sub-mode distribution:

| Sub-mode | Count | % | Signature |
|---|---|---|---|
| NO_APPROACH | 10 | 83% | `ee_to_bowl_min` 8–27 cm; `grip_close_near_bowl_steps = 0` for all 10 |
| OTHER | 1 | 8% | Got close, partial grasp, bowl drifted back |
| PLACEMENT_MISS | 1 | 8% | Bowl reached drawer area, missed AABB |

Pi0.5 never even attempts a grasp in 83% of IN_FALSE traces. Gripper command stays at −1 (open) throughout the approach phase. Bowl stays near its initial position.

**This is a structurally different failure mode than IN_TRUE.** IN_TRUE = Pi0.5 *tries* and embodiment *can't*. IN_FALSE / NO_APPROACH = Pi0.5 *doesn't try*. No single reward term targets both. Memory file: `project_in_false_diagnostic_2026_05_23.md`.

Sub-pattern worth flagging: within NO_APPROACH, 5/10 traces have `ee_to_bowl_min` 8–12cm (close, but no grasp attempt — possible perception near-miss) and 5/10 have `ee_to_bowl_min > 14cm` (way off). Two different sub-sub-modes may be hiding here.

### Joint-teleport + push closes the drawer in 2/14 traces

A diagnostic primitive that teleports robot qpos to the canonical push pose and then commands +y closes the drawer in 14% of cases — the "near-stuck" subgroup where EE was already close to the cavity. Failures break into 9/14 stuck (OSC stale state after teleport) and 3/14 bowl-ejected (teleport violently displaced the bowl).

This confirmed the **mechanism** (the action sequence works) but rejected the **implementation** (instantaneous teleport is too violent). We did not build v2 because the value of confirming the mechanism was already extracted.

---

## Hypotheses falsified

These are the negative results that shaped the path. Each cost time and produced information.

### Replay-drift was claimed to be ruled out; it isn't

Task #51 ("Rule out replay-drift confound") was marked done after adding `init_sim_state` save. The save is necessary but not sufficient: OSC controller state isn't captured by sim state. Replay drift on contact-rich trajectories is severe — up to 40cm. Every diagnostic relying on action-replay restore is compromised on the cases that matter most.

### The reachability field doesn't generalize to deployment

Trained a small MLP `f(q ∈ ℝ⁷, a ∈ ℝ⁷) → realized_EE_delta` on 10k simulator-derived samples. Training-distribution R² = 0.51 (passed gate). Deployment-distribution (the 14 failure traces + 14 matched success traces): mean R² = 0.10, y-axis R² = −0.37 (worse than predicting mean), AUROC = 0.47, inflection-zone correct calls 0/14.

The training distribution had objects pinned out of workspace to isolate the controller signal. The failure mode depends on scene contact. Cheap data was cheap because it abstracted away the relevant variable.

**Generalization to other "feasibility" learning attempts: the cheap version is structurally insufficient.** Scene-aware versions (B1: task-specific pinned objects; B2: object-state input vector; B3: workspace SDF input) all exist but face the harder question of how scene representation enters the policy. Open-ended without further work.

### Pi0.5 sampling can't be made deterministic across runs

Sprint 1 finding: cross-box JAX-RNG nondeterminism prevents reproducing the same Pi0.5 rollout across different machines or even runs. Implication: paired-seed comparisons across boxes are confounded, and any "diagnostic that depends on identical Pi0.5 output between runs" is broken.

### Hand-scripted recovery primitives have low expressiveness

The recovery-regime experiment (task #54 / day-1 diagnostic) tried hand-coded approach + grasp + place primitives. 0/5 success. Hand-scripted primitives don't have the vocabulary to handle the perturbation distribution; this isn't where the bottleneck is.

### WM-only gradient refinement on Pi0.5 hurts

Phase 1 of REASON: train a learned world model, then refine Pi0.5 action chunks via gradient descent against the WM's value head. Result: 52% → 30% task success. The WM's predicted-value-improvement is anti-correlated with actual-success on contact-rich tasks. Physics grounding is mandatory; learned-only-WM is misaligned.

---

## What survives

These are the claims that haven't been falsified and should be carried forward:

1. **Per-failure-mode diagnostics are decisive.** Aggregate success rate hides everything. Once you stratify, you can ask targeted questions and get targeted answers. This is the methodology contribution.

2. **The drawer-close failure is execution-bottlenecked, not perception- or planning-bottlenecked.** Adding scene-aware perception, better value functions, or longer planning horizons won't help this specific failure. The fix is at the controller / training-signal level.

3. **Dense, simulator-privileged supervisory signals are the unexplored lever.** Training-time access to object poses, joint torques, contact forces, and (commanded - realized) gaps is free in sim. The "reward shaping" framing is the obvious place to put this; the deeper claim is that **the structure of the supervisory signal determines what fills the model's capacity, and we have a lot of untapped signal in the simulator we're already using.**

4. **The Cosmos-RL ecosystem commoditizes the RL infrastructure but not the science.** NVIDIA's stack supports Pi0.5 + LIBERO + GRPO with 6D parallelism; their docs don't show ablations on which reward components matter. That science is open. Our differentiator is the diagnostic apparatus that lets us answer it per-failure-mode.

---

## Where this leaves the project

### The strategic claim

**We're not building a better RL framework or a better VLA architecture. We're building better supervision for VLAs, demonstrated with diagnostic precision about which supervisory signals address which failure modes.**

The artifact is: (a) the diagnostic methodology (BDDL state-jump stratification + execution-signature analysis) applied to a wide enough task distribution to identify failure-mode taxonomy, plus (b) a reward / loss / curriculum design experiment showing per-stratum lift attributable to specific dense signals.

### What this position survives

- NVIDIA shipping Cosmos-RL: we're not competing on infrastructure.
- The inference-time-wrapper ceiling: we're training-time.
- Long-context attention efficiency papers (Lighthouse etc): orthogonal to our scope.
- "Scale solves it" priors: we're proposing what to do *with* scale, not how to replace it.

### What it doesn't survive

- A finding that dense reward shaping doesn't improve per-stratum failure rates beyond what sparse reward does. That would force a deeper reframe.
- A finding that our diagnostic apparatus doesn't generalize beyond task 3 — i.e., other tasks don't have crisp per-stratum failure signatures we can target. We have no evidence yet either way on this.

### The next experiment, designed for the per-stratum story

Three-arm ablation, each arm attributable to per-stratum lift:

1. Use Cosmos-RL (or a fork) to GRPO-fine-tune Pi0.5 on libero_10 with `train_expert_only=true`.
2. Three reward variants:
   - **A (baseline)**: sparse task success only.
   - **B**: sparse + λ₁ · Σ_t max(0, ||a_cmd_t|| − ||realized_ee_t||) — *targets IN_TRUE_CLOSE_FALSE (execution-stuck)*.
   - **C**: sparse + λ₁ · tracking_error + λ₂ · approach_progress, where approach_progress = (initial_ee_to_bowl − ee_to_bowl_t) — *targets IN_TRUE AND IN_FALSE / NO_APPROACH*.
3. Evaluate per BDDL stratum using the state-jump re-stratifier.
4. **Headline result**: an A→B→C ablation table showing per-stratum failure-rate reduction attributable to each added reward component. Each row is a stratum; each column is a reward variant; the cells decompose where the lift comes from.

Pre-requisite: extend Cosmos-RL's text-shaped reward interface to accept rollout trajectory information (observations, actions, full sim state at every step). Their code is Apache-2.0; this is a contribution-shaped change, not a fork.

This experiment serves all three audiences (YC, paper, collaborator engagement) but the *framing* of the headline differs. Decide framing before writing.

### Open methodological gaps that should be closed before the next experiment

- **Transfer-test target task is unidentified.** Without a second reach-limited task to test transfer on, the per-stratum result can be called "task 3 curiosity." Candidates: task 9 (microwave + close), BEHAVIOR-1K reach-limited tasks (different sim). Stratifying failures on a second task is the cheap pre-experiment (requires fresh GPU rollouts).
- **NO_APPROACH sub-sub-modes.** 5/10 of NO_APPROACH traces are at `ee_to_bowl_min` 8–12cm (close, but no grasp attempt) and 5/10 at >14cm (way off). May be different sub-sub-modes requiring different signals. ~30 min of follow-up analysis on existing local data.
- **The 1 OTHER + 1 PLACEMENT_MISS** in the IN_FALSE diagnostic deserve a closer look before treating them as noise — small N but the PLACEMENT_MISS one specifically tells us whether the AABB metric and the policy's place-target are mis-aligned.

---

## Files and locations

- Source corpus: 80 task-3 traces (HF dataset `arif101/openpi-contact-mpc-2026-05-19`, folder `recovery_source_traces/`)
- Failed reachability field model + dataset: same HF dataset, folder `reachability_field/`
- BDDL re-stratification result (v2, state-jump): same HF dataset, file `failure_strata_bddl_v2.json`
- Diagnostic scripts: `scripts/restratify_failures_bddl.py`, `scripts/test_field_on_failure_traces.py`, `scripts/find_drawer_push_pose.py`, `scripts/test_drawer_recovery_primitive.py`
- Older memory files (now superseded by this document for the high-level read; still useful for specific details): the seven `project_*_2026_05_*.md` files in this memory directory.

## What this document is for

This is the document to read first the next time the project resumes after a long gap. The individual memory files have details; this is the synthesis. If something in this document conflicts with a specific memory file, this is the more recent and more honest version — the older file is preserved for historical detail but the headline claim has been revised.
