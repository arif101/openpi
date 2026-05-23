---
name: Recovery regime diagnostic — day 1 (2026-05-18)
description: Hand-scripted recovery primitives failed on 5/5 grip-loss failures across both task 0 (10cm) and task 3 (5cm). Verdict provisionally REPERTOIRE-LIMITED but diagnostic itself has a state-replay confound that needs to be ruled out before treating verdict as final.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
**The gate, as committed to before running:**
- ≥3/5 recovered by primitives → perception-limited → privileged-state planner viable → build data engine
- ≤1/5 → repertoire-limited → privileged state buys nothing → proposal distribution IS the problem → rethink

**Run 1: task 0 (multi-object, basket placement), 7.5-10cm perturbations**
- 5 grip-loss failures sampled (closest_approach <8cm, object_motion <5cm)
- 0/5 recovered by any of 5 primitives (open_then_close, retry_grasp@obj, retry_grasp@goal, lift_then_move, push_and_regrasp)
- Verdict reading: REPERTOIRE-LIMITED

**Run 2: task 3 (single-object, drawer placement), 5cm only (the publishable regime where MPPI got +30pp at seed=42)**
- Only 1 trace matched the grip-loss filter at this strict perturbation level
- 0/1 recovered
- Verdict reading: same, but small sample

**Caveat that must be resolved before the verdict is final:**
The diagnostic replays each trace by re-executing recorded actions from a default env.reset() up to T-60, then runs the primitive. We do NOT save init_state in the physics-trace NPZs, so the replayed state at T-60 is reached from a different starting state than the original failure trace had. Even with deterministic dynamics, divergence accumulates over 200+ replayed steps. The "failure state" we're recovering from may not actually be the real failure state. Three possibilities:

1. Replay state ≈ real failure state. Primitives genuinely cannot recover. REPERTOIRE-LIMITED stands.
2. Replay state is drifted but still in a "failure-like" region. Primitives can't recover from any nearby failure state. Still REPERTOIRE-LIMITED, but with weaker evidence.
3. Replay state is a totally different failed state from the actual trace's failure. The diagnostic itself is invalid.

**Cheapest way to distinguish:** run the primitives from a controlled mid-trajectory point of a *known-successful* episode (freeze at random t, run primitive, check if it can still complete the task). If primitives succeed in that controlled setting, they're functional and we have hypothesis (3) — diagnostic is broken. If primitives fail in that controlled setting too, we have stronger evidence for REPERTOIRE-LIMITED.

**Why this matters:**
If REPERTOIRE-LIMITED is confirmed: every architecture we've been discussing (PhysVLA latent, A+B+C, distilled corrector) is downstream of "a planner can find recoveries." If the planner can't because the recovery isn't expressible in the primitive vocabulary, the project's binding constraint is not architecture — it's the recovery action space itself.

That has two implications:
1. Scripted primitives are insufficient as a planner. We'd need either (a) a much larger primitive vocabulary, (b) a learned proposal distribution that can find behaviors outside hand-designed primitives, or (c) admit that these failures aren't recoverable from observable state.
2. The wedge of the project shifts. Rather than "physics-grounded refinement gives capability," the contribution becomes either (a) "what we can recover with structured primitives — a smaller capability claim" or (b) "we identify and characterize a class of contact failures that are NOT recoverable from observation, with empirical evidence — a publishable negative result."

**Action items for tomorrow:**
1. Add init_state save to the physics-trace logger so future traces can replay deterministically.
2. Run the controlled-mid-trajectory diagnostic to rule out the replay-drift confound.
3. If diagnostic v2 confirms REPERTOIRE-LIMITED, the next decision is whether to expand the primitive library (might be sufficient given LIBERO's manipulation scope) or to step further back and ask whether LIBERO at high perturbation is even the right testbed.

**How to apply:**
- DO NOT build PhysVLA dynamics distillation on the current premise. The data engine question is unresolved.
- DO NOT generate more PhysVLA training data via the current Pi0.5 rollout collection. Same reason.
- The Phase A +6.7pp on task 3 result IS still the best concrete signal we have. Defer pushing past it until the recovery regime is clarified.

---

**Update 2026-05-18 late afternoon: REPERTOIRE-LIMITED verdict retracted; diagnostic is INCONCLUSIVE.**

Ran two follow-up diagnostics:

1. **Controlled mode** (primitives from mid-trajectory of a successful trace, task 3 5cm filter — 4 of 5 traces ended up being 0cm, 1 was 2.5cm): 0/5 primitives completed the task. But the test asks "can primitives complete LIBERO from a partial-success state?" — different from "can primitives recover a specific failure." Open-loop primitives failing task completion is expected and uninformative.

2. **Replay-sanity mode** (just re-execute the recorded action sequence end-to-end and check done=True): **4/5 faithful for 0cm traces, 0/1 faithful for 2.5cm perturbed trace.** This proves what was suspected: perturbations are not reproducible across env.reset() because we don't save init_state. The perturbation is sampled deterministically from `perturb_rng = np.random.default_rng(args.seed + 1000)`, but the LIBERO env's initial state when env.reset() is called differs from run to run, so the perturbed state we tried to reach is genuinely irrecoverable.

**Implication:** Every "perturbed-trace primitive failure" we have so far tested states we never actually reached. The original 0/5 on task 0 10cm and task 3 5cm was the wrong experiment, not a finding.

**True verdict: INCONCLUSIVE.** We don't know whether primitives could recover real failures. We also don't know whether the failures themselves are repertoire-limited because we can't reproducibly visit them.

**Required fix before any further diagnostic:**
Extend the physics-trace logger to save `sim.get_state().flatten()` at episode start. With this, perturbed traces become replayable deterministically and the diagnostic question becomes well-defined again. ~30 minutes of code.

**Better recovery-specific controlled test, once init_state save is added:**
1. Take a successful trace.
2. Replay to mid-trajectory using saved init_state — verify we hit the actual state (deterministic).
3. INJECT a grip-loss: forcibly open the gripper for 5 steps so the held object falls.
4. Try each recovery primitive from that artificial-but-known failure state.

This is the test that actually maps "primitives can recover grip-loss" → directly answers the perception-vs-repertoire question without the replay-faithfulness confound.

**What this changes about the project plan:**
- The data-engine viability gate remains unanswered. We don't know yet whether privileged-state planning can find recoveries.
- We do NOT have evidence that primitives are inadequate. We thought we did this morning; we don't.
- Next action sequence: (1) add init_state save, (2) re-collect a small batch of perturbed-trace data, (3) redo failure-diagnostic with reproducible perturbed states, (4) THEN draw a verdict.

Estimated cost: ~1h of code + ~30 min of small data re-collection + ~1h of diagnostic re-run = half a day. That's the right investment to make before any architecture decision.

---

**Final day-1 verdict (2026-05-18 evening): scripted primitives are inadequate, but for an architecture-relevant reason, not a "failures are unrecoverable" reason.**

After fixing init_sim_state save and confirming replay-with-restore drift is <1cm for baseline-mode traces:

Ran injection-mode diagnostic (restore mid-trajectory → force grip open 8 steps → try primitives) on n=3 task-3 5cm baseline traces. All 5 primitives failed on all 3 traces (0/15 attempts succeeded), including:
- retry_grasp@static_xyz (pre-drop position): closes on empty air after object falls
- retry_grasp@goal_xyz: closes inside drawer where there's no object
- lift_then_move: moves to goal, drops nothing into drawer
- push_and_regrasp: nudges, then retries — same as static
- **PRIVILEGED_retry_grasp** (queries live sim.data.body_xpos[obj_id], gets ground-truth current position post-drop, approaches, descends, closes, lifts): still 0/3

The PRIVILEGED variant has the perfect ground-truth state the user's strategic reframe argued was the data-engine premise. It still fails. Why:

The injection point (T-80) on task 3 is often inside the drawer-placement subtask. After grip loss, the recovery required isn't "regrasp bowl" — it might be "regrasp, place, close drawer" or "close drawer" (if bowl is already in). None of my primitives do "close drawer."

This isn't "failures are unrecoverable from observable state" — it's "the primitive vocabulary I hand-crafted doesn't contain the actions LIBERO recovery requires." With a closed-drawer primitive added, some of these would likely recover. With another 10 primitives per task, more would. The scaling property of this approach is hand-engineering effort per task.

**This empirically confirms the strategic point from the conversation:** scripted-primitive recovery has a hard cap at the scripted vocabulary. You can only ever distill what you wrote.

**What this means for the project:**
- Path 1 (scripted primitives as planner action space): viable but caps capability at hand-engineering. Not a research contribution.
- Path 2 (learned proposal from successful trajectories): the only path that can scale beyond what we explicitly script. Cold-start problem is real but tractable — initial proposals come from the successful-trajectory corpus we already have (~250 successful baseline rollouts at 5cm), and the dynamics model would be trained on the diversified state space the planner explores.
- Path 3 (write up Phase A and stop): the Phase A +6.7pp on task 3 across 3 seeds at N=10 is real, reproducible, and publishable as a robustness amplifier result with honest scope. The negative diagnostic results add credibility to the writeup rather than undermine it.

**My recommendation:** pursue path 2 + path 3 in parallel. Phase A writeup begins; planner v2 uses a learned-proposal mechanism bootstrapped from the successful corpus. This is the structurally correct path forward; primitive expansion is a dead end.

**Code state at end of day 1:**
- scripts/diagnose_recovery_regime.py — failure/controlled/replay-sanity/injection modes
- init_sim_state save in physics-trace logger
- diagnostic corpus at data/contact_mpc/physvla_traces_diag (still collecting in background)
- 5 primitives in the library, all confirmed inadequate on task 3 alone
- PhysVLA Week 1 corpus (500 episodes, no init_sim_state) — still valid for dynamics training if we resume that thread later

---

**Update 2026-05-19 — Stratification reveals the diagnostic has been targeting the wrong failure mode.**

Ran scripts/stratify_failures.py over 25 task-3 baseline failures from the recovery-source corpus. Failure-mode distribution:

  PLACEMENT_OK_SUBPRED_FAIL: 48% (12 traces) — object reached goal area, sub-predicate unsatisfied (for task 3: bowl is at/in drawer but drawer wasn't closed)
  APPROACH_FAIL:             24% (6) — EE never got within 10cm of the object
  OTHER:                     16% (4) — ambiguous
  PLACEMENT_FAIL:            12% (3) — object moved meaningfully, ended far from EE and goal (true grip-loss in-transit)
  GRIP_LOSS:                  0% (0) — EE got close, object never moved >5cm

**Implications:**

1. The "FAILED MANIPULATION 71%" verdict from the early failure-trace analyzer was using a definition (closest_approach < 8cm AND object_motion < 5cm) that conflated PLACEMENT_OK_SUBPRED_FAIL and GRIP_LOSS together. The dominant failure mode on task 3 is NOT grip loss. It's drawer-close.

2. All 5 recovery primitives I built (retry_grasp, lift_then_move, push_and_regrasp, etc.) were designed for grip-loss recovery. None of them attempt drawer-close. The injection-diagnostic verdict ("primitives can't recover") is consistent with this — we were testing primitives that don't address the dominant failure mode.

3. For task 3 specifically, recovery requires: detect drawer-not-closed state → move EE behind drawer face → push forward to close. A trivial primitive vocabulary for this would be `close_drawer(direction, force)`. We never wrote it because we were targeting the wrong failure type.

4. This is a third instance of the averaging error: dominant failure types in mixed data get conflated with rare ones, and architectures designed for the rare type don't address what's actually happening.

**Implications for the project:**

- The real-P3 v2 paired-comparison test should run on PLACEMENT_OK_SUBPRED_FAIL (the majority stratum) first, not on a grip-loss stratum that doesn't exist. Decisive question: can heavy MPPI or CEM find drawer-close recoveries?
- Backward reachability is partially derisked: it uses the recorded successful action sequence, which DOES include the drawer-close subroutine. So the recovery pairs generated by the backward-reachability data engine will naturally include drawer-close behaviors, unlike our hand-scripted primitives.
- The paper framing tightens: 'we identify the dominant failure modes per task via BDDL-sub-predicate stratification and show that previous failure-trace diagnostics conflate them.' This is the kind of methodological contribution that's robust against reviewer pushback.

---

**Update 2026-05-19 — Real-P3 v2 on PLACEMENT_OK_SUBPRED_FAIL: effectively UNRECOVERABLE.**

After fixing the four critique-identified issues (stratified, paired, privileged-cost, leakage-logged), real-P3 v2 ran on the dominant task-3 failure stratum (PLACEMENT_OK_SUBPRED_FAIL = drawer-close sub-predicate, 48% of failures).

Config: prior-centered K=48 σ=0.3 iters=2 vs prior-free CEM K=64 elite 15% iters=2. Horizon 10. Max 120 recovery steps. 3 leakage perturbations per recovery.

Result on N=5:
  prior-centered recovery rate: 1/5 = 20%
  prior-free CEM recovery rate: 0/5 = 0%
  Leakage robustness on the 1 recovered solution: 0.00 (zero perturbations preserved success)

**Interpretation:**
- prior-free 0/5 rules out repertoire-limited hypothesis for this stratum. Forgetting Pi0.5's prior doesn't help.
- prior-centered 1/5 with leakage 0.00 means the apparent recovery exploited millimeter-level privileged state — would not survive distillation to an image-conditioned model.
- Functionally: 0/5 deployable recoveries.

**This is a publishable negative for PLACEMENT_OK_SUBPRED_FAIL specifically.** Drawer-close sub-predicate failures, when restored to mid-trajectory, are not recoverable via inference-time search of any kind tested (heavy MPPI around prior, prior-free CEM). Recovery would require either:
- policy improvement at training time (RL fine-tune that learns drawer-close-from-disoriented-state)
- a fundamentally different proposal mechanism (e.g., scripted close-drawer primitive in a learned planner)
- avoiding the failure state in the first place

**What this means for the project plan:**
- Backward reachability is partially derisked for this stratum: it uses the recorded successful trajectory's drawer-close subroutine, which IS a successful drawer-close action sequence by construction. So Path 1 (backward reachability) bypasses the unrecoverability of this stratum by not searching forward from the failed state at all.
- Path 2 (learned proposal with RL fine-tune) becomes more important: it's the only way to fix capability in this stratum because no inference-time search will find recoveries.
- Phase A's +6.7pp result remains the strongest concrete signal — and is consistent with "MPPI catches some easy failures but not sub-predicate failures."

**Caveats before generalizing:**
- N=5 is small. We should rerun at N=10 once we have more APPROACH_FAIL and PLACEMENT_FAIL data.
- restore_frac=0.5 = mid-failure. Earlier restore (e.g., 30%) might give the planner more time to recover.
- horizon=10 in MPPI may be too short to see drawer-close repositioning (which takes 30-50 steps).
- Leakage perturbations are state-space (±2mm) not observation-space — the 0.00 robustness reflects open-loop fragility, not strictly privileged-state dependence.

**Action items:**
- Run APPROACH_FAIL stratum (24% of failures, in flight now, ETA 60 min)
- Run PLACEMENT_FAIL stratum (12%, smaller N but worth testing)
- After all three strata results, draw the cross-stratum verdict and update the project plan
