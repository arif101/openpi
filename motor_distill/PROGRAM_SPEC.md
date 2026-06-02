# Build Spec — Learned Re-targetable Action Head on Pi0.5 (committed program)

**Claim:** replacing/augmenting Pi0.5's imitation action generation with a *learned,
goal-grounded, re-targetable* head improves novel-object / perturbation OOD where the
stock action expert collapses. **Benchmark:** LIBERO-PRO (primary) + VLABench/AGNOSTOS.

## Key architectural discovery (reshapes the plan) — THIS IS OUR OWN WAYPOINT BRANCH
Pi0.5 ALREADY has a goal channel: `target_state` (next-keyframe waypoint, [B,8]) →
`target_state_proj` (pi0.py:97) → appended as a token in the action-expert suffix
(`embed_suffix`, ~pi0.py:156-162). The action expert is a 300M Gemma sharing attention
with the frozen 2B PaliGemma VLM (dual-stream MoE) — NOT cleanly separable, but the
**suffix is extensible**, and a goal token is exactly how Pi0.5 already conditions.

CRUCIAL: this `waypoint-conditioning` branch ALREADY built the infrastructure:
- `target_state` plumbed end-to-end (libero_policy.py:86, transforms.py:337, model.py:120/142/222).
- `LeRobotLiberoWaypointDataConfig` (config.py:360) maps `target_state` from a waypoint dataset,
  "e.g. **libero90_waypoints**" — which is OUR HF dataset (arif101/libero90_waypoints, has the
  `target_state` next-keyframe column).
- A training config (config.py:812) trains `target_state_proj` + LoRA on that waypoint dataset.
- `target_state_proj` is a NEW layer NOT in the base checkpoint (weight_loaders.py:53, random-init);
  base pi05_libero has the PLUMBING but the goal channel is UNTRAINED until the waypoint fine-tune.
So we EXPLOIT/LEARN/STRUCTURE an existing, half-built goal channel — the program is the continuation
of this branch's waypoint work, now with a CLAIM (OOD re-targetability) + VALIDATION (the DMP proof).

Concrete dims (pi05_libero): action_dim=32 (7 joints+1 grip+pad), action_horizon=10,
state_dim=8, VLM width 2048, action-expert width 1024. Action gen = flow-matching Euler
denoise (sample_actions, pi0.py:265-335); loss = velocity MSE (compute_loss, pi0.py:194-224);
VLM feature read point = extract_vlm_features (pi0.py:225, [B,2048]).

## Architecture (graft points)
- KEEP frozen: PaliGemma VLM (perception+language).
- GOAL HEAD (new, learned): extract_vlm_features [B,2048] -> MLP -> predicted target_state /
  affordance goal. The goal is LEARNED from perception (generalizes to novel objects) — this
  is the answer to "don't hand-design primitives/goals".
- RE-TARGETABILITY: decided by Stage 0:
  (a) if Pi0.5's existing target_state conditioning already re-targets -> trust it, done.
  (b) if it's also a demo-mimic -> add attractor structure (NDP-style: constrain action_out so
      it converges to the goal) OR retrain the action expert with GOAL DIVERSITY (hindsight-
      relabel target_state with many goals so it learns target->action as a MAP not a correlation).
      (The DMP result [[project_dmp_retarget_beats_pi05]] is the existence proof of the property.)
- VERIFICATION (fold in, we have it): eps commanded-vs-realized self-residual as a 2nd channel.

## Staged plan — cheapest-decisive-first (each stage can kill the program)

### Stage 0 (linchpin): does a WAYPOINT-conditioned Pi0.5 re-target?
RESOLVED (0a): target_state_proj is UNTRAINED in base pi05_libero — it's a new layer trained only
by the waypoint fine-tune (config.py:812) on a waypoint dataset (libero90_waypoints). So Stage 0
needs a WAYPOINT-FINE-TUNED checkpoint:
0a. Check if a waypoint-conditioned checkpoint already exists from this branch's prior work; if not,
    run the existing waypoint fine-tune (config.py:812 config + arif101/libero90_waypoints) — the
    config + dataset already exist. (GPU; LoRA + target_state_proj only, cheap.)
0b. With the waypoint checkpoint: at INFERENCE feed the CORRECT target_state for a PERTURBED object
    (keyframe goal from the sim object pose / a matching reference success). Measure grasp/task
    success on perturbed scenes WITH vs WITHOUT the correct waypoint.
    -> Decisive: does an EXPLICIT correct goal make Pi0.5 re-target (degrade gracefully) where the
       stock policy collapses? If yes -> the goal channel works, Stage 1 (learn the goal) is the
       program. If the waypoint conditioning is ALSO a demo-mimic -> Stage 2 (attractor structure /
       goal-diversity). Reuses our perturbation corpus (HF: arif101/libero10-pi05-perturbation-traces).

### Stage 1 (Risk 1 — learned goal transfer): goal head predicts target_state from VLM
- Train goal head: VLM features -> target_state, on demos (hindsight target = future keyframe).
  Freeze VLM. Tiny head, fast.
- Eval: does predicted target_state transfer to NOVEL objects (LIBERO-PRO object-swap axis)?
  Metric: goal-pred error on novel objects + downstream grasp when fed the predicted goal.
- KILL: if the goal head can't predict a usable goal for a swapped object, the whole re-targeting
  premise has nothing correct to aim at.

### Stage 2 (only if Stage 0 = sub-option b): add re-targetability structure
- NDP/attractor-structured action output, or goal-diversity retrain of the action expert.
- Validate it matches Pi0.5 grasp precision in-dist (Risk 2 / LAPA warning) before trusting OOD.

### Stage 3: integrate end-to-end -> eval on LIBERO-PRO PER-AXIS vs Pi0.5 baseline.
### Stage 4: fold in eps verification channel (Phase-2f dispatch math).

## Benchmark integration (parallel track)
- LIBERO-PRO (arXiv 2510.03827): find if code/data are open; integrate its 4 axes (object swap,
  init state, instruction, environment) into our harness. Our 5/10cm position perturbation is ONE
  sub-axis; LIBERO-PRO adds object/instruction/environment. Reproduce Pi0.5 baseline PER-AXIS
  (honest: combined "0%" is worst-case; pi0.5 more robust than peers).

## Risks / kill-criteria (restate)
1. Goal head must be LEARNED + transfer to novel objects (else DMP dead end). — Stage 1 gate.
2. Re-targetable head must preserve grasp precision (LAPA warning). — Stage 2 gate; DMP+residual hedge.
3. Claims PER-AXIS, not the combined-worst-case headline.
4. Fast-moving field — re-verify novelty before months of work.
FALLBACK: if 1-2 fatal, pivot to the metacognition/verification wedge (eps already validated).

## First concrete task (no GPU): Stage 0a
Verify whether `target_state` is a live, trained channel in pi05_libero — read
src/openpi/policies/libero_policy.py + data transforms + check `target_state_proj` params in the
checkpoint. This single fact determines whether Stage 0 (no-training re-targeting test) is runnable
and is the linchpin of the whole program.
