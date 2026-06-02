# Held-out compositional generalization eval — design (the SCAN-for-manipulation gap)

Goal: an eval where the surprise-gated structural-retrieval architecture can show a real
generalization gain over (a) parametric-only Pi0.5 and (b) naive surface retrieval, on
compositions never seen as a single training/stored episode.

## The honest constraint: LIBERO varies ONE axis per suite

| suite | varies | held constant | "composition" available |
|---|---|---|---|
| libero_object (10) | target object | skill (pick→basket), scene set | object only (1 axis) |
| libero_spatial (10) | spatial relation | object (black bowl), scene | relation only (1 axis) |
| libero_goal (10) | goal/verb | scene | goal only (1 axis) |
| libero_10 (long) | multi-step compound | — | whole task |

There is **no native 2-axis (object × instruction) grid** to hold out within a suite — each
suite is a single-axis sweep. A clean SCAN-style "train AB, AC, BC → test BD" requires data we
don't have. So the compositional test must be built from what single-axis sweeps + cross-suite
structure DO give us.

## Three viable designs (ranked)

### Design 1 (RECOMMENDED, buildable now): Leave-one-referent-out, retrieval-as-knowledge
Uses libero_object (cleanest: shared skill + shared scene, only target object varies).
- The SKILL "pick up X, place in basket" is shared across all 10 objects.
- Held-out object X: remove its demos from the episodic store; X is still PRESENT in scenes.
- Query "pick up the {X}": the *combination* (this skill applied to X) is absent from the store,
  but the SKILL exists (9 other objects) and the OBJECT X exists (as a distractor in stored episodes
  / in pretraining).
- Claim under test: retrieve structurally-similar episodes (same skill, neighbour objects) →
  recombine/condition → succeed on held-out X better than parametric-only and naive retrieval.
- Why clean: skill held constant, single held-out variable, structural retrieval has a clear target
  (same-skill episodes), and the surface baseline (retrieve by scene) retrieves the same-scene
  distractor-grasp which is WRONG → naive-vs-structural separates.

### Design 2 (external, published): LIBERO-CF CF-OOD suite (entirely unseen objects, 15 tasks)
- Off-the-shelf held-out-object benchmark with SOTA numbers (VLAs ~0-13%).
- Pro: published, comparable, hardest. Con: not ours to control the train/store split; tests
  unseen-object, not object×context recombination.

### Design 3 (cross-suite): object × context transfer
- Objects appearing in multiple suites/contexts (e.g. black bowl in spatial + goal). Train context A,
  test same object in held-out context B.
- Pro: a real 2-axis composition. Con: instructions differ across suites (confounds), fewer shared objects.

## THE deeper open question (must decide before Exp 3): how does retrieval CONDITION Pi0.5?

"Recombine retrieved episodes" is the hard, unsolved part for a flow-matching VLA. Options:
- (a) **Action-space guidance**: bias the flow-matching denoiser toward retrieved action chunks
  (like classifier guidance). Cheapest; no retraining. Risk: retrieved action is for a different
  object location → needs spatial adaptation.
- (b) **Memory-token injection**: append retrieved (obs/action) features as extra prefix tokens; needs
  a small LoRA to teach the policy to use them (a training step).
- (c) **In-context demonstrations**: prepend retrieved (obs,action) as context. Pi0.5 not trained for
  this; likely weak without training.
- (d) **Sub-goal / waypoint retrieval**: retrieve a high-level plan, not raw actions; execute with the
  base policy. Sidesteps trajectory-stitching.

Recommendation: start with (a) action-space guidance (no retraining, fastest signal); if the spatial-
adaptation problem dominates, move to (b) memory-token injection with a small LoRA.

## Minimal buildable now (offline, this module)
1. Define the leave-one-referent-out split: for each held-out object, the store = all other-object
   episodes + the held-out object's episodes REMOVED; query set = held-out object episodes.
2. Metadata: per episode (suite, target referent, scene id, episode frame range) — for store/query
   construction and structural-vs-surface retrieval labels.
3. (Exp 3, GPU) wire retrieval-conditioning mechanism (a) and run the 4-way ablation on this split.

## Open decisions for the user
- Recombination mechanism (a/b/c/d) — start with (a)?
- Which split: Design 1 (we control, clean) as primary + Design 2 (CF-OOD, published) as external check?
