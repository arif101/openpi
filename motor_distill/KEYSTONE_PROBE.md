# Keystone Probe — is Pi0.5's motor manifold separable from its memorization?

**The one question this answers (go/no-go for the whole hybrid):** can we distill a
head that maps `(goal-relative target, proprio) -> action chunk` — with pixels and
language DROPPED — that still executes competent grasps when *told* the target? If
yes, Pi0.5's motor competence (HOW to move) is separable from its scene-selection
(WHAT to aim at, which memorizes and fails OOD), and we can keep the motor manifold
while replacing the selection with our structured WM + verification + correction.

A *probe* stubs out everything except the component under test. Here: the WM,
MCTS planner, correction memory and affordance proposer are all replaced by a
**scripted target generator** (geometric, reads object poses from sim). The only
variable is the distilled motor head. Logic: if even a *perfect* (scripted)
planner can't drive the distilled head, the keystone is dead and the planner is
moot; if it can, building the real WM-planner to replace the script is worth it.

## Scope (honest)
- Tests **pose/arrangement novelty** (perturbed object positions / novel counts) —
  the axis where Pi0.5 memorizes and goal-relative + equivariant conditioning should
  generalize by construction.
- Does **not** test shape novelty (new object needing a new grasp) — that needs
  affordances/grasp synthesis, a separate bet. Keep to known object types.

## Status
- [x] **Re-keying bridge** (`rekey.py`, `test_rekey.py`) — reads physics-trace NPZ →
  `(g_pos, g_quat, proprio, chunk)` pairs. `g` = EE pose at t+H in the target
  object's frame at t. 6/6 unit tests pass, incl. global-yaw invariance (the
  equivariance property Test B depends on) and the MuJoCo[wxyz]/robosuite[xyzw]
  quaternion-convention gotcha.
- [ ] **Fresh collection** (needs GPU) — see command below.
- [ ] **Distilled head** (`head.py`) — small flow-matching `(g,proprio)->chunk`,
  plain + SE(2)-equivariant variants. No VLM → no OOM.
- [ ] **Scripted target generator** (`target_gen.py`) — grasp-pose→lift→goal→release
  from current object poses; the stand-in for the future WM-planner.
- [ ] **Test A / Test B** closed-loop in sim.

## Fresh collection — ready command (run on the H100)
Baseline Pi0.5 rollouts with the object-pose channel. The harness loops
`(baseline, mppi)`; we use the **baseline**-tagged NPZs. No replay (object poses
read live each step), `init_sim_state` saved.

```bash
# in-distribution (pert 0) — for Test A (manifold preserved)
for TASK in 0 3 6 9; do
  python scripts/run_reason_v3_mppi.py \
    --task-suite libero_10 --task-idx $TASK --seed 7 \
    --num-trials 50 --perturbation-cm 0 \
    --log-physics-traces data/keystone/indist
done

# novel arrangements (pert 5cm + 10cm) — for Test B (memorization stripped)
for PCM in 5 10; do for TASK in 0 3 6 9; do
  python scripts/run_reason_v3_mppi.py \
    --task-suite libero_10 --task-idx $TASK --seed 7 \
    --num-trials 50 --perturbation-cm $PCM \
    --log-physics-traces data/keystone/pert${PCM} \
    --no-validate-perturbations   # keep ALL perturbed inits, incl. hard ones
done; done
```
Output NPZ schema (per episode): `ee_pos[T,3]`, `ee_quat[T,4]`(robosuite xyzw),
`gripper_qpos[T,2]`, `action[T,A]`, `qpos/qvel`, `object_pos[T,n,3]`,
`object_quat[T,n,4]`(mujoco wxyz), `object_names[n]`, `success`, `init_sim_state`.
Filter to `PHYS_OK_*_baseline_*.npz` for the distillation corpus.

> Efficiency option (not required): the eval-mode loop at run_reason_v3_mppi.py:1282
> hardcodes `("baseline","mppi")`. A `--modes baseline` flag would halve collection
> cost. One-line edit; do it only if GPU time is tight.

## The two tests + pass bars
- **Test A — manifold preserved? (in-distribution)** distilled head + scripted
  targets, closed-loop. PASS: success ≥ ~80% of baseline Pi0.5 on the same tasks.
- **Test B — memorization stripped? (novel arrangements)** same head + scripted
  targets computed from the *perturbed* object pose, vs raw Pi0.5 (which infers the
  target from pixels). PASS: distilled-head-with-correct-target beats raw Pi0.5 by
  ≥ ~15pp, AND equivariant > plain.

## Reading the result
- A✓ B✓ → keystone holds; build the full hybrid (WM, MCTS, correction).
- A✓ B✗ → manifold transfers but goal-relative conditioning doesn't generalize;
  fix target representation / equivariance.
- A✗ → manifold + memorization entangled; fallback = freeze Pi0.5's head and only
  re-route its conditioning rather than distilling.
