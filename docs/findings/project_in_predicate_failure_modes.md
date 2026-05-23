---
name: BDDL In() predicate failure modes — predicted before re-stratification (2026-05-18)
description: Specific physical configurations where BDDL In(bowl, drawer) AABB test will disagree with human judgment of "bowl is in the drawer". Written BEFORE the re-run so contamination findings are predicted, not discovered.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
LIBERO `In(obj, region)` semantics — full read of `libero/envs/predicates/base_predicates.py`,
`libero/envs/object_states/base_object_states.py`, `libero/envs/objects/site_object.py`:

```
In(bowl, drawer_region)
  = drawer_region_site.check_contact(bowl)   # SiteObjectState always returns True (line 176)
    AND drawer_region_site.check_contain(bowl)
  = drawer_region_site.in_box(bowl_center_position)
  # AABB test:
  total_size = |drawer_region_site.rotation_mat @ drawer_region_site.size|
  ub = site_position + total_size
  lb = site_position - total_size
  lb[2] -= 0.01   # 1cm floor slack
  return all(bowl_pos > lb) AND all(bowl_pos < ub)
```

**Critical property:** the `drawer_region` site is a child of the drawer slide body. The AABB **moves
with the drawer** as it opens/closes. It is NOT anchored to the cabinet interior.

**Close(drawer_region)** iterates the region's associated joint(s) and returns `qpos < max(default_close_ranges)` for WhiteCabinet (per `articulated_objects.py:215`). For `bottom_region`, only the `bottom_level` slider matters — not `top_level` or `middle_level`.

**Why:** The current stratifier uses Euclidean `obj_to_goal_cm < 10` as proxy for In(). That proxy
diverged on 4/5 PLACEMENT_OK_SUBPRED_FAIL traces. Re-stratifier using actual BDDL eval is the
correct move, but BDDL itself is a proxy — a point-in-AABB test with no contact, no orientation,
no stability, no body-extent. Predicting BDDL's failure modes before the re-run breaks the pattern
of discovering metric pathologies after they contaminate.

**How to apply:** Use these predictions to (a) categorize the re-stratifier output, (b) flag specific
traces for visual inspection, (c) decide whether to layer a stricter "human-like containment" check.

### Predicted false-positives of BDDL In() (BDDL says True, a human would say "no/precarious")

1. **Bowl tilted / upside-down with center inside AABB.** BDDL ignores orientation. Probably rare for task 3.

2. **Bowl held by gripper, gripper has moved it into the AABB region but hasn't released.** BDDL fires
   `In = True` while the bowl is being held mid-air inside the cavity. For end-of-rollout success
   evaluation this is fine if the gripper release happened before max_steps; for mid-trajectory
   restore points, this matters.

3. **Bowl center inside the floor-extended AABB (`lb[2] -= 0.01`) but bowl is resting on the drawer's
   front lip, not the interior floor.** Center happens to clip the 1cm slack zone. Rare; needs tall
   bowl + low drawer.

4. **Bowl with momentary inside-AABB position during ballistic fall (after gripper release).** BDDL
   evaluates instant. If sim terminates at exactly this frame, false-positive. Unlikely for the
   3-step end-of-rollout BDDL eval but possible.

5. **Bowl center inside AABB but bowl body width exceeds drawer interior width.** Center is inside,
   geometry collides with drawer walls. Sim contact would prevent rest, but BDDL doesn't check.

### Predicted false-negatives of BDDL In() (BDDL says False, a human would say "yes/clearly placed")

6. **Bowl rim sitting on drawer floor with center 1cm above ub_z.** If drawer interior site has a
   conservative z-extent, a tall-rim bowl with most of its body inside the drawer can have center
   above ceiling → BDDL False. Likely the dominant false-negative for task 3 if the akita_black_bowl
   geometry is tall.

7. **Bowl in the wrong drawer (middle or top, when goal is bottom).** BDDL False on
   `In(bowl, bottom_region)`. Semantically distinct from "placement failed" — the policy
   demonstrated drawer-targeting competence, just hit the wrong one. Treat as `WRONG_DRAWER`.

8. **Drawer half-open, bowl inside cabinet body but the bottom_region site has translated forward
   with the drawer.** The "cabinet interior behind the open drawer" is NOT in the AABB. Bowl in
   that volume → BDDL False, but visually "in the cabinet". This is the geometry that the
   dead-state check could hit at restore_frac=1.0 if the policy partially closed the drawer
   trapping the bowl behind/under it.

9. **Bowl just outside lb_x or lb_y by < 1cm, resting against the drawer interior wall.** Center is
   strictly outside but body is fully inside. Common for objects sized close to the drawer interior.

### Implications for the re-stratifier

- The right re-stratifier reads each task's BDDL goal expression, evaluates each atomic against the
  final sim state of every failure trace, and labels by **which atomic(s) were False**. Not by Euclidean.
- Add an `In_AABB_slack` debug field: `(bowl_pos - AABB_center) / AABB_half_extent` per axis.
  Values just outside ±1 are likely false-negative candidates (mode 6, 9). Values just inside ±1
  with the gripper still gripping are mode 2.
- Add a `which_drawer_region` field: evaluate `In(bowl, top_region)`, `In(bowl, middle_region)`,
  `In(bowl, bottom_region)` to detect WRONG_DRAWER (mode 7).
- The stricter dead-state check should teleport BOTH bowl-to-site-center AND drawer-to-closed.
  Current script only teleports the drawer.

### Predicted re-stratification distribution for the 25 task-3 failures (gut estimate, written
before run, to be compared with actual)

- Real `(In bowl bottom_region) = False ∧ (Close ...) = True`: 5–8 of 25 (placement off, drawer closed)
- Real `(In bowl bottom_region) = False ∧ (Close ...) = False`: 10–15 of 25 (placement off + drawer open — combination failure; the dominant mode)
- Real `(In bowl bottom_region) = True ∧ (Close ...) = False`: 2–5 of 25 (the only legitimate
  PLACEMENT_OK_SUBPRED_FAIL; the 1/5 endstate-recovered trace likely lives here)
- WRONG_DRAWER (`In(bowl, top|middle_region) = True`): 0–2 of 25 (rare)
- Mode-6 false-negative borderline (bowl in drawer but center above ub_z): 1–3 of 25

If actual distribution diverges substantially from this prediction, that's the signal to inspect
the geometry of `akita_black_bowl`'s body vs `bottom_region`'s AABB rather than re-run.

### N-sizing

After re-stratification, the legitimate `BOWL_IN_DRAWER_OPEN` subset will be small (predicted
2–5 traces). Recovery-test N needs to be larger than this for any confidence interval to be
meaningful. Either (a) collect more failure traces from the existing 110-trace corpus first,
filtered through BDDL eval, or (b) accept that this stratum is intrinsically rare and report
the result with explicit "N=5, CI=±40pp" caveats.
