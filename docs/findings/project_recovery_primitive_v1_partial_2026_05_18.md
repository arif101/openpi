---
name: Recovery primitive v1 (joint teleport + push) — partial success (2026-05-18)
description: Joint-teleport + 60-step +y push primitive succeeds on 2/14 IN_TRUE_CLOSE_FALSE traces. Mechanism validated; naive implementation rejected. Need controller-mediated motion for deployable version.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
## Result

Tested the diagnostic primitive on 14 IN_TRUE_CLOSE_FALSE task-3 traces.
Procedure: state-jump env to trace's inflection point (no action replay) →
overwrite robot qpos[0:7] with canonical push pose extracted from 53/54
PHYS_OK traces → 10 no-op settle steps → 60 steps of +y EE-delta push.

| Outcome | Count | % |
|---|---|---|
| SUCCESS (In ∧ Close both True) | 2 | 14% |
| STILL_IN_TRUE_CLOSE_FALSE | 9 | 64% |
| BOWL_FELL_OUT (In became False) | 3 | 21% |

## What worked

- The 2 successes are in the **"near-stuck" subgroup** (gap to drawer face 4–6 cm at inflection). When teleport distance is small, settling works and push transmits force through the bowl.
- Confirms the **mechanism**: joint reset to a feasible push pose + forward push can close the drawer. Pi0.5's failure mode is exactly the OSC kinematic limitation it appears to be.
- Canonical push pose's EE position is (+0.03, +0.15, +0.94) — **inside the open drawer cavity, behind the bowl**. Successful trials close the drawer by pushing the bowl into the drawer's back wall, not by pushing the drawer face directly.

## What failed

- **Joint teleport is too violent.** Instantaneous qpos write displaces objects in the EE's swept volume → bowl ejection in 3/14 cases (bowl_pos drifted out of drawer AABB).
- **OSC controller state staleness.** After teleport, OSC's internal target pose / integrator are inconsistent with the new joint positions. First few +y commands fight the stale target before clean push begins.
- In 9/14 cases the drawer ends UNCHANGED (qpos around −0.06) or DRIVEN FURTHER OPEN (qpos −0.16). The recovery action isn't engaging the close mechanism — likely due to the OSC stale state, since the push pose itself was the correct one from successful trials.

## Why the canonical push pose finding is itself interesting

Mean EE position at the moment of drawer closing across 53 PHYS_OK traces:
(+0.031, +0.154, +0.939). This is NOT at the drawer face (y≈0.25) but **at
y=0.15, near the bowl's position inside the open drawer cavity**.

Successful traces close the drawer by:
1. Placing bowl in drawer cavity.
2. Maintaining EE position near the bowl (not retreating).
3. Pushing +y → EE pushes bowl → bowl pushes drawer's back wall → drawer slides closed.

Failing traces retreat to EE y=0.04 (10cm behind bowl), losing the contact
path. Pushing from there pushes empty air.

## How to apply

1. **Mechanism is validated. Don't pivot to subgoal search or other architectural
   changes** — the recovery action sequence works.
2. **Naive teleport implementation rejected.** A deployable primitive needs
   controller-mediated smooth motion to the push pose.
3. **Three implementation options for v2:**
   - **A: Controller mode switch** to JOINT_POSITION for recovery period. Smooth
     joint ramp via the joint controller. Cleanest if LIBERO exposes runtime
     controller switching.
   - **B: Multi-step OSC waypoint** sequence — explicit EE-delta commands to
     go back to a safe pose, lift, then forward, then push. Multi-stage but
     stays within OSC.
   - **C: Direct `sim.data.ctrl` write** during recovery. Bypass env interface.
     Diagnostic only; not deployable as a real recovery wrapper.
4. **The detector signature is solid and ready to deploy:**
   `mean(action_dy over 30 steps) > 0.4 AND mean(joint_motion/EE_motion ratio) > 5x`
   verified across 14/14 IN_TRUE_CLOSE_FALSE traces.

## Files

- Canonical push pose: `data/contact_mpc/drawer_push_pose.npz` on GPU box
- Primitive test results: `data/contact_mpc/recovery_primitive_test.json`
- Scripts: `scripts/find_drawer_push_pose.py`, `scripts/test_drawer_recovery_primitive.py` (commit 3cebc87)
