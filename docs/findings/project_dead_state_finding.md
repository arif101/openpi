---
name: Dead-state check + stratifier bug (2026-05-18)
description: PLACEMENT_OK_SUBPRED_FAIL stratum is 80% misclassified — bowl-not-in-drawer cases falsely labeled as drawer-close-subpredicate failures.
type: project
originSessionId: 53e17618-aeb4-4007-9d29-d56bfc1900c3
---
Ran `scripts/dead_state_check.py` on the 5 PLACEMENT_OK_SUBPRED_FAIL traces from real-P3 v2. Teleported `white_cabinet_1_{top,middle,bottom}_level` slide joints to closed qpos and checked if BDDL `done` fires.

- `restore_frac=0.5` (mid-trajectory): 0/5 done. **Misleading** — mid-trajectory the bowl hasn't been placed yet.
- `restore_frac=1.0` (end-state): **1/5 done**. The remaining 4 traces: bowl isn't actually IN the drawer at failure point, so closing the drawer cannot satisfy `(In bowl drawer)`.

**Why:** `stratify_failures.py` uses `obj_to_goal_cm < 10` (Euclidean distance to a goal point) as proxy for "bowl placed in drawer". But BDDL's `In()` is topological containment, not distance. So the stratum mixes:
- 20%: real `(Close drawer)` subpredicate failure
- 80%: `(In bowl drawer)` failure where bowl is near the drawer but not contained

**How to apply:**
- Don't draw recovery conclusions from real-P3 v2's 0/5 number until re-stratification.
- Re-stratifier should evaluate the actual BDDL goal predicates against the final sim state, not infer from object position.
- The 4 "bowl-near-drawer-not-in" traces are real `PLACEMENT_FAIL` cases — real-P3 v2's "close drawer" recovery was the wrong action against them.
- Phase A's +6.7pp score itself uses BDDL `done`, so the *number* is clean. What's contaminated is our diagnostic interpretation of where the failure modes lie.
