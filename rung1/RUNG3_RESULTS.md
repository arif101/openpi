# Rung 3 — surprise-gated memory makes an imperfect world model usable (PASSED, strong)

Fixes the Rung 2 boundary (structured WM errs on hard multi-step -> 22%/2%) WITHOUT retraining:
as the agent acts, it compares WM prediction to reality; on divergence (epsilon high = SURPRISE) it
writes the true (state,action)->outcome to memory and plans with the corrected model.

| condition | WM-only (Rung2) | +surprise-mem | oracle | mem writes/ep |
|---|---|---|---|---|
| no blocker | 85% | 100% | 100% | 0.1 |
| BLOCKER on path (novel) | 0% | 100% | 100% | 1.0 |
| blocker + 2 clutter (novel) | 0% | 70% | 90% | 3.2 |

~1-3 surprise-triggered corrections/episode transform a useless planner (0%) into a near-oracle one.
THE THESIS CONVERGES on one diagnostic: world-model (predict, Rung1) + verification (reason, Rung2)
+ surprise (epsilon detects model error) + memory (write correction) = imperfect WM made usable for
reasoning, ONLINE, NO retraining, NON-EXPLOITABLE (BFS + grounded corrections, no opt-through-model;
respects REASON 52->30).

## Honest caveat + the convergence to come
Memory is keyed on EXACT (state,action) -> helps WITHIN an episode (agent re-encounters states while
looping) but corrections DON'T transfer across episodes. Generalizing them needs a STRUCTURAL/relational
key (store the local interaction config, retrieve by structural similarity) -> directly the Exp 2a
finding (structural retrieval keys) and the hippocampus/retrieval thread. The threads converge:
surprise-gated STRUCTURAL memory = corrections that generalize. That is the natural Rung 4.

Still a state-toy (grid); thesis proven in principle, not yet on perception/real manipulation.

## Files
rung1/rung3_surprise.py (Corrected model + surprise-gated memory + BFS planning eval).
