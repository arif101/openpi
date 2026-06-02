# Rung 2 — verification-reasoning enabled by a structured world model (PASSED, with boundary)

Substrate: DETERMINISTIC Sokoban-style grid (no contact noise; push succeeds or is blocked -> clean
yes/no). This isolated the claim after the continuous toy's contact-noise drowned the signal (and we
learned: simple greedy/short-MPC planners are too weak — the PLANNER, not the WM, was the bottleneck;
fixed with a competent BFS planner over the WM's dynamics).

## Prediction (push events — the discriminating dynamics: push succeeds iff destination free)
| objects | MonoWM | InterWM |
|---|---|---|
| K=3 (trained) | 0.41 | 0.87 |
| K=4-6 (unseen) | 0.39-0.44 | 0.83-0.88 |
| K=8 (unseen) | 0.55 | 0.71 |
Structured WM LEARNS the relational push/block rule far better even in-distribution (occupancy is a
pairwise check the flat MLP can't express) AND generalizes to unseen counts. Monolithic ~chance.

## Verification-planning (BFS plan with WM -> execute in TRUE env -> replan)
| condition | Plan-Mono | Plan-Inter | Oracle(TrueWM) |
|---|---|---|---|
| no blocker | 0% | 90% | 100% |
| BLOCKER on path (novel) | 0% | 22% | 100% |
| blocker + 2 clutter (novel) | 0% | 2% | 95% |

PRIMARY RESULT: Monolithic WM is USELESS for planning (0% everywhere) — a memorizing world model is
worthless as a reasoning substrate. Structured WM ENABLES planning (90% easy) >> monolithic.
BOUNDARY: structured WM's residual error (0.87 not 1.0) COMPOUNDS over the longer plans needed to
route around blockers -> 22%/2% vs oracle 100%/95%. The task is solvable (oracle); planning success
is bottlenecked by WM accuracy. => quantified WM-accuracy -> planning-success link; motivates more
causal structure (north star) for a more accurate model.

Non-exploitable: BFS over discrete actions, no gradient-optimization-through-model (respects REASON).

## Files
rung1/gridworld_rung2.py (sim + Mono/Inter WM + push-acc + BFS planning eval).
