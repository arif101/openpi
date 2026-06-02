# Rung 1 — structured world model generalizes by construction (PASSED)

Thesis (north-star brick): a world model that captures per-object CAUSAL structure extrapolates
to novel scenes; a monolithic one that memorizes configs cannot. This is the substrate for
"simulate a candidate approach on a novel object and verify it."

## Setup
2D multi-object pusher physics (CPU, no GPU/rendering — isolates structure from perception).
Both world models trained ONLY on K=3 objects (small radii). Predict per-object next-state delta.
- MonolithicWM: flat MLP over the whole (padded) scene = "coverage" (memorize joint configs).
- InteractionWM: object-centric interaction net (shared per-object + pairwise messages, relative
  positions) = "structure" (one local rule for every object; handles any count by construction).

## Result — 12-step rollout simulation MSE (the WM use case; errors compound)
| eval            | Monolithic | Interaction |
|---|---|---|
| K3 (in-dist)    | 0.0118 | 0.0116  (tie) |
| K4 (unseen N)   | 0.0359 | 0.0108  STRUCT |
| K6 (unseen N)   | 0.0422 | 0.0117  STRUCT |
| K8 (unseen N)   | 0.0495 | 0.0091  STRUCT |

Generalization gap K3->K8 (OOD / in-dist):  Monolithic 4.21x   Interaction 0.79x

The structured WM is as accurate on 8 unseen objects as on the 3 it trained on (0.79x); the
monolithic degrades 4.2x. Generalization by construction, not coverage. Monolithic is also
architecturally CAPPED at n_max objects; the interaction net is unbounded.

## Honest caveats
- Cleanest axis = object COUNT. Size-generalization tied (gentle near-linear physics).
- Low-dim STATE toy, not pixels. Next: does it hold with perception / on a richer sim.
- Single-step contact-MSE showed the same direction but weakly; multi-step is where it's decisive.

## Files
rung1/physics2d.py (sim + collectors), world_models.py (Monolithic vs Interaction), train_eval.py.

## HARDENED (mass ∝ size², 3 seeds) — the honest boundary
| eval (12-step rollout) | Monolithic | Interaction | degradation |
|---|---|---|---|
| K3 in-dist  | 0.0104±.003 | 0.0102±.002 | 1.0x / 1.0x |
| K8 unseen count | 0.0591±.013 | 0.0151±.004 | **5.68x / 1.47x** (STRUCT) |
| K3_tiny unseen size | 0.0233±.004 | 0.0228±.004 | 2.24x / 2.23x (TIE) |

REFINED CLAIM (more honest + more useful):
- COMPOSITIONAL novelty (object count / arrangement): object-centric structure beats coverage
  BY CONSTRUCTION, robustly across seeds (5.7x vs 1.5x degradation).
- PROPERTY novelty (object size/mass outside training range): object-factorization ALONE does NOT
  solve it (tie, both ~2.2x). The per-object dynamics is a learned MLP — MLPs don't extrapolate
  past training inputs. => "new arrangement is free; a new KIND of object needs an ADDITIONAL
  structural prior" (scale/shape-equivariance for size/pose; affordances for how-to-interact).

This is the diagnostic-proven boundary of object-centricity. Object-centric = necessary foundation,
not sufficient. Motivates the next structural ingredient for property/shape novelty (the user's
"new object needs new approach" case).
