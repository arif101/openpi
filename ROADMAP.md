# Roadmap

*Last updated: 2026-04-25. Anchored to Phase 5 results in [SESSION_LOG.md](SESSION_LOG.md) and the strategic intel in `~/.claude/projects/-Users-arifahmed-projects-openpi/memory/`.*

---

## North star (12 months)

**`pisearch` is the open-source inference-time deliberation layer that ships with every open VLA.** Three tiers (efficient / accurate / reasoning), three VLA backbones (Pi0.5, OpenVLA, GR00T), CoRL paper, ≥3 design-partner deployments.

We don't compete with foundation VLAs. We're the layer they wrap when they ship.

---

## Sprint 1 — days 1-14 (application-critical)

Lock the LIBERO-PRO claim, add the diversity sampler, ship the SDK skeleton.

| Day | Workstream | Deliverable |
|---|---|---|
| 1 | Restore GPU artifacts | Cached state restored |
| 1-3 | Evolutionary diffusion sampler (VLA-Pilot recipe) | K=32 candidates, 10 evolution steps |
| 3-4 | Robometer-4B drop-in as alternative PRM | `--prm=robometer` flag |
| 4-5 | Re-run Experiment A with new sampler + PRM combos | LIBERO-10 ablation table |
| 5 | Seed=21 replication on LIBERO-10 | 3-seed CI on headline number |
| 6-7 | Perturbation curve at 3, 5, 7, 10cm | Curve plot |
| 8-9 | `pip install pisearch` v0.1 | Public GitHub release |
| 10-11 | Draft methods paper (arXiv preprint) | 4-6 page draft |
| 12-13 | Demo video + YC application body | 60-second Loom |
| 14 | Polish, submit YC + arXiv | Submitted |

**Acceptance criteria:**
- ≥3-seed mean lift on LIBERO-10 perturbed, with confidence interval
- Evolutionary diffusion shown to help or not (decisively)
- arXiv preprint posted
- `pip install pisearch` works end-to-end on a fresh box

---

## Sprint 2 — days 14-28 (application-strengthening)

Cross-VLA + ablations + writeup polish.

| Day | Workstream | Deliverable |
|---|---|---|
| 15-17 | OpenVLA port: extract features, train WM | OpenVLA WM + Experiment A result |
| 18-20 | Component-wise ablations (VICReg-only, NCE-only, plain-MSE) | Ablation table for paper |
| 21-23 | Sonnet-4.6 PRM track (Tier 2.5) | `--prm=sonnet` flag |
| 24-25 | arXiv polish, address self-review | v2 preprint |
| 26-28 | Reach out to 5 PI-ecosystem teams | First customer conversations |

**Acceptance criteria:**
- Cross-VLA: Pi0.5 + OpenVLA both show lift on their respective LIBERO-PRO baselines
- Ablation table proves which components matter
- Sonnet-PRM tier ships with degraded-but-acceptable performance

---

## Phase 2 — months 2-4 (Tier 3: reasoning + replanning)

Attack the LIBERO-90 / unseen-task gap that wrappers can't currently fix.

### Month 2 — Hierarchical decomposition (Hi Robot adapted)
- Sonnet-4.6 reads task + first frame, emits Pi0.5-friendly sub-goal
- Pi0.5 generates candidates conditioned on the rewritten sub-goal
- Target: +5pp on LIBERO-90 perturbed
- Honest framing: "wrapper for any high-level VLM + low-level VLA"

### Month 3 — Iterative replanning (Critic-in-the-Loop)
- After K execution steps, lightweight critic checks progress
- On stagnation, trigger Sonnet-4.6 to revise plan
- Closest implementation of PI's π0.7 quote: *"reflect on outcomes, revise the task plan"*
- Test on long-horizon LIBERO tasks

### Month 4 — ECoT integration
- Chain-of-thought before each action chunk
- Reasoning conditions the search candidate generation
- Borrow from RD-VLA / dVLA / DualCoT-VLA literature

**Acceptance criteria:**
- LIBERO-90 perturbed lift moves from +1pp toward +5pp+
- Decoupled tier API: customers opt into reasoning costs only when needed
- Second arXiv preprint or extension to first

---

## Phase 3 — months 4-6 (production + customers)

Hardware reality + commercial validation.

### Month 4-5 — Hardware partnership
- Borrow / rent Franka or UR5 (or UR5-AC academic loaner)
- Real-arm demo: a public failure case → recovery via our wrapper
- 60-second canonical clip

### Month 5 — Latency optimization
- Batched MCTS expansion (10× speedup, validated technique)
- Optional: MCTS distillation into feed-forward variant for on-device

### Month 6 — Design partner deployments
- Target: 1–3 paying or LOI partners from Pi ecosystem
- Customer-facing tooling: dashboards, A/B harnesses, regression tests

**Acceptance criteria:**
- Real-robot demo published
- ≥1 production deployment (sim or real)
- Latency under 100ms amortized for Tier 1 mode

---

## Decision criteria

### After Sprint 1 (day 14)

| Sprint 1 outcome | Move |
|---|---|
| Evo diffusion adds ≥+3pp on LIBERO-10 | Strong signal; commit Sprint 2 fully |
| Evo diffusion neutral; Robometer adds ≥+5pp | Tier 2 is the moat; reposition pitch around foundation-PRM |
| Both modest; cross-VLA pulls weight in Sprint 2 | "Universal wrapper" claim survives, narrower than ideal |
| Nothing helps beyond +1pp | Pivot to Phase 2 hierarchical earlier; LIBERO-90 becomes Tier 3 wedge |

### After Phase 2 (month 4)

| LIBERO-90 perturbed | Verdict |
|---|---|
| +5pp+ | Universal generalization story. Push CoRL / NeurIPS-tier paper. |
| +2-4pp | Modest but real. Workshop paper. Focus on production. |
| <+2pp | Phase 2 reasoning is also bounded. Full pivot to "robustness amplifier." Skip reasoning research depth, focus on commercial moat. |

---

## Key risks

| Phase | Top risk | Mitigation |
|---|---|---|
| Sprint 1 | Evo diffusion doesn't help | Robometer alone may rescue; have backup |
| Sprint 2 | OpenVLA recipe doesn't transfer | Recipe-doesn't-transfer is itself publishable; reframe as Pi0.5-specialist |
| Phase 2 (hierarchy) | Sonnet costs too much per call | Cache aggressively; use only on contact events |
| Phase 2 (replanning) | Latency breaks for real-time control | Async pipeline behind chunked execution |
| Phase 3 (hardware) | No arm access | Partner with research lab; shared cluster |
| All | Cortex 2.0 ships open SDK | Methods recipe + multi-tier + open-source still differentiates |

---

## Multi-tier architecture (the moat)

Cortex 2.0 is single-tier. We are three:

```
TIER 1 — Efficient (production)
  4M latent WM + 525K VF + light MCTS
  <100ms decision latency, fits on edge GPU
  Wins on LIBERO-10 perturbed (+8-10pp)

TIER 2 — Accurate (server-side)
  Our search architecture + Robometer-4B PRM (or Sonnet)
  1-5s latency, zero per-task fine-tuning required
  Wins where Cortex 2.0 wins, but ZERO-SHOT

TIER 3 — Reasoning (novel tasks)
  Hierarchical: Sonnet-4.6 high-level prompt rewriting
  Iterative replanning: critic-in-the-loop correction
  Targets the LIBERO-90 / unseen-task gap
```

All three share:
- Our published methods recipe (VICReg + L2 InfoNCE for cone-shaped frozen-VLM features)
- Our diagnostic framework
- Our search architecture
- Cross-VLA support (Pi0.5, OpenVLA, eventually GR00T)

---

## SDK API target (after Sprint 1)

```python
from pisearch import wrap

# Tier 1: efficient, on-device
policy = wrap(pi05, mode="efficient")

# Tier 2: accurate, server-side
policy = wrap(pi05, mode="accurate", prm="robometer")
policy = wrap(pi05, mode="accurate", prm="sonnet")

# Tier 3: reasoning-augmented (Phase 2)
policy = wrap(pi05, mode="reasoning",
              high_level="sonnet",
              replanning=True)

# Same API for any backbone
policy = wrap(openvla_model, mode="efficient")
```

Tier-based pricing is what gives the SDK commercial structure. Free at Tier 1; paid API for Tier 2 (foundation-VLM costs) and Tier 3 (reasoning costs).
