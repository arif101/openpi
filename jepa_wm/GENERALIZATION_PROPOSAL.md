# Research Proposal: Generalization in VLAs via Surprise-Gated Structural Retrieval
### "Adding a hippocampus to the VLA"

*Status: COMMITTED direction (2026-05-31), backed by two adversarial deep-research passes — see
[[project_generalization_strategy_two_research_passes_2026_05_31]].*

---

## 0. One-paragraph thesis

VLAs generalize poorly because they are **all neocortex, no hippocampus** — pure slow parametric
memorizers (Complementary Learning Systems theory: such a system suffers catastrophic interference and
cannot learn from single novel events). Humans generalize to novel tasks via a *fast episodic system*
that, **when novelty is detected**, retrieves **structurally**-similar past experiences (by relational
structure, not surface appearance — Gentner), **recombines** them, and **simulates** before committing.
We add this to a VLA: a **surprise-gated structural retrieval** module on Pi0.5, where (a) our validated
prediction-error signal ε gates retrieval, (b) a counterfactual-consistency objective shapes the
embedding so retrieval keys on causal/relational structure rather than pixels, (c) retrieved episodes are
recombined and (d) verified by a world-model monitor. Target: compositional generalization to held-out
object×instruction(×spatial) combinations.

## 1. Why this is the right bet (research-verified)

- **Novel & unoccupied** (Pass 2, medium-confidence over a fast field): every retrieval-augmented robot
  policy retrieves by SURFACE similarity, ALWAYS-ON, and COPIES — EchoVLA, MemoryVLA, Behavior Retrieval,
  VINN. NONE do structural + surprise-gated + recombined. Closest = EchoVLA (surface, always-on reads).
- **Rock-solid grounded** (Pass 2): CLS theory (McClelland/McNaughton/O'Reilly 1995; Kumaran/Hassabis/
  McClelland 2016); surprise-gated memory has direct neuroscience precedent (Sinclair&Barense 2021 PNAS:
  prediction errors switch hippocampus stabilize↔update); structure-mapping (Gentner 1983); compositional
  failure of parametric nets (SCAN, Lake&Baroni 2018).
- **Escapes the crowded lane** (Pass 1): language-counterfactual alone is NARROW + crowded (CAST, CAG,
  GateFlow on Pi0.5). Here counterfactual is demoted to a COMPONENT (representation-shaper), not the claim.
- **Uses our validated assets**: ε gate (AUROC 0.97), counterfactual repr-shaping (Brick 1, training now),
  world-model monitor (toy JEPA world⊥self 2×2). The research literally named our ε as the gate.

## 2. Architecture (each session-piece → one mechanism)

```
         obs, instruction
              │
        ┌─────▼──────┐   ε high (novel/surprised)?
        │ Pi0.5 VLA  │────────────────┐ no → act (fast/System-1)
        │ (neocortex)│                │ yes (HIPPOCAMPUS path):
        └─────┬──────┘                ▼
   ε = self/world prediction-error    RETRIEVE top-k by STRUCTURAL key
   (the GATE)                         (counterfactual-shaped embedding)
                                      │
                                      RECOMBINE retrieved (obs,instr,action) episodes
                                      │
                                      SIMULATE candidate via world-model monitor (REASON gate: monitor,
                                      not optimize-through)
                                      │
                                      act
```
| Component | Role | Status |
|---|---|---|
| Counterfactual-consistency | shapes embedding → retrieval keys are causal/relational not pixel | **Brick 1, training now** |
| Prediction-error ε (self+world) | the GATE: retrieve only when novel | validated (AUROC 0.97; toy 2×2) |
| Episodic store + structural retrieval | the missing HIPPOCAMPUS | **to build (Exp 2)** |
| World-model monitor | simulate recombination before commit | toy validated |

## 3. The experiment ladder (each gated, smallest-first)

**Brick 1 — Representation shaping (running).** Counterfactual-consistency LoRA on Pi0.5.
Gate: val_gap grows on held-out episodes (✓ seen: +0.73→+1.71) with bounded competence cost
(watch val_bc; mild rise observed → keep _best).

**Exp 2 — Does counterfactual shaping make a BETTER retrieval key?** (cheap, offline, the pivotal test)
Build an episodic store of (obs, instruction, action-chunk) embeddings. Compare retrieval quality of
BASELINE Pi0.5 embedding vs COUNTERFACTUAL-SHAPED embedding for held-out compositions:
- Metric: retrieval precision = does top-k retrieve episodes with the SAME instruction-relevant structure
  (correct object/relation), and does an embedding-kNN action predictor get lower action error on
  held-out compositions with shaped vs baseline keys?
Gate: shaped > baseline retrieval precision / lower kNN action error. If NO, the repr-shaping brick
doesn't help retrieval → rethink the key.

**Exp 3 — Surprise-gated structural retrieval beats baselines (the headline, closed-loop).**
Four-way ablation on held-out compositions:
1. parametric-only Pi0.5 (does retrieval help?)
2. + naive retrieval (always-on, surface-similarity) — vs EchoVLA/Behavior-Retrieval style
3. + structural keys (counterfactual-shaped) — tests "is STRUCTURE needed?"
4. + ε-gating (retrieve only when surprised) — tests "is GATING better than always-on?"
Full = 3+4. Deltas between rows = the contribution. Report data-efficiency curve (success vs #train pairs).

## 4. Benchmark — held-out compositional split (the one open design choice, now decided)

Both research passes flagged: no standard manipulation benchmark has a clean held-out-COMPOSITION protocol
(SCAN-for-manipulation gap). We build one on our LIBERO harness:

- **LIBERO-object/spatial templates are slot-based** → construct held-out (object × scene) or
  (relation × scene) combinations: train on a subset of pairs, test on pairings PRESENT-IN-SCENE but
  NEVER PAIRED in training. Example: train "pick up the milk" only in scenes A,B; test in scene C (milk
  present, never instructed there).
- External stress checks: LIBERO-CF CF-OOD suite (entirely unseen objects), LIBERO-PRO generalized
  (multi-axis, 0.0% SOTA = max headroom).
- Primary metric: success on held-out compositions, vs the 4 ablations, at matched training data.

## 5. Differentiation (must-cite, verified)
- **EchoVLA** (#1): brain-inspired memory but surface retrieval + always-on reads + attention-fuse. We:
  structural keys + surprise-gated reads + recombination.
- **MemoryVLA**: intra-episode temporal working memory, not cross-episode novel-task retrieval.
- **Behavior Retrieval / VINN**: offline/test-time surface-similarity kNN, no gating, no recombination.
- **CAST / CAG / GateFlow** (language-shortcut fixes on Pi0.5): we subsume counterfactual as the
  repr-shaper, not the claim.
- **"Don't Blind Your VLA"** (vision-bottleneck): motivates multi-axis (not language-only) repr shaping.

## 6. Risks / honesty
- Ambitious (full architecture) → de-risk via the ladder; Exp 2 is the cheap go/no-go before closed-loop.
- "Structural retrieval" operationalization is non-trivial → Exp 2 tests whether counterfactual-shaped
  embeddings are a sufficient structural key (vs needing explicit object-centric/relational graphs).
- Novelty is medium-confidence over a fast field → re-check prior art before any writeup.
- Multi-axis counterfactual (visual+spatial invariance) is the broadening if language-only key is weak.

## 7. Immediate next steps
1. Finish Brick-1 run; capture full val curve + _best.
2. Build episodic store + Exp 2 (shaped vs baseline retrieval key) — the pivotal cheap test.
3. Construct held-out-composition LIBERO split.
4. If Exp 2 passes → Exp 3 four-way ablation (closed-loop, the headline).
