# Paper thesis — surprise as the substrate (working draft, pre-lit-search)

## The unifying principle (avoids the kitchen-sink trap)

A substantial VLA architecture paper needs ONE idea from which memory, metacognition,
tool-calling, and language grounding all follow as consequences — not four bolted-on modules
(reviewers reject those; nothing ablates cleanly).

**Unifying principle: prediction error (surprise) is the organizing signal of the policy.**
Active inference for VLAs. The four "first-class citizens" are consequences:

| Citizen | Falls out of surprise as... |
|---|---|
| **Metacognition** | the surprise signal itself — multi-channel: world, self, grounding |
| **Memory** | surprise-gated writes (store what violated prediction) + episodic retrieval |
| **Tool-calling** | surprise-triggered action (high error → emit query/recovery/skill, not next action) |
| **Grounding** | prediction must be language-conditioned; a prompt-ignoring policy has a characteristic surprise signature |

Each is independently ablatable (turn off one channel / one gate) — that's what makes it a paper.

## Evidence already in hand (this session)
- Two channels proven complementary on a toy: world-surprise (1.000 on object anomaly) ⟂
  self-residual (0.997 on effector-stuck), neither subsumes the other (M3).
- Closed loop: surprise-gated recovery +92pp on the toy (M4').
- Real-robot self-residual (ε) catches execution-stuck at AUROC 0.97 (prior).
- World model used as MONITOR, never optimized through (REASON 52→30 exploitability gate).

## The generalization angle (what makes it "beat benchmarks", not just recover)
VLAs fail to generalize because they shortcut-learn (visual context → action), ignoring
language (Schwager ALT, Mueller memorization). The GROUNDING channel attacks this:
- Diagnose shortcut via counterfactual-prompt sensitivity + attention/ablation + neuron selectivity.
- Train against it: counterfactual-consistency objective (penalize prompt-invariance). The
  diagnostic and the objective are the SAME instrument.
- Evaluate on compositional generalization (held-out object×instruction).

## Open decisions the lit search must settle
- Novelty delta vs: memory-augmented policies, hierarchical/System-2 VLAs (pi0.7, Hi Robot,
  Helix), uncertainty-aware VLAs (Sentinel), counterfactual grounding objectives.
- Benchmark target: must make memory/tool-calling/grounding LOAD-BEARING. LIBERO (single-
  instruction, memoryless) won't. Candidates: LIBERO-LONG, CALVIN long-horizon, RoboCasa,
  VLABench, BEHAVIOR. Pick where current VLAs provably fail for lack of memory/grounding.
- Scope for v1 paper: which channels/citizens are ready to claim vs. future work. Benchmark-
  WINNING is the long-term north star; v1 needs ONE crisp axis with deep evidence.

## Competitive landscape (deep lit search, 2026-05-30, 29 sources, 24/25 claims verified)

**Gap CONFIRMED:** mainstream VLAs Markovian/no-memory; memory added as external banks
(MemoryVLA, EchoVLA, MAP-VLA, MTIL); failure-detection exclusively external monitors
(Sentinel/STAC, SAFE, FAIL-Detect, INSIGHT) — none use surprise as a closed-loop CONTROL signal;
existing "surprise" is action-consistency/latent-OOD/token-entropy, NOT world-model prediction
error; NO VLA emits memory/tool/help as a native learned output conditioned on its own uncertainty
(UPS/INSIGHT wrap frozen policies with external classifiers).

**Must-differentiate-against:** UPS (RSS'26), INSIGHT (Oct'25), Sentinel/STAC (CoRL'24),
MemoryVLA / EchoVLA, SAFE / FAIL-Detect.

**Our delta:** prediction-error (world+self) as the single organizing signal; memory-write /
tool-call / help-request as NATIVE learned policy outputs gated by surprise. Unattested framing.

**Confirmed risks:** (1) crowded, monthly releases from strong labs — need speed + sharp delta;
(2) novel framing must be PROVEN to beat bolt-ons via ablation, not just asserted; (3) our traction
is the metacognition/execution-stuck axis (ε AUROC 0.97), but the cleanest beatable gap per the
search is MEMORY (Mikasa-Robo, LIBERO-LONG, SimplerEnv-Bridge-long) — partly different axis.

**Coverage gaps (need follow-up search):** grounding/memorization novelty (Schwager ALT, Mueller,
counterfactual-consistency objective); world-model-as-monitor; benchmark SOTA numbers/holders.

## Risk register
- Kitchen-sink rejection → mitigated by the single-principle framing + per-channel ablations.
- Novelty → gated by lit search (running).
- "Recovery wrapper, not generalization" → the grounding channel + compositional-gen eval is
  the answer; must show a real held-out gain, not just recovery.
