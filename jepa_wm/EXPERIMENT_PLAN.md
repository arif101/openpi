# Experiment plan — grounding via instruction-conditioned surprise

Focus: the IDEA, the METHODOLOGY, and clean EXECUTION. (No paper framing yet; that's an
output at the end.) v1 axis chosen with the lit search: LANGUAGE GROUNDING, unified under
the surprise-as-substrate thesis. Memory + tool-calling sequenced to later.

## Why this axis (lit search, 2026-05-30)
- Phenomenon dramatic & cited: pi0.5 keeps 96.2% success under CONTRADICTORY instructions;
  null-prompt recovers cosine 0.999 (pi0.5 73-77% with NO language); counterfactual prompting
  effect size eta^2=0.012. LIBERO-CF: SOTA VLAs 0-13% (OpenVLA-OFT 0.4%, pi0 9.6%, pi0.5 13.2%).
- Novelty CONFIRMED OPEN: no action-level counterfactual-consistency / instruction-contrastive
  TRAINING objective for VLAs. Closest: CF-VLM (VLMs only), CAST (data aug + standard BC),
  IGAR/CAG (inference-time, no training).
- Benchmark beatable & sim: LIBERO-CF (0-13%), grounding load-bearing by construction.
- World-model-as-monitor-not-planner CONFIRMED externally (WoVR: planning-through-model fails via
  spurious-success / model-exploitation; AtomVLA uses WM only as scorer).
- CAUTION (verified): the strong "diffusion = pure nearest-neighbor lookup table" claim was
  REFUTED in verification. Say "shortcut / ignore-language", NOT "memorization". Grounding is
  suite-DEPENDENT (load-bearing when scene has multiple candidate objects; ignorable when the
  scene disambiguates).

## The idea (best version)
Root cause: BC maximizes p(action | image, instruction). When the IMAGE alone predicts the
action, the gradient w.r.t. instruction tokens vanishes → language is a free rider. Fix = make
language carry gradient.

Principled objective: **maximize I(action ; instruction | image)**.

Implementable on Pi0.5 (flow-matching head, no clean likelihood but a usable FM-loss energy):
**Contrastive flow-matching for grounding.** For demo (image, L, A_demo), sample counterfactual
instruction L' (another object actually present). Train so A_demo's flow-matching loss is LOW
under L and HIGH under L' (margin / InfoNCE form = MI lower bound). 
- Makes language carry gradient through the existing FM head (LoRA-able, minimal arch change).
- The per-instruction FM energy gap FM(A|L') - FM(A|L) IS the inference-time grounding-surprise
  signal → same quantity is the training objective AND the metacognition monitor. One idea.
- Pitfall: the counterfactual needs a TARGET (the other object's action) where available, not just
  "be different" (which rewards incoherence). Margin form + targeted supervision handles it.

## Methodology (where rigor lives)
- **D0 diagnostic-first**: reproduce linguistic blindness on OUR pi0.5/LIBERO. Run correct /
  counterfactual / null prompts; measure Language-Grounding-Score = SR(correct) - SR(counterfactual)
  and action-divergence under swaps. Gate: pick the suite where pi0.5 is demonstrably prompt-
  ignoring (e.g. libero_goal/spatial where scene shares objects). Sets the number to beat.
- **Controls (3, not 1)**: bare pi0.5; CAST-style data-aug + standard BC (isolates objective vs
  data); IGAR/CAG inference-time (the competitor to beat — not just bare pi0.5).
- **The tradeoff IS the experiment**: lift LIBERO-CF grounding WITHOUT regressing standard LIBERO.
  Always report both axes.
- Multiple seeds + CIs; held-out object×instruction for compositional generalization;
  grounding-surprise AUROC for the monitor.
- Final novelty re-check + verify LIBERO-CF leaderboard before claiming.

## LOCKED Phase-1 data recipe (verified on box 2026-05-30)
- Base: `HuggingFaceVLA/libero` (the dataset pi05_libero_finetuned trained on): 1693 episodes,
  273465 frames, 10fps, 2x 256x256 cams + 8-dim state -> 7-dim action. 4 suites x 10 tasks (40),
  task_index 0-9 long / 10-19 goal / 20-29 object / 30-39 spatial.
- Templates: object "pick up the {X} and place it in the basket"; spatial "pick up the black bowl
  {relation} and place it on the plate"; goal "put the {obj} on the {target}".
- Counterfactual L_cf = swap ONE slot to another object/relation PRESENT IN THAT EPISODE'S SCENE
  (per-task pool parsed from BDDL :objects, NOT the full suite set — out-of-scene swaps are too easy).
  Verified: object task0 scene has 7 objects (alphabet_soup, salad_dressing, cream_cheese, milk,
  tomato_sauce, butter, basket); spatial task0 has 2 black bowls (relation disambiguates).
- Grounding loss: contrastive flow-matching — down FM(A|img,L_correct), up FM(A|img,L_cf) (margin).
  No action label needed for L_cf.
- Metacognition (self-supervised, free): world head -> enc(image_{t+k}); self head -> state_{t+1}-state_t.
- Split: hold out (object x instruction) combos for compositional-gen eval; LIBERO-CF external test.
- Augment (planned): paraphrases + extra in-scene swaps (only ~10 instructions/suite otherwise).
- Build scripts (laptop): jepa_wm/inspect_libero_data.py (done), d0_patch.py (done). Next: cf dataset
  wrapper + LoRA trainer with contrastive-FM + metacognition heads.

## Execution status
- Harness: LeRobot Pi0.5 + LIBERO on H100 (216.81.245.29:15354). venv on LOCAL disk
  (/root/vla-venv; /workspace network mount threw stale-file-handle on LIBERO's many small files).
  HF_HOME=/workspace/hf_cache. Installing lerobot[pi,libero,peft].
- Next: smoke-test (pi05_libero produces actions; baseline LIBERO success matches published) →
  D0 blindness diagnostic.
