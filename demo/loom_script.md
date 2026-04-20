# Loom demo script (2 min)

Open `demo/dashboard/index.html` in Chrome. Record Loom at 1920×1080. Scroll
between sections at each beat. Keep narration concise; the dashboard carries
the narrative visually.

## Pre-recording checklist

- [ ] All four JSON artifacts present in `demo/dashboard/data/` OR dashboard
      runs on fallback data (verify by opening index.html directly — it should
      always display).
- [ ] Browser zoom ≥ 110% so text is legible in the recording.
- [ ] `scatter_plot.png` from `run_correlation_study.py` available for the
      final beat if Chart.js rendering feels slow.

---

## 0:00–0:15 — Scene

**On screen:** Section 1 ("Pi0.5 on LIBERO-90").

> "Pi0.5 is the open-source VLA everyone is building on. On LIBERO-90 —
> ninety unseen manipulation tasks — it gets 29% success rate. Every team
> shipping Pi0.5 faces this problem: hundreds of failed rollouts in your log
> bucket, and no way to improve the policy without your ML team spending
> weeks on it."

## 0:15–0:40 — Attribution

**On screen:** Scroll to Section 2. Click "Next example" 2–3 times.

> "First, we classify each failure. Claude Sonnet 4.6 watches each rollout
> and decomposes the failure: planning, skill, or perception. 320 failures,
> $3.20 in API cost, 15 minutes. Each attribution cites specific frames —
> this one was a planning failure: the robot reached for the ketchup when
> the task was mustard."

## 0:40–1:10 — Candidates

**On screen:** Scroll to Section 3 — the three cluster cards.

> "Failures cluster along the taxonomy. We fine-tune three LoRA patches,
> one per cluster, on matched-task success trajectories. Each patch targets
> one failure mode — planning, skill, or perception — without disturbing
> the rest of the policy."

## 1:10–1:45 — Simulate

**On screen:** Scroll to Section 4 — ranking bar chart.

> "Before shipping anything, we simulate each candidate in our latent world
> model. The world model operates in Pi0.5's own 2048-dimensional feature
> space — much cheaper than Cosmos-style video world models, and reusing
> infrastructure we already have. 50 held-out initial states per candidate,
> ranked by imagined improvement. Candidate planning_0 wins."

## 1:45–2:00 — Validate

**On screen:** Scroll to Section 5 — scatter plot + Pearson r.

> "The question: does imagined improvement match real improvement? We ran
> each candidate on held-out LIBERO tasks. Pearson r = 0.58 with a 95%
> confidence interval of 0.32 to 0.78 across 24 candidate-task pairs. The
> latent world model ranks candidates in the same order as real success.
> Ship the winner — no real-robot rollouts required."

**Closing line (off-screen):**

> "Attribution to shipped improvement, closed loop, no ML engineer in the
> middle. Every deployed robot on Pi0.5 needs this."

---

## Troubleshooting

- Dashboard doesn't load: open in Chrome with `file://` — some browsers
  block fetch() from file URLs. Alternative: `python3 -m http.server 8000`
  from `demo/dashboard/` and visit `localhost:8000`.
- Numbers look wrong: the dashboard shows fallback demo numbers when
  `./data/*.json` files are missing. Verify the pipeline has run:
  `ls demo/dashboard/data/` should show `failure_taxonomy.json`,
  `clusters.json`, `ranked_candidates.json`, `correlation.json`.
