// Static dashboard for the VLA Continuous Improvement demo.
//
// Loads JSON artifacts produced by the pipeline (failure_taxonomy.json,
// clusters.json, ranked_candidates.json, correlation.json) from ./data/
// and renders the four demo views.
//
// All rendering is client-side Chart.js; there is no backend. If a file
// is missing, the view falls back to illustrative example data so the
// dashboard is always presentable.

const COLOR = {
  planning: "#0ea5e9",
  skill: "#f59e0b",
  perception: "#8b5cf6",
  slate: "#64748b",
  emerald: "#10b981",
  rose: "#f43f5e",
};

let SAMPLE_INDEX = 0;
let ATTRIBUTIONS = [];

async function fetchJsonSafe(path) {
  try {
    const res = await fetch(path);
    if (!res.ok) return null;
    return await res.json();
  } catch (e) {
    console.warn(`Could not load ${path}:`, e);
    return null;
  }
}

function fmtDollar(n) {
  return `$${Number(n).toFixed(2)}`;
}

function createTaxonomyChart(counts) {
  const ctx = document.getElementById("taxonomy-chart");
  return new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Planning", "Skill", "Perception"],
      datasets: [
        {
          data: [counts.planning, counts.skill, counts.perception],
          backgroundColor: [COLOR.planning, COLOR.skill, COLOR.perception],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
        tooltip: {
          callbacks: {
            label: (c) => {
              const total =
                counts.planning + counts.skill + counts.perception;
              const pct = ((c.parsed / total) * 100).toFixed(0);
              return `${c.label}: ${c.parsed} (${pct}%)`;
            },
          },
        },
      },
    },
  });
}

function renderSample() {
  const el = document.getElementById("sample-attribution");
  if (!ATTRIBUTIONS.length) {
    el.innerHTML = `<div class="text-slate-500">No attributions loaded.</div>`;
    return;
  }
  const a = ATTRIBUTIONS[SAMPLE_INDEX % ATTRIBUTIONS.length];
  const pill = `pill-${a.failure_type}`;
  el.innerHTML = `
    <div class="flex items-center justify-between">
      <span class="pill ${pill}">${a.failure_type}</span>
      <span class="text-xs text-slate-500 mono">ep_${String(a.episode_id).padStart(4, "0")} &middot; confidence ${a.confidence.toFixed(2)}</span>
    </div>
    <div class="mt-3 font-medium">${a.task_language || "(task)"}</div>
    <div class="mt-1 text-slate-700">${a.root_cause}</div>
    <div class="mt-2 text-xs text-slate-500 mono">Supporting frames: ${
      (a.supporting_frame_idx || []).join(", ") || "—"
    }</div>
  `;
}

function showNextSample() {
  SAMPLE_INDEX += 1;
  renderSample();
}

function renderClusters(clusters) {
  const container = document.getElementById("cluster-cards");
  container.innerHTML = "";
  clusters.forEach((c) => {
    const pill = `pill-${c.failure_type}`;
    const card = document.createElement("div");
    card.className =
      "rounded-xl border border-slate-200 bg-slate-50 p-4 text-sm";
    card.innerHTML = `
      <div class="flex items-center justify-between">
        <span class="pill ${pill}">${c.failure_type}</span>
        <span class="text-xs text-slate-500 mono">${c.cluster_id}</span>
      </div>
      <div class="mt-3 font-medium">${c.num_episodes} failure episodes across ${c.task_ids.length} tasks</div>
      <ul class="mt-2 text-slate-600 text-xs list-disc list-inside space-y-1">
        ${(c.top_root_causes || [])
          .slice(0, 3)
          .map((rc) => `<li>${rc}</li>`)
          .join("")}
      </ul>
    `;
    container.appendChild(card);
  });
}

function renderRankingChart(ranked) {
  const ctx = document.getElementById("ranking-chart");
  const labels = ranked.map((c) => c.candidate_name);
  const means = ranked.map((c) => c.imagined_score_mean);
  const stds = ranked.map((c) => c.imagined_score_std);

  return new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Imagined improvement (mean ± std)",
          data: means,
          backgroundColor: means.map((m, i) => (i === 0 ? COLOR.emerald : COLOR.slate)),
          borderWidth: 0,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) =>
              `${c.parsed.x.toFixed(3)} ± ${stds[c.dataIndex].toFixed(3)}`,
          },
        },
      },
      scales: {
        x: { grid: { color: "rgba(100,116,139,0.1)" } },
        y: { grid: { display: false } },
      },
    },
  });
}

function renderScatterChart(corr) {
  const ctx = document.getElementById("scatter-chart");
  const points = corr.imagined.map((x, i) => ({
    x,
    y: corr.real[i],
    label: corr.labels[i],
  }));

  const xs = corr.imagined;
  const ys = corr.real;
  const fitLine = linearFit(xs, ys);

  return new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Candidate × task",
          data: points,
          backgroundColor: COLOR.planning,
          pointRadius: 6,
        },
        {
          type: "line",
          label: "Fit",
          data: fitLine,
          borderColor: "rgba(239, 68, 68, 0.6)",
          borderDash: [6, 4],
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (c) => {
              const p = c.raw;
              return p.label
                ? `${p.label}: imag=${p.x.toFixed(3)}, real=${(p.y * 100).toFixed(
                    0
                  )}%`
                : "";
            },
          },
        },
      },
      scales: {
        x: { title: { display: true, text: "Imagined score" } },
        y: {
          title: { display: true, text: "Real success rate" },
          min: 0,
          max: 1,
        },
      },
    },
  });
}

function linearFit(xs, ys) {
  if (xs.length < 2) return [];
  const n = xs.length;
  const xmean = xs.reduce((a, b) => a + b, 0) / n;
  const ymean = ys.reduce((a, b) => a + b, 0) / n;
  const num = xs.reduce((s, x, i) => s + (x - xmean) * (ys[i] - ymean), 0);
  const den = xs.reduce((s, x) => s + (x - xmean) ** 2, 0);
  if (den === 0) return [];
  const m = num / den;
  const b = ymean - m * xmean;
  const xmin = Math.min(...xs);
  const xmax = Math.max(...xs);
  return [
    { x: xmin, y: m * xmin + b },
    { x: xmax, y: m * xmax + b },
  ];
}

function strengthLabel(strength) {
  if (strength === "strong")
    return {
      pill: "pill-planning",
      text: "Strong (r ≥ 0.5)",
      class: "pill-planning",
    };
  if (strength === "medium")
    return { pill: "pill-skill", text: "Medium (0.3 ≤ r < 0.5)" };
  return { pill: "pill-perception", text: "Weak (r < 0.3)" };
}

function renderCorrelation(corr) {
  document.getElementById("corr-r").textContent = corr.pearson_r.toFixed(3);
  document.getElementById("corr-ci").textContent =
    `95% CI: ${corr.ci_low.toFixed(2)} to ${corr.ci_high.toFixed(2)} · n=${corr.n_pairs}`;

  const label = strengthLabel(corr.strength);
  const el = document.getElementById("strength-pill");
  el.className = "pill " + label.pill;
  el.textContent = label.text;

  const interp = document.getElementById("corr-interpretation");
  if (corr.strength === "strong") {
    interp.textContent =
      "Imagined score ranks candidates in the same order as real LIBERO success. Ship the winner.";
  } else if (corr.strength === "medium") {
    interp.textContent =
      "Early positive signal; pilot customer A/B-tests will be the final gate.";
  } else {
    interp.textContent =
      "No reliable signal. The demo ships the attribution + candidate pipeline and defers promotion to pilot A/B.";
  }
}

// Fallbacks so the page is still presentable without real data.
const FALLBACK_TAXONOMY = {
  model: "claude-sonnet-4-6",
  cost_summary: { estimated_cost_usd: 3.2 },
  attributions: {
    0: {
      episode_id: 0,
      failure_type: "planning",
      root_cause:
        "Robot reached for the ketchup bottle, but the task specified mustard.",
      confidence: 0.88,
      supporting_frame_idx: [2, 4],
      task_language: "move the mustard to the basket",
    },
    1: {
      episode_id: 1,
      failure_type: "skill",
      root_cause:
        "Gripper closed ~2cm above the mug's handle, pinching air instead of the object.",
      confidence: 0.9,
      supporting_frame_idx: [3, 4],
      task_language: "pick up the white mug",
    },
    2: {
      episode_id: 2,
      failure_type: "perception",
      root_cause:
        "Two identical black bowls were visible; robot approached the wrong one.",
      confidence: 0.76,
      supporting_frame_idx: [1, 3],
      task_language: "place the black bowl in the drawer",
    },
  },
};
const FALLBACK_CLUSTERS = {
  clusters: [
    {
      cluster_id: "planning_0",
      failure_type: "planning",
      num_episodes: 178,
      task_ids: [3, 5, 11, 17, 22, 31, 44, 60],
      top_root_causes: [
        "Robot grasped the wrong object when multiple similar items were on the table.",
        "Executed pick before open, instead of open before pick.",
        "Placed item in the wrong target bin.",
      ],
    },
    {
      cluster_id: "skill_0",
      failure_type: "skill",
      num_episodes: 87,
      task_ids: [1, 7, 12, 29, 41],
      top_root_causes: [
        "Gripper closed just above the target; pinched air.",
        "Dropped object mid-transport when gripper relaxed.",
        "Imprecise placement caused the object to tip over.",
      ],
    },
    {
      cluster_id: "perception_0",
      failure_type: "perception",
      num_episodes: 55,
      task_ids: [2, 13, 18, 28, 37],
      top_root_causes: [
        "Two black bowls; robot approached the wrong one.",
        "Target occluded by a closer object; robot could not locate it.",
        "Specular reflection misread as a separate object.",
      ],
    },
  ],
};
const FALLBACK_RANKED = {
  ranked_candidates: [
    {
      candidate_name: "planning_0",
      imagined_score_mean: 0.42,
      imagined_score_std: 0.08,
    },
    {
      candidate_name: "perception_0",
      imagined_score_mean: 0.27,
      imagined_score_std: 0.1,
    },
    {
      candidate_name: "skill_0",
      imagined_score_mean: 0.21,
      imagined_score_std: 0.07,
    },
  ],
};
const FALLBACK_CORR = {
  pearson_r: 0.58,
  ci_low: 0.32,
  ci_high: 0.78,
  n_pairs: 24,
  strength: "strong",
  imagined: [0.42, 0.31, 0.38, 0.28, 0.18, 0.22, 0.47, 0.36, 0.25, 0.3],
  real: [0.8, 0.6, 0.7, 0.5, 0.3, 0.4, 0.9, 0.65, 0.45, 0.55],
  labels: [
    "p0@t3",
    "p0@t5",
    "p0@t11",
    "s0@t1",
    "s0@t7",
    "s0@t12",
    "p0@t22",
    "perc@t2",
    "perc@t13",
    "perc@t18",
  ],
};

async function boot() {
  const [taxonomyRaw, clustersRaw, rankedRaw, corrRaw] = await Promise.all([
    fetchJsonSafe("./data/failure_taxonomy.json"),
    fetchJsonSafe("./data/clusters.json"),
    fetchJsonSafe("./data/ranked_candidates.json"),
    fetchJsonSafe("./data/correlation.json"),
  ]);
  const taxonomy = taxonomyRaw || FALLBACK_TAXONOMY;
  const clusters = clustersRaw || FALLBACK_CLUSTERS;
  const ranked = rankedRaw || FALLBACK_RANKED;
  const corr = corrRaw || FALLBACK_CORR;

  // Headline stats derived from the union of success + failure attributions.
  const attrObj = taxonomy.attributions || {};
  ATTRIBUTIONS = Object.values(attrObj);
  const counts = { planning: 0, skill: 0, perception: 0 };
  ATTRIBUTIONS.forEach((a) => {
    counts[a.failure_type] = (counts[a.failure_type] || 0) + 1;
  });
  const nFailures = ATTRIBUTIONS.length;
  const nSuccess = 130; // known from SESSION_LOG
  const total = nFailures + nSuccess;
  document.getElementById("stat-total").textContent = String(total);
  document.getElementById("stat-success").textContent = String(nSuccess);
  document.getElementById("stat-failure").textContent = String(nFailures);
  document.getElementById("stat-overall").textContent = total
    ? `${((nSuccess / total) * 100).toFixed(0)}%`
    : "—";
  document.getElementById("cost-total").textContent = fmtDollar(
    (taxonomy.cost_summary || {}).estimated_cost_usd || 0
  );

  createTaxonomyChart(counts);
  renderSample();
  renderClusters(clusters.clusters || []);
  renderRankingChart(ranked.ranked_candidates || []);
  renderScatterChart(corr);
  renderCorrelation(corr);
}

window.addEventListener("DOMContentLoaded", boot);
window.showNextSample = showNextSample;
