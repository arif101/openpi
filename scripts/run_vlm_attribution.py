"""Run the Claude failure-attribution judge over all failure episodes.

Reads `frames/metadata.json` produced by `replay_failure_frames.py`, calls
the judge on each failure episode, and writes a consolidated
`failure_taxonomy.json`.

Uses prompt caching on the long system prompt (~1.5k tokens), so per-call
input tokens drop ~90% after the first call. Expected cost for 320 failures
at ~8 frames each with Claude Sonnet 4.6: under $30.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    PYTHONPATH=src python3 scripts/run_vlm_attribution.py \
        --frames-dir data/contact_mpc/frames \
        --output data/contact_mpc/failure_taxonomy.json

Smoke test (5 episodes, dry-run, still calls API):
    ... --max-episodes 5

Resume from partial output:
    ... --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", default="data/contact_mpc/frames")
    p.add_argument("--output", default="data/contact_mpc/failure_taxonomy.json")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--max-frames", type=int, default=8)
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Cap episodes processed (for smoke testing).")
    p.add_argument("--resume", action="store_true",
                   help="Load existing --output file and skip already-judged episodes.")
    p.add_argument("--api-key", default=None,
                   help="Defaults to ANTHROPIC_API_KEY env var.")
    p.add_argument("--only-failures", action="store_true", default=True)
    return p.parse_args()


def load_metadata(frames_dir: pathlib.Path) -> dict:
    meta_path = frames_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata.json at {meta_path}. Run replay_failure_frames.py first."
        )
    raw = json.loads(meta_path.read_text())
    # Keys in JSON are strings; normalize to int.
    return {int(k): v for k, v in raw.items()}


def load_existing_output(path: pathlib.Path) -> dict:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {int(k): v for k, v in raw.get("attributions", {}).items()}


def build_frame_paths(frames_dir: pathlib.Path, episode_id: int, meta: dict) -> list[pathlib.Path]:
    ep_dir = frames_dir / f"ep_{episode_id:04d}"
    return [ep_dir / frame["path"] for frame in meta["frames"]]


def save_output(
    path: pathlib.Path,
    attributions: dict[int, dict],
    cost_summary: dict,
    model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "num_attributions": len(attributions),
        "cost_summary": cost_summary,
        "attributions": {str(k): v for k, v in attributions.items()},
    }
    path.write_text(json.dumps(payload, indent=2))


def summarize_costs(attributions: dict[int, dict]) -> dict:
    total_in = sum(a.get("input_tokens", 0) for a in attributions.values())
    total_out = sum(a.get("output_tokens", 0) for a in attributions.values())
    total_cache_read = sum(a.get("cache_read_tokens", 0) for a in attributions.values())
    total_cache_create = sum(a.get("cache_creation_tokens", 0) for a in attributions.values())

    # Claude Sonnet 4.6 pricing (as of 2026-04):
    # $3/MTok input, $15/MTok output, $0.30/MTok cached read, $3.75/MTok cache write.
    in_cost = total_in * 3.0 / 1e6
    out_cost = total_out * 15.0 / 1e6
    cache_read_cost = total_cache_read * 0.30 / 1e6
    cache_create_cost = total_cache_create * 3.75 / 1e6

    return {
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cache_read_tokens": total_cache_read,
        "total_cache_creation_tokens": total_cache_create,
        "estimated_cost_usd": round(
            in_cost + out_cost + cache_read_cost + cache_create_cost, 4
        ),
    }


def main():
    args = parse_args()
    frames_dir = pathlib.Path(args.frames_dir)
    output_path = pathlib.Path(args.output)

    # Late import so --help works without the anthropic dep installed.
    from openpi.contact_mpc.attribution import FailureJudge

    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY is not set and --api-key not provided.")
        sys.exit(1)

    judge = FailureJudge(
        model=args.model,
        max_frames=args.max_frames,
        api_key=api_key,
    )

    metadata = load_metadata(frames_dir)
    logger.info(f"Loaded metadata for {len(metadata)} episodes")

    existing = load_existing_output(output_path) if args.resume else {}
    if existing:
        logger.info(f"Resuming: {len(existing)} episodes already attributed, will skip those.")

    # Target: failures only (default)
    target_ids = sorted([
        eid for eid, m in metadata.items()
        if (not m["is_success"]) or (not args.only_failures)
    ])
    target_ids = [eid for eid in target_ids if eid not in existing]
    if args.max_episodes is not None:
        target_ids = target_ids[: args.max_episodes]

    logger.info(f"Running attribution on {len(target_ids)} episodes")

    attributions: dict[int, dict] = dict(existing)
    t_start = time.time()

    for i, eid in enumerate(target_ids):
        meta = metadata[eid]
        frame_paths = build_frame_paths(frames_dir, eid, meta)

        # Verify all frames exist on disk before spending API tokens
        missing = [p for p in frame_paths if not p.exists()]
        if missing:
            logger.warning(
                f"Episode {eid}: skipping, {len(missing)} frames missing "
                f"(first: {missing[0]})"
            )
            continue

        try:
            attribution = judge.judge(
                episode_id=eid,
                task_instruction=meta["task_language"],
                frame_paths=frame_paths,
            )
        except Exception as e:
            logger.exception(f"Episode {eid}: judge failed: {e}")
            continue

        record = attribution.to_dict()
        record["task_id"] = meta["task_id"]
        record["task_language"] = meta["task_language"]
        attributions[eid] = record

        if (i + 1) % 10 == 0 or i == len(target_ids) - 1:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_min = (len(target_ids) - i - 1) / rate / 60 if rate > 0 else 0
            cost = summarize_costs(attributions)
            logger.info(
                f"[{i+1}/{len(target_ids)}] ep={eid} type={record['failure_type']} "
                f"conf={record['confidence']:.2f} | "
                f"elapsed={elapsed/60:.1f}min eta={eta_min:.1f}min | "
                f"cost=${cost['estimated_cost_usd']:.2f}"
            )

            # Periodic save so we don't lose progress on a crash
            save_output(output_path, attributions, cost, args.model)

    final_cost = summarize_costs(attributions)
    save_output(output_path, attributions, final_cost, args.model)

    # Headline stats
    type_counts: dict[str, int] = {}
    for a in attributions.values():
        type_counts[a["failure_type"]] = type_counts.get(a["failure_type"], 0) + 1

    logger.info("=" * 60)
    logger.info(f"Attribution complete: {len(attributions)} episodes")
    logger.info(f"  Planning:   {type_counts.get('planning', 0)}")
    logger.info(f"  Skill:      {type_counts.get('skill', 0)}")
    logger.info(f"  Perception: {type_counts.get('perception', 0)}")
    logger.info(f"Estimated cost: ${final_cost['estimated_cost_usd']:.2f}")
    logger.info(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
