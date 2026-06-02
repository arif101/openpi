"""Inspect HuggingFaceVLA/libero: structure, size, features, instructions, and the
scene-object metadata needed to build 'hard' counterfactual instruction pairs."""
import json
from collections import defaultdict

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

REPO = "HuggingFaceVLA/libero"
meta = LeRobotDatasetMetadata(REPO)

print("=== dataset size ===")
print("episodes:", meta.total_episodes, "frames:", meta.total_frames, "fps:", meta.fps)
print("chunks/videos:", getattr(meta, "total_videos", "?"))

print("\n=== features (key: dtype shape) ===")
for k, v in meta.features.items():
    print(f"  {k}: {v.get('dtype')} {v.get('shape')}")

print("\n=== tasks (language instructions) ===")
tasks = meta.tasks
try:
    items = list(tasks.items()) if hasattr(tasks, "items") else list(enumerate(tasks))
except Exception:
    items = []
print("num unique task strings:", len(items))
for i, t in items[:20]:
    print(f"  [{i}] {t!r}")

# episode -> task mapping (how many episodes per instruction)
print("\n=== episodes per task (first 12) ===")
per = defaultdict(int)
try:
    for ep_idx in range(meta.total_episodes):
        ep = meta.episodes[ep_idx]
        t = ep.get("tasks", ep.get("task", None))
        per[str(t)] += 1
except Exception as e:
    print("episode iteration failed:", repr(e)[:120])
for k, c in list(per.items())[:12]:
    print(f"  {c:4d}  {k[:70]}")
