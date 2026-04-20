"""Cluster failure episodes by (failure_type × hidden-state similarity).

The primary grouping is by failure_type (planning / skill / perception) so
every cluster has an interpretable label for the demo narrative. When a
primary group is large (> `subcluster_threshold`), we optionally apply
k-means on the episode-level mean-pooled hidden states to split it into
sub-clusters, giving the improvement pipeline 3–5 targeted cohorts to
train LoRAs against.

Episode-level features are the mean of per-decision hidden states within
the episode, consistent with the pooling used elsewhere in contact_mpc.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

FAILURE_TYPES = ("planning", "skill", "perception")


@dataclasses.dataclass
class Cluster:
    """One cohort of failure episodes to target with a single LoRA patch."""

    cluster_id: str
    failure_type: str
    episode_ids: list[int]
    task_ids: list[int]
    centroid: np.ndarray  # [hidden_dim] or zeros if not computed
    top_root_causes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "failure_type": self.failure_type,
            "episode_ids": self.episode_ids,
            "task_ids": sorted(set(self.task_ids)),
            "num_episodes": len(self.episode_ids),
            "centroid_norm": float(np.linalg.norm(self.centroid)),
            "top_root_causes": self.top_root_causes,
        }


def episode_level_features(
    hidden_states: np.ndarray,
    episode_ids: np.ndarray,
) -> dict[int, np.ndarray]:
    """Mean-pool per-decision hidden states into one vector per episode."""
    result: dict[int, np.ndarray] = {}
    for eid in np.unique(episode_ids):
        mask = episode_ids == eid
        result[int(eid)] = hidden_states[mask].mean(axis=0)
    return result


def _top_root_causes(root_causes: list[str], k: int = 3) -> list[str]:
    """Return the k most representative root-cause strings (by frequency).

    Many root_cause strings are unique (free-form text), so we fall back to
    returning the first k alphabetically-sorted strings when frequencies
    are uniform.
    """
    from collections import Counter

    counts = Counter(root_causes)
    most_common = [rc for rc, _ in counts.most_common(k)]
    if len(most_common) < k:
        remaining = [rc for rc in sorted(set(root_causes)) if rc not in most_common]
        most_common.extend(remaining[: k - len(most_common)])
    return most_common


def cluster_failures(
    attributions: dict[int, dict],
    episode_features: dict[int, np.ndarray],
    *,
    subcluster_threshold: int = 80,
    max_subclusters: int = 2,
    random_state: int = 0,
) -> list[Cluster]:
    """Build 3–5 interpretable failure cohorts.

    Args:
        attributions: {episode_id: {failure_type, root_cause, task_id, ...}}
        episode_features: {episode_id: [hidden_dim] mean-pooled feature}
        subcluster_threshold: primary groups larger than this are split.
        max_subclusters: max k for subcluster k-means.
        random_state: for KMeans reproducibility.

    Returns:
        list of Cluster (ordered: planning first, then skill, then perception).
    """
    clusters: list[Cluster] = []

    for ftype in FAILURE_TYPES:
        members = [
            eid for eid, a in attributions.items() if a["failure_type"] == ftype
        ]
        if not members:
            logger.info(f"No failures of type {ftype!r}; skipping.")
            continue

        # Primary grouping
        if len(members) <= subcluster_threshold:
            clusters.append(
                _build_cluster(
                    cluster_id=f"{ftype}_0",
                    failure_type=ftype,
                    member_ids=members,
                    attributions=attributions,
                    episode_features=episode_features,
                )
            )
            continue

        # Large group → split by hidden-state k-means
        feats = np.stack([episode_features[eid] for eid in members])
        k = min(max_subclusters, max(2, len(members) // subcluster_threshold + 1))
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(feats)
        labels = kmeans.labels_

        for sub_idx in range(k):
            sub_members = [members[i] for i in range(len(members)) if labels[i] == sub_idx]
            if not sub_members:
                continue
            clusters.append(
                _build_cluster(
                    cluster_id=f"{ftype}_{sub_idx}",
                    failure_type=ftype,
                    member_ids=sub_members,
                    attributions=attributions,
                    episode_features=episode_features,
                )
            )

    return clusters


def _build_cluster(
    *,
    cluster_id: str,
    failure_type: str,
    member_ids: list[int],
    attributions: dict[int, dict],
    episode_features: dict[int, np.ndarray],
) -> Cluster:
    feats = np.stack([episode_features[eid] for eid in member_ids])
    centroid = feats.mean(axis=0)
    root_causes = [attributions[eid]["root_cause"] for eid in member_ids]
    task_ids = [attributions[eid]["task_id"] for eid in member_ids]
    return Cluster(
        cluster_id=cluster_id,
        failure_type=failure_type,
        episode_ids=sorted(member_ids),
        task_ids=task_ids,
        centroid=centroid,
        top_root_causes=_top_root_causes(root_causes),
    )


def load_attributions(path: pathlib.Path) -> dict[int, dict]:
    """Load failure_taxonomy.json and normalize keys to int."""
    raw = json.loads(pathlib.Path(path).read_text())
    return {int(k): v for k, v in raw.get("attributions", raw).items()}


def save_clusters(clusters: list[Cluster], path: pathlib.Path) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"num_clusters": len(clusters), "clusters": [c.to_dict() for c in clusters]}
    pathlib.Path(path).write_text(json.dumps(payload, indent=2))
