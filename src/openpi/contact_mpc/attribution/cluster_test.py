"""Tests for failure clustering."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from openpi.contact_mpc.attribution.cluster import (
    Cluster,
    _top_root_causes,
    cluster_failures,
    episode_level_features,
    load_attributions,
    save_clusters,
)


class TestEpisodeLevelFeatures:
    def test_mean_pools_per_episode(self):
        hs = np.array(
            [
                [1.0, 0.0],
                [3.0, 0.0],  # episode 0 mean = [2, 0]
                [0.0, 10.0],  # episode 1 mean = [0, 10]
            ]
        )
        eids = np.array([0, 0, 1])
        feats = episode_level_features(hs, eids)
        np.testing.assert_allclose(feats[0], [2.0, 0.0])
        np.testing.assert_allclose(feats[1], [0.0, 10.0])

    def test_preserves_all_episode_ids(self):
        hs = np.random.randn(20, 4)
        eids = np.random.randint(0, 5, 20)
        feats = episode_level_features(hs, eids)
        assert set(feats.keys()) == set(int(e) for e in np.unique(eids))


class TestTopRootCauses:
    def test_returns_top_by_frequency(self):
        causes = ["A"] * 5 + ["B"] * 3 + ["C"] * 1
        top = _top_root_causes(causes, k=2)
        assert top == ["A", "B"]

    def test_pads_when_fewer_unique_than_k(self):
        causes = ["A", "A", "B"]
        top = _top_root_causes(causes, k=5)
        assert len(top) == 2  # only 2 unique
        assert "A" in top
        assert "B" in top


class TestClusterFailures:
    def _make_inputs(self, n_planning=5, n_skill=3, n_perception=0, hidden_dim=8):
        attributions = {}
        features = {}
        eid = 0
        for ftype, n in [("planning", n_planning), ("skill", n_skill), ("perception", n_perception)]:
            for _ in range(n):
                attributions[eid] = {
                    "failure_type": ftype,
                    "root_cause": f"{ftype} root cause {eid}",
                    "task_id": eid % 3,
                }
                features[eid] = np.random.randn(hidden_dim)
                eid += 1
        return attributions, features

    def test_one_cluster_per_type_when_small(self):
        attributions, features = self._make_inputs(5, 3, 0)
        clusters = cluster_failures(attributions, features, subcluster_threshold=100)
        assert len(clusters) == 2  # planning and skill (perception has 0)
        assert clusters[0].failure_type == "planning"
        assert len(clusters[0].episode_ids) == 5
        assert clusters[1].failure_type == "skill"
        assert len(clusters[1].episode_ids) == 3

    def test_perception_skipped_when_empty(self):
        attributions, features = self._make_inputs(1, 1, 0)
        clusters = cluster_failures(attributions, features, subcluster_threshold=100)
        types = [c.failure_type for c in clusters]
        assert "perception" not in types

    def test_subclusters_when_above_threshold(self):
        # Force planning type to exceed threshold
        attributions, features = self._make_inputs(n_planning=20, n_skill=3)
        clusters = cluster_failures(
            attributions, features, subcluster_threshold=5, max_subclusters=2
        )
        planning_clusters = [c for c in clusters if c.failure_type == "planning"]
        assert len(planning_clusters) >= 2

        # All original planning episodes should appear exactly once across subclusters
        all_ids = []
        for c in planning_clusters:
            all_ids.extend(c.episode_ids)
        assert sorted(all_ids) == list(range(20))

    def test_cluster_has_centroid_of_correct_shape(self):
        attributions, features = self._make_inputs(5, 3, 0, hidden_dim=16)
        clusters = cluster_failures(attributions, features, subcluster_threshold=100)
        for c in clusters:
            assert c.centroid.shape == (16,)

    def test_returns_empty_when_no_failures(self):
        clusters = cluster_failures({}, {})
        assert clusters == []

    def test_cluster_ids_are_unique(self):
        attributions, features = self._make_inputs(30, 30, 30)
        clusters = cluster_failures(
            attributions, features, subcluster_threshold=10, max_subclusters=3
        )
        ids = [c.cluster_id for c in clusters]
        assert len(ids) == len(set(ids))


class TestSerialization:
    def test_save_clusters_writes_valid_json(self, tmp_path: pathlib.Path):
        clusters = [
            Cluster(
                cluster_id="planning_0",
                failure_type="planning",
                episode_ids=[1, 2, 3],
                task_ids=[0, 0, 1],
                centroid=np.zeros(4),
                top_root_causes=["x", "y"],
            )
        ]
        out = tmp_path / "clusters.json"
        save_clusters(clusters, out)
        loaded = json.loads(out.read_text())
        assert loaded["num_clusters"] == 1
        assert loaded["clusters"][0]["cluster_id"] == "planning_0"
        assert loaded["clusters"][0]["num_episodes"] == 3
        assert loaded["clusters"][0]["task_ids"] == [0, 1]  # deduped + sorted

    def test_load_attributions_handles_nested_format(self, tmp_path: pathlib.Path):
        p = tmp_path / "tax.json"
        p.write_text(json.dumps({"attributions": {"5": {"failure_type": "skill"}}}))
        out = load_attributions(p)
        assert 5 in out
        assert out[5]["failure_type"] == "skill"

    def test_load_attributions_handles_flat_format(self, tmp_path: pathlib.Path):
        p = tmp_path / "tax.json"
        p.write_text(json.dumps({"5": {"failure_type": "skill"}}))
        out = load_attributions(p)
        assert 5 in out
