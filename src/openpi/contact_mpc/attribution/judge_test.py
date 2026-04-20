"""Tests for the failure-attribution judge.

These tests do not require the anthropic package or an API key; the Claude
client is injected as a fake that returns canned responses.
"""

from __future__ import annotations

import dataclasses
import io
import pathlib
import tempfile
from typing import Any

import numpy as np
import pytest
from PIL import Image

from openpi.contact_mpc.attribution.judge import (
    FailureAttribution,
    FailureJudge,
    _build_content_blocks,
    _encode_jpeg_as_base64,
    subsample_frames,
)


# --- Fake Claude client for dependency injection ---------------------------


@dataclasses.dataclass
class _FakeToolUseBlock:
    type: str = "tool_use"
    name: str = "record_failure"
    input: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class _FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclasses.dataclass
class _FakeResponse:
    content: list[Any]
    usage: _FakeUsage
    stop_reason: str = "tool_use"


class _FakeClient:
    """Records the last-seen create() kwargs and returns a canned response."""

    def __init__(self, response: _FakeResponse):
        self._response = response
        self.last_kwargs: dict[str, Any] | None = None

        class _Messages:
            def __init__(outer, parent: "_FakeClient"):
                outer._parent = parent

            def create(outer, **kwargs):
                outer._parent.last_kwargs = kwargs
                return outer._parent._response

        self.messages = _Messages(self)


def _make_fake_response(
    failure_type: str = "planning",
    root_cause: str = "reached for ketchup when task was mustard",
    confidence: float = 0.9,
    supporting_frame_idx: list[int] | None = None,
    usage: _FakeUsage | None = None,
) -> _FakeResponse:
    if supporting_frame_idx is None:
        supporting_frame_idx = [2, 5]
    if usage is None:
        usage = _FakeUsage(
            input_tokens=120, output_tokens=45,
            cache_read_input_tokens=1500, cache_creation_input_tokens=0,
        )
    return _FakeResponse(
        content=[
            _FakeToolUseBlock(
                input={
                    "failure_type": failure_type,
                    "root_cause": root_cause,
                    "confidence": confidence,
                    "supporting_frame_idx": supporting_frame_idx,
                }
            )
        ],
        usage=usage,
    )


# --- Image fixture helpers -------------------------------------------------


@pytest.fixture
def tmp_frames(tmp_path: pathlib.Path) -> list[pathlib.Path]:
    """Create 10 tiny solid-color JPEGs for use as fake frame paths."""
    paths: list[pathlib.Path] = []
    for i in range(10):
        img = Image.fromarray(
            np.full((32, 32, 3), fill_value=i * 25, dtype=np.uint8)
        )
        p = tmp_path / f"t{i:04d}.jpg"
        img.save(p, format="JPEG", quality=50)
        paths.append(p)
    return paths


# --- Tests ------------------------------------------------------------------


class TestSubsampleFrames:
    def test_all_frames_kept_when_under_max(self):
        paths = [pathlib.Path(f"x{i}") for i in range(5)]
        out, idx = subsample_frames(paths, max_frames=8)
        assert len(out) == 5
        assert idx == [0, 1, 2, 3, 4]

    def test_first_and_last_always_kept(self):
        paths = [pathlib.Path(f"x{i}") for i in range(20)]
        out, idx = subsample_frames(paths, max_frames=5)
        assert idx[0] == 0
        assert idx[-1] == 19

    def test_respects_max_frames(self):
        paths = [pathlib.Path(f"x{i}") for i in range(100)]
        out, idx = subsample_frames(paths, max_frames=8)
        assert len(out) <= 8
        assert len(idx) == len(out)

    def test_indices_are_sorted_and_unique(self):
        paths = [pathlib.Path(f"x{i}") for i in range(30)]
        _, idx = subsample_frames(paths, max_frames=7)
        assert idx == sorted(idx)
        assert len(idx) == len(set(idx))

    def test_empty_returns_empty(self):
        out, idx = subsample_frames([], max_frames=8)
        assert out == []
        assert idx == []


class TestEncodeJpeg:
    def test_roundtrip(self, tmp_path: pathlib.Path):
        arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        p = tmp_path / "x.jpg"
        Image.fromarray(arr).save(p, format="JPEG")
        b64 = _encode_jpeg_as_base64(p)
        # Should be a non-trivial base64 string
        assert isinstance(b64, str)
        assert len(b64) > 100


class TestBuildContentBlocks:
    def test_contains_one_image_per_frame(self, tmp_frames):
        blocks = _build_content_blocks("pick up the cup", tmp_frames[:4])
        image_blocks = [b for b in blocks if b["type"] == "image"]
        assert len(image_blocks) == 4

    def test_ends_with_instruction_text(self, tmp_frames):
        blocks = _build_content_blocks("pick up the cup", tmp_frames[:3])
        assert blocks[-1]["type"] == "text"
        assert "record_failure" in blocks[-1]["text"]

    def test_mentions_task_instruction_verbatim(self, tmp_frames):
        blocks = _build_content_blocks("grab the red mug", tmp_frames[:2])
        full_text = " ".join(b["text"] for b in blocks if b["type"] == "text")
        assert "grab the red mug" in full_text

    def test_image_uses_base64(self, tmp_frames):
        blocks = _build_content_blocks("pick", tmp_frames[:1])
        img_block = next(b for b in blocks if b["type"] == "image")
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/jpeg"
        assert isinstance(img_block["source"]["data"], str)


class TestFailureJudgeCall:
    def test_returns_parsed_attribution(self, tmp_frames):
        client = _FakeClient(_make_fake_response())
        judge = FailureJudge(client=client)
        result = judge.judge(
            episode_id=42,
            task_instruction="move the ketchup to the basket",
            frame_paths=tmp_frames,
        )
        assert isinstance(result, FailureAttribution)
        assert result.episode_id == 42
        assert result.failure_type == "planning"
        assert result.confidence == pytest.approx(0.9)
        assert result.supporting_frame_idx == [2, 5]

    def test_subsamples_to_max_frames(self, tmp_frames):
        client = _FakeClient(_make_fake_response())
        judge = FailureJudge(client=client, max_frames=4)
        result = judge.judge(
            episode_id=1, task_instruction="x", frame_paths=tmp_frames
        )
        assert result.num_frames_shown == 4

        image_blocks = [
            b for b in client.last_kwargs["messages"][0]["content"]
            if b["type"] == "image"
        ]
        assert len(image_blocks) == 4

    def test_uses_prompt_cache_on_system_prompt(self, tmp_frames):
        client = _FakeClient(_make_fake_response())
        judge = FailureJudge(client=client)
        judge.judge(episode_id=1, task_instruction="x", frame_paths=tmp_frames[:3])
        system = client.last_kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_forces_tool_use(self, tmp_frames):
        client = _FakeClient(_make_fake_response())
        judge = FailureJudge(client=client)
        judge.judge(episode_id=1, task_instruction="x", frame_paths=tmp_frames[:3])
        assert client.last_kwargs["tool_choice"] == {
            "type": "tool",
            "name": "record_failure",
        }

    def test_captures_usage_tokens(self, tmp_frames):
        client = _FakeClient(
            _make_fake_response(
                usage=_FakeUsage(
                    input_tokens=100, output_tokens=50,
                    cache_read_input_tokens=2000, cache_creation_input_tokens=1500,
                )
            )
        )
        judge = FailureJudge(client=client)
        result = judge.judge(
            episode_id=1, task_instruction="x", frame_paths=tmp_frames[:3]
        )
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cache_read_tokens == 2000
        assert result.cache_creation_tokens == 1500

    def test_raises_when_no_tool_use_block(self, tmp_frames):
        # Response with no tool_use block (model refused / hit max tokens)
        bad_response = _FakeResponse(
            content=[{"type": "text", "text": "I cannot classify this"}],
            usage=_FakeUsage(),
            stop_reason="end_turn",
        )
        client = _FakeClient(bad_response)
        judge = FailureJudge(client=client)
        with pytest.raises(ValueError, match="did not invoke record_failure"):
            judge.judge(episode_id=1, task_instruction="x", frame_paths=tmp_frames[:3])

    def test_rejects_empty_frame_list(self):
        client = _FakeClient(_make_fake_response())
        judge = FailureJudge(client=client)
        with pytest.raises(ValueError, match="No frames provided"):
            judge.judge(episode_id=1, task_instruction="x", frame_paths=[])


class TestFailureAttributionSerialization:
    def test_to_dict_roundtrips_fields(self):
        a = FailureAttribution(
            episode_id=7,
            failure_type="skill",
            root_cause="gripper missed the mug",
            confidence=0.8,
            supporting_frame_idx=[3],
            num_frames_shown=8,
            input_tokens=50,
            output_tokens=30,
            cache_read_tokens=1500,
            cache_creation_tokens=0,
        )
        d = a.to_dict()
        assert d["failure_type"] == "skill"
        assert d["root_cause"] == "gripper missed the mug"
        assert d["supporting_frame_idx"] == [3]
        assert d["cache_read_tokens"] == 1500
