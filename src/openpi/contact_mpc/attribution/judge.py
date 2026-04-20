"""Claude-based failure attribution judge.

Wraps the Anthropic SDK to send (task instruction + keyframes) to a VLM
and get back a structured failure classification. Uses prompt caching on
the system prompt (repeated across hundreds of calls) and forced tool use
for reliable structured output.

Requires the `anthropic` package (>=0.40.0 for cache_control). Not in
pyproject.toml yet; add via:
    uv add anthropic

Model ID: claude-sonnet-4-6 (per user selection; can be overridden).
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import pathlib
from typing import Any, Literal

import numpy as np

from openpi.contact_mpc.attribution.prompts import (
    RECORD_FAILURE_TOOL,
    SYSTEM_PROMPT,
    build_user_message_text,
)

logger = logging.getLogger(__name__)

FailureType = Literal["planning", "skill", "perception"]

DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_FRAMES = 8
DEFAULT_MAX_TOKENS = 1024


@dataclasses.dataclass
class FailureAttribution:
    """Structured result from one judge call."""

    episode_id: int
    failure_type: FailureType
    root_cause: str
    confidence: float
    supporting_frame_idx: list[int]
    num_frames_shown: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    raw_tool_input: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "failure_type": self.failure_type,
            "root_cause": self.root_cause,
            "confidence": self.confidence,
            "supporting_frame_idx": self.supporting_frame_idx,
            "num_frames_shown": self.num_frames_shown,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
        }


def subsample_frames(
    frame_paths: list[pathlib.Path],
    max_frames: int,
) -> tuple[list[pathlib.Path], list[int]]:
    """Return up to max_frames evenly-spaced frames.

    Always keeps the first and last frame. Returns the (subsampled_paths,
    indices_into_original_list) so downstream callers can map the VLM's
    supporting_frame_idx back to the original timesteps.
    """
    n = len(frame_paths)
    if n == 0:
        return [], []
    if n <= max_frames:
        return list(frame_paths), list(range(n))
    idx = np.linspace(0, n - 1, max_frames).round().astype(int)
    # Ensure uniqueness (can collapse when n is small relative to max_frames)
    idx = sorted(set(idx.tolist()))
    return [frame_paths[i] for i in idx], idx


def _encode_jpeg_as_base64(path: pathlib.Path) -> str:
    data = pathlib.Path(path).read_bytes()
    return base64.standard_b64encode(data).decode("ascii")


def _build_content_blocks(
    task_instruction: str,
    frame_paths: list[pathlib.Path],
) -> list[dict[str, Any]]:
    """Interleave images with the prompt text. Images come first so the
    model sees them before the classification instruction."""
    blocks: list[dict[str, Any]] = []
    for i, path in enumerate(frame_paths):
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": _encode_jpeg_as_base64(path),
                },
            }
        )
        # Short label helps the model refer back to frames in its reasoning
        blocks.append({"type": "text", "text": f"(frame {i})"})

    blocks.append(
        {"type": "text", "text": build_user_message_text(task_instruction, len(frame_paths))}
    )
    return blocks


class FailureJudge:
    """Call Claude to classify a single failure rollout.

    The Anthropic client is injected (or lazily constructed from env) to
    make unit-testing trivial without an API key.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: Any | None = None,
        max_frames: int = DEFAULT_MAX_FRAMES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
        max_retries: int = 3,
    ):
        self.model = model
        self.max_frames = max_frames
        self.max_tokens = max_tokens

        if client is not None:
            self.client = client
        else:
            # Lazy import so the module loads without anthropic installed.
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "The 'anthropic' package is required. Install with `uv add anthropic`."
                ) from e
            self.client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)

    def judge(
        self,
        episode_id: int,
        task_instruction: str,
        frame_paths: list[pathlib.Path],
    ) -> FailureAttribution:
        """Run one attribution call. Returns a FailureAttribution.

        Raises ValueError if the model does not use the record_failure tool
        (rare given forced tool choice, but possible on refusal).
        """
        sub_paths, _ = subsample_frames(frame_paths, self.max_frames)
        if not sub_paths:
            raise ValueError(f"No frames provided for episode {episode_id}")

        content_blocks = _build_content_blocks(task_instruction, sub_paths)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=[RECORD_FAILURE_TOOL],
            tool_choice={"type": "tool", "name": "record_failure"},
            messages=[{"role": "user", "content": content_blocks}],
        )

        return self._parse_response(
            response=response,
            episode_id=episode_id,
            num_frames_shown=len(sub_paths),
        )

    def _parse_response(
        self,
        response: Any,
        episode_id: int,
        num_frames_shown: int,
    ) -> FailureAttribution:
        tool_block = _extract_tool_use_block(response, tool_name="record_failure")
        if tool_block is None:
            raise ValueError(
                f"Episode {episode_id}: model did not invoke record_failure. "
                f"stop_reason={getattr(response, 'stop_reason', None)}"
            )

        args = tool_block.input if hasattr(tool_block, "input") else tool_block["input"]

        usage = _extract_usage(response)

        return FailureAttribution(
            episode_id=episode_id,
            failure_type=args["failure_type"],
            root_cause=args["root_cause"],
            confidence=float(args["confidence"]),
            supporting_frame_idx=list(args.get("supporting_frame_idx", [])),
            num_frames_shown=num_frames_shown,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
            raw_tool_input=dict(args),
        )


def _extract_tool_use_block(response: Any, tool_name: str) -> Any | None:
    """Find the first tool_use block matching tool_name.

    Handles both anthropic-SDK response objects (attribute access) and
    plain dicts (for unit tests).
    """
    content = getattr(response, "content", None)
    if content is None and isinstance(response, dict):
        content = response.get("content", [])
    if content is None:
        return None

    for block in content:
        btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
        bname = getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else None)
        if btype == "tool_use" and bname == tool_name:
            return block
    return None


def _extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage", {})
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "__dict__"):
        return {k: int(v) for k, v in usage.__dict__.items() if isinstance(v, (int, float))}
    return dict(usage) if isinstance(usage, dict) else {}
