"""Failure attribution for VLA rollouts.

Given a failed rollout (keyframes + task instruction), classify the
failure into planning / skill / perception categories using a VLM judge
(Claude Sonnet 4.6 by default).

Public interface:
    FailureJudge — runs one attribution call
    FailureAttribution — structured result dataclass
"""

from openpi.contact_mpc.attribution.judge import FailureAttribution
from openpi.contact_mpc.attribution.judge import FailureJudge

__all__ = ["FailureAttribution", "FailureJudge"]
