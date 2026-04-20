"""Offline evaluation and validation for LoRA improvement candidates.

Public interface:
    score_hidden_action_pairs — core scorer (pure numpy/torch, testable)
    score_candidate_from_rollouts — load a candidate's rollout npz and score
    CandidateScore — per-candidate aggregated result
"""

from openpi.contact_mpc.eval.offline_evaluator import CandidateScore
from openpi.contact_mpc.eval.offline_evaluator import score_candidate_from_rollouts
from openpi.contact_mpc.eval.offline_evaluator import score_candidate_per_task
from openpi.contact_mpc.eval.offline_evaluator import score_hidden_action_pairs

__all__ = [
    "CandidateScore",
    "score_candidate_from_rollouts",
    "score_candidate_per_task",
    "score_hidden_action_pairs",
]
