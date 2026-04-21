"""Direct Preference Optimization for cluster-targeted LoRA candidates.

Given a failure cluster (from contact_mpc.attribution.cluster), we can
construct paired (success, failure) trajectories on matched tasks — a
preference dataset aligned natively with the cluster structure. DPO
trains Pi0.5's LoRA to prefer the success trajectory over the failure
trajectory, with gradient directly shaped by the failure signal.

Phase 1 (MVP) public interface:
    build_pairs_for_cluster  — construct DPO pairs from rollout + taxonomy data
    DPOPair                  — dataclass describing one (obs, action_w, action_l) triple

Phase 2 (follow-up) will replace the DPO loss with policy-gradient
objectives using differentiable flow-matching rollouts through the
latent world model.
"""

from openpi.contact_mpc.dpo.pair_builder import DPOPair
from openpi.contact_mpc.dpo.pair_builder import build_pairs_for_cluster

__all__ = ["DPOPair", "build_pairs_for_cluster"]
