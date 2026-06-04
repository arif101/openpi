"""Eval the trained target-binding adapter on cf_harness reached_target (the GO/KILL metric).

Wraps frozen pi0.5 with the trained binder/adapter (BoundPolicy.infer == stock infer but injects
g = adapter(language features) via sample_actions_bound). Runs PAIRED rollouts (same seed/init) of
baseline (stock pi0.5) vs bound on each LIBERO-PRO TASK scene, reports reached-named-target rate.

For the 2x2: run with --params runs/cf_bind_on.pkl and runs/cf_bind_off.pkl. GO if ON lifts grounding
40%->>=60% and OFF does not.
"""
from __future__ import annotations

import argparse
import glob
import pathlib
import pickle

import jax
import jax.numpy as jnp
import numpy as np

import cf_bind as B
from cf_harness import parse_bddl, run_episode


class BoundPolicy:
    """Same .infer(obs) contract as openpi Policy, but injects the language-derived target vector g."""

    def __init__(self, policy, params):
        from openpi.models import model as _model
        from openpi.shared import nnx_utils
        self._model_mod = _model
        self.p = policy
        self.model = policy._model
        self.binder = jax.tree.map(jnp.asarray, params["binder"])
        self.adapter = jax.tree.map(jnp.asarray, params["adapter"])
        self._rng = jax.random.key(0)
        # jit the two heavy nnx forward passes (else eager execution makes each replan ~17s).
        # binder_apply is a static function arg -> fun-arg position 4 (after state,rng,observation,g).
        self._extract = nnx_utils.module_jit(self.model.extract_vlm_spatial_features)
        self._sample_bound = nnx_utils.module_jit(self.model.sample_actions_bound, static_argnums=(4,))

    def infer(self, obs: dict) -> dict:
        inputs = self.p._input_transform(jax.tree.map(lambda x: x, obs))
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], inputs)
        observation = self._model_mod.Observation.from_dict(inputs)
        pf, _ = self._extract(observation)
        n_img = int(pf.shape[1] - observation.tokenized_prompt.shape[1])  # image tokens precede language
        g = B.adapter_apply(self.adapter, pf, n_img)
        g = g / (jnp.linalg.norm(g, axis=-1, keepdims=True) + 1e-8)
        self._rng, k = jax.random.split(self._rng)
        actions = self._sample_bound(k, observation, g, B.binder_apply, self.binder)
        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0]), outputs)
        return self.p._output_transform(outputs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_10_task")
    p.add_argument("--params", required=True, help="runs/cf_bind_{on,off}.pkl")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--horizon", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    import torch
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    with open(args.params, "rb") as fh:
        params = pickle.load(fh)
    bound = BoundPolicy(policy, params)
    print(f"loaded {args.params}", flush=True)

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    rows = []
    for bf in bddls:
        instruction, objs, targets, distractors = parse_bddl(bf)
        if not targets or not distractors:
            continue
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        base = run_episode(policy, env, instruction, targets, distractors, horizon=args.horizon, device=dev)
        env.seed(args.seed)
        bnd = run_episode(bound, env, instruction, targets, distractors, horizon=args.horizon, device=dev)
        env.close()
        rec = {"scene": pathlib.Path(bf).stem[:34], "tgt": targets[0],
               "b_reach": base["reached_target"], "n_reach": bnd["reached_target"],
               "b_rt": base["reach_tgt_cm"], "b_rd": base["reach_dist_cm"],
               "n_rt": bnd["reach_tgt_cm"], "n_rd": bnd["reach_dist_cm"]}
        rows.append(rec)
        print(f"  {rec['scene']:36s} tgt={rec['tgt']:18s} | BASE reach_t/d={rec['b_rt']}/{rec['b_rd']} "
              f"reached={rec['b_reach']} || BOUND reach_t/d={rec['n_rt']}/{rec['n_rd']} reached={rec['n_reach']}",
              flush=True)
    n = len(rows)
    if n:
        br = sum(r["b_reach"] for r in rows); nr = sum(r["n_reach"] for r in rows)
        print(f"\n=== {args.params}  (N={n}) ===", flush=True)
        print(f"  BASELINE reached-target {br}/{n}={br/n*100:.0f}%", flush=True)
        print(f"  BOUND    reached-target {nr}/{n}={nr/n*100:.0f}%  (delta {(nr-br)/n*100:+.0f}pp)", flush=True)
    print("CF_BIND_EVAL_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
