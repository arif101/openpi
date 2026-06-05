"""CAG baseline (classifier-free language guidance) on the captured object scenes, for the head-to-head
table. CAGPolicy.infer == stock infer but samples with sample_actions_cfg (cond=instruction, uncond=empty
prompt, guidance w). Measures reached_target (same metric as our factored pipeline). Our must-beat baseline.
"""
from __future__ import annotations

import argparse, glob, pathlib
import jax, jax.numpy as jnp, numpy as np
from cf_harness import parse_bddl, run_episode, build_obs


class CAGPolicy:
    def __init__(self, policy, w):
        from openpi.models import model as _model
        from openpi.shared import nnx_utils
        self._m = _model; self.p = policy; self.model = policy._model; self.w = w
        self._rng = jax.random.key(0)
        # jit the CFG sampler (else eager double-forward-pass per step is ~10-50x slower)
        self._cfg = nnx_utils.module_jit(self.model.sample_actions_cfg, static_argnames=("w", "num_steps"))

    def _obs(self, obs_dict):
        inp = self.p._input_transform(jax.tree.map(lambda x: x, obs_dict))
        inp = jax.tree.map(lambda x: jnp.asarray(x)[None], inp)
        return self._m.Observation.from_dict(inp)

    def infer(self, obs):
        oc = self._obs(obs)
        ou = self._obs({**obs, "prompt": ""})            # unconditional = empty prompt
        self._rng, k = jax.random.split(self._rng)
        a = np.asarray(self._cfg(k, oc, ou, w=self.w)[0])
        return self.p._output_transform({"state": np.zeros(8, np.float32), "actions": a})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="pi05_libero")
    ap.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--w", type=float, default=3.0); ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    import torch
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from libero.libero.envs import OffScreenRenderEnv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    cag = CAGPolicy(policy, args.w)
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    rows = []
    for bf in bddls:
        instruction, objs, targets, distractors = parse_bddl(bf)
        if not targets or not distractors:
            continue
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        r = run_episode(cag, env, instruction, targets, distractors, horizon=args.horizon, device=dev)
        env.close()
        rows.append(r["reached_target"])
        print(f"  {pathlib.Path(bf).stem[:24]:26s} reach_t={r['reach_tgt_cm']} reach_d={r['reach_dist_cm']} "
              f"reached={r['reached_target']}", flush=True)
    n = len(rows)
    print(f"\n=== CAG (w={args.w}, N={n}) reaches named target: {sum(rows)}/{n} = {sum(rows)/n*100:.0f}% ===", flush=True)
    print("EVAL_CAG_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
