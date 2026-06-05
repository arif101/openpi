"""CFG amplification probe: does turning up the gain on pi0.5's language condition redirect the
grasp to the named target T (and does the motor survive)? Sizes whether the existing weak
language->target circuit is salvageable by gain alone, or whether we need a new architecture.

Also = the CAG inference-time baseline (classifier-free language guidance).

Per captured object scene: obs_cond = instr names T; obs_uncond = empty prompt. Sweep w; for each w
sample with sample_actions_cfg; report action direction toward T vs memorized M, and action norm
(blow-up = motor wrecked).
"""
from __future__ import annotations

import argparse
import glob
import pathlib

import jax
import jax.numpy as jnp
import numpy as np

from cf_harness import parse_bddl, resolve_bodies, body_pos, build_obs
from cf_bind_diag import name_of


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", default="pi05_libero")
    p.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_libero")
    p.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--ws", default="1,2,3,5,8")
    args = p.parse_args()
    import torch
    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config
    from openpi.shared import download
    from openpi.models import model as _model
    from libero.libero.envs import OffScreenRenderEnv
    download.maybe_download(args.checkpoint + "/assets"); download.maybe_download(args.checkpoint + "/params")
    policy = _policy_config.create_trained_policy(_config.get_config(args.config_name), args.checkpoint)
    model = policy._model
    ws = [float(x) for x in args.ws.split(",")]
    noise = jax.random.normal(jax.random.key(0), (1, 10, 32))

    def to_obs(obs_dict):
        inp = policy._input_transform(jax.tree.map(lambda x: x, obs_dict))
        inp = jax.tree.map(lambda x: jnp.asarray(x)[None], inp)
        return _model.Observation.from_dict(inp)

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    agg = {w: {"toT": [], "toM": [], "norm": []} for w in ws}
    for bf in bddls:
        instr_T, objs, targets, distractors = parse_bddl(bf)
        if not targets or not distractors:
            continue
        T, M = targets[0], distractors[0]
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=256, camera_widths=256)
        env.seed(7); env.reset(); obs = env.reset()
        sim = env.env.sim; rb = resolve_bodies(sim, [T, M])
        ee = np.asarray(obs["robot0_eef_pos"], np.float64)
        uT = (body_pos(sim, rb[T]) - ee); uT /= np.linalg.norm(uT) + 1e-8
        uM = (body_pos(sim, rb[M]) - ee); uM /= np.linalg.norm(uM) + 1e-8
        img, wr = np.asarray(obs["agentview_image"]), np.asarray(obs["robot0_eye_in_hand_image"])
        eq, gq = obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]
        oc = to_obs(build_obs(img, wr, obs["robot0_eef_pos"], eq, gq, instr_T))
        ou = to_obs(build_obs(img, wr, obs["robot0_eef_pos"], eq, gq, ""))     # empty prompt = unconditional
        env.close()
        line = f"  {pathlib.Path(bf).stem[:26]:28s} T={name_of(T)[:12]:12s}"
        for w in ws:
            a = np.asarray(model.sample_actions_cfg(jax.random.key(0), oc, ou, w=w, noise=noise))[0]
            d = a[:, 0:3].sum(0); nrm = float(np.linalg.norm(d)); d = d / (nrm + 1e-8)
            agg[w]["toT"].append(float(d @ uT)); agg[w]["toM"].append(float(d @ uM)); agg[w]["norm"].append(nrm)
            line += f" | w{w:g}:T{float(d@uT):+.2f}/M{float(d@uM):+.2f}"
        print(line, flush=True)
    print(f"\n=== CFG AMPLIFICATION SWEEP (N={len(agg[ws[0]]['toT'])}) ===", flush=True)
    print(f"  {'w':>4} {'dir->T':>8} {'dir->M':>8} {'win(T>M)':>9} {'|disp|':>8}", flush=True)
    for w in ws:
        toT = np.array(agg[w]["toT"]); toM = np.array(agg[w]["toM"])
        win = float(np.mean(toT > toM))
        print(f"  {w:>4g} {toT.mean():>+8.2f} {toM.mean():>+8.2f} {win:>9.0%} {np.mean(agg[w]['norm']):>8.2f}", flush=True)
    print("  (salvageable if dir->T overtakes dir->M as w grows WITHOUT |disp| blowing up)", flush=True)
    print("CF_BIND_CFG_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
