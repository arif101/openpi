"""Stage B — the headline test: does object-relative conditioning make the
pixel-free head ROBUST to object-position perturbation, where pixel-based Pi0.5
collapses?

We run the head closed-loop on the PERTURBED init states (already collected),
driving it with an object-relative g-sequence taken from an UNPERTURBED success
of the same task. Because g is object-relative and the head reads the LIVE
(perturbed) object pose each step, the targets adapt to the perturbation for
free -- no scripted target_gen. We compare the head's success rate to raw Pi0.5
on the SAME perturbed scenes (the PHYS_OK fraction of the perturbed corpus).

The claim under test: head + object-relative target degrades gracefully under
perturbation (e.g. ~50% @10cm) while raw Pi0.5 drops to ~18%. In-distribution
Pi0.5 is better (94.5%); the win is robustness, not in-dist accuracy.

  python eval_stageb.py --ckpt ckpt/head_plain.pt --task-idx 3 \
      --pert-dir data/keystone/pert10 --ref-dir data/keystone/pert0 --replan 2
"""
from __future__ import annotations

import argparse
import glob
import os
import pathlib

import numpy as np
import torch

import rekey
from eval_a0 import load_head, rollout

torch.set_num_threads(min(8, os.cpu_count() or 8))

MAX_STEPS = {"libero_10": 520}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--task-idx", type=int, required=True)
    p.add_argument("--pert-dir", required=True, help="perturbed traces -> init states + Pi0.5 baseline")
    p.add_argument("--ref-dir", default="data/keystone/pert0", help="unperturbed successes -> g-sequences")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--replan", type=int, default=2)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--g-mode", choices=["step", "nearest"], default="nearest")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    net, H_ = load_head(args.ckpt)
    tag = "equiv" if net.equivariant else "plain"

    # reference object-relative g-sequences from unperturbed successes
    ref_files = sorted(glob.glob(str(pathlib.Path(args.ref_dir) /
                       f"PHYS_OK_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))
    ref_pool = []
    for rf in ref_files[:30]:
        o = rekey.build_pairs(rf, rekey.RekeyConfig(horizon=args.horizon))
        ref_pool.append((o["g_pos"], o["g_quat"], o["proprio"], o["target_name"]))
    if not ref_pool:
        raise FileNotFoundError(f"no unperturbed successes in {args.ref_dir}")

    # perturbed scenes: ALL episodes (OK+FAIL) so head rate is comparable to Pi0.5's
    pert_files = sorted(glob.glob(str(pathlib.Path(args.pert_dir) /
                        f"PHYS_*_{args.task_suite}_task{args.task_idx}_*baseline*.npz")))[: args.n]
    pi05_ok = sum("PHYS_OK_" in os.path.basename(f) for f in pert_files)

    bm = benchmark.get_benchmark_dict()[args.task_suite]()
    task = bm.get_task(args.task_idx)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(args.seed)
    max_steps = MAX_STEPS[args.task_suite]
    print(f"task {args.task_idx} [{tag}] pert={args.pert_dir.split('pert')[-1]} : "
          f"{len(pert_files)} perturbed scenes, {len(ref_pool)} ref skills", flush=True)

    head_ok = 0
    for i, pf in enumerate(pert_files):
        d = np.load(pf, allow_pickle=True)
        gp, gq, pr, tname = ref_pool[i % len(ref_pool)]
        ref = {"init_state_libero": d["init_state_libero"], "g_pos": gp, "g_quat": gq, "proprio": pr}
        ok = rollout(net, env, ref, tname, args.replan, max_steps, g_mode=args.g_mode)
        head_ok += ok
        print(f"  [{i+1}/{len(pert_files)}] {'OK' if ok else 'FAIL'} "
              f"(head {head_ok}/{i+1})", flush=True)
    n = len(pert_files)
    print(f"\nSTAGE B [{tag}] task {args.task_idx} pert={args.pert_dir.split('pert')[-1]}: "
          f"HEAD {head_ok}/{n} = {head_ok/n*100:.1f}%  vs  Pi0.5 {pi05_ok}/{n} = {pi05_ok/n*100:.1f}%  "
          f"(delta {((head_ok-pi05_ok)/n*100):+.1f}pp)", flush=True)


if __name__ == "__main__":
    main()
