"""Definitive baseline: reproduce the ORIGINAL state-machine motor (motor_head.pt, prop_dim=9: goal_rel(3)+proprio(5)+
held(1), plain cnn/prop/head) under MATCHED conditions (hide distractors, body_pos goals, held-out inits) so it is
apples-to-apples with the new no-state-machine selector. held = (gripper closed >8 steps AND lifted >2cm) -> switch goal
to container. This tells us the true target the learned-switch motor must hit.
"""
from __future__ import annotations
import argparse, glob, pathlib
import numpy as np, torch, torch.nn as nn
from cf_harness import parse_bddl, resolve_bodies, body_pos, _quat2axisangle


class WristMotorV0(nn.Module):
    def __init__(self, prop_dim=9, chunk=10, act=7):
        super().__init__(); self.chunk, self.act = chunk, act
        c = lambda i, o, s: nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.GroupNorm(min(8, o), o), nn.ReLU())
        self.cnn = nn.Sequential(c(3, 32, 2), c(32, 64, 2), c(64, 128, 2), c(128, 128, 2), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.prop = nn.Sequential(nn.Linear(prop_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, chunk * act))

    def forward(self, img, vec):
        return self.head(torch.cat([self.cnn(img), self.prop(vec)], -1)).view(-1, self.chunk, self.act)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="data/motor_head.pt"); p.add_argument("--bddl-dir", required=True)
    p.add_argument("--init-dir", default=""); p.add_argument("--container", default="basket")
    p.add_argument("--n", type=int, default=10); p.add_argument("--trials", type=int, default=3)
    p.add_argument("--horizon", type=int, default=280); p.add_argument("--replan", type=int, default=5)
    p.add_argument("--seed", type=int, default=5); p.add_argument("--img", type=int, default=128)
    p.add_argument("--init-start", type=int, default=20); p.add_argument("--res", type=int, default=256); p.add_argument("--hide", type=int, default=1)
    args = p.parse_args()
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.head, map_location=dev, weights_only=False)
    net = WristMotorV0(prop_dim=ck["prop_dim"]).to(dev); net.load_state_dict(ck["state"]); net.eval()
    vm, vs = ck["vm"], ck["vs"]
    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    succ = []; grasp = []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets: continue
        stem = pathlib.Path(bf).stem; graspables = [o for o in objs if args.container not in o]
        import re as _re
        m = _re.match(r"pick_up_the_(.+)$", stem); T = targets[0]
        if m:
            toks = m.group(1).split("_"); cut = len(toks)
            for s in ("between", "next", "on", "from", "in", "and"):
                if s in toks: cut = min(cut, toks.index(s))
            tc = "_".join(toks[:cut]); cand = next((o for o in graspables if tc and tc in o), None)
            if cand: T = cand
        inits = None
        if args.init_dir:
            fi = pathlib.Path(args.init_dir) / f"{stem}.pruned_init"
            if fi.exists():
                try: inits = np.asarray(torch.load(fi, weights_only=False))
                except Exception: inits = None
        s0 = args.init_start; nt = min(args.trials, len(inits) - s0) if inits is not None else args.trials
        for t in range(nt):
            ti = s0 + t
            env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=args.res, camera_widths=args.res, camera_depths=False)
            env.seed(args.seed + ti); env.reset(); sim = env.env.sim
            obs = env.set_init_state(inits[ti]) if inits is not None else env.reset()
            rb = resolve_bodies(sim, graspables + [args.container + "_1"]); cb = rb.get(args.container + "_1")
            if cb is None:
                cand = [b for b in (sim.model.body_id2name(i) for i in range(sim.model.nbody)) if b and args.container in b]
                cb = cand[0] if cand else None
            if rb.get(T) is None or cb is None: env.close(); succ.append(0); grasp.append(0); continue
            if args.hide:
                for o in graspables:
                    if o == T or rb.get(o) is None: continue
                    bid = sim.model.body_name2id(rb[o])
                    for g in range(sim.model.ngeom):
                        if sim.model.geom_bodyid[g] == bid: sim.model.geom_rgba[g, 3] = 0.0
            z0 = body_pos(sim, rb[T])[2]; lifted = 0.0; held = False; close_cnt = 0; chunk = None; ci = 0
            for step in range(args.horizon):
                ee = np.asarray(obs["robot0_eef_pos"], np.float32)
                active = body_pos(sim, cb) if held else body_pos(sim, rb[T])
                goal_rel = (active.astype(np.float32) - ee)
                if chunk is None or ci >= args.replan:
                    wr_raw = np.asarray(obs["robot0_eye_in_hand_image"])
                    wr = image_tools.resize_with_pad(np.ascontiguousarray(wr_raw[::-1, ::-1]), args.img, args.img).astype(np.uint8)
                    prop = np.concatenate((_quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
                                           np.asarray(obs["robot0_gripper_qpos"], np.float32))).astype(np.float32)
                    vec = ((np.concatenate([goal_rel, prop, [float(held)]]).astype(np.float32) - vm) / vs).astype(np.float32)
                    img = torch.tensor(np.transpose(wr.astype(np.float32) / 255.0, (2, 0, 1)))[None].to(dev)
                    with torch.no_grad():
                        chunk = net(img, torch.tensor(vec)[None].to(dev)).cpu().numpy()[0]
                    ci = 0
                a = chunk[ci]; ci += 1; grip = 1.0 if a[6] > 0 else -1.0
                close_cnt = close_cnt + 1 if grip > 0 else 0
                act = a[:7].copy(); act[6] = grip
                obs, _, done, _ = env.step(act.tolist())
                lifted = max(lifted, body_pos(sim, rb[T])[2] - z0)
                if (not held) and close_cnt > 8 and lifted > 0.02: held = True
                if done: break
            try: ok = bool(env.env._check_success())
            except Exception: ok = False
            succ.append(int(ok)); grasp.append(int(lifted > 0.03)); env.close()
        print(f"  {stem[:30]:32s} succ={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  grasp={np.mean(grasp):.2f}", flush=True)
    print(f"\n=== OLD-V0 state-machine motor [{pathlib.Path(args.head).name}] hide={args.hide} on {pathlib.Path(args.bddl_dir).name}: "
          f"succ={np.mean(succ):.3f} ({sum(succ)}/{len(succ)})  grasp_rate={np.mean(grasp):.3f} ===", flush=True)
    print("OLDV0_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
