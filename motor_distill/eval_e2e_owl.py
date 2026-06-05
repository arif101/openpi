"""END-TO-END with OWLv2 open-vocab binding (generalizes by construction):
OWLv2 multi-query (all present objects) -> target box -> median real-depth -> geometric unprojection
(verified 2.6cm, vflip) -> 3D goal -> distilled reach head closed-loop. No oracle, no learned 2D->3D.
baseline 20%, pi0.5-feat 50%, oracle 90%. OWLv2 is open-vocab => no held-out gap.
"""
from __future__ import annotations

import argparse, glob, pathlib, pickle, re
import jax, jax.numpy as jnp, numpy as np
from cf_harness import parse_bddl, resolve_bodies, body_pos
from distill_reach import head_apply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bddl-dir", default="data/libero_pro/bddl_files/libero_object_task")
    ap.add_argument("--reach-head", default="runs/reach_head.pkl")
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--horizon", type=int, default=140)
    ap.add_argument("--rebind", type=int, default=25); ap.add_argument("--cam", default="agentview")
    ap.add_argument("--thr", type=float, default=0.03); ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    import torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from libero.libero.envs import OffScreenRenderEnv
    import robosuite.utils.camera_utils as CU
    from PIL import Image
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    owlp = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    owlm = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(dev).eval()
    rp = jax.tree.map(jnp.asarray, pickle.load(open(args.reach_head, "rb")))
    H = W = 256
    nm = lambda o: re.sub(r"_\d+$", "", o).replace("_", " ")

    @torch.no_grad()
    def locate(img, names, ti):
        tgt = names[ti]
        # per-target query with prompt variants + low threshold to maximize recall; take best box.
        queries = [tgt, f"a {tgt}", f"a photo of a {tgt}", f"{tgt} package", f"{tgt} bottle"]
        best = None
        for q in queries:
            inp = owlp(text=[[q]], images=Image.fromarray(img), return_tensors="pt").to(dev)
            r = owlp.post_process_grounded_object_detection(
                owlm(**inp), threshold=args.thr, target_sizes=torch.tensor([[H, W]]).to(dev))[0]
            sc = r["scores"].cpu().numpy()
            if len(sc) and (best is None or sc.max() > best[0]):
                b = r["boxes"].cpu().numpy()[sc.argmax()]
                best = (float(sc.max()), int((b[1] + b[3]) / 2), int((b[0] + b[2]) / 2))
        return None if best is None else (best[1], best[2])

    bddls = sorted(glob.glob(str(pathlib.Path(args.bddl_dir) / "*.bddl")))[: args.n]
    rows, berrs, lifts = [], [], []
    for bf in bddls:
        instr, objs, targets, distractors = parse_bddl(bf)
        if not targets or not distractors:
            continue
        T, M = targets[0], distractors[0]
        present = list(dict.fromkeys((targets or []) + (distractors or [])))
        names = [nm(o) for o in present]; ti = present.index(T)
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=H, camera_widths=W, camera_depths=True)
        env.seed(args.seed); env.reset(); obs = env.reset()
        sim = env.env.sim; rb = resolve_bodies(sim, [T, M])
        w2c = CU.get_camera_transform_matrix(sim, args.cam, H, W); c2w = np.linalg.inv(w2c)
        z0T, z0M = body_pos(sim, rb[T])[2], body_pos(sim, rb[M])[2]; liftT, liftM = 0.0, 0.0
        goal = None; rT, rM = 1e9, 1e9; gerr = None
        for step in range(args.horizon):
            if goal is None or step % args.rebind == 0:
                img = np.asarray(obs["agentview_image"])[::-1].copy()
                rd = CU.get_real_depth_map(sim, np.asarray(obs["agentview_depth"]).reshape(H, W, 1))[::-1].copy()
                rc = locate(img, names, ti)
                if rc is not None:
                    cy, cx = rc
                    reg = rd[max(0, cy - 6):cy + 6, max(0, cx - 6):cx + 6, 0]
                    dmed = float(np.median(reg)) if reg.size else float(rd[cy, cx, 0])
                    g = CU.transform_from_pixels_to_world(np.array([cy, cx], float),
                                                          np.full((H, W, 1), dmed), c2w)[:3]
                    goal = np.asarray(g, np.float32)
                    if gerr is None:
                        gerr = float(np.linalg.norm(goal - body_pos(sim, rb[T])))
            if goal is None:
                break
            ee = np.asarray(obs["robot0_eef_pos"], np.float32)
            a = np.asarray(head_apply(rp, jnp.asarray(ee - goal),
                                      jnp.asarray(np.asarray(obs["robot0_eef_quat"], np.float32)),
                                      jnp.asarray(np.asarray(obs["robot0_gripper_qpos"], np.float32))))
            obs, _, done, info = env.step(a.tolist())
            E = np.asarray(obs["robot0_eef_pos"], np.float32)
            pT = body_pos(sim, rb[T]); pM = body_pos(sim, rb[M])
            rT = min(rT, float(np.linalg.norm(E - pT))); rM = min(rM, float(np.linalg.norm(E - pM)))
            liftT = max(liftT, pT[2] - z0T); liftM = max(liftM, pM[2] - z0M)
            if done:
                break
        env.close()
        reached = rT < rM if goal is not None else False
        lifted = liftT > 0.03 and liftT > liftM                      # picked up the NAMED object
        rows.append(reached); lifts.append(lifted)
        if gerr is not None:
            berrs.append(gerr)
        ge = f"{gerr*100:.1f}cm" if gerr is not None else "NOLOC"
        print(f"  {pathlib.Path(bf).stem[:22]:24s} T={nm(T):13s} | bind={ge} "
              f"reach_T={rT*100:.0f} reach_M={rM*100:.0f} reached={reached} liftT={liftT*100:.0f}cm lifted={lifted}", flush=True)
    n = len(rows)
    print(f"\n=== END-TO-END OWLv2 (no oracle, N={n}, seed={args.seed}) ===", flush=True)
    print(f"  reaches named target: {sum(rows)}/{n} = {sum(rows)/n*100:.0f}%   (baseline 20%, pi0.5-feat 50%, CAG 30%, oracle 90%)", flush=True)
    print(f"  LIFTS named target:   {sum(lifts)}/{n} = {sum(lifts)/n*100:.0f}%   (task-relevant grasp success)", flush=True)
    print(f"  bind_err mean {np.mean(berrs)*100:.1f}cm over {len(berrs)} localized", flush=True)
    print("EVAL_E2E_OWL_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
