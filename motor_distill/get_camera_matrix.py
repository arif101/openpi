"""Option A step 1: extract the agentview (base) camera world->pixel matrix for
task 3, and VERIFY it by overlaying projected object positions on real corpus
frames. Camera/flip conventions are the classic footgun -> eyeball before trust.

Saves data/keystone/camera_t3.npz: M (4x4 world->pixel), H, W, z_table.
Writes overlay PNGs to data/keystone/overlay_*.png (red = M as-is, green = v-flipped).
"""
from __future__ import annotations

import glob
import pathlib

import numpy as np


def project(M, X):                       # X [...,3] world -> pixel (u,v)
    Xh = np.concatenate([X, np.ones(X.shape[:-1] + (1,))], -1)
    p = Xh @ M.T                          # [...,4]
    return p[..., :2] / p[..., 2:3]


def stamp(img, px, py, color, r=4):
    H, W = img.shape[:2]
    x, y = int(round(px)), int(round(py))
    img[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1)] = color


def main():
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils import camera_utils
    import imageio

    H = W = 256
    bm = benchmark.get_benchmark_dict()["libero_10"]()
    task = bm.get_task(3)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=H, camera_widths=W)
    env.reset()
    sim = env.env.sim
    M = camera_utils.get_camera_transform_matrix(sim, "agentview", H, W)
    print("M=\n", np.asarray(M).round(3), flush=True)

    out = pathlib.Path("data/keystone"); out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob("data/keystone/pert0/PHYS_OK_libero_10_task3_*baseline*.npz"))[:3]
    zt = []
    for i, f in enumerate(files):
        d = np.load(f, allow_pickle=True)
        imgs, it, op = d["image"], d["image_t"], d["object_pos"]
        t0 = int(np.clip(it[0], 0, op.shape[0] - 1))
        Xobj = op[t0, 0].astype(np.float64)          # target bowl (index 0)
        zt.append(Xobj[2])
        u, v = project(np.asarray(M), Xobj)
        img = imgs[0].copy()
        stamp(img, u, v, [255, 0, 0])                # M as-is (red)
        stamp(img, u, H - 1 - v, [0, 255, 0])        # v-flipped (green)
        imageio.imwrite(out / f"overlay_{i}.png", img)
        print(f"frame {i}: obj={Xobj.round(3)} -> pixel as-is=({u:.1f},{v:.1f}) vflip=({u:.1f},{H-1-v:.1f})", flush=True)

    z_table = float(np.mean(zt))
    np.savez(out / "camera_t3.npz", M=np.asarray(M), H=H, W=W, z_table=z_table)
    print(f"saved camera_t3.npz z_table={z_table:.3f}", flush=True)
    print("CAM_EXIT=0", flush=True)


if __name__ == "__main__":
    main()
