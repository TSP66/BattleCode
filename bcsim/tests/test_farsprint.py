"""The far sprints (BC_FARSPRINT, cpp/bc_obs.hpp VecEnv::far_path): ids 51-74, one per 7x7 window
tile 4-6 steps away, walked by the shortest open path inside the window.

Random legal play on the official and generated maps, biased towards far sprints. Every turn, for
every far target:

  path     the simulator's path equals an independent breadth-first search over the dragon's own
           grid planes (kelp and portal sides block, bodies block, the target may be a head), with
           the same forward/left/right/back tie-break: same steps, ending on the target, at most 8 steps
  mask     a far sprint the mask allows has a path

and every far sprint actually played lands the dragon's head on its target's world tile, or kills
it only in a head-on there. Negative control: the same check with the targets mirrored must fail.

    python3 tests/test_farsprint.py [--steps 1500]
"""

from __future__ import annotations

import argparse
import ctypes
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("BCSIM_LIB", str(ROOT / "bcsim" / "libbcvec_s2_g15p.so"))
sys.path.insert(0, str(ROOT))
import bcsim  # noqa: E402

G = bcsim.GRID_SIDE
H = G // 2
EGO = [(0, -1), (1, 0), (0, 1), (-1, 0)]          # ego N (forward) E (right) S W: (dx, dy)
ORDER = [0, 3, 1, 2]                              # forward, left, right, back


def ego_to_world(f, ox, oy):
    return [(ox, oy), (-oy, ox), (-ox, -oy), (oy, -ox)][f]


def bfs(grid, tx, ty):
    """Shortest path in ego steps from (0, 0) to (tx, ty) over the window, or None."""
    occ = grid[[11, 12, 13, 14, 19]].max(0) > 0
    occ[H, H] = True
    heads = (grid[11] > 0) | (grid[13] > 0)
    frm = {(0, 0): None}
    q = [(0, 0)]
    for c in q:
        if (tx, ty) in frm:
            break
        ox, oy = c
        r, col = oy + H, ox + H
        for ed in ORDER:
            dx, dy = EGO[ed]
            n = (ox + dx, oy + dy)
            if not (-3 <= n[0] <= 3 and -3 <= n[1] <= 3) or n in frm:
                continue
            if grid[ed, r, col] or grid[4 + ed, r, col]:          # kelp / portal on that side
                continue
            nr, nc = n[1] + H, n[0] + H
            if occ[nr, nc] and not (n == (tx, ty) and heads[nr, nc]):
                continue
            frm[n] = (c, ed)
            if n != (tx, ty):
                q.append(n)
    if (tx, ty) not in frm:
        return None
    n_steps, c = 0, (tx, ty)
    while frm[c] is not None:
        c = frm[c][0]
        n_steps += 1
    if n_steps > 8:                                  # FAR_MAX_STEPS
        return None
    out, c = [], (tx, ty)
    while frm[c] is not None:
        c, ed = frm[c]
        out.append(ed)
    return out[::-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--envs", type=int, default=128)
    a = p.parse_args()
    first, targets = bcsim.BattlecodeVecEnv.far_targets()
    assert first == 51 and len(targets) == 24 and bcsim.N_ACTIONS == 75, (first, len(targets), bcsim.N_ACTIONS)
    assert len(set(targets)) == 24 and all(4 <= abs(x) + abs(y) <= 6 for x, y in targets)
    lib = bcsim.env._lib
    lib.bcv_far_path.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.bcv_dragon_head.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    files = sorted((ROOT.parent / "maps-train-official").glob("*.map")) + sorted((ROOT.parent / "maps-loong").glob("*.map"))[:60]
    texts = bcsim.load_maps([str(f) for f in files])
    env = bcsim.BattlecodeVecEnv(texts, num_envs=a.envs, num_threads=8, seed=7, sonar=True, grid=True)
    for i in range(a.envs):
        env.set_sonar2(i, (0, 1))
    rng = np.random.default_rng(0)
    h = ctypes.c_void_p(env._h)
    obs = env.reset()
    buf = np.zeros(16, np.int32)
    st = dict(paths=0, path_bad=0, mask_bad=0, mirror_bad=0, mirror_n=0, played=0, landed=0, headon=0, other=0)
    bad = []
    for step in range(a.steps):
        grid = env.grid
        mask = obs.mask.astype(bool)
        for e in range(a.envs):
            face = int(np.argmax(obs.scalar[e, 4:8]))
            for k, (tx, ty) in enumerate(targets):
                n = lib.bcv_far_path(h, e, k, buf.ctypes.data)
                ego = [(int(d) - face) & 3 for d in buf[:n]]
                ref = bfs(grid[e], tx, ty)
                st["paths"] += n > 0
                if (ref or []) != ego:
                    st["path_bad"] += 1
                    bad.append((step, e, k, ego, ref))
                if mask[e, first + k] and n == 0:
                    st["mask_bad"] += 1
                if k % 6 == 0:                          # control: mirrored target
                    mref = bfs(grid[e], -tx, ty)
                    st["mirror_n"] += 1
                    st["mirror_bad"] += (mref or []) != ego
        lg = np.where(mask, rng.gumbel(size=mask.shape), -np.inf)
        lg[:, first:first + 24] += 2.5
        lg[:, 48] -= 4
        act = lg.argmax(1).astype(np.int32)
        want = {}
        for e in np.flatnonzero(act >= first):
            did = int(obs.dragon_id[e])
            head = lib.bcv_dragon_head(h, int(e), did)
            # map width/height from the scalars (head_x = x / w): recover w from map_w (w / 64)
            w, hh = int(round(obs.scalar[e, 10] * 64)), int(round(obs.scalar[e, 11] * 64))
            face = int(np.argmax(obs.scalar[e, 4:8]))
            ox, oy = targets[act[e] - first]
            wx, wy = ego_to_world(face, ox, oy)
            want[e] = (did, ((head // w + wy) % hh) * w + (head % w + wx) % w)
        obs, _, eps = env.step(act)
        ended = set(eps.rows[:, 0].tolist()) if len(eps.rows) else set()
        for e, (did, tgt) in want.items():
            if e in ended:
                continue
            st["played"] += 1
            dead = {int(r[0]): int(r[1]) for r in env.last_deaths(int(e))}
            if did in dead:
                if dead[did] == ord("H"):
                    st["headon"] += 1
                else:
                    st["other"] += 1
                    bad.append(("died", step, int(e), did, chr(dead[did])))
            elif lib.bcv_dragon_head(h, int(e), did) == tgt:
                st["landed"] += 1
            else:
                st["other"] += 1
                bad.append(("missed", step, int(e), did))
    print(st)
    fail = st["path_bad"] or st["mask_bad"] or st["other"] or st["mirror_bad"] == 0 or st["landed"] == 0
    if bad:
        print("e.g.", bad[:5])
    print("FAIL" if fail else "OK")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
