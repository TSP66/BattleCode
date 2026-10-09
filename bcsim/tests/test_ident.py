"""The identity planes (grid_cfg BIRTH..ID43, 2026-10-03): birth round / 500, sin(id / 7), sin(id / 43).

    python tests/test_ident.py [--lib NEW.so] [--old OLD.so] [--steps 400]

Checks, on random play over the gate maps:
  * planes 54-56 are constant, and ID7 / ID43 match obs.dragon_id;
  * BIRTH is fixed for a dragon's life: 0 if it spawned with the map (the queens always) and, for a
    split child, the round it split off: never after the first round it is seen acting, at most one before;
  * negative control: with --old (a 54-channel build of the same sources before the change), the
    first 54 planes are bit-identical step for step -- the CNN's input did not move.
Each library runs in its own process (bcsim binds one library per process).
"""

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def play(lib: str, steps: int, out: str) -> None:
    os.environ["BCSIM_LIB"] = lib
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import bcsim
    maps = bcsim.load_maps([str(p) for p in sorted((ROOT.parent / "maps-gate-1001").glob("*.map"))])
    N = 16
    env = bcsim.BattlecodeVecEnv(maps, num_envs=N, num_threads=4, seed=7, grid=True, sonar=True)
    for i in range(N):
        env.set_opponent(i, team=-1, bot=0, map_index=i % len(maps))
        env.set_sonar2(i, (0, 1))
    obs = env.reset()
    rng = np.random.default_rng(0)
    grids, ids, rounds, uids = [], [], [], []
    for _ in range(steps):
        grids.append(env.grid.copy())
        ids.append(obs.dragon_id.copy())
        rounds.append(obs.round.copy())
        uids.append(obs.uid.copy())
        # uniform over the legal moves (the same in both builds: actions never read the grid)
        p = obs.mask.astype(np.float64) + 1e-12
        a = np.array([rng.choice(len(r), p=r / r.sum()) for r in p], np.int32)
        obs = env.step(a)[0]
    np.savez_compressed(out, grid=np.stack(grids), id=np.stack(ids), round=np.stack(rounds), uid=np.stack(uids))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lib", default=str(ROOT / "bcsim" / "libbcvec_s2_g15p.so"))
    p.add_argument("--old", default="")
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--play", nargs=2, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.play:
        play(a.play[0], a.steps, a.play[1])
        return
    import numpy as np
    tmp = pathlib.Path(os.environ.get("TMPDIR", "/tmp"))
    runs = {"new": a.lib} | ({"old": a.old} if a.old else {})
    for k, lib in runs.items():
        subprocess.run([sys.executable, __file__, "--steps", str(a.steps), "--play", lib,
                        str(tmp / f"ident_{k}.npz")], check=True)
    d = np.load(tmp / "ident_new.npz")
    g, ids, rnd, uid = d["grid"], d["id"], d["round"], d["uid"]
    assert g.shape[2] == 57, g.shape
    bad = 0
    pl = g[:, :, 54:57]
    flat = pl.reshape(*pl.shape[:3], -1)
    bad += int((flat.max(-1) != flat.min(-1)).sum())
    print(f"  constant planes: {bad} non-constant")
    c = g.shape[-1] // 2
    v = pl[..., c, c]                                                # (steps, envs, 3)
    e7 = np.abs(v[..., 1] - np.sin(ids / 7.0)).max()
    e43 = np.abs(v[..., 2] - np.sin(ids / 43.0)).max()
    print(f"  sin(id/7) max err {e7:.2e}, sin(id/43) max err {e43:.2e}")
    first: dict[int, int] = {}
    births: dict[int, float] = {}
    for s in range(len(uid)):
        for e in range(uid.shape[1]):
            u = int(uid[s, e])
            first.setdefault(u, int(rnd[s, e]))
            b = float(v[s, e, 0]) * 500.0
            if births.setdefault(u, b) != b:
                bad += 1
                print(f"  uid {u}: birth changed {births[u]} -> {b}")
    spawned = [u for u, b in births.items() if b == 0]
    kids = [(u, b, first[u]) for u, b in births.items() if b > 0]
    off = [(u, b, f) for u, b, f in kids if not (0 <= f - round(b) <= 1)]
    print(f"  {len(births)} dragons: {len(spawned)} with birth 0, {len(kids)} split children; "
          f"first seen - birth: {sorted({f - round(b) for _, b, f in kids})}; {len(off)} off")
    # a child is born after round 0 unless it split in round 0: its id is past the spawned ones
    late = [u for u, b, f in kids if (u & 0xFFF) < 2]
    bad += len(off) + len(late) + (e7 > 1e-6) + (e43 > 1e-6)
    if a.old:
        o = np.load(tmp / "ident_old.npz")["grid"]
        assert o.shape[2] == 54, o.shape
        same = np.array_equal(o, g[:, :, :54])
        print(f"  negative control: planes 0-53 identical to the old build: {same}")
        bad += not same
    print("PASS" if bad == 0 else f"FAIL ({bad})")
    sys.exit(int(bad != 0))


if __name__ == "__main__":
    main()
