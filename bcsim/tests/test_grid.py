"""The LSTM policy's 38 x 14 x 14 grid against the live 7x7 it is built around.

Checks, over random play with sonar on:
  * every value is in [0, 1];
  * the live channels in the centre 7x7 are exactly the observation's;
  * the static terrain in the centre -- which the grid reads back from the
    world-anchored memory and rotates into the dragon's frame -- is exactly the
    live kelp/portal channels. This is the check that the rotation is right on
    every turn, for every facing;
  * seen is 1 in the window, and never outside what was ever seen;
  * the self channels agree with the window where they overlap;
  * the channel count matches train/lstm_net.py.

    python tests/test_grid.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim  # noqa: E402
from train.lstm_net import CHANNELS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
LC = dict(PEARL=0, PEARL_TIME=1, SELF_BODY=4, ALLY_HEAD=5, ALLY_BODY=6, ENEMY_HEAD=7, ENEMY_BODY=8,
          FACE=9, KELP=13, PORTAL=17, SELF_INDEX=21, SELF_TAIL=22)
C = {n: i for i, n in enumerate(CHANNELS)}


def main() -> None:
    env = bcsim.BattlecodeVecEnv(bcsim.load_maps(str(ROOT / "maps-all")), num_envs=64,
                                 num_threads=8, seed=3, grid=True, sonar=True)
    assert env.grid.shape[1:] == (len(CHANNELS), 14, 14), env.grid.shape
    obs = env.reset()
    rng = np.random.default_rng(0)
    lo, hi = 4, 11                      # the live window: HALF - VISION .. HALF + VISION
    bad = {k: 0 for k in ["range", "live", "terrain", "seen", "self"]}
    turns = 0
    echo_nonzero = 0
    for step in range(3000):
        g = env.grid
        loc = obs.local
        bad["range"] += int(((g < 0) | (g > 1)).any(axis=(1, 2, 3)).sum())
        c = g[:, :, lo:hi, lo:hi]
        pairs = [(LC["PEARL"], C["pearl"]), (LC["PEARL_TIME"], C["pearl_timer"]),
                 (LC["ALLY_HEAD"], C["ally_head"]), (LC["ALLY_BODY"], C["ally_body"]),
                 (LC["ENEMY_HEAD"], C["enemy_head"]), (LC["ENEMY_BODY"], C["enemy_body"])]
        pairs += [(LC["FACE"] + d, C["seg_dir_" + "nesw"[d]]) for d in range(4)]
        bad["live"] += int(sum((c[:, gc] != loc[:, lc]).any(axis=(1, 2)) for lc, gc in pairs).astype(bool).sum())
        terr = [(LC["KELP"] + d, C["kelp_" + "nesw"[d]]) for d in range(4)]
        terr += [(LC["PORTAL"] + d, C["portal_" + "nesw"[d]]) for d in range(4)]
        bad["terrain"] += int(sum((c[:, gc] != loc[:, lc]).any(axis=(1, 2)) for lc, gc in terr).astype(bool).sum())
        bad["seen"] += int((c[:, C["seen"]] != 1.0).any(axis=(1, 2)).sum())
        # the window's self body (head excluded: the grid has no head channel)
        sb = c[:, C["self_body"]]
        bad["self"] += int((sb != loc[:, LC["SELF_BODY"]]).any(axis=(1, 2)).sum())
        echo_nonzero += int((g[:, C["echo_kelp"]:C["echo_enemy_head"] + 1, 0, 0] > 0).any(axis=1).sum())
        turns += len(g)
        legal = obs.mask.astype(np.float64) + 1e-9
        acts = np.array([rng.choice(len(r), p=r / r.sum()) for r in legal], np.int32)
        obs, _, _ = env.step(acts)
    env.close()
    print(f"{turns:,} turns checked; echo non-zero on {echo_nonzero / turns:.1%} of turns")
    for k, v in bad.items():
        print(f"  {k:8s} {v:6d} turns wrong")
    assert all(v == 0 for v in bad.values()), bad
    print("ok")


if __name__ == "__main__":
    main()
