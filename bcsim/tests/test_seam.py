"""Simulator vs engine where the torus wraps.

test_vecenv.py plays random moves, which seldom produce a collision across the
map edge. Here the moves are steered toward the nearest map edge and toward
enemy heads in view, so dragons meet across the seam, and every turn is
checked in lockstep against the engine (identical blocks, so identical deaths,
lengths and results). Coverage is reported: seam turns with another dragon in
view, and deaths on those turns.

    python tests/test_seam.py
"""

from __future__ import annotations

import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bcsim                                    # noqa: E402
from blockparse import Block                    # noqa: E402
from oracle import OracleGame                   # noqa: E402
from test_vecenv import VLIB, vec_block, check_observation   # noqa: E402
import ctypes                                   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root

DIRS = "NESW"
STEP = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}


def wrapped(d: int, n: int) -> int:
    d %= n
    return d - n if d > n // 2 else d


def play(map_text: str, w: int, h: int, seed: int) -> dict:
    rng = random.Random(seed)
    env = bcsim.BattlecodeVecEnv([map_text], num_envs=1, num_threads=1, seed=0,
                                 egocentric=False, random_pearl_seed=False)
    env.reset()
    died = bcsim.REWARD_COMPS.index("died")
    stats = {"turns": 0, "seam_turns": 0, "seam_deaths": 0, "deaths": 0}

    def bridge(dragon_id: int, block: str) -> str:
        if VLIB.bcv_acting_dragon(ctypes.c_void_p(env._h), 0) != dragon_id:
            raise AssertionError("turn order")
        ours = vec_block(env, 0)
        if ours != block:
            raise AssertionError(f"block mismatch on dragon {dragon_id}\n--- engine\n{block}"
                                 f"\n--- ours\n{ours}")
        check_observation(env, block)
        stats["turns"] += 1
        b = Block(block)
        hx, hy = b.head
        others = [(x, y) for (x, y), (_t, did, _f, head) in b.bodies.items() if did != dragon_id]
        seam = (hx < 3 or hx > w - 4 or hy < 3 or hy > h - 4) and bool(others)
        stats["seam_turns"] += seam

        safe = b.safe_dirs() or list(DIRS)
        # steer: toward an enemy head in view, else toward the nearest edge
        heads = [(x, y) for (x, y), (t, did, _f, head) in b.bodies.items()
                 if head and did != dragon_id]
        if heads and rng.random() < 0.7:
            tx, ty = min(heads, key=lambda p: abs(wrapped(p[0] - hx, w)) + abs(wrapped(p[1] - hy, h)))
            dx, dy = wrapped(tx - hx, w), wrapped(ty - hy, h)
        else:
            dx = -1 if hx < w / 2 else 1
            dy = -1 if hy < h / 2 else 1
        prefer = [d for d in safe if STEP[d][0] * dx > 0 or STEP[d][1] * dy > 0]
        pick = rng.choice(prefer if prefer and rng.random() < 0.8 else safe)
        kind, n_steps = np.zeros(1, np.int8), np.ones(1, np.int8)
        dirs = np.zeros((1, 8), np.int8)
        dirs[0, 0] = DIRS.index(pick)
        _, cl, _ = env.step_raw(kind, n_steps, dirs, np.zeros(1, np.int16),
                                np.zeros(1, np.int8), np.zeros(1, np.uint32))
        n_died = int(cl.comps[:, died].sum()) if len(cl.uid) else 0
        stats["deaths"] += n_died
        if seam:
            stats["seam_deaths"] += n_died
        return f"MOVE {pick}\nENDTURN\n"

    game = OracleGame(map_text, bridge)
    game.run()
    env.close()
    return stats


def main() -> int:
    total = {"turns": 0, "seam_turns": 0, "seam_deaths": 0, "deaths": 0}
    for path in sorted(ROOT / "maps".glob("*.map")):
        text = path.read_text()
        first = next(l for l in text.splitlines() if l.startswith("MAP "))
        w, h = (int(v) for v in first.split()[1:3])
        for seed in range(6):
            s = play(text, w, h, seed)
            for k in total:
                total[k] += s[k]
        print(f"  {path.stem:32s} ok")
    print(f"engine and simulator agree on {total['turns']} turns; "
          f"{total['seam_turns']} had the window wrapping with another dragon in view, "
          f"with {total['seam_deaths']} deaths on those turns ({total['deaths']} deaths in all)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
