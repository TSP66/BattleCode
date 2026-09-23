"""bc_memory.hpp must equal train/clone_features.py's MemoryTracker, exactly.

The clones' weights were fitted through the Python tracker, so it is the
authority: if the simulator's memory drifts from it by even a cell, every
feature after that turn is wrong for the rest of that dragon's life, and a net
trained on one is fed inputs it never saw by the other.

A single-turn check cannot see that class of bug, so this drives whole dragon
lifetimes and compares every turn:

    BCSIM_LIB=/path/to/libbcvec.so python -m tests.parity_memory

Run it against any build that emits 708 scalars, before that build trains
anything.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                        # noqa: E402
from train.clone_features import MemoryTracker      # noqa: E402

N_BASE, N_MEM, N_FAR = 14, 676, 18


def main() -> None:
    # The memory block must still be exactly where it was. The row is allowed to
    # be longer -- features are appended on the end, the sonar echoes were the
    # first -- but everything in [0, 708) has to stay put, which is what lets the
    # frozen league keep playing. Anything shorter is not a memory build.
    if bcsim.N_SCALARS < N_BASE + N_MEM + N_FAR:
        raise SystemExit(f"this build gives {bcsim.N_SCALARS} scalars, fewer than the "
                         "708 the memory features need -- point BCSIM_LIB at a "
                         "memory build")
    root = pathlib.Path(__file__).resolve().parents[2]
    names = ["default_small", "devil", "queen_of_spades", "schooltime", "big_empty", "trophy"]
    texts = [(root / "maps" / f"{n}.map").read_text() for n in names]

    rng = np.random.default_rng(0)
    worst_mem = worst_far = 0.0
    rows_checked = 0

    for name, txt in zip(names, texts):
        env = bcsim.BattlecodeVecEnv([txt], num_envs=16, num_threads=2, seed=7)
        obs = env.reset()
        tr = MemoryTracker()
        prev_round = np.array(obs.round, dtype=np.int64).copy()
        idx = np.arange(len(obs.round))
        for step in range(1500):
            # a finished game restarts the env, and every dragon in it is new
            cur = np.asarray(obs.round, dtype=np.int64)
            for e in np.flatnonzero(cur < prev_round):
                tr.forget(lambda k, e=int(e): k[0] == e)
            prev_round = cur.copy()

            keys = [(int(e), int(d)) for e, d in zip(idx, obs.dragon_id)]
            got = tr.step(keys, obs.local, obs.scalar[:, :N_BASE], obs.round,
                          want=["mem", "memfar"])
            want = np.concatenate([got["mem"], got["memfar"]], 1).astype(np.float32)
            # exactly the memory block; anything appended after it is not ours
            have = obs.scalar[:, N_BASE:N_BASE + N_MEM + N_FAR]
            d_mem = np.abs(have[:, :N_MEM] - want[:, :N_MEM]).max()
            d_far = np.abs(have[:, N_MEM:] - want[:, N_MEM:]).max()
            worst_mem = max(worst_mem, float(d_mem))
            worst_far = max(worst_far, float(d_far))
            rows_checked += len(keys)
            if d_mem > 0 or d_far > 0:
                bad = int(np.argmax(np.abs(have - want).max(1)))
                col = int(np.argmax(np.abs(have[bad] - want[bad])))
                where = "mem" if col < N_MEM else "memfar"
                raise SystemExit(
                    f"MISMATCH on {name} at step {step}, env {bad}, {where} index "
                    f"{col if col < N_MEM else col - N_MEM}: "
                    f"sim {have[bad, col]!r} vs tracker {want[bad, col]!r}")

            m = obs.mask.astype(np.float64)
            s = m.sum(1, keepdims=True)
            p = np.where(s > 0, m / np.maximum(s, 1e-9), 0.0)
            act = np.array([rng.choice(p.shape[1], p=pi) if pi.sum() > 0.5 else 0
                            for pi in p], dtype=np.int64)
            obs, _, _ = env.step(act)
        env.close()
        print(f"  {name:<18} ok", flush=True)

    print(f"\nparity holds: {rows_checked:,} turns, max |diff| mem {worst_mem:g}, "
          f"memfar {worst_far:g}")


if __name__ == "__main__":
    main()
