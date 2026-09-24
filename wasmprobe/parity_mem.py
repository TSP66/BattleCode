"""Checks the C++ bot's remembered inputs are the ones the clone was trained on.

`mem` (676) and `memfar` (18) are computed from a dragon's own earlier turns
(bcsim/train/clone_features.py), so unlike the window they cannot be checked
against a single turn's protocol block: they depend on the whole sequence. This
replays the sequence instead. Each -DBC_DUMP file is one dragon's turns in
order, so its observations are fed one at a time to clone_features.MemoryTracker
-- the reference the checkpoint's win rates were measured through -- and every
extra scalar the bot appended must equal what the tracker produces.

It needs no simulator: parity_obs.py already establishes that the window and
the 14 scalars the bot builds are the simulator's, and this takes those as
given and checks what the bot then remembers of them.

    python parity_mem.py dump.txt [dump2.txt ...]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT / "bcsim"))

import bcsim                                        # noqa: E402
from train.clone_features import MemoryTracker      # noqa: E402

# S is the BASE scalar count, which is what the dump puts before the remembered
# block. bcsim.N_SCALARS is the full row now that cpp/bc_memory.hpp appends mem
# and memfar to it, so using that here would make `extra` come out zero.
C, S, A = bcsim.N_CHANNELS, len(bcsim.SCALARS), bcsim.N_ACTIONS
N_MEM, N_FAR = 4 * 13 * 13, 18
# The bot appends the 5 sonar echo counts after the remembered features, so the
# dump's trailing block is 694 or 699 wide. The echoes come straight off the
# ECHOES line and are checked by parity_obs.py against the simulator; this file
# only owns mem and memfar, so it slices them by width rather than "the rest".
N_ECHO = 5
TOL = 1e-6          # both sides are float32 of the same arithmetic


def rows_of(path: str) -> list[np.ndarray]:
    return [np.array(l.split()[1:], np.float64) for l in open(path) if l.startswith("DUMP")]


def main() -> None:
    files = sys.argv[1:]
    worst = {"mem": 0.0, "memfar": 0.0}
    turns = bad = 0
    for path in files:
        rows = rows_of(path)
        if not rows:
            continue
        extra = len(rows[0]) - (2 + C * 49 + S + 2 * A)
        if extra not in (N_MEM + N_FAR, N_MEM + N_FAR + N_ECHO):
            print(f"{pathlib.Path(path).name}: {extra} extra scalars, expected "
                  f"{N_MEM + N_FAR} or {N_MEM + N_FAR + N_ECHO} (with sonar echoes)")
            sys.exit(1)
        # one tracker per file: a dump file is one dragon, and its memory starts
        # empty exactly as a fresh dragon's process does
        tracker = MemoryTracker()
        errs = []
        for t, r in enumerate(rows):
            loc = r[2:2 + C * 49].reshape(1, C, 7, 7)
            sc = r[2 + C * 49:2 + C * 49 + S].reshape(1, S)
            got = r[2 + C * 49 + S:2 + C * 49 + S + extra]
            rnd = np.rint(sc[:, 0] * 500).astype(np.int64)
            want = tracker.step([0], loc, sc, rnd)
            for name, ref, mine in (("mem", want["mem"][0], got[:N_MEM]),
                                    ("memfar", want["memfar"][0],
                                     got[N_MEM:N_MEM + N_FAR])):
                err = float(np.abs(ref - mine).max())
                worst[name] = max(worst[name], err)
                if err > TOL:
                    errs.append((t, name, err, int(np.abs(ref - mine).argmax())))
            turns += 1
        bad += len(errs)
        print(f"{pathlib.Path(path).name:32s} {len(rows):3d} turns, {len(errs)} differ")
        for t, name, err, where in errs[:3]:
            print(f"    turn {t}: {name}[{where}] off by {err:.3g}")
    print(f"\n{turns} turns compared over {len(files)} dragons, {bad} differ "
          f"(max error mem {worst['mem']:.3g}, memfar {worst['memfar']:.3g}, "
          f"tolerance {TOL})")
    sys.exit(1 if bad or turns == 0 else 0)


if __name__ == "__main__":
    main()
