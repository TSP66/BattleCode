"""Reads a protocol transcript on stdin and saves, per turn, the observation
the Python bot builds from it (archive/dbgbot/obs.py): local, scalars, mask, facing.

    python dump_obs.py out.npz < transcript.txt
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT / "archive/dbgbot"))

import numpy as np          # noqa: E402

import helper as unswbc     # noqa: E402
import obs as O             # noqa: E402

out = sys.argv[1]
sys.stdout = open("/dev/null", "w")     # the helper writes ENDTURN; nobody reads it
ct, game = unswbc.init()
rows = {"local": [], "scalar": [], "mask": [], "facing": []}
while unswbc.update(ct, game):
    snap = O.Snapshot(ct, game)
    rows["local"].append(np.asarray(snap.local(), np.float32))
    rows["scalar"].append(np.asarray(snap.scalars(), np.float32))
    rows["mask"].append(np.asarray(snap.mask(), np.uint8))
    rows["facing"].append(snap.facing)
    unswbc.end_turn()
np.savez(out, **{k: np.stack(v) for k, v in rows.items()})
