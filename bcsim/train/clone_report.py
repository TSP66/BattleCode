"""Held-out accuracy of imitate2.py runs, broken down.

    python -m train.clone_report --cache ../runs/clone_cache/devtest_1302 ../runs/i2/*_300

For each run it reads val_pred.npy (the best epoch's argmax on the held-out
rows) and prints accuracy over all held-out rows, over the novel ones (whose
exact observation occurs in no training game, see obs_hash in clone_cache.py),
and per class of the team's actual move.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from train.clone_features import action_class     # noqa: E402

CLASSES = ["straight", "left", "right", "sprint", "split"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--val-cap", type=int, default=300_000)
    p.add_argument("runs", nargs="+")
    a = p.parse_args()
    cache = pathlib.Path(a.cache)
    meta = json.loads((cache / "meta.json").read_text())
    game = np.load(cache / "game.npy")
    keep = np.load(cache / "keep.npy")
    val = np.isin(game, np.array(meta["val_games"]))
    rows = np.flatnonzero(val & keep)[:a.val_cap]
    act = np.load(cache / "action.npy")[rows].astype(np.int64)
    alt = np.load(cache / "alt.npy")[rows].astype(np.int64)
    h = np.load(cache / "obs_hash.npy")
    novel = ~np.isin(h[rows], h[~val])
    cls = action_class(act)
    print(f"{len(rows):,} held-out rows, {novel.mean():.1%} novel; class shares "
          + " ".join(f"{c} {np.mean(cls == i):.3f}" for i, c in enumerate(CLASSES)))
    print(f"{'run':28s} {'all':>6s} {'novel':>6s} {'dup':>6s} " + " ".join(f"{c:>8s}" for c in CLASSES))
    for r in a.runs:
        f = pathlib.Path(r) / "val_pred.npy"
        if not f.exists():
            continue
        pred = np.load(f).astype(np.int64)
        hit = (pred == act) | ((alt >= 0) & (pred == alt))
        print(f"{pathlib.Path(r).name:28s} {hit.mean():6.4f} {hit[novel].mean():6.4f} {hit[~novel].mean():6.4f} "
              + " ".join(f"{hit[cls == i].mean():8.4f}" for i in range(len(CLASSES))))


if __name__ == "__main__":
    main()
