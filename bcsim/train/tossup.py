"""Marks left/right "toss-ups" in a clone_cache.py cache, for downweighting.

Some of a team's left-or-right choices are decided by something no input
shows (a tie-break, a hash); fitting them spends capacity that the tactics
need. A row counts as a toss-up only when all of these hold:

  * the team turned left or right;
  * a trained clone is split between the two: P(left) + P(right) >= 0.7 and
    |P(left) - P(right)| <= 0.3 (a well trained clone is rarely split: at
    0.8 / 0.2 it is 2.9% of turns, at 0.7 / 0.3 5.2%);
  * the two steps look alike from the window (clone_features "moves": same
    pearl on the cell, same BFS distance to a pearl, same reachable-or-not,
    same enemy and ally heads next to the cell, reachable areas within 5
    cells);
  * memory does not favour a side: memfar's remembered-pearl weight to the
    left and to the right differ by <= 0.1 (a distant pearl seen earlier is
    information, not a coin flip).

    python -m train.tossup --cache ../runs/clone_cache/devtest_1302 \
        --ckpt ../runs/i2/full_mem_memfar/best.pt --weight 0.5

writes <cache>/tossup_weight.npy (float32 per row: --weight on toss-ups, 1
elsewhere) and <cache>/tossup_prob.npy (the clone's P(left), P(right)).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from train.clone_eval import load                      # noqa: E402
from train.clone_features import MEMFAR_K              # noqa: E402
from train.imitate2 import Data                        # noqa: E402
from train.net import masked_logits                    # noqa: E402

# moves layout: 12 per move (straight, left, right), see clone_features.moves
M_OK, M_PEARL, M_BFS, M_NONE, M_SOON, M_AREA, M_AREA3, M_AREA6, M_MAN, M_ENEMY, M_ALLY, M_FREE = range(12)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--weight", type=float, default=0.5)
    p.add_argument("--chunk", type=int, default=400_000)
    p.add_argument("--reuse", action="store_true", help="reuse tossup_prob.npy instead of scoring")
    a = p.parse_args()
    cache = pathlib.Path(a.cache)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    net, feats = load(a.ckpt, dev)
    n = len(np.load(cache / "game.npy", mmap_mode="r"))
    act = np.load(cache / "action.npy")
    prob = np.zeros((n, 2), np.float16)
    games = np.load(cache / "game.npy")
    uniq = np.unique(games)
    # Data takes games; walk them in slices of about --chunk rows
    step = max(1, int(len(uniq) * a.chunk / n))
    if a.reuse:
        prob = np.load(cache / "tossup_prob.npy")
        uniq = uniq[:0]
    for s in range(0, len(uniq), step):
        d = Data(cache, set(int(g) for g in uniq[s:s + step]), feats)
        # Data keeps only trainable rows; score every row of these games anyway
        d_rows = d.rows
        with torch.inference_mode():
            for b in range(0, d.n, 8192):
                idx = np.arange(b, min(b + 8192, d.n))
                loc, sc, mask, *_ = d.batch(idx, dev)
                logits, _ = net(loc, sc)
                pr = F.softmax(masked_logits(logits.float(), mask), 1)[:, 1:3]
                prob[d_rows[idx]] = pr.cpu().numpy()
        print(f"{min(s + step, len(uniq))}/{len(uniq)} games", flush=True)
        del d
    mv = np.load(cache / "feat_moves.npy", mmap_mode="r")
    mf = np.load(cache / "feat_memfar.npy", mmap_mode="r")
    weight = np.ones(n, np.float32)
    tossups = 0
    for s in range(0, n, 1_000_000):
        sl = slice(s, s + 1_000_000)
        m = np.asarray(mv[sl]).astype(np.float32)
        L, R = m[:, 12:24], m[:, 24:36]
        same = np.ones(len(m), bool)
        for k in (M_OK, M_PEARL, M_BFS, M_NONE, M_ENEMY, M_ALLY):
            same &= np.abs(L[:, k] - R[:, k]) < 1e-3
        same &= np.abs(L[:, M_AREA] - R[:, M_AREA]) * 49 <= 5
        f = np.asarray(mf[sl]).astype(np.float32)
        cone_l, cone_r = f[:, 4 * MEMFAR_K + 1], f[:, 4 * MEMFAR_K + 2]
        same &= np.abs(cone_l - cone_r) <= 0.1
        pl, pr = prob[sl, 0].astype(np.float32), prob[sl, 1].astype(np.float32)
        split = (pl + pr >= 0.7) & (np.abs(pl - pr) <= 0.3)
        turned = (act[sl] == 1) | (act[sl] == 2)
        t = same & split & turned
        weight[sl][t] = a.weight
        tossups += int(t.sum())
    np.save(cache / "tossup_prob.npy", prob)
    np.save(cache / "tossup_weight.npy", weight)
    turned_all = int(((act == 1) | (act == 2)).sum())
    print(f"toss-ups: {tossups:,} of {turned_all:,} left/right turns "
          f"({tossups / max(turned_all, 1):.1%}), {tossups / n:.2%} of all rows", flush=True)


if __name__ == "__main__":
    main()
