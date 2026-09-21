"""Does a policy die more when enemies meet it across the map edge?

Self-play with greedy moves on the maps whose edges are open. Every turn with
an enemy head within two cells is labelled seam (the head is within reach of
the enemy only through the wrap: their raw coordinates are far apart) or
plain, and the death rate before the dragon's next turn is compared.

    python scripts/seam_deaths.py runs/v2/snapshots/turns_X.pt
"""

import collections
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                     # noqa: E402
from train import augment                        # noqa: E402
from train.yardstick import greedy, load_net     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root

OPEN = ["Colloseum", "big_empty", "default", "help", "queen_of_spades", "trophy"]


def main() -> None:
    dev = torch.device("cuda")
    net, _ = load_net(sys.argv[1], dev)
    act = greedy(net, dev)
    root = ROOT / "maps"
    texts = [(root / f"{n}.map").read_text() for n in OPEN]
    dims = [(augment.parse(t).w, augment.parse(t).h) for t in texts]
    n = 384
    env = bcsim.BattlecodeVecEnv(texts, num_envs=n, num_threads=8, seed=5)
    for i in range(n):
        env.set_opponent(i, map_index=i % len(texts))
    obs = env.reset()
    ch = bcsim.CHANNELS
    eh = ch.index("enemy_head")
    si = bcsim.SCALARS
    died_i = bcsim.REWARD_COMPS.index("died")
    last = {}                          # (env, uid) -> label of its latest turn
    count = collections.Counter()
    deaths = collections.Counter()
    for _ in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6000):
        heads = obs.local[:, eh]
        for e in range(n):
            ys, xs = np.nonzero(heads[e])
            if not len(xs):
                continue
            near = [(r, c) for r, c in zip(ys, xs) if abs(r - 3) + abs(c - 3) <= 2]
            if not near:
                continue
            w, h = dims[e % len(texts)]
            hx = obs.scalar[e, si.index("head_x")] * w
            hy = obs.scalar[e, si.index("head_y")] * h
            # a window that wraps and an enemy head in its wrapped part
            wraps_x = hx < 3 or hx > w - 4
            wraps_y = hy < 3 or hy > h - 4
            label = "seam" if (wraps_x or wraps_y) else "plain"
            last[(e, int(obs.uid[e]))] = label
            count[label] += 1
        obs, cl, _ = env.step(act(obs.local, obs.scalar, obs.mask))
        for e, uid, c in zip(cl.env, cl.uid, cl.comps):
            lab = last.pop((int(e), int(uid)), None)
            if lab and c[died_i] > 0:
                deaths[lab] += 1
    for lab in ("plain", "seam"):
        k = count[lab]
        rate = deaths[lab] / max(k, 1)
        se = (rate * (1 - rate) / max(k, 1)) ** 0.5
        print(f"{lab:6s} turns with an enemy head within 2: {k:6d}  died before next turn: "
              f"{deaths[lab]:5d}  rate {rate:.3f} ± {se:.3f}")


if __name__ == "__main__":
    main()
