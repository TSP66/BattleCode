"""Compares critics on the same held-out games, split by where the game came from.

Adding invented-map self-play to the corpus also changes the held-out set, so
the `ev` a pretrain run prints is not comparable with an earlier run's. This
scores any number of critics on one fixed split, reported separately for the
replay games (real server play) and the invented-map games (train.team_critic
selfplay), which is the comparison that actually answers whether the new data
bought anything.

    python -m train.critic_compare ../runs/team_critic/pretrained_pre20260923.pt \
                                   ../runs/team_critic/pretrained.pt
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from train import team_critic                       # noqa: E402
from train.team_critic import OUT, OUTCOME_VALUE, held_out, load  # noqa: E402

SELFPLAY_BASE = 900_000_000


def subset(files):
    rec = {k: [] for k in ("local", "scalar", "priv", "round", "outcome")}
    for f in files:
        d = np.load(f)
        for k in rec:
            rec[k].append(d[k])
    return {k: np.concatenate(v) for k, v in rec.items()}


def score(net, dev, data, n_sc, batch=8192):
    probs = []
    with torch.inference_mode():
        for i in range(0, len(data["round"]), batch):
            sl = slice(i, i + batch)
            lg, _ = net(torch.from_numpy(data["local"][sl]).float().to(dev),
                        torch.from_numpy(data["scalar"][sl][:, :n_sc]).float().to(dev),
                        torch.zeros(len(data["round"][sl]), 1, device=dev),
                        torch.from_numpy(data["priv"][sl]).float().to(dev))
            probs.append(lg.float().softmax(-1).cpu().numpy())
    p = np.concatenate(probs)
    y = data["outcome"].astype(np.int64)
    target = np.array(OUTCOME_VALUE)[y]
    pred = p[:, 0] - p[:, 2]
    ev = 1 - ((target - pred) ** 2).mean() / (target.var() + 1e-9)
    ll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)).mean()
    return {"n": len(y), "ev": round(float(ev), 4), "ll": round(float(ll), 4),
            "acc": round(float((p.argmax(1) == y).mean()), 4)}


def main() -> None:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = sorted((OUT / "data").glob("*.npz"), key=lambda f: int(f.stem))
    va = [f for f in files if held_out(int(f.stem))]
    groups = {
        "replay (real server play)": [f for f in va if int(f.stem) < SELFPLAY_BASE],
        "invented maps (self-play)": [f for f in va if int(f.stem) >= SELFPLAY_BASE],
    }
    loaded = {name: subset(fs) for name, fs in groups.items() if fs}
    for name, d in loaded.items():
        print(f"{name}: {len(groups[name])} games, {len(d['round']):,} positions")
    print()
    print(f"{'critic':<34}" + "".join(f"{g[:22]:>24}" for g in loaded))
    for path in sys.argv[1:]:
        net, cfg = load(path, dev, n_context=1)
        net.eval()
        n_sc = net.scalar[0].weight.shape[1]
        cells = ""
        for name, d in loaded.items():
            m = score(net, dev, d, n_sc)
            cells += "{:>24}".format("ev {}  ll {}".format(m["ev"], m["ll"]))
        print(f"{pathlib.Path(path).name:<34}{cells}")


if __name__ == "__main__":
    main()
