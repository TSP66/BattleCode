"""How clones treat super-sprints (super_sprint.py) on held-out games.

    python -m train.sprint_report --cache ../runs/clone_cache/devtest_1302 ckpt1.pt ckpt2.pt

For every held-out super-sprint row: does the clone pick exactly the labelled
sprint, or any sprint at all? For the other held-out rows (the first
--sample of them): how often does it sprint when the team did not, and how
often does it agree with the team?
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from train.clone_eval import load                  # noqa: E402
from train.imitate2 import Data                    # noqa: E402
from train.net import masked_logits                # noqa: E402


def predict(net, d, dev):
    out = []
    with torch.inference_mode():
        for b in range(0, d.n, 8192):
            idx = np.arange(b, min(b + 8192, d.n))
            loc, sc, mask, *_ = d.batch(idx, dev)
            logits, _ = net(loc, sc)
            out.append(masked_logits(logits.float(), mask).argmax(1).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--sample", type=int, default=300_000)
    p.add_argument("ckpts", nargs="+")
    a = p.parse_args()
    cache = pathlib.Path(a.cache)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    meta = json.loads((cache / "meta.json").read_text())
    val = set(meta["val_games"])
    lab = np.load(cache / "sprint_label.npy")
    cat = np.load(cache / "sprint_cat.npy")
    for ck in a.ckpts:
        net, feats = load(ck, dev)
        d = Data(cache, val, feats, relabel="sprint_label")
        L = lab[d.rows]
        sp = np.flatnonzero(L >= 0)
        rest = np.flatnonzero(L < 0)[:a.sample]
        dsp = Data.__new__(Data)
        pred_all = predict(net, d, dev)
        p_sp, p_rest = pred_all[sp], pred_all[rest]
        is_sprint = lambda x: (x >= 3) & (x < 39)
        team = np.load(cache / "action.npy")[d.rows[rest]]
        c = cat[d.rows[sp]]
        print(f"{ck}\n  super-sprint rows {len(sp)}: exact {np.mean(p_sp == L[sp]):.3f}, any sprint "
              f"{np.mean(is_sprint(p_sp)):.3f} (last-dragon {np.mean(is_sprint(p_sp[c == 2])):.3f}, "
              f"longest {np.mean(is_sprint(p_sp[c == 1])):.3f})\n"
              f"  other rows {len(rest):,}: sprints {np.mean(is_sprint(p_rest)):.4f} "
              f"(team sprinted {np.mean(is_sprint(team)):.4f}), agreement {np.mean(p_rest == team):.4f}",
              flush=True)
        del d, dsp


if __name__ == "__main__":
    main()
