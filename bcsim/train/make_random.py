"""The two checkpoints a from-scratch run starts from (user, 2026-10-01).

    learner  an ffl policy with freshly initialised weights (the clones' architecture, the
             simulator's full grid: 54 channels, 15 x 15; 51 actions, or 75 with the far sprints of
             the portal build), told its temperature
    random   the uniform-random player, version 0 of the run: the same net with its output
             layer zeroed (equal logits) and args["uniform"] set, so every player -- training
             opponents, the KL teacher, the gate's greedy play -- picks uniformly among the legal
             moves (ff_net.build, ratchet_lstm_train.Actor, distill_lstm.LSTMGreedy)

    python -m train.make_random --out ../runs/scratch_1003/init        # the portal build (default)
    BCSIM_LIB=bcsim/libbcvec_priv_s2_g15.so python -m train.make_random --out ...   # the 51-action one
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("BCSIM_LIB", str(pathlib.Path(__file__).resolve().parents[1] / "bcsim" /
                                       "libbcvec_priv_s2_g15p.so"))

import torch                                  # noqa: E402

import bcsim                                  # noqa: E402
from train.ff_net import build                # noqa: E402

# the clones' (ffl15_pi, ffl15_cutlery) layer sizes
ARCH = {"arch": "ffl", "c1": 48, "b1": 1, "c2": 112, "b2": 2, "squeeze": 16, "embed": 256, "hidden": 128,
        "layers": 2}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temp-ref", type=float, default=0.4)
    p.add_argument("--temp-min", type=float, default=0.1)
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # the CNN reads the first 54 channels; the identity planes after them (57) only ever feed the MLP
    io = {"in_ch": min(bcsim.GRID_CH, 54), "n_actions": bcsim.N_ACTIONS, "grid": bcsim.GRID_SIDE, "sonar2": True,
          "portal": bool(bcsim.PORTAL_BUILD)}
    if (io["in_ch"], io["grid"]) != (54, 15) or io["n_actions"] not in (51, 75):
        raise SystemExit(f"expected the 15x15 queen build (54 channels, 51 or 75 actions), got {io}")

    torch.manual_seed(a.seed)
    learner_args = {**ARCH, **io, "temp_in": True, "temp_ref": a.temp_ref, "temp_min": a.temp_min,
                    "from_scratch": True}
    learner = build(learner_args)
    torch.save({"net": learner.state_dict(), "args": learner_args}, out / "learner.pt")

    rand_args = {**ARCH, **io, "sonar2": False, "temp_in": False, "uniform": True}
    rand = build(rand_args)
    with torch.no_grad():
        rand.pi.weight.zero_()
        rand.pi.bias.zero_()
    torch.save({"net": rand.state_dict(), "args": rand_args}, out / "random.pt")
    n = sum(q.numel() for q in learner.parameters())
    print(f"wrote {out}/learner.pt ({n:,} parameters, {io}) and {out}/random.pt (uniform over legal moves)")


if __name__ == "__main__":
    main()
