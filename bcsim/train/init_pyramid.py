"""Writes a fresh PyramidActorCritic checkpoint for the ratchet to start from.

There is no pyramid ancestor to resume: the architecture changes what the
network reads, so no weight of a 708-scalar ActorCritic transfers (board-frame
planes against a flattened 13x13, a different fuse width, a different scalar
count). This is a re-train, not a fine-tune.

What carries the old policy's competence across is the KL term, not the
weights. ratchet_train keeps `--kl-coef` against a frozen teacher, and that
teacher may be a flat net: both architectures emit 48 logits for the same
state, so the KL is well defined across the two. Point --teacher at v13 and the
new network is pulled towards what the old one knows while it learns to use the
planes.

    /usr/bin/python3 -m train.init_pyramid --out ../runs/pyramid0.pt

Then hand that to the ratchet as --start, with --teacher the flat checkpoint.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                    # noqa: E402
from train.net import PyramidActorCritic        # noqa: E402


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="checkpoint to write")
    p.add_argument("--near-width", type=int, default=48)
    p.add_argument("--near-blocks", type=int, default=3)
    p.add_argument("--wide-width", type=int, default=24)
    p.add_argument("--wide-blocks", type=int, default=2)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    a = parse()
    torch.manual_seed(a.seed)
    if bcsim.WIDE_CH == 0:
        raise SystemExit("this bcsim build has no wide planes; run `make -C bcsim`")

    net = PyramidActorCritic(bcsim.N_CHANNELS, bcsim.WIDE_CH, bcsim.N_ACTIONS,
                             near_width=a.near_width, near_blocks=a.near_blocks,
                             wide_width=a.wide_width, wide_blocks=a.wide_blocks,
                             wide_side=bcsim.WIDE_SIDE, hidden=a.hidden)

    # every field load_policy and ratchet_train read off a checkpoint
    ckpt = {"net": net.state_dict(), "iter": 0, "total_turns": 0, "cand_turns": 0,
            "args": {"arch": "pyramid", "seed": a.seed, "hidden": a.hidden,
                     "near_width": a.near_width, "near_blocks": a.near_blocks,
                     "wide_width": a.wide_width, "wide_blocks": a.wide_blocks,
                     # load_policy reads these off old checkpoints; name them so
                     # anything that prints a shape has something to print
                     "width": a.near_width, "blocks": a.near_blocks}}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)

    n_par = sum(q.numel() for q in net.parameters())
    print(f"wrote {out}")
    print(f"  near {a.near_width}x{a.near_blocks}  wide {a.wide_width}x{a.wide_blocks}  "
          f"hidden {a.hidden}")
    print(f"  wide planes {bcsim.WIDE_CH} x {bcsim.WIDE_SIDE} x {bcsim.WIDE_SIDE}")
    print(f"  {n_par:,} parameters")


if __name__ == "__main__":
    main()
