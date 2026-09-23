"""Widens a 14-scalar checkpoint to the 708 scalars the memory env gives.

Adding mem (676) and memfar (18) to bcsim's observation changes N_SCALARS from
14 to 708, which would strand every network trained before it: gen0-gen4 (over a
billion turns of ratchet lineage), v9, v10 and the whole clone league.

None of that is necessary. The scalar branch starts with one Linear, so padding
its weight with a zero column per new input leaves the function it computes
*exactly* unchanged -- the new inputs are multiplied by zero whatever they hold.
The widened net plays identically to the original, and PPO then learns to use
memory from a zero start, which is a warm start rather than a cold one.

The user asked (2026-09-23) to keep gen4 as a frozen opponent through the
switch. This is what allows that: gen4 widened is gen4, move for move.

    python -m train.migrate_scalars --out ../runs/anchors708 \
        ../runs/ratchet/anchors/gen4.pt ../runs/submitted/v10.pt

Every checkpoint is verified before it is written: the widened net must agree
with the original to the bit on random observations, with the 694 new scalars
set to large random values so that any leak shows up.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from train.net import ActorCritic          # noqa: E402

N_BASE = 14
N_MEM, N_FAR = 676, 18
N_WIDE = N_BASE + N_MEM + N_FAR           # 708
KEY = "scalar.0.weight"


def strip(sd: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def build(sd: dict, ck: dict, n_sc: int) -> ActorCritic:
    a = ck["args"]
    hidden = next(v for k, v in sd.items() if k.endswith("fuse.0.weight")).shape[0]
    n_ch = sd["stem.0.weight"].shape[1]
    net = ActorCritic(n_ch, n_sc, 48, width=a["width"], blocks=a["blocks"], hidden=hidden)
    net.load_state_dict(sd)
    net.eval()
    return net


def widen(path: pathlib.Path, out: pathlib.Path, rows: int = 128) -> str:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = strip(ck["net"])
    have = sd[KEY].shape[1]
    if have == N_WIDE:
        return "already 708, skipped"
    if have != N_BASE:
        return f"REFUSED: {KEY} is {have} wide, expected {N_BASE} or {N_WIDE}"

    W = sd[KEY]
    big = torch.zeros(W.shape[0], N_WIDE, dtype=W.dtype)
    big[:, :N_BASE] = W
    wide = dict(sd)
    wide[KEY] = big

    # verify before writing: the new inputs must not move the output at all
    old, new = build(sd, ck, N_BASE), build(wide, ck, N_WIDE)
    g = torch.Generator().manual_seed(0)
    n_ch = sd["stem.0.weight"].shape[1]
    loc = torch.rand(rows, n_ch, 7, 7, generator=g)
    base = torch.rand(rows, N_BASE, generator=g)
    junk = torch.randn(rows, N_MEM + N_FAR, generator=g) * 100.0
    with torch.inference_mode():
        l0, v0 = old(loc, base)
        l1, v1 = new(loc, torch.cat([base, junk], 1))
    dl = (l0 - l1).abs().max().item()
    dv = (v0 - v1).abs().max().item()
    if dl != 0.0 or dv != 0.0:
        return f"REFUSED: widened net differs (logits {dl:g}, value {dv:g})"

    ck["net"] = wide
    ck["n_scalars"] = N_WIDE
    ck.setdefault("migrated_from", str(path))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    return f"ok, identical (max logit diff {dl:g})"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="directory for the widened checkpoints")
    p.add_argument("ckpts", nargs="+")
    a = p.parse_args()
    out = pathlib.Path(a.out)
    bad = 0
    for c in a.ckpts:
        src = pathlib.Path(c)
        msg = widen(src, out / src.name)
        bad += msg.startswith("REFUSED")
        print(f"{src.name:<28} {msg}", flush=True)
    if bad:
        raise SystemExit(f"{bad} checkpoint(s) refused")


if __name__ == "__main__":
    main()
