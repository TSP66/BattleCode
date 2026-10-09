"""A policy trained before BC_PORTALREP, made into a portal policy (2026-10-02).

    /usr/bin/python3 -m train.portal_init --src runs/x/cands/g004_s4/seg0.pt --out runs/y/init/learner.pt

Grid channels 39-42 meant the teammates' move intents; in the portal simulator (make -C bcsim
s2g15p) they are the portal planes (cpp/bc_memory.hpp PT_*). The copy gets args["portal"] = True
and ZERO stem weights on those four channels, so it plays exactly as the original does with the
channels blinded (what an old policy reads in a portal simulator, ff_net.build) until training
moves them. Adam's moments for those weights are zeroed too, so nothing learnt from the intents
carries over. Everything else (the critic, the optimizer, the run counters) is copied as it is.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from train.lstm_net import PORTAL_PLANES  # noqa: E402

STEMS = ("stem.weight", "body.0.weight")      # ffl / lstm, ff


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ck = torch.load(a.src, map_location="cpu", weights_only=False)
    args = ck["args"]
    if args.get("portal"):
        raise SystemExit(f"{a.src} is already a portal policy")
    if args.get("arch") not in ("ffl", "lstm", "ff") or not args.get("sonar2") or int(args.get("grid", 14)) != 15:
        raise SystemExit(f"{a.src}: need a 15x15 sonar-v2 grid policy (arch {args.get('arch')}, "
                         f"sonar2 {args.get('sonar2')}, grid {args.get('grid')})")
    lo, hi = PORTAL_PLANES
    key = next((k for k in STEMS if k in ck["net"]), None)
    if key is None:
        raise SystemExit(f"{a.src}: no stem weight ({' / '.join(STEMS)})")
    w = ck["net"][key]
    if w.shape[1] < hi:
        raise SystemExit(f"{a.src}: the stem reads {w.shape[1]} channels, fewer than {hi}")
    before = w[:, lo:hi].abs().sum().item()
    w = w.clone()
    w[:, lo:hi] = 0
    ck["net"][key] = w
    # the optimizer's moments for that tensor: found by shape, and only if exactly one tensor has it
    opt = ck.get("opt")
    n_opt = 0
    if isinstance(opt, dict) and "state" in opt:
        hits = [st for st in opt["state"].values()
                if isinstance(st.get("exp_avg"), torch.Tensor) and st["exp_avg"].shape == w.shape]
        if len(hits) == 1:
            for k in ("exp_avg", "exp_avg_sq"):
                if k in hits[0]:
                    hits[0][k][:, lo:hi] = 0
                    n_opt += 1
        else:
            print(f"  {len(hits)} optimizer tensors shaped like the stem: dropping the optimizer state")
            ck.pop("opt")
    ck["args"] = {**args, "portal": True, "portal_from": str(a.src)}
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out)
    print(f"{out}: {key}[:, {lo}:{hi}] zeroed (|w| was {before:.4f}), {n_opt} Adam moments zeroed, args portal=True")


if __name__ == "__main__":
    main()
