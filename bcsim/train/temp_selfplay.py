"""Greedy vs sampled self-play (user, 2026-10-09): does a little randomness help a gated model?

One checkpoint plays itself: one side samples softmax(logits / T) at T = the lowest temperature it
trained at (args temp_min, and is told that T), the other side plays greedy (as submitted). Every map
in --maps, half the games on each side. Runs in rounds (a fresh seed per round) on the CPU, so it does
not take the GPU from a live PPO run, and appends one line per round to --out with the cumulative score
of the SAMPLED side and its standard error. Stop it whenever the answer is clear.

    BC_QUEEN_GUARD=1 nice -n 19 /usr/bin/python3 -u -m train.temp_selfplay \
        --ckpt ../runs/scratch_1002/anchors/gen12.pt --out ../runs/temp_selfplay_1009/log.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bcsim"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--maps", default=str(ROOT / "maps-server-1003"))
    p.add_argument("--out", required=True)
    p.add_argument("--temp", type=float, default=-1.0, help="sampled side's temperature; -1 = args temp_min")
    p.add_argument("--games", type=int, default=4, help="per map per round (half on each side)")
    p.add_argument("--rounds", type=int, default=1000)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--threads", type=int, default=4, help="simulator threads")
    p.add_argument("--torch-threads", type=int, default=6)
    p.add_argument("--lib", default="libbcvec_s2_g15p.so", help="the run's simulator library")
    a = p.parse_args()
    os.environ["BCSIM_LIB"] = str(ROOT / "bcsim/bcsim" / a.lib)          # before bcsim is imported
    os.environ.setdefault("BC_EVAL_QUEUE", "0")                             # the queued path pins CUDA memory
    if os.environ.get("BC_QUEEN_GUARD") != "1":
        raise SystemExit("set BC_QUEEN_GUARD=1, as scratch_1002 trains and gates with it")
    import numpy as np
    import torch
    torch.set_num_threads(a.torch_threads)
    import bcsim
    from train.distill_lstm import LSTMGreedy
    from train.yardstick import evaluate, load_net

    dev = torch.device("cpu")
    files = sorted(pathlib.Path(a.maps).glob("*.map"))
    maps, names = bcsim.load_maps([str(f) for f in files]), [f.stem for f in files]
    net, ck = load_net(a.ckpt, dev)
    args = ck["args"]
    t_min = float(args.get("temp_min", args.get("temp", 0.4)))
    T = t_min if a.temp < 0 else a.temp
    if getattr(net, "temp_in", False):
        net.temp_default.fill_(t_min)          # greedy is told the lowest training temperature, as in the gate
    n_envs = 2 * len(maps) * max(1, a.games // 2)

    def player(temp: float):
        f = LSTMGreedy(net, dev, n_envs)
        f.n_act = int(args.get("n_actions", 48))
        f.grid_model = int(args.get("grid", 14))
        f.speaks = bool(args.get("sonar2", False))
        f.keep_probs = f.speaks
        f.temp = temp
        return f

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prev = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
    tot = prev[-1]["total"] if prev else {"n": 0, "win": 0, "draw": 0, "loss": 0}
    by_map = prev[-1]["by_map"] if prev else {}
    print(f"{a.ckpt}: sampled T={T} vs greedy (told T={t_min}), {len(maps)} maps, "
          f"{n_envs} games a round, resuming at round {len(prev)}", flush=True)
    for r in range(len(prev), a.rounds):
        res = evaluate(player(T), [{"name": "greedy", "act": player(0.0)}], maps, names,
                       games=a.games, threads=a.threads, seed=a.seed + 7919 * r, max_seconds=1e9,
                       sonar=True, fast=True)
        cells = res["cells"]["greedy"]
        for m, c in cells.items():
            d = by_map.setdefault(m, {"n": 0, "win": 0, "draw": 0, "loss": 0})
            for k in d:
                d[k] += c[k]
                tot[k] += c[k]
        n = tot["n"]
        s = (tot["win"] + 0.5 * tot["draw"]) / n
        # per-game score in {0, .5, 1}: its sample variance, so draws count as what they are
        var = (tot["win"] + 0.25 * tot["draw"]) / n - s * s
        se = math.sqrt(max(var, 0.0) / max(n - 1, 1))
        rs = sum(c["win"] + 0.5 * c["draw"] for c in cells.values()) / sum(c["n"] for c in cells.values())
        row = {"round": r, "time": time.strftime("%F %T"), "seconds": res["seconds"], "temp": T,
               "round_score": round(rs, 4), "score": round(s, 4), "se": round(se, 4), "z": round((s - 0.5) / se, 2) if se else None,
               "total": tot, "by_map": by_map}
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[{row['time']}] round {r}: {rs:.3f} this round ({res['seconds']:.0f}s) | sampled "
              f"{s:.4f} +- {se:.4f} over {n} games (W {tot['win']} D {tot['draw']} L {tot['loss']}), "
              f"z {row['z']}", flush=True)


if __name__ == "__main__":
    main()
