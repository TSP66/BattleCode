"""Round robin between clone checkpoints (with or without memory features).

Every pair plays --games per map, half on each side, greedy, on the live
maps; the pairs are batched as clone_eval does, one candidate against all the
ones after it per pass. Results accumulate in --out (one JSON line per pass),
and passes already there are skipped, so an interrupted round robin resumes.

    python -m train.round_robin --out ../runs/evals/rr.jsonl \
        v10=../runs/submitted/v10.pt mm=../runs/i2/full_mem_memfar/best.pt ...

With --learner, only that candidate's pass runs, so passes can run as
parallel processes; --table prints the matrix from what --out holds.

Prints the score matrix (row's score against column, a draw half a win) and
each candidate's mean score.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                         # noqa: E402
from train.clone_eval import Policy, load            # noqa: E402
from train.yardstick import evaluate                 # noqa: E402


def table(names, done):
    """Score matrix over every pass in `done` (several seeds merge: a pair's
    score is the game-weighted mean), each candidate's mean over its
    opponents, and that mean's standard error from the games played."""
    n = len(names)
    pts = np.zeros((n, n))
    games = np.zeros((n, n), int)
    for row in done:
        i = names.index(row["learner"])
        for opp, s in row["summary"].items():
            if opp not in names:
                continue
            j = names.index(opp)
            pts[i, j] += s["score"] * s["n"]
            pts[j, i] += (1 - s["score"]) * s["n"]
            games[i, j] += s["n"]
            games[j, i] += s["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        m = pts / games
    w = max(len(x) for x in names) + 2
    print(" " * w + "".join(f"{x[:10]:>11s}" for x in names) + "       mean     se")
    for i, x in enumerate(names):
        cells = "".join("          -" if i == j else
                        ("          ." if games[i, j] == 0 else f"{m[i, j]:11.3f}")
                        for j in range(n))
        g = games[i].sum()
        mean = pts[i].sum() / g if g else float("nan")
        se = np.sqrt(mean * (1 - mean) / g) if g else float("nan")
        print(f"{x:<{w}s}{cells}  {mean:9.3f} {se:6.3f}")
    print(f"games per pair: {sorted(set(games[games > 0].tolist()))}")
    return m, games


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True,
                   help="results file; with --table, several comma separated are merged")
    p.add_argument("--maps", default="../runs/eval_maps")
    p.add_argument("--games", type=int, default=12)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-seconds", type=float, default=6 * 3600,
                   help="per pass; games unfinished by then are dropped, the rest kept")
    p.add_argument("--learner", default="",
                   help="play only this candidate's pass (to run passes as parallel processes "
                        "appending to the same --out); without it, every pass, then the table")
    p.add_argument("--table", action="store_true", help="only print the table from --out")
    p.add_argument("cands", nargs="+", help="name=checkpoint")
    a = p.parse_args()
    torch.set_num_threads(4)
    dev = torch.device(a.device)
    cands = [c.split("=", 1) for c in a.cands]
    names = [n for n, _ in cands]
    maps = bcsim.load_maps(a.maps)
    map_names = [f.stem for f in sorted(pathlib.Path(a.maps).glob("*.map"))]
    outs = [pathlib.Path(x) for x in a.out.split(",")]
    out = outs[0]
    done = [json.loads(l) for o in outs if o.exists() for l in o.read_text().splitlines()]
    for i, (name, path) in enumerate(cands[:-1]):
        if a.table or (a.learner and name != a.learner):
            continue
        rest = cands[i + 1:]
        # pairs already played (either way round) are not played again
        have = {(r["learner"], o) for r in done for o in r["summary"]}
        have |= {(o, l) for l, o in have}
        rest = [(n, pth) for n, pth in rest if (name, n) not in have]
        if not rest:
            continue
        t0 = time.perf_counter()
        net, feats = load(path, dev)
        opps = []
        for n, pth in rest:
            onet, ofeats = load(pth, dev)
            opps.append({"name": n, "act": Policy(onet, ofeats, dev), "path": pth})
        res = evaluate(Policy(net, feats, dev), opps, maps, map_names, games=a.games,
                       threads=a.threads, seed=a.seed + i, max_seconds=a.max_seconds, progress=300)
        row = {"learner": name, "ckpt": path, "games": a.games, "time": time.time(), **res}
        done.append(row)
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{name} vs {[n for n, _ in rest]}: " + "  ".join(
            f"{k} {v['score']:.3f}" for k, v in res["summary"].items())
            + f"  ({time.perf_counter() - t0:.0f}s)", flush=True)
    table(names, done)


if __name__ == "__main__":
    main()
