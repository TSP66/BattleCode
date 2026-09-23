"""Every saved agent against every other, on every map, both sides.

A gate measures one candidate against one anchor, which says whether the
candidate beat that anchor and nothing else. Comparing two *architectures*
needs more than that: a net can beat the anchor by exploiting it while being
worse against the field. So this plays the full matrix and ranks the field
from it.

Output is a score matrix (row's score against column, draws as half a win)
plus a Bradley-Terry fit, which is the ranking implied by every game at once
rather than by one column of it.

    python -m train.roundrobin --agents a.pt,b.pt,c.pt --games 8
    python -m train.roundrobin --agents-file agents.txt --maps ../maps-live

Each (row, column) pair is played twice over -- once with the row as learner
and once with the column as learner -- because `evaluate` drives one side. The
two halves are pooled, so `--games 8` means 8 per (pair, map, side) from each
half, 16 per (pair, map) in total.

Cost is quadratic in the number of agents: 10 agents on 9 maps at --games 8 is
about 6.5k games. Start with --games 2 to see the shape of the matrix, then
raise it for the pairs that matter.
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

import bcsim                                    # noqa: E402
from train.yardstick import evaluate, greedy, load_net   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", default="",
                   help="comma separated checkpoints; name=path to label one")
    p.add_argument("--agents-file", default="",
                   help="one 'name=path' or 'path' per line; # comments allowed")
    p.add_argument("--maps", default=str(ROOT / "maps-live"))
    p.add_argument("--games", type=int, default=8,
                   help="per (pair, map, side) in each half of the matrix")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-seconds", type=float, default=3600.0, help="per row")
    p.add_argument("--out", default="", help="write the full result as json here")
    return p.parse_args()


def read_agents(a) -> list[tuple[str, pathlib.Path]]:
    spec: list[str] = []
    if a.agents:
        spec += [x for x in a.agents.split(",") if x.strip()]
    if a.agents_file:
        for line in pathlib.Path(a.agents_file).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                spec.append(line)
    out = []
    for s in spec:
        name, _, path = s.partition("=")
        if not path:
            name, path = "", name
        p = pathlib.Path(path).expanduser()
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise SystemExit(f"no such checkpoint: {p}")
        out.append((name or p.parent.name + "/" + p.stem, p))
    if len(out) < 2:
        raise SystemExit("give at least two agents")
    names = [n for n, _ in out]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate agent names: {names}")
    return out


def bradley_terry(wins: np.ndarray, games: np.ndarray, iters: int = 5000,
                  tol: float = 1e-10) -> np.ndarray:
    """Maximum-likelihood strengths from the pooled matrix, as log-odds.

    Standard MM iteration on p_ij = s_i / (s_i + s_j). Pairs that were never
    played contribute nothing, so a partially filled matrix still fits. The
    result is centred, and is only identified if the win graph is connected --
    an agent that won or lost every single game has no finite strength, so its
    total is nudged by half a game to keep the fit bounded.
    """
    n = len(wins)
    w = wins.astype(np.float64).copy()
    g = games.astype(np.float64).copy()
    # Keep the fit finite: an undefeated or never-winning agent would otherwise
    # run off to infinity and take the centring with it.
    tot = w.sum(1)
    played = g.sum(1)
    for i in range(n):
        if played[i] and (tot[i] == 0 or tot[i] == played[i]):
            j = int(np.argmax(g[i]))
            w[i, j] += 0.5 if tot[i] == 0 else -0.5
    s = np.ones(n)
    for _ in range(iters):
        prev = s.copy()
        for i in range(n):
            num = w[i].sum()
            den = 0.0
            for j in range(n):
                if i != j and g[i, j]:
                    den += g[i, j] / (s[i] + s[j])
            s[i] = num / den if den > 0 else s[i]
        s /= np.exp(np.log(s).mean())
        if np.abs(s - prev).max() < tol:
            break
    return np.log(s)


def main() -> None:
    a = parse()
    agents = read_agents(a)
    dev = torch.device("cuda")
    maps = bcsim.load_maps(a.maps)
    map_names = [f.stem for f in sorted(pathlib.Path(a.maps).glob("*.map"))]
    n_ag = len(agents)

    # Rows for a single `evaluate` call: every other agent, both sides, per map.
    n_envs = (n_ag - 1) * len(maps) * 2 * max(1, a.games // 2)
    print(f"{n_ag} agents, {len(maps)} maps, {a.games} games per cell per side, "
          f"{n_envs} envs per row", flush=True)

    # One callable per net, reused for every row of the matrix.
    acts, archs = {}, {}
    for name, path in agents:
        net, ck = load_net(path, dev)
        acts[name] = greedy(net, dev)
        archs[name] = ck["args"].get("arch", "flat")
        print(f"  {name:28s} {archs[name]:8s} "
              f"{sum(q.numel() for q in net.parameters()):,} params", flush=True)

    wins = np.zeros((n_ag, n_ag))        # pooled, draws as half
    games = np.zeros((n_ag, n_ag))
    cells: dict = {}
    t0 = time.perf_counter()
    for i, (name, _) in enumerate(agents):
        others = [(j, m) for j, (m, _) in enumerate(agents) if m != name]
        opps = [{"name": m, "act": acts[m]} for _, m in others]
        res = evaluate(acts[name], opps, maps, map_names, games=a.games,
                       threads=a.threads, seed=a.seed + i,
                       max_seconds=a.max_seconds, progress=120)
        cells[name] = res["summary"]
        line = []
        for j, m in others:
            s = res["summary"].get(m)
            if not s or not s["n"]:
                continue
            # score is (win + draw/2) / n for `name`; pool both halves
            wins[i, j] += s["score"] * s["n"]
            games[i, j] += s["n"]
            wins[j, i] += (1.0 - s["score"]) * s["n"]
            games[j, i] += s["n"]
            line.append(f"{m} {s['score']:.3f}")
        print(f"[{i + 1}/{n_ag}] {name}: " + "  ".join(line) +
              f"   ({res['seconds']:.0f}s)", flush=True)

    played = games > 0
    score = np.where(played, wins / np.maximum(games, 1), np.nan)
    strength = bradley_terry(wins, games)
    order = np.argsort(-strength)
    names = [n for n, _ in agents]

    w = max(len(n) for n in names) + 1
    print("\nscore matrix (row's score against column, both halves pooled)")
    print(" " * w + "".join(f"{names[j][:7]:>8s}" for j in order))
    for i in order:
        row = "".join("     -  " if i == j or not played[i, j]
                      else f"{score[i, j]:8.3f}" for j in order)
        print(f"{names[i]:{w}s}{row}")

    print("\nranking (Bradley-Terry, log-odds; mean score over all opponents)")
    for i in order:
        n_i = games[i].sum()
        mean = float(np.nansum(wins[i]) / max(n_i, 1))
        se = float(np.sqrt(max(mean * (1 - mean), 1e-9) / max(n_i, 1)))
        print(f"  {names[i]:{w}s} {strength[i]:+7.3f}   "
              f"mean {mean:.4f} +/- {se:.4f} over {int(n_i)} games   [{archs[names[i]]}]")

    print(f"\n{int(games.sum() / 2)} games, {time.perf_counter() - t0:.0f}s")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps({
            "agents": [{"name": n, "path": str(p), "arch": archs[n]} for n, p in agents],
            "games": games.tolist(), "wins": wins.tolist(),
            "strength": strength.tolist(), "cells": cells,
            "maps": map_names, "games_per_cell": a.games,
            "seconds": round(time.perf_counter() - t0, 1),
        }, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
