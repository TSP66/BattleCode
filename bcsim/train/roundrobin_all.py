"""Round robin with every pair playing at once, in one env batch.

roundrobin.py plays one row of the matrix per `evaluate` call, and a row takes
as long as its longest game (~11 minutes on maps-live, 2026-09-24), so 14 agents
cost 2.5 hours however few games each cell has. Here every (pair, map, side,
repeat) game is its own env in a single batch: the whole matrix takes about as
long as one row did. Each turn, the rows are split by which agent owns the side
to move, and every agent is called once on its own rows.

Same output as roundrobin.py: score matrix (draws are half a win) and a
Bradley-Terry fit.

    python -m train.roundrobin_all --agents-file agents.txt --games 2 --sonar --out rr.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                    # noqa: E402
from train.roundrobin import bradley_terry, read_agents   # noqa: E402
from train.yardstick import COLS, _call, greedy, load_net  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agents", default="")
    p.add_argument("--agents-file", default="")
    p.add_argument("--maps", default=str(ROOT / "maps-live"))
    p.add_argument("--games", type=int, default=2, help="per (pair, map, side)")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-seconds", type=float, default=5400.0)
    p.add_argument("--sonar", action="store_true", help="every dragon broadcasts four ways (needed by LSTM agents)")
    p.add_argument("--out", default="")
    a = p.parse_args()

    agents = read_agents(a)
    dev = torch.device("cuda")
    maps = bcsim.load_maps(a.maps)
    map_names = [f.stem for f in sorted(pathlib.Path(a.maps).glob("*.map"))]
    n_ag = len(agents)
    layout = [(i, j, mi, side) for i, j in itertools.combinations(range(n_ag), 2)
              for mi in range(len(maps)) for side in (0, 1) for _ in range(a.games)]
    n = len(layout)
    print(f"{n_ag} agents, {len(maps)} maps, {a.games} games per (pair, map, side): {n} games at once, "
          f"sonar {'on' if a.sonar else 'off'}", flush=True)

    acts, archs = [], []
    for name, path in agents:
        net, ck = load_net(path, dev)
        arch = ck["args"].get("arch", "flat")
        if arch == "lstm":
            from train.distill_lstm import LSTMGreedy
            acts.append(LSTMGreedy(net, dev, n))
        elif getattr(net, "recurrent", False):
            from train.recurrent import RecurrentGreedy
            acts.append(RecurrentGreedy(net, dev, n, per_env=256))
        else:
            acts.append(greedy(net, dev))
        archs.append(arch)
        print(f"  {name:28s} {arch:8s} {sum(q.numel() for q in net.parameters()):,} params", flush=True)

    any_wide = any(getattr(f, "wants_wide", False) for f in acts)
    any_grid = any(getattr(f, "wants_grid", False) for f in acts)
    env = bcsim.BattlecodeVecEnv(maps, num_envs=n, num_threads=a.threads, seed=a.seed,
                                 closure_capacity=max(8192, n * 160), wide=any_wide,
                                 sonar=a.sonar, grid=any_grid)
    for e, (_, _, mi, _) in enumerate(layout):
        env.set_opponent(e, team=-1, bot=0, map_index=mi)
    agent_i = np.array([i for i, _, _, _ in layout])
    agent_j = np.array([j for _, j, _, _ in layout])
    side_i = np.array([s for _, _, _, s in layout], np.int8)      # the team agent i plays
    stateful = [f for f in acts if getattr(f, "stateful", False) or getattr(f, "wants_grid", False)]

    done = np.zeros(n, bool)
    result = [None] * n
    obs = env.reset()
    t0 = time.perf_counter()
    last = t0
    while not done.all():
        if time.perf_counter() - t0 > a.max_seconds:
            print(f"  hit the {a.max_seconds:.0f}s limit with {int((~done).sum())} games unfinished", flush=True)
            break
        if time.perf_counter() - last > 120:
            last = time.perf_counter()
            print(f"  {int(done.sum())}/{n} games done, {last - t0:.0f}s", flush=True)
        owner = np.where(obs.team == side_i, agent_i, agent_j)
        acts_now = np.zeros(n, np.int32)
        for k in range(n_ag):
            rows = owner == k
            if rows.any():
                acts_now[rows] = _call(acts[k], obs, rows, env.wide, env.grid)
        obs, _, eps = env.step(acts_now)
        for row in eps.rows:
            e = int(row[0])
            for f in stateful:
                if hasattr(f, "forget"):
                    f.forget(e)
            if not done[e]:
                done[e] = True
                result[e] = dict(zip(COLS, row.tolist()))
    env.close()

    wins = np.zeros((n_ag, n_ag))
    games = np.zeros((n_ag, n_ag))
    per_map: dict = {}
    for e, r in enumerate(result):
        if r is None:
            continue
        i, j, mi, s = layout[e]
        sc = 0.5 if r["winner"] < 0 else float(r["winner"] == s)
        wins[i, j] += sc
        wins[j, i] += 1 - sc
        games[i, j] += 1
        games[j, i] += 1
        cell = per_map.setdefault(f"{agents[i][0]}|{agents[j][0]}", {}).setdefault(map_names[mi], [0.0, 0])
        cell[0] += sc
        cell[1] += 1

    names = [nm for nm, _ in agents]
    played = games > 0
    score = np.where(played, wins / np.maximum(games, 1), np.nan)
    strength = bradley_terry(wins, games)
    order = np.argsort(-strength)
    w = max(len(x) for x in names) + 1
    print("\nscore matrix (row's score against column; draws are half)")
    print(" " * w + "".join(f"{names[j][:7]:>8s}" for j in order))
    for i in order:
        print(f"{names[i]:{w}s}" + "".join("     -  " if i == j or not played[i, j]
                                           else f"{score[i, j]:8.3f}" for j in order))
    print("\nranking (Bradley-Terry log-odds; mean score over all opponents)")
    for i in order:
        n_i = games[i].sum()
        mean = float(wins[i].sum() / max(n_i, 1))
        se = float(np.sqrt(max(mean * (1 - mean), 1e-9) / max(n_i, 1)))
        print(f"  {names[i]:{w}s} {strength[i]:+7.3f}   mean {mean:.4f} +/- {se:.4f} over {int(n_i)} games   [{archs[i]}]")
    print(f"\n{int(games.sum() / 2)} games, {time.perf_counter() - t0:.0f}s")
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps({
            "agents": [{"name": nm, "path": str(pth), "arch": archs[k]} for k, (nm, pth) in enumerate(agents)],
            "games": games.tolist(), "wins": wins.tolist(), "strength": strength.tolist(),
            "per_map": per_map, "maps": map_names, "games_per_cell": a.games, "sonar": a.sonar,
            "seconds": round(time.perf_counter() - t0, 1)}, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
