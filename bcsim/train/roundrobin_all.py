"""Round robin with every pair playing at once, in one env batch.

roundrobin.py plays one row of the matrix per `evaluate` call, and a row takes
as long as its longest game (~11 minutes on maps-live, 2026-09-24), so 14 agents
cost 2.5 hours however few games each cell has. Here every (pair, map, side,
repeat) game is its own env in a single batch: the whole matrix takes about as
long as one row did. Each turn, the rows are split by which agent owns the side
to move, and every agent is called once on its own rows.

Same output as roundrobin.py: score matrix (draws are half a win) and a
Bradley-Terry fit.

--temps 0,0.1,0.2,0.3,0.5 (user, 2026-10-01): each side of each game draws its own temperature
from the list (0 = greedy); the json then lists every game with both temperatures, so strength
can be fitted per (agent, temperature). --record DIR also writes every game as critic
pretraining data (train/critic_data.py): the critic view and Phi of every turn.

    python -m train.roundrobin_all --agents-file agents.txt --games 2 --sonar --out rr.json
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import pathlib
import sys
import time

# --s2: the next-generation simulator (sonar v2 + the self-kill), chosen before bcsim loads.
# --grid 15 (2026-10-01): the 15x15-grid build (51 actions), so the 15x15 clones can play; agents
# trained on 14x14 read a crop of it (LSTMGreedy.grid_model) and their own 49 actions, as train/panel.py
_GRID = int(sys.argv[sys.argv.index("--grid") + 1]) if "--grid" in sys.argv else 14
if "--s2" in sys.argv:
    os.environ["BCSIM_LIB"] = str(pathlib.Path(__file__).resolve().parents[1] / "bcsim" /
                                  {14: "libbcvec_s2.so", 15: "libbcvec_s2_g15.so"}[_GRID])

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
    p.add_argument("--s2", action="store_true",
                   help="BC_SONAR2 simulator: agents trained with it (in_ch 43 / 49 actions) speak the "
                        "v2 packet with their own probabilities; older agents play as before (their 38 "
                        "grid channels, their 48 actions)")
    p.add_argument("--sample-games", type=int, default=0,
                   help="instead of every (pair, map, side) x --games: this many games, each a random pair, map "
                        "and side (critic data on many unique maps, e.g. ../maps-critic-gen)")
    p.add_argument("--temps", default="", help="each side of each game draws a temperature from this list "
                   "(0 = greedy); empty = everyone greedy")
    p.add_argument("--record", default="", help="write the games as critic pretraining data (train/critic_data.py)")
    p.add_argument("--keep", type=float, default=0.01, help="with --record: share of turns kept as samples")
    p.add_argument("--cview", type=int, default=27, help="with --record: the critic view's crop")
    p.add_argument("--source", default="roundrobin", choices=("roundrobin", "generated"),
                   help="with --record: how critic_pretrain labels these games")
    p.add_argument("--grid", type=int, default=14, choices=(14, 15),
                   help="with --s2: the simulator's grid; 15 lets 15x15 agents play, 14x14 ones get a crop")
    a = p.parse_args()

    agents = read_agents(a)
    dev = torch.device("cuda")
    maps = bcsim.load_maps(a.maps)
    map_names = [f.stem for f in sorted(pathlib.Path(a.maps).glob("*.map"))]
    n_ag = len(agents)
    if a.sample_games:
        lrng = np.random.default_rng(a.seed + 31)
        pairs = list(itertools.combinations(range(n_ag), 2))
        layout = [(*pairs[lrng.integers(len(pairs))], int(lrng.integers(len(maps))), int(lrng.integers(2)))
                  for _ in range(a.sample_games)]
    else:
        layout = [(i, j, mi, side) for i, j in itertools.combinations(range(n_ag), 2)
                  for mi in range(len(maps)) for side in (0, 1) for _ in range(a.games)]
    n = len(layout)
    print(f"{n_ag} agents, {len(maps)} maps, {a.games} games per (pair, map, side): {n} games at once, "
          f"sonar {'on' if a.sonar else 'off'}", flush=True)

    acts, archs, n_act, speaks = [], [], [], []
    for name, path in agents:
        # "name@sample=path": the same checkpoint, sampling its moves instead of the argmax
        sampled = name.endswith("@sample")
        net, ck = load_net(path, dev)
        arch = ck["args"].get("arch", "flat")
        n_act.append(int(ck["args"].get("n_actions", 48)))
        speaks.append(bool(ck["args"].get("sonar2", False)) and a.s2)
        if speaks[-1] and not a.s2:
            raise SystemExit(f"{name} was trained with sonar v2: run with --s2")
        if arch in ("lstm", "ff", "ffl"):
            from train.distill_lstm import LSTMGreedy
            acts.append(LSTMGreedy(net, dev, n))
            acts[-1].keep_probs = speaks[-1]
            acts[-1].sample = sampled
            g_ = int(ck["args"].get("grid", 14))
            if g_ > a.grid:
                raise SystemExit(f"{name} reads a {g_}x{g_} grid: run with --grid {g_}")
            acts[-1].grid_model = g_                  # a crop of a bigger simulator grid
            if getattr(net, "temp_in", False):        # played greedy: told its lowest training temperature
                net.temp_default.fill_(float(ck["args"].get("temp_min", ck["args"].get("temp", 0.4))))
        elif getattr(net, "recurrent", False):
            from train.recurrent import RecurrentGreedy
            acts.append(RecurrentGreedy(net, dev, n, per_env=256))
        else:
            acts.append(greedy(net, dev))
        archs.append(arch)
        print(f"  {name:28s} {arch:8s} {sum(q.numel() for q in net.parameters()):,} params", flush=True)

    any_wide = any(getattr(f, "wants_wide", False) for f in acts)
    any_grid = any(getattr(f, "wants_grid", False) for f in acts)
    rec = None
    env = bcsim.BattlecodeVecEnv(maps, num_envs=n, num_threads=a.threads, seed=a.seed,
                                 closure_capacity=max(8192, n * 160), wide=any_wide,
                                 sonar=a.sonar, grid=any_grid, privileged=bool(a.record),
                                 cview=a.cview if a.record else 0)
    if a.record:
        from train.critic_data import Recorder
        from train.critic_v8 import GAMMA, KAPPA
        env.set_potential_gamma(GAMMA)
        env.set_reward_v8(True, KAPPA)                   # Phi in the privileged row (no effect on play)
        rec = Recorder(a.record, a.keep, seed=a.seed, cview_w=a.cview)
    for e, (i_, j_, mi, s_) in enumerate(layout):
        env.set_opponent(e, team=-1, bot=0, map_index=mi)
        if a.s2:
            env.set_sonar2(e, [t for t, k in ((s_, i_), (1 - s_, j_)) if speaks[k]])
    agent_i = np.array([i for i, _, _, _ in layout])
    agent_j = np.array([j for _, j, _, _ in layout])
    side_i = np.array([s for _, _, _, s in layout], np.int8)      # the team agent i plays
    # per-game temperatures: agent i's side and agent j's side each draw one (0 = greedy)
    temp_list = [float(x) for x in a.temps.split(",") if x != ""]
    trng = np.random.default_rng(a.seed + 77)
    temp_i = trng.choice(temp_list, n) if temp_list else np.zeros(n)
    temp_j = trng.choice(temp_list, n) if temp_list else np.zeros(n)
    if temp_list:
        for k, f in enumerate(acts):
            if not hasattr(f, "_forward"):
                raise SystemExit(f"--temps needs grid policies; {agents[k][0]} is not one")
            f.env_temp = np.where(agent_i == k, temp_i, np.where(agent_j == k, temp_j, 0.0)).astype(np.float32)
    if rec is not None:
        for e, (i_, j_, mi, s_) in enumerate(layout):
            names, temps = [None, None], [0.0, 0.0]
            names[s_], names[1 - s_] = agents[i_][0], agents[j_][0]
            temps[s_], temps[1 - s_] = float(temp_i[e]), float(temp_j[e])
            rec.start(e, sides=names, temps=temps, observed=0, map=map_names[mi], source=a.source)
    stateful = [f for f in acts if getattr(f, "stateful", False) or getattr(f, "wants_grid", False)]

    done = np.zeros(n, bool)
    result = [None] * n
    step = 0
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
        if rec is not None:
            rec.turn(np.arange(n), obs, env.cview, ~done, step)
        owner = np.where(obs.team == side_i, agent_i, agent_j)
        acts_now = np.zeros(n, np.int32)
        for k in range(n_ag):
            rows = owner == k
            if rows.any():
                # a 48-action agent in the 49-action simulator: its own legal moves only
                o = obs if n_act[k] == obs.mask.shape[1] else dataclasses.replace(obs, mask=obs.mask[:, :n_act[k]])
                acts_now[rows] = _call(acts[k], o, rows, env.wide, env.grid)
                if speaks[k]:
                    from train.sonar2 import intents_from_probs
                    env.intent[rows] = intents_from_probs(acts[k].last_probs)
        obs, _, eps = env.step(acts_now)
        for row in eps.rows:
            e = int(row[0])
            for f in stateful:
                if hasattr(f, "forget"):
                    f.forget(e)
            if not done[e]:
                done[e] = True
                result[e] = dict(zip(COLS, row.tolist()))
                if rec is not None:
                    rec.end(e, int(row[COLS.index("winner")]), int(row[COLS.index("rounds")]))
        step += 1
    env.close()
    if rec is not None:
        rec.close()

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
            "temps": temp_list,
            # every game: agent i, agent j, map, the side i played, both temperatures, i's score (None: unfinished)
            "game_list": [[int(i_), int(j_), int(mi), int(s_), float(temp_i[e]), float(temp_j[e]),
                           None if result[e] is None else
                           (0.5 if result[e]["winner"] < 0 else float(result[e]["winner"] == s_))]
                          for e, (i_, j_, mi, s_) in enumerate(layout)],
            "seconds": round(time.perf_counter() - t0, 1)}, indent=1))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
