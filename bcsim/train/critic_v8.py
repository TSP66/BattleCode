"""Pretraining data and fit for the reward-v8 critic (train/critic_net.BoardCritic).

The critic is V(s) = -Phi(s) + f_theta(s) (REWARDS.md). Phi comes from the
engine in the privileged row; f_theta is fitted here to what the acting dragon
REALLY went on to collect: its own discounted v8 return, summed from its own
reward closures at the end of the game, so a dragon that dies early is scored
by what its chain actually paid, not by the team's result.

Conditioned on the two teams (a small learnable vector per side, BoardCritic's
team embeddings): contest team ids for replays, and a registered id per league
agent for simulated games (ids.json, so the same agent keeps the same slot).

Two sources, as many games as possible, because every position in a game
shares one future and it is the number of games that bounds what a critic
learns:

    replays   every scraped replay, played again under v8 (CPU, no policy)
    league    every pair of league agents -- the opponents the learner will
              face -- on every map, both sides, with our sonar (GPU)
    pretrain  fits the value (MSE to the v8 return) and the win/draw/loss head
              (cross-entropy), holding out games by id; saves pretrained.pt

    python -m train.critic_v8 replays --keep 0.03 --workers 10
    python -m train.critic_v8 league --agents-file agents.txt --games 3000
    python -m train.critic_v8 pretrain
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("BCSIM_LIB", str(pathlib.Path(__file__).resolve().parents[1]
                                       / "bcsim" / "libbcvec_priv.so"))

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/critic_v8"
DATA = OUT / "data_phi"             # with per-turn Phi trajectories (the target is computed at fit time)
IDS = OUT / "ids.json"
GAMMA = 0.997                     # train.py's PPO gamma, which v8's potential is discounted by
KAPPA = 1.0
LEAGUE_ID_BASE = 1_000_000        # league agents' ids, clear of any contest team id
LEARNER_ID = 999_999              # the policy being trained, during PPO
LEAGUE_GID_BASE = 800_000_000     # simulated games' ids, clear of any battle id


def v8_weights():
    import bcsim
    return bcsim.reward_vector({"v8_win": 1.0, "v8_len": 1.0, "v8_queen": 1.0, "v8_kill": 1.0,
                                "v8_exp": 1.0, "outcome": 1.0})


class Returns:
    """Per-dragon discounted returns from the env's closures.

    A dragon's k-th closure closes the transition out of its k-th turn, so its
    rewards line up with its turns in order; the return is folded backwards at
    the end of the game.
    """

    def __init__(self, w):
        self.w = w
        self.turns = defaultdict(list)       # (env, uid) -> [row or -1]
        self.rew = defaultdict(list)         # (env, uid) -> [reward]
        self.mismatch = 0

    def turn(self, env: int, uid: int, row: int) -> None:
        self.turns[(env, uid)].append(row)

    def close(self, closures) -> None:
        r = closures.comps @ self.w
        for e, u, x in zip(closures.env.tolist(), closures.uid.tolist(), r.tolist()):
            self.rew[(e, u)].append(x)

    def finish(self, env: int, out: dict) -> None:
        """Writes {row: G} for every recorded row of this env's finished game."""
        for key in [k for k in self.turns if k[0] == env]:
            rows, rew = self.turns.pop(key), self.rew.pop(key, [])
            if len(rew) != len(rows):
                self.mismatch += 1
            g = 0.0
            n = min(len(rows), len(rew))
            for k in range(n - 1, -1, -1):
                g = rew[k] + GAMMA * g
                if rows[k] >= 0:
                    out[rows[k]] = g
        for key in [k for k in self.rew if k[0] == env]:
            self.rew.pop(key)


def phi_returns(rnd: np.ndarray, team: np.ndarray, phi: np.ndarray, alpha: float) -> np.ndarray:
    """The critic's target for every turn of one game: the acting team's potential
    change, summed along that team's own turns and discounted alpha per ROUND.

    From team X's turn m the reward is Phi_X at X's next turn minus Phi_X now; X's
    last turn takes the game's last observed Phi instead (v8's terms are
    antisymmetric, so Phi_X = -Phi of the other team there). No result term and
    nothing about whether the acting dragon survives: it is a team quantity.
    Closed form G_m = alpha^-R_m * sum_{k>=m} alpha^R_k * r_k, in float64."""
    n = len(rnd)
    out = np.zeros(n, np.float64)
    if n == 0:
        return out.astype(np.float32)
    last = n - 1
    for x in (0, 1):
        idx = np.flatnonzero(team == x)
        if not len(idx):
            continue
        nxt = np.append(idx[1:], last)
        q_last = phi[last] if team[last] == x else -phi[last]
        p_next = np.where(np.arange(len(idx)) < len(idx) - 1, phi[nxt], q_last)
        r = p_next - phi[idx]
        if idx[-1] == last:
            r[-1] = 0.0                      # the game's final move: its effect is never observed
        R = rnd[idx].astype(np.float64)
        w = alpha ** R * r
        S = np.cumsum(w[::-1])[::-1]
        out[idx] = S * alpha ** (-R)
    return out.astype(np.float32)


def save_game(dest: pathlib.Path, rec: dict, G: dict, rounds: int, map_name: str = "") -> int:
    keep = [i for i in range(len(rec["round"])) if i in G]
    if not keep:
        return 0
    pick = lambda k: [rec[k][i] for i in keep]     # noqa: E731
    np.savez_compressed(
        dest,
        local=np.stack(pick("local")), scalar=np.stack(pick("scalar")), priv=np.stack(pick("priv")),
        board_bits=np.stack(pick("board_bits")), board_tail=np.stack(pick("board_tail")),
        round=np.array(pick("round"), np.int16), team=np.array(pick("team"), np.int16),
        outcome=np.array(pick("outcome"), np.int16),
        team_self=np.array(pick("team_self"), np.int32), team_foe=np.array(pick("team_foe"), np.int32),
        ret=np.array([G[i] for i in keep], np.float32), rounds=np.int16(rounds),
        turn_idx=np.array(pick("turn_idx"), np.int32),
        traj_round=np.array(rec["traj_round"], np.int16), traj_team=np.array(rec["traj_team"], np.int8),
        traj_phi=np.array(rec["traj_phi"], np.float32), map=np.array(map_name))
    return len(keep)


# ------------------------------------------------------------------ replays
def replay_one(args):
    game_json, keep_p, data = args
    import bcsim
    from bcsim.env import MAX_STEPS
    from train.critic_net import board_pack
    from train.replay_dataset import game_map
    from train.replay_read import read
    gid = int(game_json.stem)
    dest = pathlib.Path(data) / f"{gid}.npz"
    if dest.exists():
        return gid, "cached", 0
    try:
        rp = read(game_json.with_suffix(".replay"))
    except Exception as e:
        return gid, f"unreadable {type(e).__name__}", 0
    if rp.winner < 0 and rp.end_reason < 0:
        return gid, "no result", 0
    try:
        meta = json.loads(game_json.read_text())
        ids = (int(meta.get("teamAId", -1)), int(meta.get("teamBId", -1)))
    except Exception:
        meta, ids = {}, (-1, -1)
    # server replays zero every TILE range; the pearls need our copy of the map and,
    # since unswbc 1.1.0, the match's 64-bit seed (see top_fetch.py)
    map_text = game_map(rp.map_text)
    if map_text is None:
        return gid, "no copy of the map", 0
    rng = np.random.default_rng(gid)
    env = bcsim.BattlecodeVecEnv([map_text], num_envs=1, num_threads=1, seed=0,
                                 random_pearl_seed=False, max_rounds=500, privileged=True, board=True)
    env.set_potential_gamma(GAMMA)
    env.set_reward_v8(True, KAPPA)
    if meta.get("seed"):
        env.set_pearl_seed64(0, int(meta["seed"], 16))
    obs = env.reset()
    R = Returns(v8_weights())
    rec = defaultdict(list)
    z = lambda dt, *s: np.zeros((1,) + s, dt)       # noqa: E731
    kind, nst, dirs, split = z(np.int8), z(np.int8), z(np.int8, MAX_STEPS), z(np.int16)
    send, value = z(np.uint8), z(np.uint64, bcsim.SONAR_DIRS)
    G: dict = {}
    status = "ok"
    for t in rp.turns:
        did, rnd, team = int(obs.dragon_id[0]), int(obs.round[0]), int(obs.team[0])
        if (did, rnd) != (t.dragon, t.round):
            status = "diverged"
            break
        row = -1
        rec["traj_round"].append(rnd)
        rec["traj_team"].append(team)
        rec["traj_phi"].append(float(obs.priv[0, bcsim.PRIV_BASE:].sum()))
        if rng.random() < keep_p:
            row = len(rec["round"])
            rec["turn_idx"].append(len(rec["traj_round"]) - 1)
            rec["local"].append(obs.local[0].astype(np.float16))
            rec["scalar"].append(obs.scalar[0][:len(bcsim.SCALARS)].copy())
            rec["priv"].append(obs.priv[0].copy())
            rec["round"].append(rnd)
            rec["team"].append(team)
            rec["outcome"].append(1 if rp.winner < 0 else (0 if rp.winner == team else 2))
            bits, tail = board_pack(env.board[:1])
            rec["board_bits"].append(bits[0])
            rec["board_tail"].append(tail[0])
            rec["team_self"].append(ids[team])
            rec["team_foe"].append(ids[1 - team])
        R.turn(0, int(obs.uid[0]), row)
        kind[0] = t.kind if t.kind >= 0 else 2
        if len(t.dirs) > MAX_STEPS:
            status = "sprint over MAX_STEPS"
            break
        nst[0] = len(t.dirs)
        dirs[0] = 0
        dirs[0, :nst[0]] = t.dirs
        split[0] = t.split_k
        send[0] = 0
        value[0] = 0
        for _d, _v in t.sonars:
            send[0] |= np.uint8(1 << _d)
            value[0, _d] = _v
        obs, closures, eps = env.step_raw(kind, nst, dirs, split, send, value)
        R.close(closures)
        if len(eps.rows):
            R.finish(0, G)
            break
    env.close()
    if status != "ok":
        # a return needs the game's real ending; a diverged game has none we trust
        return gid, status, 0
    name = next((l[9:].strip() for l in map_text.splitlines() if l.startswith("MAP_NAME ")), "")
    n = save_game(dest, rec, G, rp.rounds, name)
    return gid, ("ok" if not R.mismatch else "ok (closure count mismatch)"), n


def replays(a) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    jobs, seen = [], set()
    folders = ([ROOT / "runs/replays" / f for f in a.folders.split(",") if f] if a.folders
               else sorted((ROOT / "runs/replays").iterdir()))
    skip_teams = {int(x) for x in a.exclude_teams.split(",") if x}
    dropped = 0
    for folder in folders:
        g = folder / "games"
        if not g.is_dir():
            continue
        for j in sorted(g.glob("*.json")):
            if j.stem in seen or not j.with_suffix(".replay").exists():
                continue
            if skip_teams:
                m = json.loads(j.read_text())
                if {m.get("teamAId"), m.get("teamBId")} & skip_teams:
                    dropped += 1
                    continue
            seen.add(j.stem)
            jobs.append((j, a.keep, str(DATA)))
    if skip_teams:
        print(f"{dropped} games dropped for teams {sorted(skip_teams)}", flush=True)
    if a.max_games:
        rng = np.random.default_rng(0)
        jobs = [jobs[i] for i in rng.choice(len(jobs), min(a.max_games, len(jobs)), replace=False)]
    print(f"{len(jobs)} distinct replayed games", flush=True)
    counts, n, t0 = defaultdict(int), 0, time.time()
    with mp.Pool(a.workers) as pool:
        for k, (gid, status, got) in enumerate(pool.imap_unordered(replay_one, jobs, chunksize=4)):
            counts[status] += 1
            n += got
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)} {time.time() - t0:.0f}s {dict(counts)} {n:,} positions", flush=True)
    print(dict(counts), f"{n:,} positions", flush=True)


# ------------------------------------------------------------------ league
def league_ids(names: list[str]) -> dict[str, int]:
    """name -> stable id, registered in ids.json the first time a name is seen."""
    OUT.mkdir(parents=True, exist_ok=True)
    reg = json.loads(IDS.read_text()) if IDS.exists() else {}
    for n_ in names:
        if n_ not in reg:
            reg[n_] = LEAGUE_ID_BASE + len(reg)
    IDS.write_text(json.dumps(reg, indent=1))
    return {n_: reg[n_] for n_ in names}


def league(a) -> None:
    import itertools
    import torch
    import bcsim
    from train.critic_net import board_pack
    from train.ratchet_lstm_train import Actor
    from train.roundrobin import read_agents

    DATA.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    agents = read_agents(a)
    ids = league_ids([n_ for n_, _ in agents])
    maps_dir = pathlib.Path(a.maps)
    weights = None
    if a.gen_maps or a.per_map:
        # the ratchet's own pool (augment.training_pool): official maps with variants,
        # generated maps at --gen-share
        from train import augment
        texts, weights, names = augment.training_pool(
            str(maps_dir), a.per_map, a.seed, 0.25, "", 0.0, a.gen_maps, a.gen_share,
            a.gen_per_map, a.pearl_hotspots, log=lambda m: print(m, flush=True))
    else:
        names = sorted(f.stem for f in maps_dir.glob("*.map"))
        texts = [(maps_dir / f"{n_}.map").read_text() for n_ in names]
    pairs = list(itertools.combinations(range(len(agents)), 2))
    rng = np.random.default_rng(a.seed)
    E = a.envs
    actors = [Actor(str(p_), dev, E * 160) for _, p_ in agents]
    print(f"{len(agents)} agents ({', '.join(n_ for n_, _ in agents)}), {len(pairs)} pairs, "
          f"self-play {a.self_play:.0%}, {len(texts)} maps, {a.games} games, sonar on, "
          f"into {DATA}", flush=True)
    env = bcsim.BattlecodeVecEnv(texts, num_envs=E, num_threads=a.threads, seed=a.seed,
                                 closure_capacity=max(8192, E * 160), privileged=True, board=True,
                                 sonar=True, grid=True, wide=any(x.wants_wide for x in actors))
    if weights is not None:
        env.set_map_weights(weights)
    env.set_potential_gamma(GAMMA)
    env.set_reward_v8(True, KAPPA)
    side_a = np.zeros(E, np.int64)                  # agent playing team 0 in each env
    side_b = np.zeros(E, np.int64)

    def assign(envs):
        for e in envs:
            if rng.random() < a.self_play:
                i = j = int(rng.integers(len(agents)))   # one Actor, both sides: states key on (env, uid)
            else:
                i, j = pairs[rng.integers(len(pairs))]
                if rng.random() < 0.5:
                    i, j = j, i
            side_a[e], side_b[e] = i, j

    assign(range(E))
    recs = [defaultdict(list) for _ in range(E)]
    seen = np.zeros(E, np.int64)
    obs = env.reset()
    # resume: a restart with the same seed continues numbering after the games on disk
    # instead of overwriting them
    base = LEAGUE_GID_BASE + a.seed * 100_000
    have = [int(f.stem) - base for f in DATA.glob("*.npz") if 0 <= int(f.stem) - base < 100_000]
    done = max(have) + 1 if have else 0
    if done:
        print(f"resuming after {done} games already in {DATA} for seed {a.seed}", flush=True)
    kept = 0
    t0 = time.time()
    stop = pathlib.Path(a.stop_file) if a.stop_file else None
    while done < a.games:
        if stop is not None and stop.exists():
            print(f"stop file {stop} found", flush=True)
            break
        grid = torch.from_numpy(env.grid).to(dev)
        local = torch.from_numpy(obs.local).to(dev)
        scalar = torch.from_numpy(obs.scalar).to(dev)
        mask = torch.from_numpy(obs.mask).to(dev).bool()
        wide = torch.from_numpy(env.wide).to(dev) if env.wide is not None else None
        owner = np.where(obs.team == 0, side_a, side_b)
        action = torch.zeros(E, dtype=torch.long, device=dev)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for k, act in enumerate(actors):
                rows = np.flatnonzero(owner == k)
                if len(rows):
                    rt = torch.as_tensor(rows, device=dev)
                    action[rt] = act.act(rows, obs, grid[rt], local[rt], scalar[rt],
                                         wide[rt] if wide is not None else None, mask[rt])
        pick = np.flatnonzero(rng.random(E) < a.keep)
        if len(pick):
            bits, tail = board_pack(env.board[pick])
        pos = {e: j for j, e in enumerate(pick.tolist())}
        for e in range(E):
            r_ = recs[e]
            r_["traj_round"].append(int(obs.round[e]))
            r_["traj_team"].append(int(obs.team[e]))
            r_["traj_phi"].append(float(obs.priv[e, bcsim.PRIV_BASE:].sum()))
            if e in pos:
                # at most --cap positions a game, a uniform sample of the game (reservoir):
                # an uncapped long game held thousands of 32 KB boards per env, and the
                # 18-plane board OOM-killed this three times on 2026-09-29. The target is
                # phi_returns over traj_*, so a slot needs no per-dragon bookkeeping.
                seen[e] += 1
                j, tm = pos[e], int(obs.team[e])
                slot_ = len(r_["round"]) if len(r_["round"]) < a.cap else int(rng.integers(seen[e]))
                if slot_ < a.cap:
                    me, them = (side_a[e], side_b[e]) if tm == 0 else (side_b[e], side_a[e])
                    vals = {"turn_idx": len(r_["traj_round"]) - 1, "local": obs.local[e].astype(np.float16),
                            "scalar": obs.scalar[e][:len(bcsim.SCALARS)].copy(), "priv": obs.priv[e].copy(),
                            "round": int(obs.round[e]), "team": tm, "board_bits": bits[j].copy(),
                            "board_tail": tail[j].copy(), "team_self": ids[agents[me][0]],
                            "team_foe": ids[agents[them][0]]}
                    for k_, v_ in vals.items():
                        if slot_ == len(r_[k_]):
                            r_[k_].append(v_)
                        else:
                            r_[k_][slot_] = v_
        obs, closures, eps = env.step(action.to(torch.int32).cpu().numpy())
        if len(closures.env):
            for e_, u_, d_ in zip(closures.env.tolist(), closures.uid.tolist(), closures.done.tolist()):
                if d_:
                    for act in actors:
                        act.release(e_, u_)
        for row in eps.as_dicts():
            e, winner, rounds = int(row["env"]), int(row["winner"]), int(row["rounds"])
            r_, recs[e] = recs[e], defaultdict(list)
            seen[e] = 0
            # the stored per-dragon return is unused (targets come from traj_phi at fit time)
            G = {i: 0.0 for i in range(len(r_["round"]))}
            r_["outcome"] = [1 if winner < 0 else (0 if winner == tm else 2) for tm in r_["team"]]
            if r_["round"] and done < a.games:
                mi = int(row["map"])
                kept += save_game(DATA / f"{base + done}.npz", r_, G, rounds,
                                  names[mi] if 0 <= mi < len(names) else "")
                done += 1
            for act in actors:
                act.release_env(e)
            assign([e])
            if done % 100 == 0 and done:
                print(f"  {done}/{a.games} games, {kept:,} positions, {time.time() - t0:.0f}s", flush=True)
    env.close()
    print(f"{done} games, {kept:,} positions into {DATA}", flush=True)


# ------------------------------------------------------------------ pretrain
def held_out(gid: int) -> bool:
    return (gid * 2654435761) % 1000 < 50          # a fixed 5% by game id


def pretrain(a) -> None:
    """Streams games from disk in shuffled chunks: the whole set is far larger than
    RAM or the GPU (~11 KB a position), and positions within a game share one future,
    so at most --per-game random positions are taken from each game. What bounds a
    critic is the number of games, not positions."""
    import torch
    import torch.nn.functional as F
    import bcsim
    from train.critic_net import BoardCritic, TeamSlots, board_unpack

    dev = torch.device("cuda")
    files = sorted(DATA.glob("*.npz"), key=lambda f: int(f.stem))
    if a.max_games:
        files = files[:a.max_games]
    val_files = [f for f in files if held_out(int(f.stem))]
    tr_files = [f for f in files if not held_out(int(f.stem))]
    rng = np.random.default_rng(0)
    prev = torch.load(OUT / "pretrained.pt", map_location="cpu", weights_only=False) if a.resume else None
    slots = TeamSlots(prev["slots"]) if prev else TeamSlots()
    # the PPO learner's row exists from the start (ratchet_lstm_train seeds it from the
    # agent it starts as), and every league agent of this critic's registry has one
    slots.slot(LEARNER_ID, add=True)
    for _name, _id in sorted((json.loads(IDS.read_text()) if IDS.exists() else {}).items(),
                             key=lambda kv: kv[1]):
        slots.slot(_id, add=True)
    KEYS = ("local", "scalar", "priv", "board_bits", "board_tail", "round", "outcome",
            "team_self", "team_foe", "ret")

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(12)             # zlib releases the GIL: 12 ms a game serial, 4 threaded

    def one(f, seed):
        d = np.load(f)
        m = len(d["round"])
        r = np.random.default_rng(seed)
        pick = r.choice(m, min(m, a.per_game), replace=False) if m > a.per_game else np.arange(m)
        g = {k: d[k][pick] for k in KEYS if k != "ret"}
        target = phi_returns(d["traj_round"], d["traj_team"], d["traj_phi"], a.alpha)
        g["ret"] = target[d["turn_idx"][pick]]
        return g

    def load(fs, cap_rows=0):
        cols = {k: [] for k in KEYS}
        n = 0
        seeds = rng.integers(1 << 31, size=len(fs))
        for g in pool.map(one, fs, seeds):
            for k in KEYS:
                cols[k].append(g[k])
            n += len(g["round"])
            if cap_rows and n >= cap_rows:
                break
        D = {k: np.concatenate(v) for k, v in cols.items()}
        D["team_self"] = np.array([slots.slot(int(x), add=True) for x in D["team_self"]], np.int64)
        D["team_foe"] = np.array([slots.slot(int(x), add=True) for x in D["team_foe"]], np.int64)
        return {k: torch.from_numpy(v) for k, v in D.items()}

    Vd = load(val_files, a.val_cap)
    Vd = {k: v.to(dev) for k, v in Vd.items()}
    # Phi goes IN as a feature, not on as an offset. The V = -Phi + f_theta anchor of
    # REWARDS.md measured ill-conditioned on real games (2026-09-25, 22k positions):
    # var(G) 0.088 but var(G + Phi) 0.31, so f has to cancel -Phi almost exactly;
    # -Phi alone explains -2.5 of the variance and even an oracle -Phi + g^left *
    # true result only -0.49 (the derivation assumes a dragon lives to the end).
    # So the value head regresses the return directly, with all 13 privileged
    # values -- Phi's five terms included -- in the conditioning (n_phi = 0).
    n_priv = Vd["priv"].shape[1]
    n_phi = 0
    n_sc = Vd["scalar"].shape[1]
    # the board layout comes from the data: 11 planes (10 packed) before 2026-09-29, 18 (12) after
    n_bin = Vd["board_bits"].shape[1] * 8 // (64 * 64)
    board_ch = n_bin + Vd["board_tail"].shape[1]
    if a.d4 and board_ch != 18:
        raise SystemExit(f"--d4 needs the 18-plane board; this data has {board_ch}")
    print(f"board: {board_ch} planes ({n_bin} packed){'; D4 augmentation' if a.d4 else ''}"
          f"{'; local window ZEROED' if a.drop_local else ''}", flush=True)
    print(f"{len(tr_files)} training games, {len(val_files)} held out ({len(Vd['ret']):,} positions); "
          f"<= {a.per_game} positions a game; priv {n_priv} + phi {n_phi}", flush=True)
    net = BoardCritic(bcsim.N_CHANNELS, n_sc, 1, board_ch, n_priv=n_priv, n_phi=n_phi,
                      resid_scale=2.0, dropout=a.dropout).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    est_rows = len(tr_files) * min(a.per_game, 150)
    steps = a.epochs * (est_rows // a.batch + len(tr_files) // a.chunk_games + 1)
    if prev:
        # more epochs on a fitted critic: its weights, a cosine decay from --lr, and the
        # held-out error it has to beat before anything is replaced
        net.load_state_dict(prev["critic"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        best_prev = min(r["val_mse"] for r in prev["log"])
        print(f"resuming from pretrained.pt (best held-out mse {best_prev:.5f})", flush=True)
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.03)
        best_prev = None

    from train import d4 as D4

    def fwd(D, ix, aug=False):
        b = board_unpack(D["board_bits"][ix], D["board_tail"][ix], board_ch, 64, device=dev, n_binary=n_bin)
        loc, sc = D["local"][ix].float(), D["scalar"][ix].float()
        if aug:
            # a random flip/rotation per position (train/d4.py, exact against the engine);
            # the partner planes are read raw, so they go back to 0..255 for the transform
            fx, fy, tr = D4.random_syms(len(ix), dev)
            W = torch.round(sc[:, D4.SC_MW] * 64).long(); H = torch.round(sc[:, D4.SC_MH] * 64).long()
            b[:, 14:18] *= 255.0
            b = D4.board(b, W, H, fx, fy, tr)
            b[:, 14:18] /= 255.0
            loc, sc = D4.local(loc, fx, fy, tr), D4.scalars(sc, fx, fy, tr)
        if a.drop_local:
            loc = torch.zeros_like(loc)
        return net(loc, sc, torch.ones(len(ix), 1, device=dev),
                   D["priv"][ix].float(), b, D["team_self"][ix], D["team_foe"][ix],
                   None, D["round"][ix].float())

    log, best, step = (list(prev["log"]) if prev else []), best_prev, 0
    t0 = time.time()
    for ep in range(a.epochs):
        order = rng.permutation(len(tr_files))
        tl = tc = n_seen = 0.0
        net.train()
        chunks = [[tr_files[i] for i in order[c0:c0 + a.chunk_games]]
                  for c0 in range(0, len(order), a.chunk_games)]
        # the next chunk loads in the background while the GPU trains on this one
        loader = ThreadPoolExecutor(1)
        nxt = loader.submit(load, chunks[0])
        for ci in range(len(chunks)):
            Dc = nxt.result()
            if ci + 1 < len(chunks):
                nxt = loader.submit(load, chunks[ci + 1])
            Dc = {k: v.to(dev, non_blocking=True) for k, v in Dc.items()}
            perm = torch.randperm(len(Dc["ret"]), device=dev)
            for s0 in range(0, len(perm), a.batch):
                ix = perm[s0:s0 + a.batch]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, v = fwd(Dc, ix, aug=a.d4)
                lv = F.mse_loss(v.float(), Dc["ret"][ix])
                lc = F.cross_entropy(logits.float(), Dc["outcome"][ix].long(), label_smoothing=a.smooth)
                loss = lv + a.ce_weight * lc
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                if step < steps - 1:
                    sched.step()
                step += 1
                tl += lv.item() * len(ix)
                tc += lc.item() * len(ix)
                n_seen += len(ix)
            del Dc
        net.eval()
        vs = []
        lcs = 0.0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s0 in range(0, len(Vd["ret"]), a.batch):
                ix = torch.arange(s0, min(s0 + a.batch, len(Vd["ret"])), device=dev)
                logits, v = fwd(Vd, ix)
                vs.append(v.float())
                lcs += F.cross_entropy(logits.float(), Vd["outcome"][ix].long(), reduction="sum").item()
        v_, r_, rnd_ = torch.cat(vs), Vd["ret"], Vd["round"]
        ev = float(1 - (r_ - v_).var() / r_.var())
        row = {"epoch": ep + (len(prev["log"]) if prev else 0), "train_mse": tl / n_seen, "train_ce": tc / n_seen,
               "val_mse": float(((r_ - v_) ** 2).mean()), "val_ev": round(ev, 4),
               "val_ce": lcs / len(r_), "positions": int(n_seen), "minutes": round((time.time() - t0) / 60, 1)}
        for lo, hi in ((0, 100), (100, 250), (250, 500)):
            m = (rnd_ >= lo) & (rnd_ < hi)
            if m.sum() > 100:
                row[f"ev_r{lo}"] = round(float(1 - (r_[m] - v_[m]).var() / r_[m].var()), 4)
        log.append(row)
        print(json.dumps(row), flush=True)
        if best is None or row["val_mse"] < best:
            best = row["val_mse"]
            name = f"pretrained_{a.tag}.pt" if a.tag else "pretrained.pt"
            tmp = OUT / (name + ".tmp")
            torch.save({"critic": net.state_dict(), "slots": slots.as_dict(), "n_priv": n_priv,
                        "n_phi": n_phi, "resid_scale": 2.0, "board_ch": board_ch, "n_scalars": n_sc,
                        "gamma": GAMMA, "kappa": KAPPA, "alpha": a.alpha, "target": "phi_returns", "log": log,
                        "drop_local": bool(a.drop_local), "d4": bool(a.d4),
                        "ids": json.loads(IDS.read_text()) if IDS.exists() else {}}, tmp)
            tmp.replace(OUT / name)      # atomic: a trainer starting now reads old or new, never half
    (OUT / (f"pretrain_log_{a.tag}.json" if a.tag else "pretrain_log.json")).write_text(json.dumps(log, indent=1))


def main() -> None:
    global OUT, DATA, IDS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=str(OUT),
                   help="critic dir: data_phi/, ids.json, pretrained.pt (runs/critic_v8b from 2026-09-28: "
                        "reward changed, so the old data_phi must not be mixed in)")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replays")
    r.add_argument("--keep", type=float, default=0.03, help="share of turns recorded")
    r.add_argument("--workers", type=int, default=10)
    r.add_argument("--max-games", type=int, default=0)
    r.add_argument("--folders", default="", help="runs/replays subfolders, comma separated; empty = all")
    r.add_argument("--exclude-teams", default="", help="contest team ids whose games are skipped")
    lg = sub.add_parser("league")
    lg.add_argument("--agents", default="")
    lg.add_argument("--agents-file", default="")
    lg.add_argument("--maps", default=str(ROOT / "maps-all"))
    lg.add_argument("--games", type=int, default=3000)
    lg.add_argument("--keep", type=float, default=0.03)
    lg.add_argument("--cap", type=int, default=300, help="stored positions per game at most (reservoir)")
    lg.add_argument("--envs", type=int, default=256)
    lg.add_argument("--threads", type=int, default=12)
    lg.add_argument("--seed", type=int, default=0)
    lg.add_argument("--self-play", type=float, default=0.0, help="share of games an agent plays itself")
    lg.add_argument("--per-map", type=int, default=0, help="augmented variants per official map (0 = none)")
    lg.add_argument("--gen-maps", default="", help="generated maps (train/loong_mapgen.py)")
    lg.add_argument("--gen-share", type=float, default=0.65)
    lg.add_argument("--gen-per-map", type=int, default=3)
    lg.add_argument("--pearl-hotspots", action="store_true")
    lg.add_argument("--stop-file", default="", help="finish cleanly once this file exists")
    t = sub.add_parser("pretrain")
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=0.05)
    t.add_argument("--dropout", type=float, default=0.1)
    t.add_argument("--smooth", type=float, default=0.05)
    t.add_argument("--ce-weight", type=float, default=0.5)
    t.add_argument("--max-games", type=int, default=0)
    t.add_argument("--per-game", type=int, default=150, help="positions taken from each game at most")
    t.add_argument("--alpha", type=float, default=0.95,
                   help="discount per ROUND of the team's potential changes (0.95; 0.96 on 28-29 Sep)")
    t.add_argument("--tag", default="", help="write pretrained_<tag>.pt (and its log) instead of pretrained.pt")
    t.add_argument("--drop-local", action="store_true", help="zero the acting dragon's 7x7 window (ablation)")
    t.add_argument("--d4", action="store_true", help="random flip/rotation per training position (18-plane data)")
    t.add_argument("--resume", action="store_true", help="continue from pretrained.pt (more epochs); it is "
                   "replaced only when held-out mse beats its best")
    t.add_argument("--chunk-games", type=int, default=1500, help="games loaded at a time")
    t.add_argument("--val-cap", type=int, default=150_000, help="held-out positions kept resident")
    a = p.parse_args()
    OUT = pathlib.Path(a.out_dir)
    DATA = OUT / "data_phi"
    IDS = OUT / "ids.json"
    {"replays": replays, "league": league, "pretrain": pretrain}[a.cmd](a)


if __name__ == "__main__":
    main()
