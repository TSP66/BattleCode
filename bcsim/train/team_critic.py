"""Team-level critic: predicts how the game ends for the acting dragon's team.

ft1-ft3 fitted a value to each dragon's own return. Win and lose are paid only
to survivors, so a dragon that dies early never sees the result, and those
returns were close to unpredictable (explained variance 0.07-0.12). The
team's final result is predictable (critic_probe: EV 0.37 offline from the 8
global features). So this critic has two heads on one trunk:

  * `team`: win / draw / loss logits for the acting dragon's team. Its value is
    p_win - p_loss, in the units of the terminal reward (win +1, lose -1).
  * `self`: the dragon's own shaped return (the dense potential terms, no
    result), one output per opponent slot, as net.Critic has.

Inputs: the 7x7 window and scalars, the 8 privileged global features and a
one-hot of the opponent slot (all zeros = unknown, as in replays).

    extract    replays are played again in the simulator; a random
               --keep share of all turns, both teams, is recorded with the
               global features and how the game ended for that team
    pretrain   fits the team head on them (held-out games by id), reporting
               log loss, EV and calibration by stage of the game; saves
               runs/team_critic/pretrained.pt for finetune_team.py

    python -m train.team_critic extract --games 250 --keep 0.08
    python -m train.team_critic pretrain
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("BCSIM_LIB", str(pathlib.Path(__file__).resolve().parents[1]
                                       / "bcsim" / "libbcvec_priv.so"))

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/team_critic"
# Positions written by this run. The board planes made the stored layout
# different, and extract_one treats an existing file as already done, so new
# data goes to its own directory rather than silently skipping every game that
# the boardless extract had already covered.
DATA = OUT / "data_board"
def replay_teams() -> list[str]:
    """Every team folder under runs/replays that actually holds replays.

    This was a hand-written list of six, which quietly ignored the other five
    folders on disk -- 6,000 games. What bounds the critic is how many *games*
    it has seen, because every position in a game shares one outcome, so the
    list is now whatever has been scraped, and a new scrape is picked up
    without editing this file.
    """
    root = ROOT / "runs/replays"
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and any((d / "games").glob("*.replay")))


TEAMS = replay_teams()
# The BASE privileged features. The engine's row is wider than this under reward
# v8 -- the extra entries are Phi's components -- and a critic that wants them
# passes n_phi and the full row. Anything here that predates v8 slices to the
# base width, which is why this is PRIV_BASE and not PRIV_COUNT.
# Must equal bcsim.PRIV_BASE; kept as a literal because bcsim is imported
# lazily here, after BCSIM_LIB is pinned to libbcvec_priv.so. Asserted in
# extract_one, which does have bcsim in scope.
N_PRIV = 8
# class index -> value, in units of the terminal reward
OUTCOME_VALUE = (1.0, 0.0, -1.0)                # win, draw, loss
ROUND_BUCKETS = (0, 100, 200, 300, 400, 501)


# ------------------------------------------------------------------ model
def build(n_channels: int, n_scalars: int, n_context: int, width: int = 128,
          blocks: int = 6, hidden: int = 512):
    import torch
    import torch.nn as nn
    from train.net import ResBlock

    class TeamCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_context = n_context
            self.stem = nn.Sequential(
                nn.Conv2d(n_channels, width, 3, padding=1, bias=False),
                nn.GroupNorm(8, width), nn.SiLU())
            self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
            self.flat = nn.Sequential(nn.Conv2d(width, 32, 1, bias=False),
                                      nn.GroupNorm(8, 32), nn.SiLU(), nn.Flatten())
            # the global features get their own path: they carry most of the
            # signal and should not have to squeeze through the scalar MLP
            self.glob = nn.Sequential(nn.Linear(N_PRIV + n_context, 128), nn.SiLU(),
                                      nn.Linear(128, 128), nn.SiLU())
            self.scalar = nn.Sequential(nn.Linear(n_scalars, 128), nn.SiLU())
            self.fuse = nn.Sequential(nn.Linear(32 * 7 * 7 + 256, hidden), nn.SiLU(),
                                      nn.Linear(hidden, hidden), nn.SiLU())
            self.team = nn.Linear(hidden, 3)
            self.self_v = nn.Linear(hidden, n_context)
            nn.init.orthogonal_(self.team.weight, 0.1)
            nn.init.zeros_(self.team.bias)
            nn.init.orthogonal_(self.self_v.weight, 1.0)
            nn.init.zeros_(self.self_v.bias)

        def forward(self, local, scalar, context, priv):
            """Returns (team logits (B, 3), own shaped value (B,)). context is a
            one-hot (B, n_context), or zeros when the opponent is unknown."""
            x = self.flat(self.blocks(self.stem(local)))
            g = self.glob(torch.cat([priv, context], 1))
            h = self.fuse(torch.cat([x, self.scalar(scalar), g], 1))
            v = (self.self_v(h) * context).sum(1)
            return self.team(h), v

    return TeamCritic()


def team_value(logits):
    """p_win - p_loss: the expected terminal reward."""
    p = logits.float().softmax(-1)
    return p[..., 0] - p[..., 2]


def load(path: str, dev, n_context: int | None = None):
    """Loads a critic saved by pretrain or finetune_team. A pretrained one has
    no opponent slots; with n_context given, its self head is rebuilt at that
    size (it was never trained in pretraining)."""
    import torch
    import bcsim
    assert N_PRIV == bcsim.PRIV_BASE, "N_PRIV is out of step with PRIV_BASE"
    ck = torch.load(path, map_location=dev, weights_only=False)
    c = ck["critic_args"]
    nc = n_context or c["n_context"]
    sd = ck["critic"]
    # Size the scalar branch from the checkpoint, not from the env. bcsim's row
    # is 708 wide now (bc_memory.hpp) while this critic was pretrained on the
    # base 14, and it is frozen, so it keeps taking the 14 it knows -- callers
    # pass scalar[:, :n_scalars]. Building it at the env's width instead would
    # just fail to load.
    n_scalars = sd["scalar.0.weight"].shape[1]
    net = build(bcsim.N_CHANNELS, n_scalars, nc, c["width"], c["blocks"]).to(dev)
    if nc != c["n_context"]:
        sd = {k: v for k, v in sd.items() if not k.startswith("self_v.")}
        # the context columns of the first global layer start at zero weight
        w = sd["glob.0.weight"]
        grown = torch.zeros(w.shape[0], N_PRIV + nc, device=w.device, dtype=w.dtype)
        grown[:, :w.shape[1]] = w
        sd["glob.0.weight"] = grown
    missing, unexpected = net.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("self_v.")] + list(unexpected)
    if bad:
        raise RuntimeError(f"critic checkpoint does not fit: {bad}")
    return net, {**c, "n_context": nc}


# ------------------------------------------------------------------ extract
def extract_one(args):
    game_json, keep_p, seed = args
    import bcsim
    from bcsim.env import MAX_STEPS
    from train.critic_net import board_pack
    from train.replay_read import read
    gid = int(game_json.stem)
    dest = DATA / f"{gid}.npz"
    if dest.exists():
        return gid, "cached", 0
    try:
        rp = read(game_json.with_suffix(".replay"))
    except Exception as e:                      # a truncated download
        return gid, f"unreadable {type(e).__name__}", 0
    if rp.winner < 0 and rp.end_reason < 0:
        return gid, "no result", 0
    rng = np.random.default_rng(seed + gid)
    # Which contest team played each side, for the critic's matchup embeddings.
    # -1 is unknown, which is what a game with no metadata gets.
    try:
        meta = json.loads(game_json.read_text())
        team_ids = (int(meta.get("teamAId", -1)), int(meta.get("teamBId", -1)))
    except Exception:
        team_ids = (-1, -1)
    env = bcsim.BattlecodeVecEnv([rp.map_text], num_envs=1, num_threads=1, seed=0,
                                 random_pearl_seed=False, max_rounds=500, privileged=True,
                                 board=True)
    obs = env.reset()
    rec = {k: [] for k in ("local", "scalar", "priv", "round", "team", "outcome",
                           "board_bits", "board_tail", "team_self", "team_foe")}
    z = lambda dt, *s: np.zeros((1,) + s, dt)
    kind, nst, dirs, split = z(np.int8), z(np.int8), z(np.int8, MAX_STEPS), z(np.int16)
    send, value = z(np.uint8), z(np.uint64, bcsim.SONAR_DIRS)
    status = "ok"
    for i, t in enumerate(rp.turns):
        did, rnd, team = int(obs.dragon_id[0]), int(obs.round[0]), int(obs.team[0])
        if (did, rnd) != (t.dragon, t.round):
            status = "diverged"
            break
        if rng.random() < keep_p:
            rec["local"].append(obs.local[0].astype(np.float16))
            # the base scalars only. The env's row is 708 wide now, but the
            # extra 694 are the policy's flattened memory features and the
            # critic has the whole board instead -- and the self-play path
            # stores the base row, so both have to agree to be mixed.
            rec["scalar"].append(obs.scalar[0][:len(bcsim.SCALARS)].copy())
            rec["priv"].append(obs.priv[0].copy())
            rec["round"].append(rnd)
            rec["team"].append(team)
            rec["outcome"].append(1 if rp.winner < 0 else (0 if rp.winner == team else 2))
            bits, tail = board_pack(env.board[:1])
            rec["board_bits"].append(bits[0])
            rec["board_tail"].append(tail[0])
            # the board is already written from the acting team's side, so
            # "self" is whichever contest team that is
            rec["team_self"].append(team_ids[team])
            rec["team_foe"].append(team_ids[1 - team])
        kind[0] = t.kind if t.kind >= 0 else 2
        if len(t.dirs) > MAX_STEPS:
            status = "sprint over MAX_STEPS"
            break
        nst[0] = len(t.dirs)
        dirs[0] = 0
        dirs[0, :nst[0]] = t.dirs
        split[0] = t.split_k
        # Every ray the dragon really cast, with its direction and full 64-bit
        # payload. Reading only one 32-bit value along the facing (which this
        # replaced) both truncated wide payloads and dropped every ray after the
        # first, so a replayed game diverged from the one the engine played.
        send[0] = 0
        value[0] = 0
        for _d, _v in t.sonars:
            send[0] |= np.uint8(1 << _d)
            value[0, _d] = _v
        obs, _, _ = env.step_raw(kind, nst, dirs, split, send, value)
    env.close()
    # a diverged game's early positions are still true positions of a game
    # whose result is known, so they are kept
    if not rec["round"]:
        return gid, "empty", 0
    np.savez_compressed(
        dest,
        **{k: np.stack(v) for k, v in rec.items()
           if k in ("local", "scalar", "priv", "board_bits", "board_tail")},
        **{k: np.array(v, np.int16) for k, v in rec.items()
           if k in ("round", "team", "outcome")},
        **{k: np.array(v, np.int32) for k, v in rec.items()
           if k in ("team_self", "team_foe")},
        rounds=np.int16(rp.rounds))
    return gid, status, len(rec["round"])


# ------------------------------------------------------- invented-map games
# Replays only exist for official maps, so a critic fitted on them alone has
# never seen maps-gen -- and those are 40% of training under --live-share 0.6.
# This plays the simulator's own games there and records them in the same
# format, so the same pretrain can use both.
SELFPLAY_BASE = 900_000_000        # game ids well clear of any server battle id


def selfplay(a) -> None:
    import torch
    import bcsim
    from train.critic_net import board_pack
    from train.finetune import load_policy
    from train.net import masked_logits

    DATA.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets = [load_policy(p_, dev)[0].eval() for p_ in a.ckpts.split(",") if p_]
    if not nets:
        raise SystemExit("--ckpts needs at least one policy")
    maps_dir = pathlib.Path(a.maps)
    names = sorted(f.stem for f in maps_dir.glob("*.map"))
    texts = [(maps_dir / f"{n}.map").read_text() for n in names]
    rng = np.random.default_rng(a.seed)
    n_base = len(bcsim.SCALARS) if a.scalars == "base" else None

    written = kept = 0
    for mi, (name, text) in enumerate(zip(names, texts)):
        env = bcsim.BattlecodeVecEnv([text], num_envs=a.envs, num_threads=a.threads,
                                     seed=a.seed + mi, privileged=True, board=True)
        obs = env.reset()
        buf = [[] for _ in range(a.envs)]
        done_here = 0
        while done_here < a.games:
            net = nets[rng.integers(len(nets))]
            with torch.inference_mode():
                lg, _ = net(torch.from_numpy(obs.local).to(dev),
                            torch.from_numpy(obs.scalar).to(dev))
                lg = masked_logits(lg.float(), torch.from_numpy(obs.mask).to(dev).bool())
                # sampled, not greedy: the critic wants the spread of positions a
                # game really visits, and greedy self-play repeats one line
                act = torch.distributions.Categorical(logits=lg).sample()
            keep = rng.random(len(buf)) < a.keep
            picked = np.flatnonzero(keep)
            if len(picked):
                bits, tail = board_pack(env.board[picked])
            for j, e in enumerate(picked):
                buf[e].append((obs.local[e].astype(np.float16),
                               obs.scalar[e][:n_base].copy(), obs.priv[e].copy(),
                               int(obs.round[e]), int(obs.team[e]), bits[j], tail[j]))
            obs, _, st = env.step(act.to(torch.int32).cpu().numpy())
            for row in st.as_dicts():
                e, winner, rounds = int(row["env"]), int(row["winner"]), int(row["rounds"])
                rec, buf[e] = buf[e], []
                if not rec or done_here >= a.games:
                    continue
                gid = SELFPLAY_BASE + mi * 100_000 + done_here
                np.savez_compressed(
                    DATA / f"{gid}.npz",
                    local=np.stack([r[0] for r in rec]),
                    scalar=np.stack([r[1] for r in rec]),
                    priv=np.stack([r[2] for r in rec]),
                    round=np.array([r[3] for r in rec], np.int16),
                    team=np.array([r[4] for r in rec], np.int16),
                    outcome=np.array([1 if winner < 0 else (0 if winner == r[4] else 2)
                                      for r in rec], np.int16),
                    board_bits=np.stack([r[5] for r in rec]),
                    board_tail=np.stack([r[6] for r in rec]),
                    # self-play has no contest team on either side: both
                    # unknown, which is the embedding slot that starts at zero
                    team_self=np.full(len(rec), -1, np.int32),
                    team_foe=np.full(len(rec), -1, np.int32),
                    rounds=np.int16(rounds))
                done_here += 1
                kept += len(rec)
        env.close()
        written += done_here
        print(f"  {name:<14} {done_here} games", flush=True)
    print(f"{written} games, {kept:,} positions into {DATA}")


def extract(a) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    jobs, seen = [], set()
    for team in TEAMS:
        games = sorted((ROOT / "runs/replays" / team / "games").glob("*.json"))
        # a game between two tracked teams sits in both folders: once is enough
        games = [g for g in games if g.with_suffix(".replay").exists() and g.stem not in seen]
        pick = rng.choice(len(games), min(a.games, len(games)), replace=False) if games else []
        for i in pick:
            seen.add(games[i].stem)
            jobs.append((games[i], a.keep, 0))
    print(f"{len(jobs)} games to extract", flush=True)
    counts, n = {}, 0
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for k, (gid, status, got) in enumerate(pool.imap_unordered(extract_one, jobs)):
            counts[status] = counts.get(status, 0) + 1
            n += got
            if k % 100 == 0:
                print(f"  {k}/{len(jobs)} {time.time() - t0:.0f}s {counts}", flush=True)
    print(counts, f"{n:,} new positions")


# ------------------------------------------------------------------ pretrain
def held_out(gid: int) -> bool:
    """A fixed split by game id, so every run holds out the same games."""
    return (gid * 2654435761) % 1000 < 100


def calibration(p: np.ndarray, y: np.ndarray, rounds: np.ndarray) -> dict:
    """EV of p_win - p_loss against the result, per stage of the game, and
    the expected calibration error of p_win over ten bins."""
    target = np.array(OUTCOME_VALUE)[y]
    pred = p[:, 0] - p[:, 2]
    out = {}
    for lo, hi in zip(ROUND_BUCKETS[:-1], ROUND_BUCKETS[1:]):
        m = (rounds >= lo) & (rounds < hi)
        # a handful of games all ending the same way has no variance to explain
        if m.sum() < 50 or target[m].var() < 0.05:
            continue
        out[f"ev_r{lo}"] = round(float(1 - ((target[m] - pred[m]) ** 2).mean()
                                       / (target[m].var() + 1e-9)), 4)
    bins = np.minimum((p[:, 0] * 10).astype(int), 9)
    won = (y == 0).astype(float)
    ece = sum(abs(p[bins == b, 0].mean() - won[bins == b].mean()) * (bins == b).mean()
              for b in range(10) if (bins == b).any())
    out["ece_win"] = round(float(ece), 4)
    if target.var() >= 0.05:
        out["ev"] = round(float(1 - ((target - pred) ** 2).mean() / target.var()), 4)
    return out


def pretrain(a) -> None:
    import torch
    import torch.nn.functional as F
    import bcsim
    from train.critic_net import BoardCritic, TeamSlots, board_unpack
    torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    dev = torch.device("cuda")
    files = sorted(DATA.glob("*.npz"))
    if not files:
        raise SystemExit(f"no positions in {DATA}; run `extract` first")
    # Everything is held in memory, and a board is 9.2 KB a position even packed,
    # so the whole scrape does not fit: ~1.1M positions is ~12.5 GB. Cap by
    # *games*, not positions, because the outcome label is per game and a random
    # subset of games keeps the held-out split (which is a hash of the game id)
    # representative.
    if a.max_games and len(files) > a.max_games:
        pick = np.random.default_rng(0).choice(len(files), a.max_games, replace=False)
        files = [files[i] for i in sorted(pick)]
        print(f"capped to {len(files)} games of the scrape", flush=True)
    # sized first and filled in place: a list-and-concatenate load would hold
    # the whole set twice, and it is several GB
    sizes = {"tr": 0, "va": 0}
    for f in files:
        sizes["va" if held_out(int(f.stem)) else "tr"] += len(np.load(f)["round"])
    probe = np.load(files[0])
    if "board_bits" not in probe.files:
        raise SystemExit(f"{files[0]} has no board planes; this is data from the "
                         f"boardless extract. Re-extract into {DATA}.")
    # the width the extract actually stored: bcsim's row grew to 708 when
    # bc_memory.hpp landed, and data written before that is still the base 14
    n_scalars = int(probe["scalar"].shape[1])
    n_bits = int(probe["board_bits"].shape[1])
    tail_shape = tuple(probe["board_tail"].shape[1:])
    board_side = tail_shape[-1]
    shapes = {"local": ((bcsim.N_CHANNELS, 7, 7), np.float16),
              "scalar": ((n_scalars,), np.float32), "priv": ((N_PRIV,), np.float32),
              "board_bits": ((n_bits,), np.uint8), "board_tail": (tail_shape, np.uint8),
              "team_self": ((), np.int32), "team_foe": ((), np.int32),
              "round": ((), np.int16), "outcome": ((), np.int16)}
    arrs = {s: {k: np.empty((n,) + sh, dt) for k, (sh, dt) in shapes.items()}
            for s, n in sizes.items()}
    fill = {"tr": 0, "va": 0}
    for f in files:
        d = np.load(f)
        s = "va" if held_out(int(f.stem)) else "tr"
        m = len(d["round"])
        for k in shapes:
            arrs[s][k][fill[s]:fill[s] + m] = d[k]
        fill[s] += m

    # Embedding slots are handed out over the training split only, in sorted id
    # order so the table is the same whatever order the files came in. A team
    # that appears only in held-out games stays unknown, which is what it would
    # be on a team we have never played.
    slots = TeamSlots()
    for tid in sorted(set(arrs["tr"]["team_self"].tolist())
                      | set(arrs["tr"]["team_foe"].tolist())):
        slots.slot(tid, add=True)
    for s in ("tr", "va"):
        for k in ("team_self", "team_foe"):
            arrs[s][k] = np.array([slots.slot(int(t)) for t in arrs[s][k]], np.int64)

    data = {s: {k: torch.from_numpy(v) for k, v in p_.items()} for s, p_ in arrs.items()}
    tr, va = data["tr"], data["va"]
    n_va_games = sum(held_out(int(f.stem)) for f in files)
    print(f"{len(files)} games: {len(tr['round']):,} train positions, {len(va['round']):,} "
          f"held out ({n_va_games} games)", flush=True)
    print(f"board {bcsim.BOARD_CH}x{board_side}x{board_side} packed to "
          f"{n_bits + int(np.prod(tail_shape))} bytes a position; "
          f"{len(slots)} contest teams have their own embedding", flush=True)
    base = np.bincount(tr["outcome"].numpy(), minlength=3) / len(tr["outcome"])
    print("outcome share (train) win/draw/loss", base.round(3), flush=True)

    torch.manual_seed(0)
    net = BoardCritic(bcsim.N_CHANNELS, n_scalars, 1, bcsim.BOARD_CH,
                      board_side=board_side, n_priv=N_PRIV,
                      near_width=a.width, near_blocks=a.blocks,
                      dropout=a.dropout).to(dev)
    print(f"{sum(q.numel() for q in net.parameters()):,} parameters", flush=True)
    # An exponential moving average of the weights is what gets evaluated and
    # saved. The failure this is here for is the one we have already hit: the
    # critic fits, then falls off a cliff within an epoch. An average over
    # recent steps does not follow it off.
    ema = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    n = len(tr["round"])
    steps = a.epochs * ((n + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    ctx0 = torch.zeros(max(a.batch, 4096), 1, device=dev)

    def fwd(d, idx):
        b = len(idx)
        board = board_unpack(d["board_bits"][idx], d["board_tail"][idx],
                             bcsim.BOARD_CH, board_side, device=dev)
        logits, _ = net(d["local"][idx].to(dev).float(), d["scalar"][idx].to(dev),
                        ctx0[:b], d["priv"][idx].to(dev), board,
                        team_self=d["team_self"][idx].to(dev),
                        team_foe=d["team_foe"][idx].to(dev),
                        iteration=None,
                        rnd=d["round"][idx].to(dev).float())
        return logits

    def evaluate(use_ema: bool = True):
        backup = None
        if use_ema:
            backup = {k: v.detach().clone() for k, v in net.state_dict().items()}
            net.load_state_dict({k: v.to(backup[k].dtype) for k, v in ema.items()})
        net.eval()
        ps = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(va["round"]), a.eval_batch):
                idx = torch.arange(s, min(s + a.eval_batch, len(va["round"])))
                ps.append(fwd(va, idx).float().softmax(1).cpu())
        net.train()
        if backup is not None:
            net.load_state_dict(backup)
        return torch.cat(ps).numpy()

    y_va = va["outcome"].numpy()
    ll_base = float(-np.log(base[y_va] + 1e-9).mean())
    best, log = None, []
    t0 = time.time()
    for ep in range(a.epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for s in range(0, n, a.batch):
            idx = perm[s:s + a.batch]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = fwd(tr, idx)
            loss = F.cross_entropy(logits.float(), tr["outcome"][idx].to(dev).long(),
                                   label_smoothing=a.smooth)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    ema[k].mul_(a.ema).add_(v.detach().float(), alpha=1.0 - a.ema)
            tot += loss.detach().item() * len(idx)
        p = evaluate()
        ll = float(-np.log(p[np.arange(len(y_va)), y_va] + 1e-9).mean())
        row = {"epoch": ep, "train_ce": round(tot / n, 4), "val_ll": round(ll, 4),
               "baseline_ll": round(ll_base, 4),
               "acc": round(float((p.argmax(1) == y_va).mean()), 4),
               **calibration(p, y_va, va["round"].numpy()), "elapsed": round(time.time() - t0)}
        log.append(row)
        print(json.dumps(row), flush=True)
        if best is None or ll < best:
            best = ll
            torch.save({"critic": {k: v for k, v in ema.items()}, "eval": row,
                        "critic_args": {"width": a.width, "blocks": a.blocks,
                                        "n_context": 1, "n_priv": N_PRIV,
                                        "arch": "board", "board_ch": bcsim.BOARD_CH,
                                        "board_side": board_side,
                                        "dropout": a.dropout,
                                        "team_slots": slots.as_dict()}},
                       OUT / "pretrained_board.pt")
    (OUT / "pretrain_board_log.json").write_text(json.dumps(log, indent=1))
    print(f"best held-out log loss {best:.4f} (baseline {ll_base:.4f}) "
          f"-> {OUT / 'pretrained_board.pt'}")

def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--games", type=int, default=250, help="per team folder")
    e.add_argument("--keep", type=float, default=0.08, help="share of turns recorded")
    e.add_argument("--workers", type=int, default=4)
    t = sub.add_parser("pretrain")
    t.add_argument("--width", type=int, default=128)
    t.add_argument("--blocks", type=int, default=6)
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--batch", type=int, default=1024)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--wd", type=float, default=0.05)
    t.add_argument("--smooth", type=float, default=0.05,
                   help="label smoothing: a result read off a position is never certain")
    t.add_argument("--gpu-frac", type=float, default=0.15,
                   help="cap on this process's share of GPU memory (other jobs run)")
    t.add_argument("--dropout", type=float, default=0.1,
                   help="in the fuse; the critic has overfitted and then collapsed before")
    t.add_argument("--ema", type=float, default=0.999,
                   help="decay of the weight average that is evaluated and saved")
    t.add_argument("--max-games", type=int, default=0,
                   help="cap how many games are loaded; 0 means all (they must fit "
                        "in memory, and a packed board is 9.2 KB a position)")
    t.add_argument("--eval-batch", type=int, default=1024,
                   help="the board trunk needs more memory a row than the old critic did")
    sp = sub.add_parser("selfplay", help="games on maps replays never cover")
    sp.add_argument("--ckpts", required=True, help="comma list of policies to play each other")
    sp.add_argument("--maps", required=True)
    sp.add_argument("--games", type=int, default=24, help="per map")
    sp.add_argument("--keep", type=float, default=0.08, help="share of turns recorded")
    sp.add_argument("--envs", type=int, default=32)
    sp.add_argument("--threads", type=int, default=8)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--scalars", choices=["base", "all"], default="base",
                    help="base keeps the row the replay extracts stored, so both mix")
    a = p.parse_args()
    {"extract": extract, "pretrain": pretrain, "selfplay": selfplay}[a.cmd](a)


if __name__ == "__main__":
    main()