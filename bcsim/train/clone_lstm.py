"""Clones a team straight into the LSTM policy, from its replays, with OUR sonar.

No flat clone in between (user, 2026-09-30): each env replays one server game in the
simulator, every dragon playing exactly the move the replay records, and the team
being cloned is trained on its own moves, turn by turn, through the LSTM with real
BPTT (the same chunked unroll as distill_lstm.py). Nothing is stored on disk: the
grid is re-rendered as the games are replayed (10.5M turns of grid would be ~150 GB).

Sonar is ours, not theirs. The cloned team speaks the v2 team packet (cpp/bc_sonar2.hpp,
a BC_SONAR2 library): every one of its dragons casts it in all four directions after
its move, and its own recorded sonar is dropped. The packet's action probabilities are
the replayed move itself (certain: we do not know theirs). The student therefore reads
the 43-channel grid with its teammates' reports in it. The other team casts exactly
what its replay says, and fails our tag.

A game is replayed only as far as it matches the server (dataset/index.jsonl from
train.replay_dataset: every turn of an exact game, the prefix of a diverged one), and
checked every turn (acting dragon and round).

    python -m train.clone_lstm --games ../runs/replays/cheji_0930 --team-id 70 --out ../runs/lstm_cheji
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import re
import sys
import time
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
os.environ.setdefault("BCSIM_LIB", str(ROOT / "bcsim/bcsim/libbcvec_replay_s2.so"))
sys.path.insert(0, str(ROOT / "bcsim"))

import numpy as np                      # noqa: E402
import torch                            # noqa: E402
import torch.nn.functional as F         # noqa: E402

import bcsim                            # noqa: E402
from bcsim.env import MAX_STEPS         # noqa: E402
from train.distill_lstm import Pool, chunks_of   # noqa: E402
from train.lstm_net import LSTMPolicy, NO_ACTION  # noqa: E402
from train.ff_net import FFPolicy, FFLPolicy  # noqa: E402
from train.board_net import BoardPolicy, board_float, coarse_of, crop_at  # noqa: E402
from train.net import masked_logits     # noqa: E402
from train import sonar2 as S2          # noqa: E402

SC = bcsim.SCALARS
I_FACE, I_LEN = SC.index("face_n"), SC.index("length_raw")
I_LENN, I_UNITS = SC.index("length"), SC.index("units")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--games", required=True, help="replay_dataset directory (games/ + dataset/index.jsonl)")
    p.add_argument("--team-id", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-share", type=float, default=0.1, help="share of series held out")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--val-envs", type=int, default=24)
    p.add_argument("--steps", type=int, default=512, help="env steps per rollout")
    p.add_argument("--chunk", type=int, default=32)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--epochs", type=float, default=4.0, help="passes over the training games")
    p.add_argument("--train-epochs", type=int, default=1, help="passes over each rollout")
    p.add_argument("--batch-chunks", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--load-workers", type=int, default=8)
    p.add_argument("--limit-games", type=int, default=0, help="smoke tests")
    p.add_argument("--min-game", type=int, default=0,
                   help="only games with this id or later (ids grow with time: the team's current upload)")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--c1", type=int, default=48)
    p.add_argument("--b1", type=int, default=1)
    p.add_argument("--c2", type=int, default=112)
    p.add_argument("--b2", type=int, default=2)
    p.add_argument("--squeeze", type=int, default=16)
    p.add_argument("--embed", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--crop", type=int, default=27, help="board: the exact window around the acting head")
    p.add_argument("--arch", choices=("lstm", "ff", "ffl", "board"), default="lstm", help="ff: small feed-forward; ffl: the LSTM net with dense layers in place of the LSTM (train/ff_net.py)")
    p.add_argument("--ff-c1", type=int, default=32)
    p.add_argument("--ff-c2", type=int, default=64)
    p.add_argument("--ff-squeeze", type=int, default=16)
    p.add_argument("--ff-hidden", type=int, default=128)
    p.add_argument("--ff-blocks", type=int, default=0)
    p.add_argument("--ff-batch", type=int, default=4096, help="rows per ff update")
    return p.parse_args()


# ---------------------------------------------------------------- games
def _load(args):
    """One replay, as compact arrays (runs in a worker)."""
    root, row, team_id = args
    from train.replay_read import read
    from train.replay_dataset import game_map
    gid = row["game"]
    meta = json.loads((root / "games" / f"{gid}.json").read_text())
    rp = read(root / "games" / f"{gid}.replay")
    mt = game_map(rp.map_text)
    if mt is None:
        return None
    n = len(rp.turns)
    limit = n
    div = row.get("diverged")
    if div:
        m = re.match(r"turn (\d+)", str(div))
        limit = int(m.group(1)) if m else max(0, n - 1)
    if limit <= 0:
        return None
    t = rp.turns[:limit]
    dlen = np.array([len(x.dirs) for x in t], np.int32)
    return {
        "game": gid, "series": meta.get("seriesId") or str(gid), "map": mt,
        "seed": int(meta["seed"], 16), "side": 0 if meta["teamAId"] == team_id else 1,
        "dragon": np.array([x.dragon for x in t], np.int32),
        "round": np.array([x.round for x in t], np.int32),
        "kind": np.array([x.kind if x.kind >= 0 else 2 for x in t], np.int8),
        "doff": np.concatenate([[0], np.cumsum(dlen)]).astype(np.int64),
        "dirs": np.array([d for x in t for d in x.dirs], np.int8),
        "split": np.array([x.split_k for x in t], np.int16),
        "send": np.array([sum(1 << d for d, _ in x.sonars) for x in t], np.uint8),
        "sval": np.array([[dict(x.sonars).get(k, 0) for k in range(4)] for x in t], np.uint64),
    }


def load_games(root: pathlib.Path, team_id: int, workers: int, limit: int = 0, min_game: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in (root / "dataset/index.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r.get("samples", 0) > 0 and int(r["game"]) >= min_game]
    if limit:
        rows = rows[::max(1, len(rows) // limit)][:limit]
    with mp.Pool(workers) as pool:
        games = [g for g in pool.imap_unordered(_load, [(root, r, team_id) for r in rows], chunksize=8) if g]
    games.sort(key=lambda g: g["game"])
    return games


def encode_turn(g: dict, c: int, facing: int, length: int) -> tuple[int, int]:
    """Codec id(s) of turn c (train.replay_dataset.encode, on the arrays)."""
    k = int(g["kind"][c])
    if k == 0:
        dirs = g["dirs"][g["doff"][c]:g["doff"][c + 1]]
        n = len(dirs)
        if not 1 <= n <= 3:
            return -1, -1
        code, f = 0, facing
        for i, d in enumerate(dirs.tolist()):
            if d == f:
                t = 0
            elif d == (f + 3) % 4:
                t = 1
            elif d == (f + 1) % 4:
                t = 2
            else:
                return -1, -1
            code += t * 3 ** i
            f = d
        return (0, 3, 12)[n - 1] + code, -1
    if k == 1 and bcsim.N_ACTIONS > 48 and not 2 <= int(g["split"][c]) <= length - 2:
        # an illegal split size kills the dragon (Game::split): Vibing++ self-kills this way (split 1,
        # 1.7% of its turns, measured 2026-10-05) -- the same move as our self-kill
        return 48, -1
    if k == 1:
        from train.replay_dataset import split_ids
        ids = split_ids(int(g["split"][c]), length)
        return (ids[0], ids[1] if len(ids) > 1 else -1) if ids else (-1, -1)
    if k == 2 and bcsim.N_ACTIONS > 48:
        return 48, -1                    # the self-kill (also every forced death)
    return -1, -1


# ---------------------------------------------------------------- main
def main() -> None:
    a = parse()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")

    t0 = time.time()
    games = load_games(pathlib.Path(a.games), a.team_id, a.load_workers, a.limit_games, a.min_game)
    val = [g for g in games if (zlib.crc32(g["series"].encode()) % 1000) < a.val_share * 1000]
    train = [g for g in games if (zlib.crc32(g["series"].encode()) % 1000) >= a.val_share * 1000]
    n_train_turns = sum(int((g["dragon"] >= 0).sum()) for g in train)
    maps = sorted({g["map"] for g in games})
    map_idx = {m: i for i, m in enumerate(maps)}
    print(f"{len(games)} games ({len(train)} train, {len(val)} held-out series), {len(maps)} distinct maps, "
          f"{n_train_turns / 1e6:.1f}M training turns (both teams); loaded in {time.time() - t0:.0f}s", flush=True)

    E = a.envs + a.val_envs
    env = bcsim.BattlecodeVecEnv(maps, num_envs=E, num_threads=a.threads, seed=a.seed,
                                 random_pearl_seed=False, closure_capacity=max(8192, E * 160), grid=True,
                                 board=a.arch == "board")
    assert env.sonar2, f"{os.environ['BCSIM_LIB']} is not a BC_SONAR2 library (make -C bcsim s2)"
    C, G = env.grid.shape[1], env.grid.shape[2]
    A = bcsim.N_ACTIONS
    ff = a.arch in ("ff", "ffl", "board")
    bd = a.arch == "board"
    if bd:
        # the true board, not a bot's view (user, 2026-10-01: an opponent model, never submitted)
        arch = {"crop": a.crop}
        net = BoardPolicy(crop=a.crop, n_actions=A).to(dev)
    elif a.arch == "ffl":
        arch = {k: getattr(a, k) for k in ("c1", "b1", "c2", "b2", "squeeze", "embed", "hidden", "layers")}
        net = FFLPolicy(**arch, in_ch=C, n_actions=A, grid=G).to(dev)
    elif ff:
        arch = {k: getattr(a, "ff_" + k) for k in ("c1", "c2", "squeeze", "hidden", "blocks")}
        net = FFPolicy(**arch, in_ch=C, n_actions=A, grid=G).to(dev)
    else:
        arch = {k: getattr(a, k) for k in ("c1", "b1", "c2", "b2", "squeeze", "embed", "hidden", "layers")}
        net = LSTMPolicy(**arch, in_ch=C, n_actions=A, grid=G).to(dev)
    total_iters = int(a.epochs * n_train_turns / (a.steps * a.envs))
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_iters))
    it0, best = 0, -1.0
    if a.resume and (out / "latest.pt").exists():
        ck = torch.load(out / "latest.pt", map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        it0, best = ck["iter"] + 1, ck.get("best", -1.0)
    print(f"{type(net).__name__} {arch} in_ch {C}, {A} actions (self-kill = 48); {total_iters} iterations of {a.steps} x {a.envs} env steps "
          f"(~{a.epochs} passes); {a.val_envs} envs replay held-out games", flush=True)

    # every env replays games from its own queue; the last val_envs play held-out games
    queues = {"train": [], "val": []}
    epoch = {"train": 0, "val": 0}

    def next_game(kind):
        q = queues[kind]
        if not q:
            src = train if kind == "train" else val
            q.extend(rng.permutation(len(src)).tolist())
            epoch[kind] += 1
        return (train if kind == "train" else val)[q.pop()]

    role = ["train"] * a.envs + ["val"] * a.val_envs
    cur = [None] * E
    cursor = np.zeros(E, np.int64)
    pool = Pool(E * 160, 0 if bd else a.layers, a.hidden, dev, grow=True, no_action=net.no_action)
    # board: each env's coarse view of the cloned team's board, refreshed at its first turn of a round
    W = a.crop
    co_cache = torch.zeros(E, 18, 16, 16, dtype=torch.float16, device=dev) if bd else None
    co_round = np.full(E, -1, np.int64)

    def load(e):
        g = next_game(role[e])
        cur[e] = g
        cursor[e] = 0
        env.set_opponent(e, team=-1, bot=0, map_index=map_idx[g["map"]])
        env.set_pearl_seed64(e, g["seed"])
        env.set_sonar2(e, (g["side"],))
        env.restart_env(e)
        pool.release_env(e)
        co_round[e] = -1

    obs = env.reset()
    for e in range(E):
        load(e)

    T = a.steps
    kind_a = np.zeros(E, np.int8); nst_a = np.zeros(E, np.int8)
    dirs_a = np.zeros((E, MAX_STEPS), np.int8); split_a = np.zeros(E, np.int16)
    send_a = np.zeros(E, np.uint8); sval_a = np.zeros((E, 4), np.uint64)
    B_grid = torch.zeros(T, a.envs, C, G, G, dtype=torch.float16, device=dev) if not bd else None
    if bd:
        B_crop = torch.zeros(T, a.envs, 18, W, W, dtype=torch.uint8, device=dev)
        B_coarse = torch.zeros(T, a.envs, 18, 16, 16, dtype=torch.float16, device=dev)
        B_extra = torch.zeros(T, a.envs, 6, dtype=torch.float16, device=dev)
    B_mask = torch.zeros(T, a.envs, A, dtype=torch.bool, device=dev)
    B_prev = torch.zeros(T, a.envs, dtype=torch.long, device=dev)
    B_tgt = torch.full((T, a.envs, 2), -1, dtype=torch.long, device=dev)
    B_h = torch.zeros(T, a.envs, a.layers, a.hidden, dtype=torch.float16, device=dev)
    B_c = torch.zeros(T, a.envs, a.layers, a.hidden, dtype=torch.float16, device=dev)
    B_life = np.full((T, a.envs), -1, np.int64)
    t_start = time.time()
    val_hits = val_n = 0

    for it in range(it0, total_iters):
        t_roll = time.time()
        net.eval()
        B_life[:] = -1
        B_tgt[:] = -1
        mismatches = 0
        for t in range(T):
            ours = np.zeros(E, bool)
            tgt = np.full((E, 2), -1, np.int64)
            for e in range(E):
                for _ in range(3):
                    g, c = cur[e], cursor[e]
                    if c < len(g["dragon"]) and (int(obs.dragon_id[e]), int(obs.round[e])) == (int(g["dragon"][c]), int(g["round"][c])):
                        break
                    mismatches += c > 0
                    load(e)
                g, c = cur[e], int(cursor[e])
                k = int(g["kind"][c])
                d0, d1 = g["doff"][c], g["doff"][c + 1]
                kind_a[e] = k
                nst_a[e] = min(d1 - d0, MAX_STEPS)
                dirs_a[e] = 0
                dirs_a[e, :nst_a[e]] = g["dirs"][d0:d0 + nst_a[e]]
                split_a[e] = g["split"][c]
                if int(obs.team[e]) == g["side"]:
                    ours[e] = True
                    facing = int(np.argmax(obs.scalar[e, I_FACE:I_FACE + 4]))
                    tgt[e] = encode_turn(g, c, facing, int(obs.scalar[e, I_LEN]))
                    env.intent[e] = S2.intent_from_turn(k, g["dirs"][d0:d1], facing)
                    send_a[e] = 0
                    sval_a[e] = 0
                else:
                    send_a[e] = g["send"][c]
                    sval_a[e] = g["sval"][c]
            # the student sees its own team's turns, teacher-forced
            rows = np.flatnonzero(ours)
            if len(rows):
                slots = pool.get(rows, obs.uid[rows])
                s = torch.as_tensor(slots, device=dev)
                mask = torch.from_numpy(obs.mask[rows]).to(dev).bool()
                prev = pool.prev[s]
                if bd:
                    board = torch.from_numpy(env.board[rows]).to(dev)
                    crop8 = crop_at(board, W)
                    bf = board_float(board)
                    fresh = np.flatnonzero(co_round[rows] != obs.round[rows])
                    if len(fresh):
                        co_cache[torch.as_tensor(rows[fresh], device=dev)] = coarse_of(bf[fresh]).half()
                        co_round[rows[fresh]] = obs.round[rows[fresh]]
                    coarse = co_cache[torch.as_tensor(rows, device=dev)].float()
                    coarse[:, 8:10] = coarse_of(bf[:, 8:10])
                    sc = obs.scalar[rows]
                    extra = torch.from_numpy(np.concatenate([sc[:, I_FACE:I_FACE + 4], sc[:, [I_LENN, I_UNITS]]], 1)
                                             .astype(np.float32)).to(dev)
                    del bf, board
                else:
                    grid = torch.from_numpy(env.grid[rows]).to(dev, non_blocking=True)
                state = [(pool.h[l, s], pool.c[l, s]) for l in range(0 if bd else a.layers)]
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    if bd:
                        logits = net(board_float(crop8), coarse, prev, extra)
                        new = []
                    elif ff:
                        logits, _ = net(grid, prev)
                        new = []
                    else:
                        logits, _, new = net(grid, prev, state)
                tg = torch.as_tensor(tgt[rows], device=dev)
                tr = rows < a.envs
                if tr.any():
                    ti = torch.as_tensor(np.flatnonzero(tr), device=dev)
                    er = rows[tr]
                    if bd:
                        B_crop[t, er] = crop8[ti]
                        B_coarse[t, er] = coarse[ti].half()
                        B_extra[t, er] = extra[ti].half()
                    else:
                        B_grid[t, er] = grid[ti].half()
                    B_mask[t, er] = mask[ti]
                    B_prev[t, er] = prev[ti]
                    B_tgt[t, er] = tg[ti]
                    if not ff:
                        B_h[t, er] = torch.stack([pool.h[l, s[ti]] for l in range(a.layers)], 1).half()
                        B_c[t, er] = torch.stack([pool.c[l, s[ti]] for l in range(a.layers)], 1).half()
                    B_life[t, er] = pool.life[slots[tr]]
                if (~tr).any():
                    vi = np.flatnonzero(~tr)
                    lg = masked_logits(logits[vi].float(), mask[vi])
                    top = lg.argmax(1).cpu().numpy()
                    tv = tgt[rows[vi]]
                    ok = tv[:, 0] >= 0
                    val_hits += int(((top == tv[:, 0]) | (top == tv[:, 1]))[ok].sum())
                    val_n += int(ok.sum())
                for l, (h, c_) in enumerate(new):
                    pool.h[l, s] = h.float()
                    pool.c[l, s] = c_.float()
                pa = torch.as_tensor(np.where(tgt[rows, 0] >= 0, tgt[rows, 0], net.no_action), device=dev)
                pool.prev[s] = pa
            obs, closures, eps = env.step_raw(kind_a, nst_a, dirs_a, split_a, send_a, sval_a)
            cursor += 1
            ended = set(int(r[0]) for r in eps.rows)
            for e in range(E):
                if e in ended or cursor[e] >= len(cur[e]["dragon"]):
                    load(e)
            if len(closures.env):
                for e_, u_, d_ in zip(closures.env.tolist(), closures.uid.tolist(), closures.done.tolist()):
                    if d_:
                        pool.release(e_, u_)
        t_roll = time.time() - t_roll

        # ---- train on the rollout, regrouped by dragon
        t_train = time.time()
        net.train()
        flat_life = B_life.reshape(-1)
        valid_rows = np.flatnonzero(flat_life >= 0)
        ch_local = chunks_of(flat_life[valid_rows], a.chunk)
        ch = torch.as_tensor(np.where(ch_local >= 0, valid_rows[np.maximum(ch_local, 0)], -1), device=dev)
        fg = B_grid.reshape(T * a.envs, C, G, G) if not bd else None
        if bd:
            fcrop, fco, fex = B_crop.reshape(T * a.envs, 18, W, W), B_coarse.reshape(T * a.envs, 18, 16, 16), \
                B_extra.reshape(T * a.envs, 6)
        fm, fprev, ft = B_mask.reshape(T * a.envs, A), B_prev.reshape(-1), B_tgt.reshape(T * a.envs, 2)
        fh, fc = B_h.reshape(T * a.envs, a.layers, a.hidden), B_c.reshape(T * a.envs, a.layers, a.hidden)
        tot_loss = tot_hit = tot_n = 0.0
        vr_t = torch.as_tensor(valid_rows, device=dev)
        for _ in range(a.train_epochs):
            perm = torch.randperm(len(vr_t) if ff else len(ch), device=dev)
            step = a.ff_batch if ff else a.batch_chunks
            for b0 in range(0, len(perm), step):
                if bd:
                    rows_ = vr_t[perm[b0:b0 + step]]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits = net(board_float(fcrop[rows_]), fco[rows_].float(), fprev[rows_], fex[rows_])
                    logits = logits.float()
                elif ff:
                    rows_ = vr_t[perm[b0:b0 + step]]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = net(fg[rows_].float(), fprev[rows_])
                    logits = logits.float()
                else:
                    idx = ch[perm[b0:b0 + a.batch_chunks]]
                    valid = idx >= 0
                    rows_ = idx[valid]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        x = net.encode(fg[rows_].float(), fprev[rows_])
                    X = torch.zeros(*idx.shape, x.shape[1], device=dev, dtype=x.dtype)
                    X[valid] = x
                    first = idx[:, 0]
                    state = [(fh[first, l].float(), fc[first, l].float()) for l in range(a.layers)]
                    outs = []
                    for k in range(a.chunk):
                        lg, _, state = net.step(X[:, k].float(), state)
                        outs.append(lg)
                    logits = torch.stack(outs, 1)[valid]
                logp = F.log_softmax(masked_logits(logits, fm[rows_]), dim=1)
                tg = ft[rows_]
                ok = tg[:, 0] >= 0
                ok &= fm[rows_].gather(1, tg[:, :1].clamp(min=0)).squeeze(1)
                if not ok.any():
                    continue
                lp0 = logp.gather(1, tg[:, :1].clamp(min=0)).squeeze(1)
                lp1 = torch.where(tg[:, 1] >= 0, logp.gather(1, tg[:, 1:].clamp(min=0)).squeeze(1),
                                  torch.full_like(lp0, -1e9))
                loss = -torch.logaddexp(lp0, lp1)[ok].mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    top = logp.argmax(1)
                    n_ = int(ok.sum())
                    tot_loss += loss.item() * n_
                    tot_hit += ((top == tg[:, 0]) | (top == tg[:, 1]))[ok].float().sum().item()
                    tot_n += n_
        sched.step()
        t_train = time.time() - t_train
        row = {"iter": it, "loss": tot_loss / max(tot_n, 1), "acc": tot_hit / max(tot_n, 1), "samples": int(tot_n),
               "val_acc": val_hits / max(val_n, 1), "val_n": val_n, "epoch": epoch["train"],
               "mismatches": mismatches, "live_slots": len(pool.key),
               "rollout_s": round(t_roll, 1), "train_s": round(t_train, 1), "lr": sched.get_last_lr()[0],
               "elapsed": round(time.time() - t_start, 1)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"it {it:4d}/{total_iters} | loss {row['loss']:.4f} | acc {row['acc']:.3f} | held-out acc "
              f"{row['val_acc']:.3f} ({val_n:,}) | pass {epoch['train']} | {t_roll:.0f}s + {t_train:.0f}s", flush=True)
        ck = {"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "iter": it,
              "best": best, "args": {**vars(a), "arch": a.arch, **arch, "in_ch": C, "n_actions": A, "grid": G, "sonar2": True}}
        if val_n >= 20000 and row["val_acc"] > best:
            best = row["val_acc"]
            ck["best"] = best
            torch.save({k: v for k, v in ck.items() if k not in ("opt", "sched")}, out / "best.pt")
        if val_n >= 20000:
            val_hits = val_n = 0
        torch.save(ck, out / "latest.pt.tmp")
        (out / "latest.pt.tmp").replace(out / "latest.pt")
    env.close()


if __name__ == "__main__":
    main()
