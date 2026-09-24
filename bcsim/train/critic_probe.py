"""Offline test of critic inputs: how well can each predict who wins?

A critic can only be as good as what it sees. Before building a new one into
PPO, this measures it offline, as supervised learning on replays with known
results:

    extract   replays are played again in the simulator (as replay_dataset
              does); every --every rounds, the first dragon of each team to
              act is recorded, with its local view, scalars, the 8
              privileged features and the whole board, all relative to that
              team, plus how the game ended for that team. Both teams of
              every game are recorded, so wins and losses are balanced.
    train     the same held-out games are scored by predictors that see:
                priv        the 8 privileged features and the round
                local       the 7x7 window and scalars (the plain critic)
                local+priv  both (the ft3 critic)
                board       the whole board, the features and the round
                board+local all of it
              each a win/draw/loss classifier.

    python -m train.critic_probe extract --games 150
    python -m train.critic_probe train

Reports held-out log loss and accuracy, and the explained variance of the
predicted result (p_win - p_loss against +1/0/-1), by stage of the game.
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
OUT = ROOT / "runs/critic_probe"
TEAMS = ["vibing", "sss", "sabotage_d", "shink_ai_6500"]


# ------------------------------------------------------------------ extract
def extract_one(args):
    game_json, every = args
    import bcsim
    from bcsim.env import MAX_STEPS
    from train.replay_read import read
    gid = int(game_json.stem)
    dest = OUT / "data" / f"{gid}.npz"
    if dest.exists():
        return gid, "cached"
    rp = read(game_json.with_suffix(".replay"))
    if rp.winner < 0 and rp.end_reason < 0:
        return gid, "no result"
    w, h = [int(v) for v in rp.map_text.split("\n", 1)[0].split()[1:3]]
    env = bcsim.BattlecodeVecEnv([rp.map_text], num_envs=1, num_threads=1, seed=0,
                                 random_pearl_seed=False, max_rounds=500,
                                 privileged=True, board=True)
    obs = env.reset()
    keep = {k: [] for k in ("local", "scalar", "priv", "board", "round", "team", "outcome")}
    seen = set()
    z = lambda dt, *s: np.zeros((1,) + s, dt)
    kind, nst, dirs, split = z(np.int8), z(np.int8), z(np.int8, MAX_STEPS), z(np.int16)
    send, value = z(np.uint8), z(np.uint64, bcsim.SONAR_DIRS)
    status = "ok"
    for i, t in enumerate(rp.turns):
        did, rnd, team = int(obs.dragon_id[0]), int(obs.round[0]), int(obs.team[0])
        if (did, rnd) != (t.dragon, t.round):
            status = f"diverged at turn {i}"
            break
        if rnd % every == 0 and (rnd, team) not in seen:
            seen.add((rnd, team))
            keep["local"].append(obs.local[0].astype(np.float16))
            keep["scalar"].append(obs.scalar[0].copy())
            keep["priv"].append(obs.priv[0].copy())
            keep["board"].append(env.board[0, :, :h, :w].copy())
            keep["round"].append(rnd)
            keep["team"].append(team)
            # this team's result: 0 win, 1 draw, 2 loss
            keep["outcome"].append(1 if rp.winner < 0 else (0 if rp.winner == team else 2))
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
    if not keep["round"]:
        return gid, "empty"
    np.savez_compressed(dest, **{k: np.stack(v) if k in ("local", "scalar", "priv", "board")
                                 else np.array(v, np.int16) for k, v in keep.items()},
                        rounds=np.int16(rp.rounds))
    return gid, status


def extract(a) -> None:
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    jobs = []
    for team in TEAMS:
        games = sorted((ROOT / "runs/replays" / team / "games").glob("*.json"))
        games = [g for g in games if g.with_suffix(".replay").exists()]
        pick = rng.choice(len(games), min(a.games, len(games)), replace=False)
        jobs += [(games[i], a.every) for i in pick]
    print(f"{len(jobs)} games to extract", flush=True)
    counts = {}
    with mp.Pool(a.workers) as pool:
        for gid, status in pool.imap_unordered(extract_one, jobs):
            counts[status.split(" at ")[0]] = counts.get(status.split(" at ")[0], 0) + 1
    print(counts)


# ------------------------------------------------------------------ train
def train(a) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import bcsim
    from train.net import ResBlock
    torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    dev = torch.device("cuda")
    files = sorted((OUT / "data").glob("*.npz"))
    rng = np.random.default_rng(0)
    held = set(rng.choice(len(files), max(1, len(files) // 5), replace=False).tolist())
    B = 64                                           # BOARD_MAX

    def load(sel):
        parts = {k: [] for k in ("local", "scalar", "priv", "board", "round", "outcome", "frac")}
        for i, f in enumerate(files):
            if (i in held) != sel:
                continue
            d = np.load(f)
            n = len(d["round"])
            bd = np.zeros((n, d["board"].shape[1], B, B), np.uint8)
            bd[:, :, :d["board"].shape[2], :d["board"].shape[3]] = d["board"]
            for k in ("local", "scalar", "priv", "round", "outcome"):
                parts[k].append(d[k])
            parts["board"].append(bd)
            # how far through the game, for the by-stage report
            parts["frac"].append(d["round"] / max(int(d["rounds"]), 1))
        return {k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()}

    tr, va = load(False), load(True)
    print(f"{len(files)} games: {len(tr['round']):,} train positions, "
          f"{len(va['round']):,} held out ({len(held)} games)", flush=True)
    print("outcome share (train): win/draw/loss",
          np.bincount(tr["outcome"].numpy(), minlength=3) / len(tr["outcome"]))

    class Head(nn.Module):
        """One predictor; `use` picks its inputs."""

        def __init__(self, use: set[str]):
            super().__init__()
            self.use = use
            feat = 0
            if "local" in use:
                self.local = nn.Sequential(
                    nn.Conv2d(bcsim.N_CHANNELS, 64, 3, padding=1), nn.SiLU(),
                    ResBlock(64), ResBlock(64), nn.Conv2d(64, 16, 1), nn.SiLU(), nn.Flatten())
                feat += 16 * 49 + bcsim.N_SCALARS
            if "board" in use:
                # 64x64 -> 8x8, then pooled: any map size fits
                self.board = nn.Sequential(
                    nn.Conv2d(8, 32, 3, padding=1), nn.SiLU(),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(), ResBlock(64),
                    nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.SiLU(), ResBlock(96),
                    nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.SiLU(), ResBlock(128))
                feat += 256
            feat += 8 + 1                            # privileged features and round
            self.mlp = nn.Sequential(nn.Linear(feat, 256), nn.SiLU(),
                                     nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 3))

        def forward(self, b):
            xs = []
            if "local" in self.use:
                xs += [self.local(b["local"]), b["scalar"]]
            if "board" in self.use:
                y = self.board(b["board"])
                xs.append(torch.cat([y.mean((2, 3)), y.amax((2, 3))], 1))
            priv = b["priv"] if "priv" in self.use or "board" in self.use else torch.zeros_like(b["priv"])
            xs += [priv, b["round"][:, None] / 500.0]
            return self.mlp(torch.cat(xs, 1))

    def batch(d, idx):
        return {"local": d["local"][idx].to(dev).float(), "scalar": d["scalar"][idx].to(dev),
                "priv": d["priv"][idx].to(dev), "board": d["board"][idx].to(dev).float(),
                "round": d["round"][idx].to(dev).float()}

    variants = {"priv": {"priv"}, "local": {"local"}, "local+priv": {"local", "priv"},
                "board": {"board", "priv"}, "board+local": {"board", "local", "priv"}}
    stages = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    base = np.bincount(tr["outcome"].numpy(), minlength=3) / len(tr["outcome"])
    results = {}
    for name, use in variants.items():
        torch.manual_seed(0)
        net = Head(use).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        n = len(tr["round"])
        best = None
        t0 = time.time()
        for ep in range(a.epochs):
            net.train()
            perm = torch.randperm(n)
            for s in range(0, n, a.batch):
                idx = perm[s:s + a.batch]
                loss = F.cross_entropy(net(batch(tr, idx)), tr["outcome"][idx].to(dev).long())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            net.eval()
            probs = []
            with torch.no_grad():
                for s in range(0, len(va["round"]), 2048):
                    idx = torch.arange(s, min(s + 2048, len(va["round"])))
                    probs.append(F.softmax(net(batch(va, idx)), 1).cpu())
            p = torch.cat(probs).numpy()
            y = va["outcome"].numpy()
            ll = float(-np.log(p[np.arange(len(y)), y] + 1e-9).mean())
            if best is None or ll < best[0]:
                best = (ll, p)
        ll, p = best
        y = va["outcome"].numpy()
        frac = va["frac"].numpy()
        target = np.array([1.0, 0.0, -1.0])[y]
        pred = p[:, 0] - p[:, 2]
        row = {}
        for lo, hi in stages:
            m = (frac >= lo) & (frac < hi)
            if not m.any():
                continue
            ll_s = float(-np.log(p[m][np.arange(m.sum()), y[m]] + 1e-9).mean())
            ll_b = float(-np.log(base[y[m]] + 1e-9).mean())
            ev = 1 - ((target[m] - pred[m]) ** 2).mean() / (target[m].var() + 1e-9)
            row[f"{lo:.2f}-{min(hi, 1):.2f}"] = {"logloss": round(ll_s, 4), "baseline": round(ll_b, 4),
                                                 "acc": round(float((p[m].argmax(1) == y[m]).mean()), 4),
                                                 "ev": round(float(ev), 4)}
        ev_all = 1 - ((target - pred) ** 2).mean() / (target.var() + 1e-9)
        results[name] = {"logloss": round(ll, 4), "ev": round(float(ev_all), 4),
                         "acc": round(float((p.argmax(1) == y).mean()), 4), "by_stage": row,
                         "seconds": round(time.time() - t0)}
        print(f"{name:12s} logloss {ll:.4f} (baseline {float(-np.log(base[y] + 1e-9).mean()):.4f}) "
              f"acc {results[name]['acc']:.3f} ev {ev_all:+.3f} | by stage ev: "
              + " ".join(f"{k} {v['ev']:+.2f}" for k, v in row.items()), flush=True)
    (OUT / "results.json").write_text(json.dumps(results, indent=1))


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--games", type=int, default=150, help="per team")
    e.add_argument("--every", type=int, default=3, help="record every this many rounds")
    e.add_argument("--workers", type=int, default=8)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=6)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--gpu-frac", type=float, default=0.5)
    a = p.parse_args()
    extract(a) if a.cmd == "extract" else train(a)


if __name__ == "__main__":
    main()
