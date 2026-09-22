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
TEAMS = ["vibing", "sss", "shink_ai_6500", "sabotage_d", "matcha_latte", "team"]
N_PRIV = 8                                      # PRIV_COUNT in bc_vec.hpp
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
    ck = torch.load(path, map_location=dev, weights_only=False)
    c = ck["critic_args"]
    nc = n_context or c["n_context"]
    net = build(bcsim.N_CHANNELS, bcsim.N_SCALARS, nc, c["width"], c["blocks"]).to(dev)
    sd = ck["critic"]
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
    from train.replay_read import read
    gid = int(game_json.stem)
    dest = OUT / "data" / f"{gid}.npz"
    if dest.exists():
        return gid, "cached", 0
    try:
        rp = read(game_json.with_suffix(".replay"))
    except Exception as e:                      # a truncated download
        return gid, f"unreadable {type(e).__name__}", 0
    if rp.winner < 0 and rp.end_reason < 0:
        return gid, "no result", 0
    rng = np.random.default_rng(seed + gid)
    env = bcsim.BattlecodeVecEnv([rp.map_text], num_envs=1, num_threads=1, seed=0,
                                 random_pearl_seed=False, max_rounds=500, privileged=True)
    obs = env.reset()
    rec = {k: [] for k in ("local", "scalar", "priv", "round", "team", "outcome")}
    z = lambda dt, *s: np.zeros((1,) + s, dt)
    kind, nst, dirs, split = z(np.int8), z(np.int8), z(np.int8, MAX_STEPS), z(np.int16)
    send, value = z(np.int8), z(np.uint32)
    status = "ok"
    for i, t in enumerate(rp.turns):
        did, rnd, team = int(obs.dragon_id[0]), int(obs.round[0]), int(obs.team[0])
        if (did, rnd) != (t.dragon, t.round):
            status = "diverged"
            break
        if rng.random() < keep_p:
            rec["local"].append(obs.local[0].astype(np.float16))
            rec["scalar"].append(obs.scalar[0].copy())
            rec["priv"].append(obs.priv[0].copy())
            rec["round"].append(rnd)
            rec["team"].append(team)
            rec["outcome"].append(1 if rp.winner < 0 else (0 if rp.winner == team else 2))
        kind[0] = t.kind if t.kind >= 0 else 2
        if len(t.dirs) > MAX_STEPS:
            status = "sprint over MAX_STEPS"
            break
        nst[0] = len(t.dirs)
        dirs[0] = 0
        dirs[0, :nst[0]] = t.dirs
        split[0] = t.split_k
        send[0] = t.sonar is not None
        value[0] = t.sonar or 0
        obs, _, _ = env.step_raw(kind, nst, dirs, split, send, value)
    env.close()
    # a diverged game's early positions are still true positions of a game
    # whose result is known, so they are kept
    if not rec["round"]:
        return gid, "empty", 0
    np.savez(dest, **{k: np.stack(v) for k, v in rec.items()
                      if k in ("local", "scalar", "priv")},
             **{k: np.array(v, np.int16) for k, v in rec.items()
                if k in ("round", "team", "outcome")},
             rounds=np.int16(rp.rounds))
    return gid, status, len(rec["round"])


def extract(a) -> None:
    (OUT / "data").mkdir(parents=True, exist_ok=True)
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
    torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    dev = torch.device("cuda")
    files = sorted((OUT / "data").glob("*.npz"))
    # sized first and filled in place: a list-and-concatenate load would hold
    # the whole set twice, and it is several GB
    sizes = {"tr": 0, "va": 0}
    for f in files:
        sizes["va" if held_out(int(f.stem)) else "tr"] += len(np.load(f)["round"])
    shapes = {"local": ((bcsim.N_CHANNELS, 7, 7), np.float16),
              "scalar": ((bcsim.N_SCALARS,), np.float32), "priv": ((N_PRIV,), np.float32),
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
    data = {s: {k: torch.from_numpy(v) for k, v in p.items()} for s, p in arrs.items()}
    tr, va = data["tr"], data["va"]
    n_va_games = sum(held_out(int(f.stem)) for f in files)
    print(f"{len(files)} games: {len(tr['round']):,} train positions, {len(va['round']):,} "
          f"held out ({n_va_games} games)", flush=True)
    base = np.bincount(tr["outcome"].numpy(), minlength=3) / len(tr["outcome"])
    print("outcome share (train) win/draw/loss", base.round(3), flush=True)

    torch.manual_seed(0)
    net = build(bcsim.N_CHANNELS, bcsim.N_SCALARS, 1, a.width, a.blocks).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    n = len(tr["round"])
    steps = a.epochs * ((n + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    ctx0 = torch.zeros(max(a.batch, 4096), 1, device=dev)

    def fwd(d, idx):
        b = len(idx)
        logits, _ = net(d["local"][idx].to(dev).float(), d["scalar"][idx].to(dev),
                        ctx0[:b], d["priv"][idx].to(dev))
        return logits

    def evaluate():
        net.eval()
        ps = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s in range(0, len(va["round"]), 4096):
                idx = torch.arange(s, min(s + 4096, len(va["round"])))
                ps.append(fwd(va, idx).float().softmax(1).cpu())
        net.train()
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
            torch.save({"critic": net.state_dict(), "eval": row,
                        "critic_args": {"width": a.width, "blocks": a.blocks, "n_context": 1,
                                        "n_priv": N_PRIV}},
                       OUT / "pretrained.pt")
    (OUT / "pretrain_log.json").write_text(json.dumps(log, indent=1))
    print(f"best held-out log loss {best:.4f} -> {OUT / 'pretrained.pt'}")


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
    a = p.parse_args()
    extract(a) if a.cmd == "extract" else pretrain(a)


if __name__ == "__main__":
    main()