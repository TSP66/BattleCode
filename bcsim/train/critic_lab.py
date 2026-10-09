"""Offline test bench for richer value critics (user, 2026-10-01: the ff value head sees no
board, so it cannot see a kill being set up).

gen: plays the ff learner (latest ratchet_ff3 candidate) against a league of versions and
     a couple of LSTM league members, as ratchet_ff_train does (temp, sonar v2, v8 reward,
     training maps), and keeps a sample of the learner's turns with everything any candidate
     critic could read:
       feat   the policy's last hidden layer (what --critic own reads, detached)
       priv   the engine's privileged row; opp the opponent-embedding id; vlive the live head
       board  planes 0-13 of the 64x64 privileged board (0-11 bit-packed into 2 bytes)
       probe  summaries of bcv_probe: what each of the acting dragon's moves would do
     plus, for every turn of every finished game, (game, team, round, Phi) for the returns,
     and every death (game, team, step, round) for the kill labels.
fit: builds PPO's value target (GAE over the v8 Phi reward, alpha per round, terminal as
     ratchet_ff_train) and fits each variant on the same rows, with auxiliary heads for the
     discounted Phi return at horizons 3/5/10/20 rounds (user, 2026-10-01: which horizon can a
     critic actually explain?); reports held-out explained variance of the value and of each
     horizon. No kill / loss labels (user, 2026-10-01).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("BCSIM_LIB", str(pathlib.Path(__file__).resolve().parents[1] / "bcsim" / "libbcvec_priv_s2.so"))

ROOT = pathlib.Path(__file__).resolve().parents[2]
P_ENEMY = 3          # rounds ahead for the death labels


# ----------------------------------------------------------------------------------- gen
def gen(a) -> None:
    import bcsim
    import train.ratchet_lstm_train as RL
    from train import augment
    from train.critic_v8 import GAMMA, KAPPA
    from train.distill_lstm import Pool
    from train.ff_net import LSTM_KEYS, PrivValue, build
    from train.net import masked_logits
    from train.sonar2 import intents_from_probs

    dev = torch.device("cuda")
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    RL.TEMP = TEMP = a.temp
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(a.policy, map_location="cpu", weights_only=False)
    arch = {k: ck["args"][k] for k in LSTM_KEYS}
    arch2 = {"in_ch": ck["args"].get("in_ch", 38), "n_actions": ck["args"].get("n_actions", 48)}
    policy = build({"arch": "ffl", **arch, **arch2}).to(dev).eval()
    policy.load_state_dict(ck["net"])
    vhead = PrivValue(arch["hidden"], bcsim.PRIV_COUNT, 64).to(dev).eval()
    vhead.load_state_dict(ck["vhead"]["net"])
    opp_ids = dict(ck["vhead"]["opp_ids"])

    names, paths, wts = [], [], []
    for spec in a.opps.split(","):
        n_, p_, w_ = spec.split("=")[0], spec.split("=")[1].split("@")[0], float(spec.split("@")[1])
        names.append(n_), paths.append(p_), wts.append(w_)
    N = a.envs
    opps = [RL.Actor(p_, dev, N * 40) for p_ in paths]
    opp_p = np.array(wts) / sum(wts)
    opp_vid = np.array([opp_ids.get(n_, 0) for n_ in names], np.int64)

    texts, map_w, _ = augment.training_pool(str(ROOT / "maps-train-official"), 48, a.seed, 0.25, "", 0.0,
                                            str(ROOT / "maps-loong"), 0.65, 3, True, log=lambda m: None)
    env = bcsim.BattlecodeVecEnv(texts, num_envs=N, num_threads=a.threads, seed=a.seed,
                                 closure_capacity=max(8192, N * 160), privileged=True, board=True,
                                 sonar=True, grid=True)
    env.set_map_weights(map_w)
    env.set_potential_gamma(GAMMA)
    env.set_reward_v8(True, KAPPA)
    PB = bcsim.PRIV_BASE

    slot = np.zeros(N, np.int64)          # 0 self-play, k = opps[k-1]
    learner = np.full(N, -1, np.int8)
    game = np.zeros(N, np.int64)
    next_game = [0]
    uid_team: list[dict] = [dict() for _ in range(N)]
    did_info: list[dict] = [dict() for _ in range(N)]     # dragon id -> (team, length at its last turn)
    I_LEN = bcsim.SCALARS.index("length_raw")

    def assign(envs):
        for e in envs:
            fz = rng.random() >= a.self_frac
            slot[e] = 1 + rng.choice(len(opps), p=opp_p) if fz else 0
            learner[e] = rng.integers(0, 2) if fz else -1
            if learner[e] < 0:
                env.set_sonar2(e, (0, 1))
            else:
                o_ = opps[slot[e] - 1]
                env.set_sonar2(e, (int(learner[e]),) + ((1 - int(learner[e]),) if o_.speaks else ()))
            game[e] = next_game[0]
            next_game[0] += 1
            uid_team[e].clear()
            did_info[e].clear()

    assign(range(N))
    lpool = Pool(N * 40, 0, 1, dev, grow=True, no_action=policy.no_action)
    turns = {k: [] for k in ("game", "team", "round", "phi", "step", "v")}   # every turn; v = live value (learner turns)
    deaths = []         # (game, dead team, step, round, killer team or -1, length)
    ends = []           # (game, winner, step)
    S = {k: [] for k in ("row_game", "row_team", "row_round", "row_step", "feat", "priv", "opp", "vlive",
                         "board", "probe", "head")}
    obs = env.reset()
    n_kept, t0, step = 0, time.perf_counter(), 0
    g_stop, drain = None, 0
    while True:
        # once enough are kept, play on (keeping nothing) until the games they came from end
        if g_stop is None and n_kept >= a.samples:
            g_stop = next_game[0]
        if g_stop is not None:
            drain += 1
            if (game >= g_stop).all() or drain > a.max_drain:
                break
        learn = (learner < 0) | (obs.team == learner)
        grid = torch.from_numpy(env.grid).to(dev)
        mask = torch.from_numpy(obs.mask).to(dev).bool()
        action = torch.zeros(N, dtype=torch.long, device=dev)
        li = np.flatnonzero(learn)
        for e, u, t, di, ln in zip(range(N), obs.uid.tolist(), obs.team.tolist(), obs.dragon_id.tolist(),
                                   obs.scalar[:, I_LEN].tolist()):
            uid_team[e][u] = t
            did_info[e][di] = (t, ln)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            lt = torch.as_tensor(li, device=dev)
            s_ = torch.as_tensor(lpool.get(li, obs.uid[li]), device=dev)
            feat = policy.features(grid[lt], lpool.prev[s_])
            lp = F.log_softmax(masked_logits(policy.pi(feat).float() / TEMP, mask[lt]), 1)
            act_ = torch.multinomial(lp.exp(), 1).squeeze(1)
            env.intent[li] = intents_from_probs(lp.exp())
            lpool.prev[s_] = act_
            action[lt] = act_
            ov_all = np.where(slot[li] > 0, opp_vid[np.maximum(slot[li] - 1, 0)], 0)
            v_all = np.full(N, np.nan, np.float32)
            v_all[li] = vhead(feat, torch.from_numpy(obs.priv[li]).to(dev),
                              torch.as_tensor(ov_all, device=dev)).float().cpu().numpy()
            for k, o in enumerate(opps):
                oi = np.flatnonzero(~learn & (slot == k + 1))
                if len(oi):
                    ot = torch.as_tensor(oi, device=dev)
                    action[ot] = o.act(oi, obs, grid[ot], None, None, None, mask[ot])
                    if o.speaks:
                        env.intent[oi] = intents_from_probs(o.last_probs)
            # sample learner turns
            keep = np.flatnonzero(rng.random(len(li)) < (a.keep if g_stop is None else 0.0))
            if len(keep):
                ki = li[keep]
                kt = torch.as_tensor(keep, device=dev)
                ov = np.where(slot[ki] > 0, opp_vid[np.maximum(slot[ki] - 1, 0)], 0)
                S["feat"].append(feat[kt].float().half().cpu().numpy())
                S["vlive"].append(v_all[ki])
                S["priv"].append(obs.priv[ki].astype(np.float32))
                S["opp"].append(ov)
                b = env.board[ki]
                S["board"].append(np.concatenate([np.packbits(b[:, :12], axis=1, bitorder="little")
                                                   .reshape(len(ki), 2, 64, 64), b[:, 12:14]], 1))
                pb = np.zeros((len(ki), 12), np.float32)
                for j, e in enumerate(ki.tolist()):
                    before, pm = env.probe(e)
                    ok = obs.mask[e, :pm.shape[0]] > 0
                    m_ = pm[ok] if ok.any() else np.zeros((1, pm.shape[1]), np.int32)
                    safe = m_[:, 0] > 0
                    pb[j] = [before[0], before[1], before[2], m_[:, 1].max(), m_[:, 6].max(), m_[:, 2].max(),
                             m_[:, 4].max(), m_[:, 5].max(), safe.sum(), m_[:, 7].max(),
                             m_[safe, 7].mean() if safe.any() else 0.0, len(m_)]
                S["probe"].append(pb)
                hd = b[:, 9].reshape(len(ki), -1).argmax(1)
                S["head"].append(np.stack([hd % 64, hd // 64], 1).astype(np.int16))
                S["row_game"].append(game[ki].copy())
                S["row_team"].append(obs.team[ki].astype(np.int8))
                S["row_round"].append(obs.round[ki].astype(np.int32))
                S["row_step"].append(np.full(len(ki), step, np.int64))
                n_kept += len(ki)
        for k_, v_ in (("game", game.astype(np.int32)), ("team", obs.team.astype(np.int8)),
                       ("round", obs.round.astype(np.int16)), ("phi", obs.priv[:, PB:].sum(1).astype(np.float32)),
                       ("step", np.full(N, step, np.int32)), ("v", v_all.astype(np.float16))):
            turns[k_].append(v_)
        obs, closures, eps = env.step(action.to(torch.int32).cpu().numpy())
        ended = set(eps.rows[:, 0].astype(int).tolist()) if len(eps.rows) else set()
        if len(closures.env):
            dead_envs = set()
            for e, u, d in zip(closures.env.tolist(), closures.uid.tolist(), closures.done.tolist()):
                if d:
                    lpool.release(e, u)
                    for o in opps:
                        o.release(e, u)
                    if e not in ended:
                        dead_envs.add(e)
            # who died, and who killed them (the engine's killer id; -1 = no killer: kelp, self, ...)
            for e in dead_envs:
                for di, _reason, killer, team in env.last_deaths(e).tolist():
                    kt_ = did_info[e].get(killer, (-1, 0))[0] if killer >= 0 else -1
                    deaths.append((game[e], team, step, int(obs.round[e]), kt_, did_info[e].get(di, (team, 0))[1]))
        if ended:
            wcol = bcsim.EpisodeStats.COLUMNS.index("winner")
            for r in eps.rows:
                e = int(r[0])
                ends.append((game[e], int(r[wcol]), step))
                lpool.release_env(e)
                for o in opps:
                    o.release_env(e)
            assign(sorted(ended))
        step += 1
        if step % 200 == 0:
            el = time.perf_counter() - t0
            print(f"step {step}: {n_kept} kept, {len(ends)} games, {step * N / el:,.0f} turns/s", flush=True)
    np.savez(out / "turns.npz", **{k: np.concatenate(v) for k, v in turns.items()},
             deaths=np.array(deaths, np.float64).reshape(-1, 6),
             ends=np.array(ends, np.int64).reshape(-1, 3))
    np.savez(out / "samples.npz", **{k: np.concatenate(v) for k, v in S.items()})
    print(f"done: {n_kept} samples, {len(ends)} games, {len(deaths)} deaths", flush=True)


# ----------------------------------------------------------------------------------- targets
HORIZONS = (3, 5, 10, 20)    # rounds: the auxiliary discounted-Phi horizons (discount 1 - 1/H)


def targets(d: pathlib.Path, alpha: float) -> dict:
    """Per sample (NaN where its game never finished):
      ret      lambda = 1 return (Phi reward, alpha per round, ratchet_ff_train's terminal)
      gae95/80 PPO's value target, GAE lambda 0.95 / 0.8 per round, bootstrapped from the live head
      disc3/5/10/20  the same Monte-Carlo return as ret, discounted 1 - 1/H per round instead of
                     alpha: horizon H rounds (disc20 at 0.95 is ret itself when alpha is 0.95)"""
    T = np.load(d / "turns.npz")
    en = T["ends"]
    S = np.load(d / "samples.npz")
    g_, tm_, rd_ = T["game"].astype(np.int64), T["team"].astype(np.int64), T["round"].astype(np.float64)
    ph_, st_, v_ = T["phi"].astype(np.float64), T["step"].astype(np.int64), T["v"].astype(np.float64)
    end_step = np.full(int(g_.max()) + 2, -1, np.int64)
    end_step[en[:, 0]] = en[:, 2]
    keep = (end_step[g_] >= 0) & (st_ <= end_step[g_])
    g_, tm_, rd_, ph_, st_, v_ = (x[keep] for x in (g_, tm_, rd_, ph_, st_, v_))
    o = np.lexsort((st_, g_))
    g_, tm_, rd_, ph_, st_, v_ = (x[o] for x in (g_, tm_, rd_, ph_, st_, v_))
    fin_of = np.full(len(end_step), -1, np.int64)
    last = np.r_[np.flatnonzero(g_[1:] != g_[:-1]), len(g_) - 1]          # each game's final turn
    fin_of[g_[last]] = last
    out = {k: np.full(len(g_), np.nan) for k in ("ret", "gae95", "gae80", *(f"disc{k}" for k in HORIZONS))}
    o2 = np.lexsort((st_, tm_, g_))
    key = g_[o2] * 2 + tm_[o2]
    cuts = np.r_[0, np.flatnonzero(key[1:] != key[:-1]) + 1, len(o2)]
    for c0, c1 in zip(cuts[:-1], cuts[1:]):
        ix = o2[c0:c1]
        ph, rd, v = ph_[ix], rd_[ix], v_[ix]
        fin = fin_of[g_[ix[0]]]
        q = ph_[fin] if tm_[fin] == tm_[ix[0]] else -ph_[fin]            # the final Phi, this team's sign
        r = np.empty(len(ix))
        r[:-1] = ph[1:] - ph[:-1]
        r[-1] = 0.0 if ix[-1] == fin else q - ph[-1]
        w = alpha ** (rd - rd[0])
        out["ret"][ix] = np.cumsum((w * r)[::-1])[::-1] / w
        if np.isfinite(v).all():                                         # a learner team: PPO's targets
            disc = np.r_[alpha ** (rd[1:] - rd[:-1]), 0.0]
            delta = r + disc * np.r_[v[1:], 0.0] - v
            for lam, name in ((0.95, "gae95"), (0.8, "gae80")):
                wl = (alpha * lam) ** (rd - rd[0])
                out[name][ix] = np.cumsum((wl * delta)[::-1])[::-1] / wl + v
        for k in HORIZONS:
            wk = (1.0 - 1.0 / k) ** (rd - rd[0])
            out[f"disc{k}"][ix] = np.cumsum((wk * r)[::-1])[::-1] / wk
    # samples -> turns
    tkey = g_ * (1 << 32) + st_
    skey = S["row_game"].astype(np.int64) * (1 << 32) + S["row_step"].astype(np.int64)
    pos = np.searchsorted(tkey, skey)
    hit = (pos < len(tkey)) & (tkey[np.minimum(pos, len(tkey) - 1)] == skey)
    res = {k: np.where(hit, v[np.minimum(pos, len(tkey) - 1)], np.nan).astype(np.float32) for k, v in out.items()}
    return res


# ----------------------------------------------------------------------------------- models
def unpack_board(b: torch.Tensor) -> torch.Tensor:
    """(B, 4, 64, 64) uint8 -> (B, 14, 64, 64) float: 12 bit planes, then 2 scaled bytes."""
    bits = torch.stack([(b[:, i // 8] >> (i % 8)) & 1 for i in range(12)], 1).float()
    return torch.cat([bits, b[:, 2:4].float() / 255.0], 1)


def crop(board: torch.Tensor, head: torch.Tensor, w: int) -> torch.Tensor:
    """w x w window of the board centred on each acting head, WRAPPED around the map (the world
    is a torus; the map's size is read off plane 5, in-map). Was zero-padded at the board's edge
    until 2026-10-01, which was wrong on 87% of 27 x 27 crops. Matches the simulator's critic
    view exactly (bcsim/tests/test_cview.py)."""
    B = board.shape[0]
    inmap = board[:, 5] > 0
    mw = inmap.any(1).sum(1).long().clamp(min=1)
    mh = inmap.any(2).sum(1).long().clamp(min=1)
    r = w // 2
    ar = torch.arange(w, device=board.device)
    ys = (head[:, 1, None].long() - r + ar[None]) % mh[:, None]
    xs = (head[:, 0, None].long() - r + ar[None]) % mw[:, None]
    bi = torch.arange(B, device=board.device)[:, None, None]
    out = board.permute(0, 2, 3, 1)[bi, ys[:, :, None], xs[:, None, :]]
    return out.permute(0, 3, 1, 2)


def tokens(board: torch.Tensor, head: torch.Tensor, k: int = 24):
    """Every head on the board (ours and theirs) as a token: side, offset from the acting
    head, and its 5x5 neighbourhood of bodies/heads; the k nearest, padded."""
    B = board.shape[0]
    heads = board[:, 1] + 2 * board[:, 3]                                   # 1 own, 2 enemy
    dev = board.device
    yy, xx = torch.meshgrid(torch.arange(64, device=dev), torch.arange(64, device=dev), indexing="ij")
    d = (yy[None] - head[:, 1, None, None]).abs() + (xx[None] - head[:, 0, None, None]).abs()
    d = torch.where(heads > 0, d.float(), torch.full_like(d.float(), 1e4)).reshape(B, -1)
    dist, idx = d.topk(k, 1, largest=False)
    valid = dist < 1e4
    hy, hx = idx // 64, idx % 64
    occ = board[:, [0, 1, 2, 3, 5]]
    pad = F.pad(occ, (2, 2, 2, 2))
    ar = torch.arange(5, device=dev)
    bi = torch.arange(B, device=dev)[:, None, None, None]
    patch = pad.permute(0, 2, 3, 1)[bi, (hy[:, :, None, None] + ar[None, None, :, None]),
                                    (hx[:, :, None, None] + ar[None, None, None, :])]      # B,k,5,5,5
    side = heads.reshape(B, -1).gather(1, idx)
    feat = torch.cat([F.one_hot(side.long().clamp(0, 2), 3).float(),
                      ((hy - head[:, 1, None]) / 16.0)[..., None], ((hx - head[:, 0, None]) / 16.0)[..., None],
                      (dist.clamp(max=64) / 16.0)[..., None], patch.reshape(B, k, -1)], 2)
    return feat * valid[..., None], valid


class Critic(nn.Module):
    """The live head's inputs (features, priv, opponent id) plus any of: probe summaries,
    tokens over all heads, a true-state window CNN, a pooled whole-board CNN."""

    def __init__(self, parts: set, feat_dim=128, n_priv=13, n_probe=12, win=21):
        super().__init__()
        self.parts, self.win = parts, win
        d = 0
        if "feat" in parts:
            d += feat_dim
        d += n_priv + 16
        self.opp = nn.Embedding(64, 16)
        if "probe" in parts:
            d += n_probe
        if "tok" in parts:
            self.tok_in = nn.Linear(3 + 3 + 125, 64)
            self.tok_att = nn.TransformerEncoder(nn.TransformerEncoderLayer(64, 4, 128, 0.0, batch_first=True), 2)
            d += 64
        if "win" in parts:
            self.wcnn = nn.Sequential(nn.Conv2d(14, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                      nn.SiLU(), nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(),
                                      nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(256, 128), nn.SiLU())
            d += 128
        if "coarse" in parts:
            self.ccnn = nn.Sequential(nn.AvgPool2d(4), nn.Conv2d(14, 32, 3, padding=1), nn.SiLU(),
                                      nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
                                      nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(256, 128), nn.SiLU())
            d += 128
        if "full" in parts:
            self.fcnn = nn.Sequential(nn.Conv2d(14, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                      nn.SiLU(), nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.SiLU(),
                                      nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.SiLU(),
                                      nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(512, 128), nn.SiLU())
            d += 128
        self.mlp = nn.Sequential(nn.Linear(d, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, N_OUT))

    def forward(self, x: dict):
        z = [x["priv"], self.opp(x["opp"])]
        if "feat" in self.parts:
            z.append(x["feat"])
        if "probe" in self.parts:
            z.append(x["probe"])
        need_board = self.parts & {"tok", "win", "coarse", "full"}
        if need_board:
            board = unpack_board(x["board"])
        if "tok" in self.parts:
            t, v = tokens(board, x["head"])
            h = self.tok_att(self.tok_in(t), src_key_padding_mask=~v)
            z.append((h * v[..., None]).sum(1) / v.sum(1, keepdim=True).clamp(min=1))
        if "win" in self.parts:
            z.append(self.wcnn(crop(board, x["head"], self.win)))
        if "coarse" in self.parts:
            z.append(self.ccnn(board))
        if "full" in self.parts:
            z.append(self.fcnn(board))
        return self.mlp(torch.cat(z, 1))          # value, then the auxiliary heads (OUTS)


# outputs: the value, then auxiliary heads (trained unless a variant says -noaux)
OUTS = ["value"] + [f"disc{k}" for k in HORIZONS]
N_OUT = len(OUTS)

BASE = {
    "live":        {"feat"},
    "probe":       {"feat", "probe"},
    "tok":         {"feat", "tok"},
    "win21":       {"feat", "win"},
    "coarse":      {"feat", "coarse"},
    "full64":      {"feat", "full"},
    "probe+tok":   {"feat", "probe", "tok"},
    "probe+tok+win": {"feat", "probe", "tok", "win"},
    "nofeat_win+probe": {"probe", "win"},
    # user, 2026-10-01: a separate critic on the true board -- a 27x27 crop about the acting head
    # plus the whole board pooled to 16x16 -- with and without the policy's features
    "crop27+coarse": {"feat", "win", "coarse"},
    "nofeat_crop27+coarse": {"win", "coarse"},
}
WIN = {"crop27+coarse": 27, "nofeat_crop27+coarse": 27}        # crop side (default 21)
# name[-noaux][@l80]: -noaux trains the value alone; @l80 fits the value to GAE lambda 0.8, not 0.95
DEFAULT_VARIANTS = list(BASE) + ["live-noaux", "probe+tok-noaux", "live@l80", "probe+tok@l80"]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(n0 * n1, 1))


def ev(t: np.ndarray, p: np.ndarray) -> float:
    return float(1 - np.var(t - p) / np.var(t))


def fit(a) -> None:
    d = pathlib.Path(a.data)
    dev = torch.device("cuda")
    t0 = time.perf_counter()
    Y = targets(d, a.alpha)
    S = np.load(d / "samples.npz")
    ok = np.isfinite(Y["ret"]) & np.isfinite(Y["gae95"])
    h = (S["row_game"].astype(np.int64) * 2654435761) % 1000
    te_ix, va_ix, tr_ix = (np.flatnonzero(ok & m) for m in (h < 200, (h >= 200) & (h < 300), h >= 300))
    print(f"targets in {time.perf_counter() - t0:.0f}s; {ok.sum()} usable samples of {len(ok)} ({len(tr_ix)} train, "
          f"{len(va_ix)} model-selection, {len(te_ix)} test)", flush=True)
    # how noisy is each target? (sd, and lag-free autocorrelation is not available: report sd only)
    print("target sd: " + ", ".join(f"{k} {np.nanstd(Y[k][ok]):.4f}" for k in ("ret", "gae95", "gae80", *(f"disc{h}" for h in HORIZONS))), flush=True)
    pr, pb = S["priv"], S["probe"]
    nrm = lambda x: ((x - x[tr_ix].mean(0)) / (x[tr_ix].std(0) + 1e-6)).astype(np.float32)   # noqa: E731
    X = {"feat": torch.from_numpy(S["feat"].astype(np.float32)).to(dev),
         "priv": torch.from_numpy(nrm(pr)).to(dev),
         "opp": torch.from_numpy(S["opp"]).to(dev),
         "probe": torch.from_numpy(nrm(pb)).to(dev),
         "board": torch.from_numpy(S["board"]).to(dev),
         "head": torch.from_numpy(S["head"].astype(np.int64)).to(dev)}
    stats = {}
    for k in ("ret", "gae95", "gae80", *(f"disc{h}" for h in HORIZONS)):
        stats[k] = (float(np.nanmean(Y[k][tr_ix])), float(np.nanstd(Y[k][tr_ix])))
    def col(k, x):
        m, s_ = stats[k]
        return torch.from_numpy(((np.nan_to_num(x) - m) / s_).astype(np.float32)).to(dev)
    tgt_aux = torch.stack([col(f"disc{h}", Y[f"disc{h}"]) for h in HORIZONS], 1)
    vt = {"gae95": col("gae95", Y["gae95"]), "gae80": col("gae80", Y["gae80"])}

    def predict(net, ix):
        outs = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(ix), 4096):
                b = torch.from_numpy(ix[i:i + 4096]).to(dev)
                outs.append(net({k: v[b] for k, v in X.items()}).float().cpu().numpy())
        return np.concatenate(outs)

    def report(name, o, vkey, aux):
        T_ = {k: Y[k][te_ix] for k in Y}
        m, s_ = stats[vkey]
        v = o[:, 0] * s_ + m
        row = {"variant": name, "ev_gae95": ev(T_["gae95"], v), "ev_ret": ev(T_["ret"], v),
               "corr_v_disc5": float(np.corrcoef(v, T_["disc5"])[0, 1])}
        if aux:
            for j, h in enumerate(HORIZONS):
                m, s_ = stats[f"disc{h}"]
                row[f"ev_disc{h}"] = ev(T_[f"disc{h}"], o[:, 1 + j] * s_ + m)
        return row

    rows = []
    vl = S["vlive"][te_ix].astype(np.float64)
    T_ = {k: Y[k][te_ix] for k in Y}
    rows.append({"variant": "live head (as trained)", "ev_gae95": ev(T_["gae95"], vl), "ev_ret": ev(T_["ret"], vl),
                 "corr_v_disc5": float(np.corrcoef(vl, T_["disc5"])[0, 1])})
    print(json.dumps(rows[-1]), flush=True)
    names = a.variants.split(",") if a.variants else DEFAULT_VARIANTS
    tr_t = torch.from_numpy(tr_ix).to(dev)
    for name in names:
        base, vkey = name.split("@")[0], ("gae80" if name.endswith("@l80") else "gae95")
        aux = not base.endswith("-noaux")
        base = base.replace("-noaux", "")
        torch.manual_seed(0)
        net = Critic(BASE[base], win=WIN.get(base, 21)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        steps = a.steps
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
        t1 = time.perf_counter()
        best, best_state = 1e9, None
        y_v = vt[vkey]
        for s in range(steps):
            net.train()
            b = tr_t[torch.randint(len(tr_t), (a.batch,), device=dev)]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = net({k: v[b] for k, v in X.items()}).float()
            loss = F.mse_loss(o[:, 0], y_v[b])
            if aux:
                ta = tgt_aux[b]
                loss = loss + a.aux * F.mse_loss(o[:, 1:], ta)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            if (s + 1) % (steps // 10) == 0:        # model selection on the value loss, separate games
                net.eval()
                ov = predict(net, va_ix)
                vloss = float(np.mean((ov[:, 0] - y_v[torch.from_numpy(va_ix).to(dev)].cpu().numpy()) ** 2))
                if vloss < best:
                    best, best_state = vloss, {k: x.detach().clone() for k, x in net.state_dict().items()}
        net.load_state_dict(best_state)
        net.eval()
        row = report(name, predict(net, te_ix), vkey, aux)
        b = torch.from_numpy(te_ix[:4096]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(3):
                net({k: v[b] for k, v in X.items()})
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            for _ in range(10):
                net({k: v[b] for k, v in X.items()})
            torch.cuda.synchronize()
        row["us_per_row"] = round((time.perf_counter() - t2) / 10 / 4096 * 1e6, 3)
        row["fit_s"] = round(time.perf_counter() - t1)
        rows.append(row)
        print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}), flush=True)
    (d / f"results{a.tag}.json").write_text(json.dumps(rows, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--out", required=True)
    g.add_argument("--policy", default=str(ROOT / "runs/ratchet_ff3/cands/g002_s2/latest.pt"))
    g.add_argument("--opps", default=",".join([
        f"anchor={ROOT}/runs/ratchet_ff3/anchors/gen0.pt@1", f"gen1={ROOT}/runs/ratchet_ff3/anchors/gen1.pt@1",
        f"v9_gen2={ROOT}/runs/ratchet_v9/anchors/gen2.pt@0.5",
        f"v8sab_gen10={ROOT}/runs/ratchet_v8_sab/anchors/gen10.pt@0.5"]))
    g.add_argument("--self-frac", type=float, default=0.4)
    g.add_argument("--temp", type=float, default=0.4)
    g.add_argument("--envs", type=int, default=512)
    g.add_argument("--threads", type=int, default=12)
    g.add_argument("--keep", type=float, default=0.01)
    g.add_argument("--samples", type=int, default=250_000)
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--max-drain", type=int, default=40000, help="steps played after the last sample at most")
    f = sub.add_parser("fit")
    f.add_argument("--data", required=True)
    f.add_argument("--alpha", type=float, default=0.95)
    f.add_argument("--variants", default="")
    f.add_argument("--steps", type=int, default=8000, help="optimizer steps per variant")
    f.add_argument("--batch", type=int, default=1024)
    f.add_argument("--wd", type=float, default=0.01)
    f.add_argument("--aux", type=float, default=1.0, help="weight of the auxiliary heads' losses")
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--tag", default="")
    a = p.parse_args()
    gen(a) if a.cmd == "gen" else fit(a)


if __name__ == "__main__":
    main()
