"""A short supervised pass on "perfect play" after each gate (supervised_learning.md, user 2026-10-02).

    /usr/bin/python3 -m train.perfect_play --init cands/g004_s4/seg0.pt --out cands/g005_s5/sl_x.pt \
        --log run/sl.jsonl --tag g004_s4_seg0 --temp-lo 0.1 --temp-hi 0.45

There is a small set of situations whose right move is known for certain, and PPO should be
helped towards them, gently -- a nudge, not a forced answer that collapses the policy:

  queen_deadend  (2026-10-05, user: "no unforced queen error") a queen with a legal move into a dead end
                 she can see the end of (one way on at every tile until none): any move that is not
                 such a dive. ("blank", one step straight on in an empty window, is no longer trained.)
  pearl          a pearl beside the head, nobody else in sight: any unpaid path that takes it first
  queen_kill     a non-queen can kill the enemy queen this move (any move that does, far sprints too)
  late_suicide   rounds 481-498, no enemy in sight, our queen within 2: the self-kill (pearls for her)
  trapped_queen  this dragon walls our queen in and no move of it frees her: the self-kill
  keep_wall      this dragon is part of a wall around the enemy queen: any move that keeps her shut

Labels come only from the simulator's true game (VecEnv::perfect, cpp/bc_obs.hpp; every kind
checked by playing it out in tests/test_perfect.py), and only where the dragon's own window shows
everything the judgement needs.

1. GENERATE, fresh every pass and never on the training maps (user: this is for generalisation):
   random maps (train/perfect_scenes.py: size, kelp, kelp walls, portals, pearl timers) and on them
   each situation built directly -- up to --per-kind positions of each kind, every one labelled by
   the simulator; a quarter of each kind is held out to measure on.
2. ORDINARY TURNS: the policy's own self-play on those same random maps (--general rows). They are
   the KL set (what must not change) and the memory donors below.
3. AUGMENT (user: aggressively, so it does not learn maps): a scene's dragon has a fresh memory, so
   with probability --aug-p everything it would remember outside the window -- terrain, seen, pearls
   expected, dragons remembered, reports, portal planes, a queen's planes where she is not in sight --
   and the map size, unit count and echoes come from a random ordinary turn. The window, our own
   body and the round are never touched: they are what the label says. The previous action is
   "none" or a random 1-3 step path.
4. TRAIN from the unmodified policy: --steps Adam steps of --batch labelled rows (kinds equally
   likely), loss = -log P(any correct action) at the row's temperature, only while that probability
   is under --cap (a learned row stops pulling), plus --kl-coef x KL(unmodified || new) on
   --kl-batch ordinary turns. After every step the KL on held-out ordinary turns is measured; past
   --max-kl the pass stops and keeps the previous step's weights.

Writes --out (the init checkpoint with the new policy weights; optimizer, critic and counters as
they were) and one row to --log, which the dashboard shows.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
ROOT = pathlib.Path(__file__).resolve().parents[2]
# PERFECT_KINDS[2:]: "blank" (straight on) is no longer trained (user, 2026-10-05), "queen_deadend" added
KINDS = ["queen_kill", "late_suicide", "trapped_queen", "pearl", "keep_wall", "queen_deadend"]

# grid channels (cpp/bc_memory.hpp grid_cfg)
MEM_SWAP = list(range(0, 9)) + list(range(22, 28)) + list(range(38, 43))
CONST_SWAP = [30, 31, 32, 33, 34, 35, 36, 37]   # units, map w/h, echoes
ALLY_QUEEN, ENEMY_QUEEN = 44, 45          # in the live window
QUEEN_MEM = {ALLY_QUEEN: (46, [48, 49, 50]), ENEMY_QUEEN: (47, [51, 52, 53])}


def _pick_lib() -> None:
    """The simulator build the policy was trained on, chosen before bcsim loads."""
    if "--init" not in sys.argv:
        return
    a = torch.load(sys.argv[sys.argv.index("--init") + 1], map_location="cpu", weights_only=False)["args"]
    grid, s2, portal = int(a.get("grid", 14)), bool(a.get("sonar2")), bool(a.get("portal"))
    if not s2:
        raise SystemExit("perfect play needs the self-kill: a sonar-v2 (--s2) policy")
    lib = "libbcvec_s2_g15p.so" if portal else {14: "libbcvec_s2.so", 15: "libbcvec_s2_g15.so"}[grid]
    os.environ.setdefault("BCSIM_LIB", str(ROOT / "bcsim" / "bcsim" / lib))


_pick_lib()
import bcsim                                       # noqa: E402
from train.distill_lstm import Pool                # noqa: E402
from train.net import masked_logits                # noqa: E402
from train.perfect_scenes import KINDS as SCENE_KINDS, SceneMaker   # noqa: E402
from train.yardstick import load_net               # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="the gated policy (also the KL anchor: it is not modified)")
    p.add_argument("--out", required=True)
    p.add_argument("--log", required=True, help="jsonl the dashboard reads (one row per pass)")
    p.add_argument("--tag", default="")
    p.add_argument("--gen", type=int, default=0)
    p.add_argument("--total-turns", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda", help="cpu for tests (slow)")
    p.add_argument("--data", default="", help=argparse.SUPPRESS)   # tests: cache the collection here
    # generation (fresh random maps, never the training maps)
    p.add_argument("--maps-n", type=int, default=256, help="random maps drawn per pass")
    p.add_argument("--per-kind", type=int, default=768, help="labelled positions generated per kind")
    p.add_argument("--scene-seconds", type=float, default=120.0, help="cap on generating them")
    p.add_argument("--holdout", type=float, default=0.25)
    # ordinary turns: the policy's self-play on the same random maps
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--general", type=int, default=8192, help="ordinary turns kept (KL set and memory donors)")
    p.add_argument("--general-rate", type=float, default=0.05, help="share of ordinary turns sampled")
    p.add_argument("--general-seconds", type=float, default=120.0)
    p.add_argument("--temp-lo", type=float, default=0.1)
    p.add_argument("--temp-hi", type=float, default=0.5)
    # training (user: 32 batches of 32, lr 0.001, KL 0.02)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="user suggested 1e-3; measured 2026-10-02 on gen4: one step at 1e-3 moved the KL on ordinary "
                        "turns to 0.14; 1e-5 runs all 32 steps at 0.009")
    p.add_argument("--kl-coef", type=float, default=0.02)
    p.add_argument("--kl-batch", type=int, default=256)
    p.add_argument("--max-kl", type=float, default=0.02,
                   help="KL(unmodified || new) on held-out ordinary turns past which the pass stops "
                        "(keeping the last step under it)")
    p.add_argument("--cap", type=float, default=0.9, help="a row stops pulling once P(correct) reaches this")
    p.add_argument("--aug-p", type=float, default=0.8, help="probability of the memory augmentation")
    return p.parse_args()


# ---------------------------------------------------------------- the policy
def policy_logits(net, grid, prev, temp):
    """Raw logits; temp is the per-row temperature the policy is told (if it reads one)."""
    if getattr(net, "temp_in", False):
        return net(grid, prev, None, temp)[0]
    return net(grid, prev)[0]


# ---------------------------------------------------------------- 1. generate
def generate(a, rng, no_action: int):
    """Synthetic labelled positions on fresh random maps: {kind: [(grid f16, prev, mask, acts, temp, round)]},
    the maps' texts, and stats."""
    maker = SceneMaker(rng)
    texts, widths = maker.maps(a.maps_n)
    N = a.maps_n
    env = bcsim.BattlecodeVecEnv(texts, num_envs=N, num_threads=a.threads, seed=a.seed, sonar=True, grid=True)
    for i in range(N):
        env.set_opponent(i, -1, 0, map_index=i)
    env.set_scenario_maps(widths)
    env.reset()
    keep = {k: [] for k in KINDS}
    built = tried = 0
    t0 = time.time()
    names = bcsim.BattlecodeVecEnv.PERFECT_KINDS
    while time.time() - t0 < a.scene_seconds:
        short = [k for k in SCENE_KINDS if len(keep[k]) < a.per_kind]
        if not short:
            break
        fresh = np.zeros(N, bool)
        for i in range(N):
            tried += 1
            spec = maker.scene(i, short[i % len(short)])
            if spec is not None and env.set_scenario(i, *spec):
                fresh[i] = True
        built += int(fresh.sum())
        kind, acts = env.perfect()
        obs = env.observation()
        g16 = env.grid.astype(np.float16)
        for i in np.flatnonzero(fresh & (kind > 0)):
            k = names[kind[i]]
            if k not in keep or len(keep[k]) >= a.per_kind:
                continue
            # the previous action: none, or a random 1-3 step path (a scene has no history)
            prev = no_action if rng.random() < 0.5 else int(rng.integers(0, 39))
            keep[k].append((g16[i].copy(), prev, obs.mask[i].copy(), acts[i].copy(),
                            float(rng.uniform(a.temp_lo, a.temp_hi)), int(obs.round[i])))
    env.close()
    return keep, texts, {"scenes_tried": tried, "scenes_built": built, "scene_seconds": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- 2. ordinary turns
def ordinary(a, net, dev, rng, texts):
    """The policy's self-play on the random maps: [(grid f16, prev, mask, temp)] and the turns played."""
    N = a.envs
    env = bcsim.BattlecodeVecEnv(texts, num_envs=N, num_threads=a.threads, seed=a.seed + 1, sonar=True, grid=True)
    for i in range(N):
        env.set_sonar2(i, (0, 1))                 # self-play: both teams speak, as in training
    team_temp = rng.uniform(a.temp_lo, a.temp_hi, (N, 2)).astype(np.float32)
    pool = Pool(N * 40, 0, 1, dev, grow=True, no_action=net.no_action)
    general = []
    obs = env.reset()
    t0, turns = time.time(), 0
    while len(general) < a.general and time.time() - t0 < a.general_seconds:
        grid = torch.from_numpy(env.grid).to(dev)
        mask = torch.from_numpy(obs.mask).to(dev).bool()
        slots = torch.as_tensor(pool.get(np.arange(N), obs.uid), device=dev)
        prev = pool.prev[slots]
        tl = torch.from_numpy(team_temp[np.arange(N), obs.team.astype(np.int64)]).to(dev)
        with torch.inference_mode(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            lg = policy_logits(net, grid, prev, tl).float()
        action = torch.multinomial(F.softmax(masked_logits(lg, mask) / tl[:, None], 1), 1).squeeze(1)
        pool.prev[slots] = action
        pick = np.flatnonzero(rng.random(N) < a.general_rate)
        if len(pick):
            g16 = env.grid[pick].astype(np.float16)
            prev_np, tl_np = prev.cpu().numpy(), tl.cpu().numpy()
            for j, e in enumerate(pick):
                general.append((g16[j], int(prev_np[e]), obs.mask[e].copy(), float(tl_np[e])))
        obs, closures, eps = env.step(action.to(torch.int32).cpu().numpy())
        turns += N
        for e_, u_, d_ in zip(closures.env.tolist(), closures.uid.tolist(), closures.done.tolist()):
            if d_:
                pool.release(e_, u_)
        if len(eps.rows):
            for e_ in eps.rows[:, 0].astype(np.int64).tolist():
                pool.release_env(e_)
                team_temp[e_] = rng.uniform(a.temp_lo, a.temp_hi, 2)
    env.close()
    return general[:a.general], turns, round(time.time() - t0, 1)


# ---------------------------------------------------------------- 3. augment
class Augmenter:
    """A scene's dragon remembers only its window: everything outside it (and the map size, unit
    count, echoes, and a queen's planes where she is not in sight) comes from an ordinary turn."""

    def __init__(self, G: int, rng, aug_p: float):
        self.G, self.H = G, G // 2
        self.lo, self.hi = self.H - 3, self.H + 3        # window rows/cols, inclusive
        self.rng, self.p = rng, aug_p
        inside = np.zeros((G, G), bool)
        inside[self.lo:self.hi + 1, self.lo:self.hi + 1] = True
        self.outside = ~inside

    def __call__(self, g: np.ndarray, donor: np.ndarray) -> None:
        """g (C, G, G) float32, modified in place."""
        if self.rng.random() >= self.p:
            return
        out = self.outside
        for ch in MEM_SWAP:
            g[ch][out] = donor[ch][out]
        for ch in CONST_SWAP:
            g[ch] = donor[ch]
        for live, (mem, vec) in QUEEN_MEM.items():
            if g[live, self.lo:self.hi + 1, self.lo:self.hi + 1].max() == 0:   # she is not in sight
                g[mem][out] = donor[mem][out]
                for ch in vec:
                    g[ch] = donor[ch]


# ---------------------------------------------------------------- 4. train
def evaluate(net, rows, dev, temp_eval: float, temp_ref: float):
    """Per kind: greedy accuracy (argmax among the correct actions, told temp_eval like a gate) and
    mean P(correct) at temp_ref."""
    out = {}
    for k, rs in rows.items():
        if not rs:
            continue
        g = torch.from_numpy(np.stack([r[0] for r in rs])).to(dev).float()
        prev = torch.tensor([r[1] for r in rs], device=dev)
        mask = torch.from_numpy(np.stack([r[2] for r in rs])).to(dev).bool()
        acts = torch.from_numpy(np.stack([r[3] for r in rs])).to(dev).bool()
        with torch.no_grad():
            lg = masked_logits(policy_logits(net, g, prev, torch.full((len(rs),), temp_eval, device=dev)).float(), mask)
            acc = acts.gather(1, lg.argmax(1, keepdim=True)).float().mean().item()
            lp = F.log_softmax(lg / temp_ref, 1)
            p = lp.masked_fill(~acts, -1e9).logsumexp(1).exp().mean().item()
        out[k] = {"n": len(rs), "acc": round(acc, 4), "p": round(p, 4)}
    return out


def kl_general(net, anchor, gen, dev, temp_ref):
    g, prev, mask, temp = gen
    with torch.no_grad():
        lp = F.log_softmax(masked_logits(policy_logits(net, g, prev, temp).float(), mask) / temp_ref, 1)
        la = F.log_softmax(masked_logits(policy_logits(anchor, g, prev, temp).float(), mask) / temp_ref, 1)
        return (la.exp() * (la - lp)).nan_to_num(0.0).sum(1).mean().item()


def main() -> None:
    a = parse()
    dev = torch.device(a.device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    t_start = time.time()
    net, ck = load_net(a.init, dev)
    args = ck["args"]
    if args.get("arch") not in ("ff", "ffl"):
        raise SystemExit(f"{a.init}: perfect play trains feed-forward grid policies (arch {args.get('arch')})")
    if getattr(net, "uniform", False):
        raise SystemExit(f"{a.init} is the uniform-random player")
    if bool(args.get("portal", False)) != bcsim.PORTAL_BUILD:
        raise SystemExit(f"{a.init}: portal {args.get('portal')} but the simulator {os.environ['BCSIM_LIB']}")
    anchor = copy.deepcopy(net).eval()
    for q in anchor.parameters():
        q.requires_grad_(False)
    temp_ref = float(args.get("temp_ref", args.get("temp", 0.4)))
    temp_eval = float(args.get("temp_min", a.temp_lo))       # what a gate tells the policy

    # 1. generate, 2. ordinary turns
    net.eval()
    if a.data and pathlib.Path(a.data).exists():      # tests: reuse a generation
        keep, general, gstats = torch.load(a.data, weights_only=False)
    else:
        keep, texts, gstats = generate(a, rng, net.no_action)
        general, turns, secs = ordinary(a, net, dev, rng, texts)
        gstats.update(general_turns=turns, general_seconds=secs)
        if a.data:
            torch.save((keep, general, gstats), a.data)
    found = {k: len(v) for k, v in keep.items()}
    print(f"generated {gstats['scenes_built']} scenes on {a.maps_n} random maps in {gstats['scene_seconds']}s: "
          + ", ".join(f"{k} {n}" for k, n in found.items())
          + f"; {len(general)} ordinary turns from {gstats.get('general_turns', 0):,} of self-play on them", flush=True)
    if len(general) < 64:
        raise SystemExit("too few ordinary turns collected")
    train_rows, held = {}, {}
    for k, rs in keep.items():
        idx = rng.permutation(len(rs))
        n_h = int(round(len(rs) * a.holdout)) if len(rs) >= 4 else 0
        held[k] = [rs[i] for i in idx[:n_h]]
        train_rows[k] = [rs[i] for i in idx[n_h:]]
    gi = rng.permutation(len(general))
    n_gh = max(len(general) // 4, 32)
    g_held = [general[i] for i in gi[:n_gh]]
    g_train = [general[i] for i in gi[n_gh:]]

    def gen_tensors(rs):
        return (torch.from_numpy(np.stack([r[0] for r in rs])).to(dev).float(),
                torch.tensor([r[1] for r in rs], device=dev),
                torch.from_numpy(np.stack([r[2] for r in rs])).to(dev).bool(),
                torch.tensor([r[3] for r in rs], device=dev))
    gh = gen_tensors(g_held)
    # the held-out positions are measured with a memory, as they will be met: augmented once, fixed
    aug = Augmenter(bcsim.GRID_SIDE, np.random.default_rng(a.seed + 5), 1.0)
    for k in held:
        rs = []
        for r in held[k]:
            g = r[0].astype(np.float32)
            aug(g, g_held[int(rng.integers(len(g_held)))][0].astype(np.float32))
            rs.append((g.astype(np.float16),) + tuple(r[1:]))
        held[k] = rs
    before = {"train": evaluate(net, train_rows, dev, temp_eval, temp_ref),
              "held": evaluate(net, held, dev, temp_eval, temp_ref)}

    # 3 + 4. augment and train
    kinds = [k for k in KINDS if train_rows[k]]
    steps_done, stop, kl_trace = 0, "", []
    if kinds:
        G = bcsim.GRID_SIDE
        aug = Augmenter(G, rng, a.aug_p)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0, eps=1e-5)
        net.train()
        last_good = copy.deepcopy(net.state_dict())
        for step in range(a.steps):
            rows, gs, ms, acs = [], [], [], []
            for _ in range(a.batch):
                k = kinds[rng.integers(len(kinds))]
                r = train_rows[k][rng.integers(len(train_rows[k]))]
                g = r[0].astype(np.float32)
                m = r[2]
                aug(g, g_train[rng.integers(len(g_train))][0].astype(np.float32))
                rows.append(r)
                gs.append(g)
                ms.append(m)
                acs.append(r[3])
            g_t = torch.from_numpy(np.stack(gs)).to(dev)
            prev = torch.tensor([r[1] for r in rows], device=dev)
            mask = torch.from_numpy(np.stack(ms)).to(dev).bool()
            acts = torch.from_numpy(np.stack(acs)).to(dev).bool()
            temp = torch.from_numpy(rng.uniform(a.temp_lo, a.temp_hi, a.batch).astype(np.float32)).to(dev)
            lp = F.log_softmax(masked_logits(policy_logits(net, g_t, prev, temp).float(), mask) / temp[:, None], 1)
            lp_ok = lp.masked_fill(~acts, -1e9).logsumexp(1)
            pull = (lp_ok.exp() < a.cap).float()          # a learned row stops pulling
            # d log pi_T / d logits scales as 1/T: weighted by T / temp_ref, as the PPO update does
            sl = -(pull * (temp / temp_ref) * lp_ok).mean()
            gi_ = rng.integers(len(g_train), size=a.kl_batch)
            gk = gen_tensors([g_train[i] for i in gi_])
            lq = F.log_softmax(masked_logits(policy_logits(net, *gk[:2], gk[3]).float(), gk[2]) / temp_ref, 1)
            with torch.no_grad():
                la = F.log_softmax(masked_logits(policy_logits(anchor, *gk[:2], gk[3]).float(), gk[2]) / temp_ref, 1)
            kl = (la.exp() * (la - lq)).nan_to_num(0.0).sum(1).mean()
            loss = sl + a.kl_coef * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            net.eval()
            klh = kl_general(net, anchor, gh, dev, temp_ref)
            net.train()
            kl_trace.append(round(klh, 5))
            if not math.isfinite(klh) or klh > a.max_kl:
                net.load_state_dict(last_good)
                stop = f"KL {klh:.4f} > {a.max_kl} after step {step + 1}: kept step {step}"
                break
            last_good = copy.deepcopy(net.state_dict())
            steps_done = step + 1
    else:
        stop = "nothing labelled"
    net.eval()
    after = {"train": evaluate(net, train_rows, dev, temp_eval, temp_ref),
             "held": evaluate(net, held, dev, temp_eval, temp_ref)}
    kl_end = kl_general(net, anchor, gh, dev, temp_ref)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    row = {"time": time.time(), "tag": a.tag, "gen": a.gen, "total_turns": a.total_turns, "init": a.init,
           "out": str(out), **gstats, "found": found,
           "general": len(general), "steps": steps_done, "steps_max": a.steps, "stopped": stop,
           "kl_held": round(kl_end, 5), "kl_trace": kl_trace, "lr": a.lr, "kl_coef": a.kl_coef,
           "max_kl": a.max_kl, "cap": a.cap, "before": before, "after": after,
           "seconds": round(time.time() - t_start, 1)}
    ck = dict(ck)
    ck["net"] = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    ck["sl"] = {k: row[k] for k in ("tag", "found", "steps", "kl_held", "before", "after")}
    tmp = out.with_suffix(".tmp")
    torch.save(ck, tmp)
    tmp.replace(out)
    with open(a.log, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    for part in ("train", "held"):
        print(f"{part:5s} " + " | ".join(
            f"{k} acc {before[part][k]['acc']:.2f}->{after[part][k]['acc']:.2f} "
            f"p {before[part][k]['p']:.2f}->{after[part][k]['p']:.2f} (n {after[part][k]['n']})"
            for k in KINDS if k in after[part]), flush=True)
    print(f"{steps_done}/{a.steps} steps, KL(held ordinary turns) {kl_end:.4f}{'; ' + stop if stop else ''}; "
          f"wrote {out} in {row['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
