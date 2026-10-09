"""Supervised fine-tune of a PPO policy on a top team's server games, with OUR sonar (user, 2026-10-05).

    BC_QUEEN_GUARD=1 /usr/bin/python3 -m train.sl_replays --init base.pt --games ../runs/replays/top3_1005/forgot_to_mention \
        --team-id 264 --out ../runs/sl_top3_1005/sl.pt

PPO is paused; the latest policy is fine-tuned to play the exemplar team's moves, with a small
KL(init || new) term so it does not forget everything else; the result is gated like any candidate.

Every env replays one server game in the simulator the policy was trained on (the replay build of
its library: s2 g15p, portal packet, far sprints, queen guard from BC_QUEEN_GUARD), every dragon
playing exactly the recorded move. The exemplar team's dragons cast OUR team packet (set_sonar2)
and their own recorded sonar is dropped; the opponent casts what its replay says (it fails our tag).
So the policy reads the grid it would read in play, with its teammates' reports in it -- what the
exemplar actually did with its sonar is ignored completely.

Games come from replay_dataset's index (--index-only is enough): exact games whole, diverged games
up to the turn they diverge. Labels: 1-3 step moves, splits (incl. k = len-2/len-3), the self-kill,
and far sprints (a 4-8 step move whose end tile is a far target: the id of that target; our own
BFS path may differ, the destination is the same). Rows whose label the mask forbids (queen guard,
codec gaps) are dropped and counted.

Loss on each row, told a temperature T ~ U(--temp-lo, --temp-hi) like the PPO rollouts:
  -(T / temp_ref) * log P_T(label or its alias)  +  --kl-coef * KL(init || new) at temp_ref
on the same rows. Held-out series (--val-share) measure greedy agreement (told temp_min, as a gate
tells it) and the KL, before and after; --max-kl stops early, keeping the last checkpoint under it.

Writes --out (the init checkpoint with the new policy weights, everything else as it was) and
log.jsonl next to it.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing as mp
import os
import pathlib
import sys
import time
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bcsim"))

import torch                            # noqa: E402


def _pick_lib() -> None:
    """The REPLAY build of the library the policy trained on (moves longer than 8 steps)."""
    a = torch.load(sys.argv[sys.argv.index("--init") + 1], map_location="cpu", weights_only=False)["args"]
    if not (a.get("sonar2") and a.get("portal") and int(a.get("grid", 14)) == 15):
        raise SystemExit("sl_replays is written for s2 / portal / 15x15 policies (libbcvec_replay_s2_g15p)")
    os.environ["BCSIM_LIB"] = str(ROOT / "bcsim/bcsim/libbcvec_replay_s2_g15p.so")


_pick_lib()
import numpy as np                      # noqa: E402
import torch.nn.functional as F         # noqa: E402

import bcsim                            # noqa: E402
from bcsim.env import MAX_STEPS         # noqa: E402
from train.clone_lstm import _load, encode_turn   # noqa: E402
from train.distill_lstm import Pool     # noqa: E402
from train.net import masked_logits     # noqa: E402
from train.perfect_play import policy_logits      # noqa: E402
from train.yardstick import load_net    # noqa: E402

SC = bcsim.SCALARS
I_FACE, I_LEN = SC.index("face_n"), SC.index("length_raw")


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", required=True, help="the policy to fine-tune (also the frozen KL reference)")
    p.add_argument("--games", required=True, help="replay_fetch dir with dataset/index.jsonl")
    p.add_argument("--team-id", type=int, required=True)
    p.add_argument("--index", default="", help="index.jsonl to use (default <games>/dataset/index.jsonl)")
    p.add_argument("--out", required=True)
    p.add_argument("--val-share", type=float, default=0.1, help="share of series held out")
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--val-envs", type=int, default=32)
    p.add_argument("--steps", type=int, default=256, help="env steps per rollout")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--epochs", type=float, default=2.0, help="passes over the training turns")
    p.add_argument("--batch", type=int, default=4096, help="rows per Adam step")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--kl-coef", type=float, default=0.1)
    p.add_argument("--max-kl", type=float, default=0.0, help="stop when held-out KL passes this (0 = never)")
    p.add_argument("--temp-lo", type=float, default=0.1)
    p.add_argument("--temp-hi", type=float, default=0.35)
    p.add_argument("--val-every", type=int, default=10, help="rollouts between held-out measurements")
    p.add_argument("--load-workers", type=int, default=8)
    p.add_argument("--limit-games", type=int, default=0, help="smoke tests")
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--ahist-in", type=int, default=0,
                   help="1 = give the policy the action-history inputs (ff_net ahist_in, zero columns) and "
                        "feed the replayed moves' ids to the simulator so the plane is filled")
    return p.parse_args()


def far_id(dirs, facing: int, length: int, ftab: dict) -> int:
    """A move of 4+ steps as a far sprint: the target whose ego offset is the path's end tile."""
    ox = oy = 0
    for d in dirs:
        rel = (int(d) - facing) % 4                 # 0 forward, 1 right, 2 back, 3 left
        ox += (0, 1, 0, -1)[rel]
        oy += (-1, 0, 1, 0)[rel]
    return ftab.get((ox, oy), -1)


def main() -> None:
    a = parse()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = (out.parent / "log.jsonl").open("a")
    t_start = time.time()

    net, ck = load_net(a.init, dev)
    args = ck["args"]
    if args.get("arch") not in ("ff", "ffl"):
        raise SystemExit(f"{a.init}: feed-forward policies only (arch {args.get('arch')})")
    if a.ahist_in:
        if bcsim.GRID_CH <= 57:
            raise SystemExit(f"--ahist-in: {os.environ['BCSIM_LIB']} has no action-history plane (make -C bcsim s2g15p)")
        net.add_ahist_input()
        ck = dict(ck)
        ck["args"] = {**ck["args"], "ahist_in": True}
        args = ck["args"]
    ref = copy.deepcopy(net).eval()
    for q in ref.parameters():
        q.requires_grad_(False)
    temp_ref = float(args.get("temp_ref", args.get("temp", 0.4)))
    temp_eval = float(args.get("temp_min", a.temp_lo))
    A = bcsim.N_ACTIONS
    assert A == int(args["n_actions"]), (A, args["n_actions"])
    first, offs = bcsim.BattlecodeVecEnv.far_targets()
    ftab = {tuple(o): first + k for k, o in enumerate(offs)}
    assert len(ftab) == 24, ftab

    # ---- games
    root = pathlib.Path(a.games)
    idx = pathlib.Path(a.index) if a.index else root / "dataset/index.jsonl"
    rows = [json.loads(l) for l in idx.read_text().splitlines()]
    rows = [r for r in rows if r.get("samples", 0) > 0 and not r.get("error")]
    if a.limit_games:
        rows = rows[:a.limit_games]
    with mp.Pool(a.load_workers) as pool:
        games = [g for g in pool.imap_unordered(_load, [(root, r, a.team_id) for r in rows], chunksize=8) if g]
    games.sort(key=lambda g: g["game"])
    val = [g for g in games if (zlib.crc32(str(g["series"]).encode()) % 1000) < a.val_share * 1000]
    train = [g for g in games if (zlib.crc32(str(g["series"]).encode()) % 1000) >= a.val_share * 1000]
    n_ours = sum(int(np.isin(g["dragon"], g["dragon"]).sum()) for g in train)   # all turns; ~half are ours
    maps = sorted({g["map"] for g in games})
    map_idx = {m: i for i, m in enumerate(maps)}
    print(f"{len(games)} games ({len(train)} train, {len(val)} held-out), {len(maps)} maps, "
          f"{n_ours / 1e6:.1f}M turns in training games; loaded in {time.time() - t_start:.0f}s", flush=True)

    E = a.envs + a.val_envs
    env = bcsim.BattlecodeVecEnv(maps, num_envs=E, num_threads=a.threads, seed=a.seed,
                                 random_pearl_seed=False, closure_capacity=max(8192, E * 160), grid=True)
    assert env.sonar2 and env.portal, os.environ["BCSIM_LIB"]
    print(f"queen guard {env.queen_guard}, grid {tuple(env.grid.shape[1:])}, {A} actions, "
          f"temp_ref {temp_ref}, told {a.temp_lo}-{a.temp_hi}, eval {temp_eval}", flush=True)
    C, G = env.grid.shape[1], env.grid.shape[2]

    queues = {"train": [], "val": []}
    passes = {"train": 0, "val": 0}

    def next_game(kind):
        q = queues[kind]
        if not q:
            src = train if kind == "train" else val
            q.extend(rng.permutation(len(src)).tolist())
            passes[kind] += 1
        return (train if kind == "train" else val)[q.pop()]

    role = ["train"] * a.envs + ["val"] * a.val_envs
    cur = [None] * E
    cursor = np.zeros(E, np.int64)
    pool = Pool(E * 160, 0, 1, dev, grow=True, no_action=net.no_action)

    def load(e):
        g = next_game(role[e])
        cur[e] = g
        cursor[e] = 0
        env.set_opponent(e, team=-1, bot=0, map_index=map_idx[g["map"]])
        env.set_pearl_seed64(e, g["seed"])
        env.set_sonar2(e, (g["side"],))
        env.restart_env(e)
        pool.release_env(e)

    obs = env.reset()
    for e in range(E):
        load(e)

    T = a.steps
    kind_a = np.zeros(E, np.int8); nst_a = np.zeros(E, np.int8)
    dirs_a = np.zeros((E, MAX_STEPS), np.int8); split_a = np.zeros(E, np.int16)
    send_a = np.zeros(E, np.uint8); sval_a = np.zeros((E, 4), np.uint64)
    stats = {"ours": 0, "no_label": 0, "masked": 0, "far": 0, "mismatch": 0}

    def rollout(collect_val: bool):
        """T env steps of every env; returns the training rows and the held-out rows (GPU tensors)."""
        nonlocal obs
        tr, va = [], []
        for _ in range(T):
            ours = np.zeros(E, bool)
            tgt = np.full((E, 2), -1, np.int64)
            step_ids = np.full(E, -1, np.int32)
            for e in range(E):
                for _ in range(3):
                    g, c = cur[e], cursor[e]
                    if c < len(g["dragon"]) and (int(obs.dragon_id[e]), int(obs.round[e])) == \
                            (int(g["dragon"][c]), int(g["round"][c])):
                        break
                    stats["mismatch"] += int(c > 0)
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
                    length = int(obs.scalar[e, I_LEN])
                    t0, t1 = encode_turn(g, c, facing, length)
                    if t0 < 0 and k == 0 and d1 - d0 >= 4:
                        t0 = far_id(g["dirs"][d0:d1], facing, length, ftab)
                        stats["far"] += int(t0 >= 0)
                    tgt[e] = (t0, t1)
                    step_ids[e] = t0
                    send_a[e] = 0                      # our packet (set_sonar2), never theirs
                    sval_a[e] = 0
                else:
                    send_a[e] = g["send"][c]
                    sval_a[e] = g["sval"][c]
            rows_ = np.flatnonzero(ours)
            if len(rows_):
                stats["ours"] += len(rows_)
                slots = torch.as_tensor(pool.get(rows_, obs.uid[rows_]), device=dev)
                prev = pool.prev[slots].clone()
                m = obs.mask[rows_].astype(bool)
                t = tgt[rows_]
                ok0 = (t[:, 0] >= 0) & m[np.arange(len(rows_)), np.maximum(t[:, 0], 0)]
                ok1 = (t[:, 1] >= 0) & m[np.arange(len(rows_)), np.maximum(t[:, 1], 0)]
                stats["no_label"] += int((t[:, 0] < 0).sum())
                stats["masked"] += int(((t[:, 0] >= 0) & ~ok0 & ~ok1).sum())
                keep = ok0 | ok1
                lab = np.where(ok0, t[:, 0], t[:, 1])
                alt = np.where(ok0 & ok1, t[:, 1], -1)
                if keep.any():
                    ki = np.flatnonzero(keep)
                    er = rows_[ki]
                    is_val = er >= a.envs
                    for sel, dst in ((~is_val, tr), (is_val, va)):
                        if sel.any() and (dst is tr or collect_val):
                            pick = ki[sel]
                            dst.append((torch.from_numpy(env.grid[rows_[pick]]).to(dev).half(),
                                        prev[torch.as_tensor(pick, device=dev)],
                                        torch.from_numpy(m[pick]).to(dev),
                                        torch.from_numpy(lab[pick]).to(dev),
                                        torch.from_numpy(alt[pick]).to(dev)))
                pa = np.where(tgt[rows_, 0] >= 0, tgt[rows_, 0], net.no_action)
                pool.prev[slots] = torch.as_tensor(pa, device=dev)
            obs, closures, eps = env.step_raw(kind_a, nst_a, dirs_a, split_a, send_a, sval_a,
                                              ids=step_ids if a.ahist_in else None)
            cursor[:] += 1
            ended = set(int(r[0]) for r in eps.rows)
            for e in range(E):
                if e in ended or cursor[e] >= len(cur[e]["dragon"]):
                    load(e)
            if len(closures.env):
                for e_, u_, d_ in zip(closures.env.tolist(), closures.uid.tolist(), closures.done.tolist()):
                    if d_:
                        pool.release(e_, u_)

        def cat(rs):
            if not rs:
                return None
            return tuple(torch.cat([r[i] for r in rs]) for i in range(5))
        return cat(tr), cat(va)

    def measure(rows):
        """Greedy agreement told temp_eval (as the gate plays), mean log P at temp_ref, KL(ref || net)."""
        g, prev, m, lab, alt = rows
        acc = lp_sum = kl_sum = 0.0
        n = len(lab)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in range(0, n, 8192):
                s = slice(b, b + 8192)
                gg = g[s].float()
                te = torch.full((len(gg),), temp_eval, device=dev)
                lg = masked_logits(policy_logits(net, gg, prev[s], te).float(), m[s])
                top = lg.argmax(1)
                acc += ((top == lab[s]) | (top == alt[s])).float().sum().item()
                tr_ = torch.full((len(gg),), temp_ref, device=dev)
                lq = F.log_softmax(masked_logits(policy_logits(net, gg, prev[s], tr_).float(), m[s]) / temp_ref, 1)
                la = F.log_softmax(masked_logits(policy_logits(ref, gg, prev[s], tr_).float(), m[s]) / temp_ref, 1)
                lp_sum += lq.gather(1, lab[s][:, None]).sum().item()
                kl_sum += (la.exp() * (la - lq)).nan_to_num(0.0).sum().item()
        return {"n": n, "acc": round(acc / n, 4), "logp": round(lp_sum / n, 4), "kl": round(kl_sum / n, 5)}

    # a fixed held-out set: the val envs' rows over the first rollouts
    net.eval()
    held_parts = []
    while sum(len(p[3]) for p in held_parts) < 60000:
        _, va = rollout(True)
        if va is not None:
            held_parts.append(va)
    held = tuple(torch.cat([p[i] for p in held_parts]) for i in range(5))
    before = measure(held)
    print(f"held-out {before['n']:,} rows before: agree {before['acc']:.4f} logp {before['logp']:.3f}; "
          f"labels: {stats}", flush=True)
    log.write(json.dumps({"time": time.time(), "phase": "before", **before, "stats": dict(stats)}) + "\n")
    log.flush()

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0, eps=1e-5)
    target_rows = a.epochs * n_ours / 2
    done_rows, it, steps = 0, 0, 0
    best = None
    stop = ""
    while done_rows < target_rows:
        t0 = time.time()
        net.eval()
        trr, _ = rollout(False)
        t_roll = time.time() - t0
        if trr is None:
            continue
        net.train()
        g, prev, m, lab, alt = trr
        n = len(lab)
        perm = torch.randperm(n, device=dev)
        sl_s = kl_s = 0.0
        nb = 0
        for b in range(0, n - a.batch // 2, a.batch):
            i = perm[b:b + a.batch]
            temp = torch.empty(len(i), device=dev).uniform_(a.temp_lo, a.temp_hi)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = policy_logits(net, g[i].float(), prev[i], temp).float()
                tr_ = torch.full((len(i),), temp_ref, device=dev)
                lgr = policy_logits(net, g[i].float(), prev[i], tr_).float()
                with torch.no_grad():
                    lga = policy_logits(ref, g[i].float(), prev[i], tr_).float()
            lp = F.log_softmax(masked_logits(lg, m[i]) / temp[:, None], 1)
            ok = torch.zeros_like(m[i])
            ok.scatter_(1, lab[i][:, None], True)
            a1 = alt[i]
            ok[a1 >= 0, a1[a1 >= 0]] = True
            lp_ok = lp.masked_fill(~ok, -1e9).logsumexp(1)
            sl = -((temp / temp_ref) * lp_ok).mean()
            lq = F.log_softmax(masked_logits(lgr, m[i]) / temp_ref, 1)
            la = F.log_softmax(masked_logits(lga, m[i]) / temp_ref, 1)
            kl = (la.exp() * (la - lq)).nan_to_num(0.0).sum(1).mean()
            loss = sl + a.kl_coef * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            sl_s += sl.detach().item(); kl_s += kl.detach().item(); nb += 1; steps += 1
        done_rows += n
        it += 1
        row = {"time": time.time(), "iter": it, "rows": done_rows, "steps": steps, "sl": round(sl_s / max(nb, 1), 4),
               "kl_train": round(kl_s / max(nb, 1), 5), "t_roll": round(t_roll, 1),
               "t_train": round(time.time() - t0 - t_roll, 1), "passes": passes["train"]}
        if it % a.val_every == 0 or done_rows >= target_rows:
            net.eval()
            row["held"] = measure(held)
            if a.max_kl and row["held"]["kl"] > a.max_kl:
                stop = f"held-out KL {row['held']['kl']} > {a.max_kl} at iter {it}"
            else:
                best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                save = dict(ck)
                save["net"] = best
                save["sl_replays"] = {"games": a.games, "team": a.team_id, "iter": it, "rows": done_rows,
                                      "before": before, "held": row["held"], "kl_coef": a.kl_coef, "lr": a.lr}
                tmp = out.with_suffix(".tmp")
                torch.save(save, tmp)
                tmp.replace(out)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"iter {it} rows {done_rows / 1e6:.2f}M/{target_rows / 1e6:.1f}M steps {steps} sl {row['sl']:.3f} "
              f"kl {row['kl_train']:.4f} roll {t_roll:.0f}s train {row['t_train']:.0f}s"
              + (f" | held agree {row['held']['acc']:.4f} logp {row['held']['logp']:.3f} kl {row['held']['kl']:.4f}"
                 if "held" in row else ""), flush=True)
        if stop:
            print("stop: " + stop, flush=True)
            break
    print(f"done in {(time.time() - t_start) / 60:.0f} min; labels {stats}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
