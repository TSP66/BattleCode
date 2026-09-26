"""Distils a teacher into the LSTM policy (train/lstm_net.py), with real BPTT.

The teacher is any flat checkpoint -- by default the clone of Sabotage-d's
submission 3952, the top of the ladder. The student reads the 38 x 14 x 14 grid
(cpp/bc_memory.hpp grid_cfg), never the teacher's inputs.

Sonar is OURS, not the cloned team's: every dragon declares protocol 3 and
broadcasts four ways every turn (bcsim sonar=True), which is what the bot will
do. The student's five echo planes come from that; the teacher reads num_msgs
under it, which HANDOFF.md measured as not hurting the clone (+5.5 points).

Why this is not distill.py's recurrent path. A step of the env is ONE dragon's
turn, and a game has 20-100 dragons, so a dragon acts about once every ~50
steps of its env. distill.py detached the state between turns, which trains a
one-step objective. Here each iteration:

  1. rolls out T steps in E envs, carrying every dragon's LSTM state in a pool
     keyed by (env, uid) and recording, per turn, the grid, the mask, the
     teacher's log-probs, the previous action and the state the dragon ENTERED
     the turn with;
  2. regroups the turns by dragon, in time order, into chunks of up to L;
  3. trains on the chunks: the CNN on every turn at once, then the LSTM unrolled
     over each chunk from its recorded starting state, so gradients flow through
     up to L of a dragon's turns.

Actions are DAgger: the teacher's for the first --teacher-iters iterations, then
the student's on a growing share of turns (down to --teacher-floor for the
teacher), so the student is trained on the states its own mistakes lead to.
The previous action it sees is always the one actually played.

    python -m train.distill_lstm --out ../runs/lstm_sab3952
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                    # noqa: E402
from train.lstm_net import LSTMPolicy, NO_ACTION  # noqa: E402
from train.net import masked_logits              # noqa: E402
from train.yardstick import evaluate, greedy, load_net  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--teacher", default=str(ROOT / "runs/imitate_sab_3952/best.pt"))
    p.add_argument("--maps", default=str(ROOT / "maps-all"))
    p.add_argument("--eval-maps", default=str(ROOT / "maps-live"))
    p.add_argument("--holdout", default="schooltime",
                   help="comma-separated map names (file stems) kept OUT of training and scored "
                        "separately in the eval, to see where the policy overfits. '' for none")
    p.add_argument("--out", default=str(ROOT / "runs/lstm_sab3952"))
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=1024, help="env steps per rollout")
    p.add_argument("--chunk", type=int, default=32, help="BPTT length, in a dragon's own turns")
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-chunks", type=int, default=384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--teacher-iters", type=int, default=20, help="iterations driven by the teacher alone")
    p.add_argument("--teacher-ramp", type=int, default=150, help="iterations to reach the floor after that")
    p.add_argument("--teacher-floor", type=float, default=0.3, help="share of turns the teacher keeps")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-games", type=int, default=8, help="per map, half each side")
    p.add_argument("--seed", type=int, default=11)
    # architecture: lstm_net.py's defaults are the metered ones
    p.add_argument("--c1", type=int, default=48)
    p.add_argument("--b1", type=int, default=1)
    p.add_argument("--c2", type=int, default=112)
    p.add_argument("--b2", type=int, default=2)
    p.add_argument("--squeeze", type=int, default=16)
    p.add_argument("--embed", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def arch_of(a) -> dict:
    return {k: getattr(a, k) for k in ("c1", "b1", "c2", "b2", "squeeze", "embed", "hidden", "layers")}


class Pool:
    """A state slot per live dragon, keyed by (env, uid).

    uid carries the episode, so a dragon of a finished game never collides with
    one of the next. `life` numbers every allocation, so a slot reused within a
    rollout still separates two dragons' turns when they are regrouped.
    """

    def __init__(self, n: int, layers: int, hidden: int, dev):
        self.h = torch.zeros(layers, n, hidden, device=dev)
        self.c = torch.zeros(layers, n, hidden, device=dev)
        self.prev = torch.full((n,), NO_ACTION, dtype=torch.long, device=dev)
        self.key: dict[tuple[int, int], int] = {}
        self.env_of = np.full(n, -1, np.int64)
        self.life = np.zeros(n, np.int64)
        self.free = list(range(n - 1, -1, -1))
        self.next_life = 0
        self.exhausted = 0

    def get(self, envs: np.ndarray, uids: np.ndarray) -> np.ndarray:
        slots = np.empty(len(envs), np.int64)
        fresh = []
        for i, (e, u) in enumerate(zip(envs.tolist(), uids.tolist())):
            s = self.key.get((e, u))
            if s is None:
                if not self.free:
                    self.exhausted += 1
                    self.release_env(e)          # never expected; keeps the run alive
                s = self.free.pop()
                self.key[(e, u)] = s
                self.env_of[s] = e
                self.life[s] = self.next_life
                self.next_life += 1
                fresh.append(s)
            slots[i] = s
        if fresh:
            f = torch.as_tensor(fresh, device=self.h.device)
            self.h[:, f] = 0
            self.c[:, f] = 0
            self.prev[f] = NO_ACTION
        return slots

    def release(self, e: int, u: int) -> None:
        s = self.key.pop((e, u), None)
        if s is not None:
            self.env_of[s] = -1
            self.free.append(s)

    def release_env(self, e: int) -> None:
        for k in [k for k in self.key if k[0] == e]:
            self.release(*k)


class LSTMGreedy:
    """Argmax LSTM policy for yardstick.evaluate's stateful-policy contract."""

    stateful = True
    wants_grid = True

    def __init__(self, net: LSTMPolicy, dev, num_envs: int, per_env: int = 160):
        self.net, self.dev = net, dev
        self.pool = Pool(num_envs * per_env, net.layers, net.hidden, dev)

    def rows(self, obs, rows, grid):
        idx = np.flatnonzero(rows)
        sl = self.pool.get(idx, obs.uid[idx])
        s = torch.as_tensor(sl, device=self.dev)
        state = [(self.pool.h[l, s], self.pool.c[l, s]) for l in range(self.net.layers)]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _, new = self.net(torch.from_numpy(grid[idx]).to(self.dev), self.pool.prev[s], state)
            m = torch.from_numpy(obs.mask[idx]).to(self.dev).bool()
            act = masked_logits(logits.float(), m).argmax(1)
        for l, (h, c) in enumerate(new):
            self.pool.h[l, s] = h.float()
            self.pool.c[l, s] = c.float()
        self.pool.prev[s] = act
        return act.to(torch.int32).cpu().numpy()

    def forget(self, env):
        self.pool.release_env(int(env))


def teacher_share(a, it: int) -> float:
    if it < a.teacher_iters:
        return 1.0
    f = min(1.0, (it - a.teacher_iters) / max(1, a.teacher_ramp))
    return 1.0 - f * (1.0 - a.teacher_floor)


def chunks_of(life: np.ndarray, L: int) -> np.ndarray:
    """Rows (in time order) regrouped by dragon into [n_chunks, L], -1 padded."""
    n = len(life)
    order = np.lexsort((np.arange(n), life))          # by dragon, then time
    sl = life[order]
    start = np.r_[0, np.flatnonzero(sl[1:] != sl[:-1]) + 1]
    first = np.repeat(start, np.diff(np.r_[start, n]))
    rank = np.arange(n) - first
    new_chunk = (rank % L) == 0
    cid = np.cumsum(new_chunk) - 1
    out = np.full((cid[-1] + 1, L), -1, np.int64)
    out[cid, rank % L] = order
    return out


def main() -> None:
    a = parse()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")

    teacher, tck = load_net(a.teacher, dev)
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    net = LSTMPolicy(**arch_of(a)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)
    it0, turns_total = 0, 0
    if a.resume and (out / "latest.pt").exists():
        ck = torch.load(out / "latest.pt", map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        it0, turns_total = ck["iter"] + 1, ck["total_turns"]
        print(f"resumed at iter {it0}", flush=True)
    n_par = sum(p_.numel() for k, m in net.named_children() if k != "v" for p_ in m.parameters())
    print(f"teacher {a.teacher} (iter {tck.get('iter')})\nstudent LSTMPolicy {arch_of(a)}: "
          f"{n_par / 1e6:.3f}M params\nsonar ON (ours: protocol 3, four-way broadcast every turn)", flush=True)

    E, T, L = a.envs, a.steps, a.chunk
    held = {h for h in a.holdout.split(",") if h}
    train_files = [f for f in sorted(pathlib.Path(a.maps).glob("*.map")) if f.stem not in held]
    assert len(train_files) < len(list(pathlib.Path(a.maps).glob("*.map"))) or not held, \
        f"--holdout {sorted(held)} names no map in {a.maps}"
    print(f"training on {len(train_files)} maps; held out: {sorted(held) or 'none'}", flush=True)
    env = bcsim.BattlecodeVecEnv(bcsim.load_maps([str(f) for f in train_files]), num_envs=E, num_threads=a.threads,
                                 seed=a.seed, closure_capacity=max(8192, E * 160), grid=True, sonar=True)
    C, G = env.grid.shape[1], env.grid.shape[2]
    A = bcsim.N_ACTIONS
    pool = Pool(E * 160, a.layers, a.hidden, dev)
    buf_grid = torch.zeros(T, E, C, G, G, dtype=torch.float16, device=dev)
    buf_mask = torch.zeros(T, E, A, dtype=torch.bool, device=dev)
    buf_logp = torch.zeros(T, E, A, dtype=torch.float16, device=dev)
    buf_prev = torch.zeros(T, E, dtype=torch.long, device=dev)
    buf_h = torch.zeros(T, E, a.layers, a.hidden, dtype=torch.float16, device=dev)
    buf_c = torch.zeros(T, E, a.layers, a.hidden, dtype=torch.float16, device=dev)
    buf_life = np.zeros((T, E), np.int64)
    envs_idx = np.arange(E)
    obs = env.reset()
    prev_round = np.asarray(obs.round, np.int64).copy()
    maps_eval = bcsim.load_maps(a.eval_maps)
    names_eval = sorted(p_.stem for p_ in pathlib.Path(a.eval_maps).glob("*.map"))
    t_start = time.perf_counter()

    for it in range(it0, a.iters):
        share = teacher_share(a, it)
        t_roll = time.perf_counter()
        net.eval()
        for t in range(T):
            slots = pool.get(envs_idx, obs.uid)
            s = torch.as_tensor(slots, device=dev)
            grid = torch.from_numpy(env.grid).to(dev, non_blocking=True)
            local = torch.from_numpy(obs.local).to(dev, non_blocking=True)
            scalar = torch.from_numpy(obs.scalar).to(dev, non_blocking=True)
            mask = torch.from_numpy(obs.mask).to(dev, non_blocking=True).bool()
            prev = pool.prev[s]
            state = [(pool.h[l, s], pool.c[l, s]) for l in range(a.layers)]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                t_logits, _ = teacher(local, scalar)
                s_logits, _, new = net(grid, prev, state)
            t_logp = F.log_softmax(masked_logits(t_logits.float(), mask), dim=1)
            s_logp = F.log_softmax(masked_logits(s_logits.float(), mask), dim=1)
            buf_grid[t] = grid.half()
            buf_mask[t] = mask
            buf_logp[t] = t_logp.half()
            buf_prev[t] = prev
            buf_h[t] = torch.stack([pool.h[l, s] for l in range(a.layers)], 1).half()
            buf_c[t] = torch.stack([pool.c[l, s] for l in range(a.layers)], 1).half()
            buf_life[t] = pool.life[slots]
            for l, (h, c) in enumerate(new):
                pool.h[l, s] = h.float()
                pool.c[l, s] = c.float()
            use_t = torch.rand(E, device=dev) < share
            logp = torch.where(use_t[:, None], t_logp, s_logp)
            action = torch.multinomial(logp.exp(), 1).squeeze(1)
            pool.prev[s] = action
            obs, closures, _ = env.step(action.to(torch.int32).cpu().numpy())
            if len(closures.env):
                for e_, u_, d_ in zip(closures.env.tolist(), closures.uid.tolist(), closures.done.tolist()):
                    if d_:
                        pool.release(e_, u_)
            cur = np.asarray(obs.round, np.int64)
            for e_ in np.flatnonzero(cur < prev_round).tolist():
                pool.release_env(e_)     # a finished game restarted this env
            prev_round = cur.copy()
        t_roll = time.perf_counter() - t_roll

        # ---- train on the rollout, regrouped by dragon
        t_train = time.perf_counter()
        net.train()
        ch = torch.as_tensor(chunks_of(buf_life.reshape(-1), L), device=dev)
        fg = buf_grid.reshape(T * E, C, G, G)
        fm, fp, fprev = buf_mask.reshape(T * E, A), buf_logp.reshape(T * E, A), buf_prev.reshape(T * E)
        fh, fc = buf_h.reshape(T * E, a.layers, a.hidden), buf_c.reshape(T * E, a.layers, a.hidden)
        tot_kl = tot_agree = tot_n = 0.0
        nb = 0
        for _ in range(a.epochs):
            perm = torch.randperm(len(ch), device=dev)
            for b0 in range(0, len(ch), a.batch_chunks):
                idx = ch[perm[b0:b0 + a.batch_chunks]]              # [B, L]
                valid = idx >= 0
                rows = idx[valid]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    x = net.encode(fg[rows].float(), fprev[rows])
                X = torch.zeros(*idx.shape, x.shape[1], device=dev, dtype=x.dtype)
                X[valid] = x
                first = idx[:, 0]
                state = [(fh[first, l].float(), fc[first, l].float()) for l in range(a.layers)]
                outs = []
                for k in range(L):
                    lg, _, state = net.step(X[:, k].float(), state)
                    outs.append(lg)
                s_logits = torch.stack(outs, 1)[valid]
                s_logp = F.log_softmax(masked_logits(s_logits, fm[rows]), dim=1)
                t_logp = fp[rows].float()
                kl = (t_logp.exp() * (t_logp - s_logp)).nan_to_num(0.0).sum(1).mean()
                opt.zero_grad(set_to_none=True)
                kl.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    n_ = len(rows)
                    tot_kl += kl.item() * n_
                    tot_agree += (s_logp.argmax(1) == t_logp.argmax(1)).float().sum().item()
                    tot_n += n_
                nb += 1
        sched.step()
        t_train = time.perf_counter() - t_train
        turns_total += T * E
        lens = (ch >= 0).sum(1).float()
        row = {"iter": it, "kl": tot_kl / tot_n, "agree": tot_agree / tot_n, "teacher_share": round(share, 3),
               "turns": turns_total, "chunks": len(ch), "chunk_len_mean": round(lens.mean().item(), 2),
               "live_slots": len(pool.key), "pool_exhausted": pool.exhausted,
               "rollout_s": round(t_roll, 1), "train_s": round(t_train, 1),
               "turns_per_s": round(T * E / (t_roll + t_train)),
               "elapsed": round(time.perf_counter() - t_start, 1)}

        if a.eval_every and (it + 1) % a.eval_every == 0:
            net.eval()
            n_env = len(names_eval) * max(1, a.eval_games // 2) * 2
            res = evaluate(LSTMGreedy(net, dev, n_env), [{"name": "teacher", "act": greedy(teacher, dev)}],
                           maps_eval, names_eval, games=a.eval_games, threads=a.threads, sonar=True)
            row["vs_teacher"] = res["summary"]["teacher"]["score"]
            row["eval_games"] = res["summary"]["teacher"]["n"]
            cells = res["cells"]["teacher"]
            for tag, keep in (("train_maps", lambda m: m not in held), ("holdout", lambda m: m in held)):
                cs = [c for m, c in cells.items() if keep(m)]
                g = sum(c["n"] for c in cs)
                if g:
                    row[f"vs_teacher_{tag}"] = round(sum(c["score"] * c["n"] for c in cs) / g, 4)
            row["vs_teacher_by_map"] = {m: c["score"] for m, c in cells.items()}
            row["eval_s"] = res["seconds"]

        log.write(json.dumps(row) + "\n")
        log.flush()
        print(f"it {it:4d} | kl {row['kl']:.4f} | agree {row['agree']:.3f} | teacher {share:.2f} "
              f"| {turns_total / 1e6:6.1f}M turns | {row['turns_per_s']:,}/s | chunk {row['chunk_len_mean']}"
              + (f" | vs teacher {row['vs_teacher']:.3f} ({row['eval_games']} games; "
                 f"train maps {row.get('vs_teacher_train_maps', float('nan')):.3f}, "
                 f"held out {row.get('vs_teacher_holdout', float('nan')):.3f})" if "vs_teacher" in row else ""),
              flush=True)
        ck = {"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "iter": it, "total_turns": turns_total,
              "args": {**vars(a), "arch": "lstm", **arch_of(a)}}
        torch.save(ck, out / "latest.pt.tmp")
        (out / "latest.pt.tmp").replace(out / "latest.pt")
        if (it + 1) % 50 == 0:
            torch.save({k: v for k, v in ck.items() if k not in ("opt", "sched")}, out / f"iter_{it + 1}.pt")
    env.close()


if __name__ == "__main__":
    main()
