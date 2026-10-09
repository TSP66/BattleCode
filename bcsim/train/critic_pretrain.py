"""Pretrain the true-board critic (train/cview.CViewCritic) before PPO (user, 2026-10-01).

Data: train/critic_data.py shards -- round-robin and generated games between our policies and
clones at per-game temperatures (roundrobin_all --record) and server replays (critic_replays.py,
observed, T = 0). One head per discount (--alphas: horizons 20 / 10 / 5 / 3 rounds); the
explained variance of each says how far ahead the critic can see. PPO then trains the head of
its own alpha (ratchet_ff_train --critic-init).

Identities: every side has a name -- an agent's ("cutlery15") or a real team's ("team:306").
--identities maps names to identities (a clone is its original team: "cutlery15=team:306"); any
other name is its own identity ("agent:<name>", "team:<id>"). Each identity seen often enough
(--min-games) gets a slot of the critic's table; the rest share slot 0 (unknown). The name map
and the slot table travel in the checkpoint.

    python -m train.critic_pretrain prepare --shards ../runs/critic_pre/rr_s1 ../runs/critic_pre/replays \\
        --identities ../runs/critic_pre/identities.txt --out ../runs/critic_pre/data
    python -m train.critic_pretrain fit --data ../runs/critic_pre/data --out ../runs/critic_pre/critic.pt

prepare writes <out>/cview.npy (memory-mapped, one critic-view row per sample) and <out>/meta.npz
(everything else, small). Splits are by game (hash) AND by map: a share of map names is held out
entirely, so the test set measures unseen boards as well as unseen games.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
import threading
import queue
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import os  # noqa: E402
os.environ.setdefault("BCSIM_LIB", str(pathlib.Path(__file__).resolve().parents[1] / "bcsim" / "libbcvec_priv_s2.so"))

import bcsim                                  # noqa: E402
from train.critic_data import returns         # noqa: E402

SOURCES = ["replay", "roundrobin", "generated"]
ALPHAS = [0.95, 0.9, 0.8, 2 / 3]


def _h(x: str) -> int:
    return int(hashlib.md5(x.encode()).hexdigest()[:8], 16)


def shard_dirs(roots: list[str]) -> list[pathlib.Path]:
    out = []
    for r in roots:
        r = pathlib.Path(r)
        out += sorted({p.parent for p in r.rglob("samples.npz")})
    return out


# ------------------------------------------------------------------------------------------ prepare
def prepare(a) -> None:
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "meta.npz").exists():
        raise SystemExit(f"{out} already holds a dataset")
    alphas = [float(x) for x in a.alphas.split(",")]
    names = {}
    if a.identities:
        for line in pathlib.Path(a.identities).read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                k, v = line.split("=", 1)
                names[k.strip()] = v.strip()
    ident_of = lambda n: names.get(n, n if n.startswith("team:") else f"agent:{n}")   # noqa: E731
    dirs = shard_dirs(a.shards)
    print(f"{len(dirs)} shard directories", flush=True)

    # pass 1: games, identities, sample counts
    per_dir, n_total, ident_games = [], 0, collections.Counter()
    stride = None
    for k, d in enumerate(dirs):
        games = {json.loads(l)["game"]: json.loads(l) for l in open(d / "games.jsonl")}
        with np.load(d / "samples.npz") as S:
            n = len(S["game"])
            w = int(S["cview_w"])
            st = bcsim.cview_layout(w)["stride"]
        if stride is None:
            stride, cview_w = st, w
        elif st != stride:
            raise SystemExit(f"{d}: crop {w}, the others {cview_w}")
        for g in games.values():
            if g.get("finished"):
                for side in g["sides"]:
                    ident_games[ident_of(side)] += 1
        per_dir.append((d, games, n))
        n_total += n
    kept_idents = sorted(i for i, c in ident_games.items() if c >= a.min_games)
    slots = {i: k + 1 for k, i in enumerate(kept_idents)}               # slot 0 = unknown
    if len(slots) + 1 > a.n_slots:
        raise SystemExit(f"{len(slots)} identities for a table of {a.n_slots}")
    print(f"{n_total:,} samples; {len(ident_games)} identities, {len(slots)} with >= {a.min_games} games "
          f"(the rest share slot 0)", flush=True)

    cv = np.lib.format.open_memmap(out / "cview.npy", mode="w+", dtype=np.uint8, shape=(n_total, stride))
    M = {k: [] for k in ("priv", "ident", "temp", "observed", "target", "game", "map", "source", "round")}
    at = 0
    for k, (d, games, n) in enumerate(per_dir):
        T = {c: np.asarray(v) for c, v in np.load(d / "turns.npz").items()}
        R = returns(T, games, alphas)
        key = T["game"].astype(np.int64) * (1 << 24) + T["step"].astype(np.int64)
        order = np.argsort(key)
        with np.load(d / "samples.npz") as S:
            sg, ss = S["game"].astype(np.int64), S["step"].astype(np.int64)
            pos = order[np.searchsorted(key[order], sg * (1 << 24) + ss)]
            if not np.array_equal(key[pos], sg * (1 << 24) + ss):
                raise SystemExit(f"{d}: a sample without its turn")
            tgt = np.stack([R[x][pos] for x in alphas], 1).astype(np.float32)
            ok = np.isfinite(tgt).all(1)
            ix = np.flatnonzero(ok)
            cv[at:at + len(ix)] = S["cview"][ix]
            team = S["team"][ix].astype(np.int64)
            gm = [games[int(g)] for g in sg[ix]]
            own = np.array([slots.get(ident_of(g["sides"][t]), 0) for g, t in zip(gm, team)], np.int16)
            foe = np.array([slots.get(ident_of(g["sides"][1 - t]), 0) for g, t in zip(gm, team)], np.int16)
            M["priv"].append(S["priv"][ix].astype(np.float32))
            M["ident"].append(np.stack([own, foe], 1))
            M["temp"].append(np.array([[g["temps"][t], g["temps"][1 - t]] for g, t in zip(gm, team)], np.float32))
            M["observed"].append(np.array([g.get("observed", 0) for g in gm], np.uint8))
            M["target"].append(tgt[ix])
            M["game"].append(np.array([_h(f"{d}|{g['game']}") for g in gm], np.int64))
            M["map"].append(np.array([_h(str(g.get("map"))) for g in gm], np.int64))
            M["source"].append(np.array([SOURCES.index(g.get("source", "generated")) for g in gm], np.int8))
            M["round"].append(S["round"][ix].astype(np.int16))
        at += len(ix)
        print(f"  {d}: {len(ix):,}/{n:,} samples (finished games)", flush=True)
    cv.flush()
    del cv
    meta = {k: np.concatenate(v) for k, v in M.items()}
    np.savez(out / "meta.npz", n=np.array(at), stride=np.array(stride), cview_w=np.array(cview_w),
             alphas=np.array(alphas), **meta)
    (out / "identities.json").write_text(json.dumps({"names": names, "slots": slots,
                                                     "games": dict(ident_games)}, indent=1))
    print(f"wrote {at:,} samples ({at * stride / 2**30:.1f} GB of critic view) to {out}", flush=True)


# ------------------------------------------------------------------------------------------ fit
def ev(t: np.ndarray, p: np.ndarray) -> float:
    return float(1 - np.var(t - p) / max(np.var(t), 1e-12))


def fit(a) -> None:
    import torch
    import torch.nn.functional as F
    from train.cview import CViewCritic

    d = pathlib.Path(a.data)
    meta = dict(np.load(d / "meta.npz"))
    n = int(meta["n"])
    cv = np.load(d / "cview.npy", mmap_mode="r")[:n]
    alphas = [float(x) for x in meta["alphas"]]
    idents = json.loads((d / "identities.json").read_text())
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    # splits: whole maps held out (unseen boards), then games
    map_hold = (meta["map"] % 1000) < a.map_holdout * 1000
    gh = (meta["game"] // 7) % 1000
    test = map_hold | (gh < a.test * 1000)
    val = ~test & (gh >= a.test * 1000) & (gh < (a.test + a.val) * 1000)
    train = ~test & ~val
    tr_ix, va_ix, te_ix = (np.flatnonzero(m) for m in (train, val, test))
    print(f"{n:,} samples: {len(tr_ix):,} train, {len(va_ix):,} model-selection, {len(te_ix):,} test "
          f"({int(map_hold.sum()):,} of them on {a.map_holdout:.0%} held-out maps)", flush=True)
    print("by source: " + ", ".join(f"{s} {int((meta['source'] == k).sum()):,}" for k, s in enumerate(SOURCES)),
          flush=True)

    net = CViewCritic(int(meta["cview_w"]), bcsim.PRIV_COUNT, alphas, a.n_slots).to(dev)
    net.set_priv_stats(torch.from_numpy(meta["priv"][tr_ix]).to(dev))
    sd = meta["target"][tr_ix].std(0)
    with torch.no_grad():
        net.ret_scale.copy_(torch.from_numpy(sd).to(dev))
    print("target sd per head: " + ", ".join(f"alpha {x:.3f} {s:.4f}" for x, s in zip(alphas, sd)), flush=True)
    G = {k: torch.from_numpy(meta[k]).to(dev) for k in ("priv", "ident", "temp", "observed", "target")}
    G["ident"] = G["ident"].long()

    def rows(ix: np.ndarray) -> torch.Tensor:
        s = np.sort(ix)
        return torch.from_numpy(np.ascontiguousarray(cv[s])).pin_memory().to(dev, non_blocking=True), s

    def batches(steps: int):
        """Background prefetch of training batches from the memory-mapped critic view."""
        q: queue.Queue = queue.Queue(maxsize=6)

        def work():
            r = np.random.default_rng(a.seed + 1)
            for _ in range(steps):
                q.put(rows(tr_ix[r.integers(len(tr_ix), size=a.batch)]))
            q.put(None)
        threading.Thread(target=work, daemon=True).start()
        while (b := q.get()) is not None:
            yield b

    def predict(ix: np.ndarray) -> np.ndarray:
        out = []
        net.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, len(ix), 8192):
                x, s = rows(ix[i:i + 8192])
                st = torch.from_numpy(s).to(dev)
                out.append((net(x, G["priv"][st], G["ident"][st], G["temp"][st], G["observed"][st]).float()
                            .cpu().numpy(), s))
        net.train()
        p = np.concatenate([o for o, _ in out])
        s = np.concatenate([s for _, s in out])
        return p, s

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    va_sub = va_ix if len(va_ix) <= a.val_rows else rng.choice(va_ix, a.val_rows, replace=False)
    best, best_state, t0 = 1e9, None, time.time()
    for step, (x, s) in enumerate(batches(a.steps)):
        st = torch.from_numpy(s).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = net.raw(x, G["priv"][st], G["ident"][st], G["temp"][st], G["observed"][st]).float()
        loss = F.mse_loss(o, G["target"][st] / net.ret_scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % a.eval_every == 0 or step + 1 == a.steps:
            p, s_ = predict(va_sub)
            y = meta["target"][s_]
            vl = float(np.mean(((p - y) / sd) ** 2))
            evs = [ev(y[:, k], p[:, k]) for k in range(len(alphas))]
            flag = ""
            if vl < best:
                best, flag = vl, " *"
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            print(f"step {step + 1}/{a.steps}: train {loss.item():.4f} val {vl:.4f} | EV " +
                  " ".join(f"{x:.2f}:{e:.3f}" for x, e in zip(alphas, evs)) + f" | {time.time() - t0:.0f}s{flag}",
                  flush=True)
    net.load_state_dict(best_state)

    # the test set, broken down
    p, s_ = predict(te_ix)
    y = meta["target"][s_]
    res = {"alphas": alphas}

    def report(name, m):
        if m.sum() < 200:
            return
        r = {f"ev_{x:.3f}": round(ev(y[m, k], p[m, k]), 4) for k, x in enumerate(alphas)}
        r["n"] = int(m.sum())
        res[name] = r
        print(f"  {name:28s} n {r['n']:>8,} | " + " ".join(f"{x:.2f}: {r[f'ev_{x:.3f}']:+.3f}" for x in alphas))
    print("\nTEST explained variance per horizon (alpha):")
    report("all", np.ones(len(s_), bool))
    report("unseen maps", map_hold[s_])
    report("seen maps, unseen games", ~map_hold[s_])
    for k, src in enumerate(SOURCES):
        report(f"source {src}", meta["source"][s_] == k)
    report("observed (server replays)", meta["observed"][s_] == 1)
    tt = meta["temp"][s_]
    for lo, hi in ((0, 1e-9), (1e-9, 0.15), (0.15, 0.35), (0.35, 9)):
        report(f"own temp [{lo:.2f}, {hi:.2f})", (tt[:, 0] >= lo) & (tt[:, 0] < hi))
    rd = meta["round"][s_]
    for lo, hi in ((0, 50), (50, 150), (150, 300), (300, 501)):
        report(f"rounds {lo}-{hi - 1}", (rd >= lo) & (rd < hi))
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"critic": {"net": net.state_dict(), "spec": net.spec(), "names": idents["names"],
                           "slots": idents["slots"]},
                "results": res, "args": vars(a)}, out)
    (out.with_suffix(".json")).write_text(json.dumps(res, indent=1))
    print(f"wrote {out}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--shards", nargs="+", required=True, help="directories; every samples.npz below each is a shard")
    q.add_argument("--identities", default="")
    q.add_argument("--out", required=True)
    q.add_argument("--alphas", default=",".join(f"{x:.6g}" for x in ALPHAS))
    q.add_argument("--min-games", type=int, default=30, help="games an identity needs for its own slot")
    q.add_argument("--n-slots", type=int, default=256)
    f = sub.add_parser("fit")
    f.add_argument("--data", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--steps", type=int, default=30000)
    f.add_argument("--batch", type=int, default=4096)
    f.add_argument("--lr", type=float, default=1e-3)
    f.add_argument("--wd", type=float, default=0.01)
    f.add_argument("--test", type=float, default=0.10, help="share of games held out for test")
    f.add_argument("--val", type=float, default=0.05, help="share of games for model selection")
    f.add_argument("--map-holdout", type=float, default=0.10, help="share of map names held out entirely (test)")
    f.add_argument("--val-rows", type=int, default=100000)
    f.add_argument("--eval-every", type=int, default=1000)
    f.add_argument("--n-slots", type=int, default=256)
    f.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    prepare(a) if a.cmd == "prepare" else fit(a)


if __name__ == "__main__":
    main()
