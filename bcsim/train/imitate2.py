"""Behaviour cloning from a clone_cache.py cache, with optional extra features.

Same losses as imitate.py (masked cross-entropy on the team's action, either
id of a split that has two, plus the value head), but the data is read from
flat memmaps, and clone_features.py can add input planes and scalars that the
simulator's observation does not have. Every extra feature is computed from
what the dragon itself has seen or done on earlier turns, so a deployed bot
can compute it too.

    python -m train.imitate2 --cache ../runs/clone_cache/devtest_1302 \
        --features hist3,mem --train-games 400 --epochs 4 --out ../runs/i2_hist_mem

Checkpoints keep train.py's layout, plus "features" (the names used) and the
channel / scalar counts, so the net can be rebuilt from the file alone.
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

from train.clone_features import load_features       # noqa: E402
from train.net import ActorCritic, masked_logits      # noqa: E402

LC_PEARL_TIME, LC_SELF_INDEX = 1, 21


class Data:
    """The rows of some games, in RAM, with their extra features."""

    def __init__(self, cache: pathlib.Path, games: set[int], features: list[str], cap: int = 0,
                 relabel: str = "", relabel_weight: float = 1.0, weights: str = ""):
        game = np.load(cache / "game.npy", mmap_mode="r")
        keep = np.load(cache / "keep.npy", mmap_mode="r")
        sel = np.isin(game, np.array(sorted(games)))
        lab = None
        if relabel:
            # a relabelled row is kept even when the team's own move was one
            # our codec cannot express: the new target is a codec move
            lab = np.load(cache / f"{relabel}.npy", mmap_mode="r")
            rows = np.flatnonzero(sel & (np.asarray(keep) | (np.asarray(lab) >= 0)))
        else:
            rows = np.flatnonzero(sel & keep)
        if cap and len(rows) > cap:
            rows = rows[:cap]
        self.rows = rows
        get = lambda k: np.load(cache / f"{k}.npy", mmap_mode="r")[rows]
        self.local = get("local")
        self.scalar = get("scalar")
        self.mask = get("mask")
        self.action = get("action").astype(np.int64)
        self.alt = get("alt").astype(np.int64)
        self.won = get("won")
        self.weight = (np.load(cache / f"{weights}.npy", mmap_mode="r")[rows].astype(np.float32)
                       if weights else np.ones(len(rows), np.float32))
        if lab is not None:
            lab = np.asarray(lab[rows]).astype(np.int64)
            hit = lab >= 0
            self.action[hit] = lab[hit]
            self.alt[hit] = -1
            self.weight[hit] = relabel_weight
        self.xloc, self.xsc = load_features(cache, features, rows)
        self.n = len(rows)

    def batch(self, idx, dev):
        loc = torch.from_numpy(self.local[idx]).to(dev).float()
        # undo the uint8 packing (see clone_cache.py)
        scale = torch.ones(loc.shape[1], device=dev)
        scale[LC_PEARL_TIME] = 1 / 99
        scale[LC_SELF_INDEX] = 1 / 255
        loc = loc * scale[None, :, None, None]
        sc = torch.from_numpy(self.scalar[idx]).to(dev)
        extra = lambda parts: [torch.from_numpy(arr[idx]).to(dev).float() * k for arr, k in parts]
        if self.xloc:
            loc = torch.cat([loc] + extra(self.xloc), 1)
        if self.xsc:
            sc = torch.cat([sc] + extra(self.xsc), 1)
        t = lambda k: torch.from_numpy(getattr(self, k)[idx]).to(dev)
        return loc, sc, t("mask").bool(), t("action"), t("alt"), t("won"), t("weight")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--features", default="", help="comma list, see clone_features.FEATURES")
    p.add_argument("--train-games", type=int, default=0, help="0 = every training game")
    p.add_argument("--val-cap", type=int, default=300_000)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--vf", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default="", help="start from this checkpoint (same features)")
    p.add_argument("--chunk-games", type=int, default=0,
                   help="load the training games this many at a time (0 = all at once); "
                        "for data bigger than RAM")
    p.add_argument("--relabel", default="",
                   help="cache array (e.g. sprint_label) of codec ids replacing the team's "
                        "action where >= 0; the held-out rows are scored on it too")
    p.add_argument("--relabel-weight", type=float, default=1.0, help="loss weight of relabelled rows")
    p.add_argument("--pct-start", type=float, default=0.05)
    p.add_argument("--weights", default="",
                   help="cache array of per-row loss weights (e.g. tossup_weight); "
                        "held-out rows are scored unweighted")
    a = p.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else
                       "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    cache, out = pathlib.Path(a.cache), pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    feats = [f for f in a.features.split(",") if f]

    meta = json.loads((cache / "meta.json").read_text())
    val_games = set(meta["val_games"])
    tr_games = [g for g in meta["games"] if g not in val_games]
    if a.train_games:
        tr_games = list(np.random.default_rng(a.seed).choice(tr_games, a.train_games, replace=False))
    t0 = time.perf_counter()
    tr_set = set(int(g) for g in tr_games)
    rl = dict(relabel=a.relabel, relabel_weight=a.relabel_weight, weights=a.weights)
    if a.chunk_games:
        g_all = np.load(cache / "game.npy", mmap_mode="r")
        n_train = int((np.isin(g_all, np.array(sorted(tr_set))) & np.load(cache / "keep.npy")).sum())
        tr = Data(cache, set(tr_games[:1]), feats, **rl)        # shapes only
    else:
        tr = Data(cache, tr_set, feats, **rl)
        n_train = tr.n
    va = Data(cache, val_games, feats, a.val_cap, relabel=a.relabel)
    np.save(out / "val_rows.npy", va.rows)
    # deterministic teams replay whole games move for move, so a held-out row
    # can be an exact copy of a training row; "novel" rows are the ones whose
    # observation never occurs in any training game (see clone_cache.py)
    novel = None
    if (cache / "obs_hash.npy").exists():
        h = np.load(cache / "obs_hash.npy")
        g = np.load(cache / "game.npy")
        novel = ~np.isin(h[va.rows], h[~np.isin(g, np.array(sorted(val_games)))])
        print(f"{novel.mean():.1%} of held-out rows are novel", flush=True)
    n_ch = tr.local.shape[1] + sum(arr.shape[1] for arr, _ in tr.xloc)
    n_sc = tr.scalar.shape[1] + sum(arr.shape[1] for arr, _ in tr.xsc)
    print(f"features {feats}: {n_ch} channels, {n_sc} scalars; {n_train:,} train rows "
          f"({len(tr_games)} games), {va.n:,} held out; loaded in {time.perf_counter() - t0:.0f}s",
          flush=True)

    net = ActorCritic(n_ch, n_sc, 48, width=a.width, blocks=a.blocks, hidden=a.hidden).to(dev)
    if a.init:
        net.load_state_dict(torch.load(a.init, map_location="cpu", weights_only=False)["net"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * (n_train // a.batch + (len(tr_games) // a.chunk_games + 1 if a.chunk_games else 0))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=a.pct_start)

    def losses(d, idx):
        loc, sc, mask, act, alt, won, w = d.batch(idx, dev)
        logits, v = net(loc, sc)
        logp = F.log_softmax(masked_logits(logits.float(), mask), dim=1)
        lp = logp.gather(1, act[:, None]).squeeze(1)
        has_alt = alt >= 0
        lp_alt = logp.gather(1, alt.clamp(min=0)[:, None]).squeeze(1)
        lp = torch.where(has_alt, torch.logaddexp(lp, lp_alt), lp)
        pred = logp.argmax(1)
        hit = (pred == act) | (has_alt & (pred == alt))
        return -(w * lp).sum() / w.sum(), ((v - (2 * won - 1)) ** 2).mean(), hit.float(), pred

    def evaluate():
        net.eval()
        tot, hits, preds = np.zeros(2), [], []
        with torch.no_grad():
            for s in range(0, va.n, 4096):
                idx = np.arange(s, min(s + 4096, va.n))
                pl, vl, hit, pred = losses(va, idx)
                tot += np.array([pl.item(), vl.item()]) * len(idx)
                hits.append(hit.cpu().numpy())
                preds.append(pred.cpu().numpy())
        net.train()
        return tot / va.n, np.concatenate(hits), np.concatenate(preds)

    log = (out / "log.jsonl").open("a")
    rng = np.random.default_rng(a.seed)
    it, best = 0, float("inf")
    def groups():
        if not a.chunk_games:
            yield tr
            return
        order = [tr_games[i] for i in rng.permutation(len(tr_games))]
        for c in range(0, len(order), a.chunk_games):
            yield Data(cache, set(int(g) for g in order[c:c + a.chunk_games]), feats, **rl)

    for ep in range(a.epochs):
        tot, nb = np.zeros(3), 0
        for d in groups():
            perm = rng.permutation(d.n)
            for s in range(0, d.n - a.batch + 1, a.batch):
                idx = np.sort(perm[s:s + a.batch])
                pl, vl, hit, _ = losses(d, idx)
                loss = pl + a.vf * vl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                if it < steps - 1:
                    sched.step()
                it += 1
                if it % 50 == 0:
                    tot += [pl.item(), vl.item(), hit.mean().item()]
                    nb += 1
            del d
        (vnll, vv), hits, preds = evaluate()
        row = {"epoch": ep, "train_nll": tot[0] / nb, "train_acc": tot[2] / nb,
               "val_nll": vnll, "val_v": vv, "val_acc": float(hits.mean()),
               **({"val_acc_novel": float(hits[novel].mean())} if novel is not None else {}),
               "elapsed": round(time.perf_counter() - t0, 1)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                         for k, v in row.items()), flush=True)
        ck = {"net": net.state_dict(), "iter": it, "total_turns": 0,
              "args": {**vars(a), "width": a.width, "blocks": a.blocks},
              "features": feats, "n_channels": n_ch, "n_scalars": n_sc, "imitation": row}
        torch.save(ck, out / "latest.pt")
        if vnll < best:
            best = vnll
            torch.save(ck, out / "best.pt")
            np.save(out / "val_pred.npy", preds.astype(np.int16))


if __name__ == "__main__":
    main()
