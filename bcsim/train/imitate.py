"""Behaviour cloning from replay datasets (see replay_dataset.py).

Trains the same ActorCritic that train.py runs, on another team's recorded
turns: cross-entropy on their action under our mask, plus the value head on
whether they went on to win. Games are split into train and held-out sets by
game id, so the reported accuracy is on games the network never saw.

    python -m train.imitate --data ../runs/replays/vibing/dataset --width 64 --blocks 4

latest.pt is the last epoch, best.pt the one with the lowest held-out NLL.
Both use train.py's layout ("net", "args", "iter", no "opt"), so either can
be loaded as a starting point or as a frozen KL anchor.
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
from train.net import ActorCritic, masked_logits  # noqa: E402


def load(files: list[pathlib.Path], weights: dict[int, float], cap: int = 0):
    """Stacks the trainable rows of these games. A row is dropped only when it
    cannot be learned under our action space: an action the codec cannot
    express, or one our mask forbids while some action was still allowed. A
    dragon with no legal action keeps its row, since masked_logits then
    ignores the mask."""
    parts = {k: [] for k in ("local", "scalar", "mask", "action", "alt", "won", "weight")}
    for f in files:
        d = np.load(f)
        w = weights.get(int(d["submission"]), 0.0)
        if w <= 0:
            continue
        act, mask = d["action"].astype(np.int64), d["mask"]
        ok = act >= 0
        legal = np.zeros(len(act), bool)
        legal[ok] = mask[np.flatnonzero(ok), act[ok]] > 0
        keep = ok & (legal | (mask.sum(1) == 0))
        if not keep.any():
            continue
        for k in ("local", "scalar", "mask", "action", "alt"):
            parts[k].append(d[k][keep])
        n = int(keep.sum())
        parts["won"].append(np.full(n, float(d["won"]), np.float32))
        parts["weight"].append(np.full(n, w, np.float32))
        if cap and sum(len(x) for x in parts["action"]) >= cap:
            break
    if not parts["action"]:
        return None
    return {k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()}


def chunks(files: list[pathlib.Path], per_chunk: int, rng):
    """Games in a fresh random order, a few dozen at a time: the whole set is
    several GB of observations, more than should sit in memory at once."""
    order = rng.permutation(len(files))
    for s in range(0, len(order), per_chunk):
        yield [files[i] for i in order[s:s + per_chunk]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--hidden", type=int, default=512, help="width of the fuse layers")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--vf", type=float, default=0.25, help="weight on the value loss")
    p.add_argument("--holdout", type=float, default=0.1, help="share of games held out")
    p.add_argument("--games-per-chunk", type=int, default=40)
    p.add_argument("--val-cap", type=int, default=200_000, help="held-out samples used")
    p.add_argument("--exclude-maps", default="",
                   help="comma-separated server map ids whose games are left out entirely, so a "
                        "map can be held out end to end (9 = Schooltime)")
    p.add_argument("--old-weight", type=float, default=0.25,
                   help="weight of games played by the team's older submissions; "
                        "the newest submission weighs 1")
    p.add_argument("--submission-weights", default="",
                   help="explicit weights per submission id, e.g. '615:1,242:0.25,98:0.05'; "
                        "overrides --old-weight, and every submission in the data must be listed")
    p.add_argument("--gpu-frac", type=float, default=0.0,
                   help="cap this process's share of GPU memory (next to a training run)")
    a = p.parse_args()
    dev = torch.device("cuda")
    if a.gpu_frac:
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    data = pathlib.Path(a.data)
    out = pathlib.Path(a.out or data.parent / "imitate")
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(data.glob("*.npz"), key=lambda f: int(f.stem))
    if a.exclude_maps:
        drop = {int(x) for x in a.exclude_maps.split(",")}
        map_of = {r["game"]: r.get("map") for r in
                  map(json.loads, (data / "index.jsonl").read_text().splitlines())}
        before = len(files)
        files = [f for f in files if map_of.get(int(f.stem)) not in drop]
        print(f"--exclude-maps {sorted(drop)}: {before - len(files)} of {before} games left out", flush=True)
    rng = np.random.default_rng(0)
    # the team's newest submission is the target; older versions still teach,
    # at a lower weight
    sub_of = {f: int(np.load(f)["submission"]) for f in files}
    newest = max(sub_of.values())
    if a.submission_weights:
        weights = {int(k): float(v) for k, v in
                   (kv.split(":") for kv in a.submission_weights.split(","))}
        missing = set(sub_of.values()) - set(weights)
        if missing:
            raise SystemExit(f"no weight given for submissions {sorted(missing)}")
    else:
        weights = {s: (1.0 if s == newest else a.old_weight) for s in set(sub_of.values())}
    counts = {s: sum(v == s for v in sub_of.values()) for s in sorted(set(sub_of.values()))}
    print(f"games per submission: {counts}; weights: {weights}", flush=True)
    # hold out the same share of every submission's games, so a version with
    # few games still has some held out
    held = set()
    for sub in counts:
        idx = [i for i, f in enumerate(files) if sub_of[f] == sub]
        k = max(1, round(len(idx) * a.holdout)) if len(idx) > 1 else 0
        held |= set(rng.choice(idx, k, replace=False).tolist()) if k else set()
    tr_files = [f for i, f in enumerate(files) if i not in held]
    # held-out accuracy is measured on the most heavily weighted submissions
    top = max(weights.values())
    va = load([f for i, f in enumerate(files) if i in held and weights[sub_of[f]] == top],
              {s: 1.0 for s, w in weights.items() if w == top}, a.val_cap)
    # the index knows each game's sample count, which sizes the LR schedule
    index = {}
    idx_path = data / "index.jsonl"
    if idx_path.exists():
        for line in idx_path.read_text().splitlines():
            r = json.loads(line)
            index[r["game"]] = r.get("samples", 0) - r.get("unrepresentable", 0) - max(0, r.get("mask_violations", 0))
    n_train = sum(index.get(int(f.stem), 6000) for f in tr_files if weights[sub_of[f]] > 0)
    print(f"{len(files)} games: ~{n_train:,} train samples in {len(tr_files)} games, "
          f"{len(va['action']) if va else 0:,} held out from {len(held)} games", flush=True)

    # The scalar width comes from the DATA, never from the env. A clone is
    # trained on what clone_cache.py stored -- the 14 base scalars -- while the
    # env's row has since grown to 708 and then 713 as mem, memfar and the sonar
    # echoes were appended. Taking bcsim.N_SCALARS here built a 713-wide first
    # Linear and fed it 14 columns, which is a shape error on the first batch and
    # is why this had stopped running at all.
    #
    # The result is therefore a 14-scalar network, exactly as before, and
    # train/migrate_scalars.py widens it afterwards by padding that Linear with
    # zero columns -- verified to play identically, which is the whole reason the
    # frozen league survived the row growing.
    n_sc = int(va["scalar"].shape[1]) if va is not None else int(
        np.load(tr_files[0])["scalar"].shape[1])
    if n_sc != bcsim.N_SCALARS:
        print(f"cloning on {n_sc} scalars (the cache's width); the env now writes "
              f"{bcsim.N_SCALARS}, so widen with train.migrate_scalars before this "
              f"net meets the env", flush=True)
    net = ActorCritic(bcsim.N_CHANNELS, n_sc, bcsim.N_ACTIONS,
                      width=a.width, blocks=a.blocks, hidden=a.hidden).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (n_train // a.batch + len(tr_files) // a.games_per_chunk + 2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)

    def batch_loss(d, idx):
        local = d["local"][idx].to(dev, non_blocking=True).float()
        scalar = d["scalar"][idx].to(dev, non_blocking=True)
        mask = d["mask"][idx].to(dev, non_blocking=True).bool()
        act = d["action"][idx].to(dev, non_blocking=True).long()
        alt = d["alt"][idx].to(dev, non_blocking=True).long()
        won = d["won"][idx].to(dev, non_blocking=True)
        w = d["weight"][idx].to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, v = net(local, scalar)
        logp = F.log_softmax(masked_logits(logits.float(), mask), dim=1)
        lp = logp.gather(1, act[:, None]).squeeze(1)
        # a split that means the same thing under two ids counts as either
        has_alt = alt >= 0
        lp_alt = logp.gather(1, alt.clamp(min=0)[:, None]).squeeze(1)
        lp = torch.where(has_alt, torch.logaddexp(lp, lp_alt), lp)
        pred = logp.argmax(1)
        hit = (pred == act) | (has_alt & (pred == alt))
        # value in [-1, 1]: win 1, loss -1, draw 0
        vloss = (w * (v.float() - (2 * won - 1)) ** 2).sum() / w.sum()
        return -(w * lp).sum() / w.sum(), vloss, hit.float().mean()

    t0 = time.perf_counter()
    log = (out / "log.jsonl").open("a")
    it = 0
    best = float("inf")
    for ep in range(a.epochs):
        net.train()
        tot = np.zeros(3)
        nb = 0
        for group in chunks(tr_files, a.games_per_chunk, rng):
            tr = load(group, weights)
            if tr is None:
                continue
            perm = torch.randperm(len(tr["action"]))
            for s in range(0, len(perm), a.batch):
                pl, vl, acc = batch_loss(tr, perm[s:s + a.batch])
                loss = pl + a.vf * vl
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                if it < steps - 1:
                    sched.step()
                tot += [pl.item(), vl.item(), acc.item()]
                nb += 1
                it += 1
            del tr
        row = {"epoch": ep, "train_nll": tot[0] / nb, "train_v": tot[1] / nb, "train_acc": tot[2] / nb}
        if va is not None:
            net.eval()
            vt, vn = np.zeros(3), 0
            with torch.no_grad():
                for s in range(0, len(va["action"]), a.batch):
                    idx = torch.arange(s, min(s + a.batch, len(va["action"])))
                    pl, vl, acc = batch_loss(va, idx)
                    vt += np.array([pl.item(), vl.item(), acc.item()]) * len(idx)
                    vn += len(idx)
            row.update(val_nll=vt[0] / vn, val_v=vt[1] / vn, val_acc=vt[2] / vn)
        row["elapsed"] = round(time.perf_counter() - t0, 1)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}"
                         for k, v in row.items()), flush=True)
        ck = {"net": net.state_dict(), "iter": it, "total_turns": 0,
              "args": {**vars(a), "width": a.width, "blocks": a.blocks}, "imitation": row}
        torch.save(ck, out / "latest.pt")
        # held-out NLL bottoms out before the last epoch once it starts to overfit
        if row.get("val_nll", float("inf")) < best:
            best = row["val_nll"]
            torch.save(ck, out / "best.pt")


if __name__ == "__main__":
    main()
