"""Packs one team's replay dataset (replay_dataset.py) into flat memmaps.

imitate.py re-reads compressed per-game .npz files every epoch, which is the
slow part of cloning on a small machine. This writes each field once, as one
array over all kept samples in game order and turn order, so the extra
features in clone_features.py (which need a dragon's earlier turns) and the
trainer in imitate2.py can index it directly.

    python -m train.clone_cache --data ../runs/replays/dev_test_1_p/dataset \
        --submission 1302 --out ../runs/clone_cache/devtest_1302

The held-out games are exactly the ones imitate.py holds out with the same
data and --holdout, so accuracies are comparable with its logs. `local` is
stored as uint8: every channel is 0/1 except pearl_time (cd/99, stored as cd,
exact) and self_index (i/(len-1), stored x255, within 0.002).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

LC_PEARL_TIME, LC_SELF_INDEX = 1, 21


def held_out(files, sub_of, holdout: float) -> set[int]:
    """imitate.py's split: the same rng, the same draws, in the same order."""
    rng = np.random.default_rng(0)
    counts = {s: sum(v == s for v in sub_of.values()) for s in sorted(set(sub_of.values()))}
    held = set()
    for sub in counts:
        idx = [i for i, f in enumerate(files) if sub_of[f] == sub]
        k = max(1, round(len(idx) * holdout)) if len(idx) > 1 else 0
        held |= set(rng.choice(idx, k, replace=False).tolist()) if k else set()
    return {int(files[i].stem) for i in held}


def write_hash(out: pathlib.Path) -> None:
    """obs_hash.npy: a 64-bit hash of each row's window and scalars. A held-out
    row whose hash occurs in a training game is an exact repeat (deterministic
    teams replay whole games), so accuracy is also reported on the rest."""
    import hashlib
    loc = np.load(out / "local.npy", mmap_mode="r")
    sc = np.load(out / "scalar.npy", mmap_mode="r")
    h = np.zeros(len(loc), np.uint64)
    for s in range(0, len(loc), 200_000):
        lb = np.asarray(loc[s:s + 200_000]).reshape(-1, 23 * 49)
        sb = np.asarray(sc[s:s + 200_000]).view(np.uint8).reshape(len(lb), -1)
        for i in range(len(lb)):
            h[s + i] = int.from_bytes(hashlib.blake2b(lb[i].tobytes() + sb[i].tobytes(),
                                                      digest_size=8).digest(), "little")
    np.save(out / "obs_hash.npy", h)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--submission", required=True,
                   help="one submission id, or several with loss weights: "
                        "\"2050:1,1302:0.5\". Several write sub_weight.npy, which "
                        "imitate2.py reads with --weights sub_weight")
    p.add_argument("--holdout", type=float, default=0.1)
    p.add_argument("--out", required=True)
    p.add_argument("--hash", action="store_true",
                   help="only (re)write obs_hash.npy for an existing cache")
    a = p.parse_args()
    data, out = pathlib.Path(a.data), pathlib.Path(a.out)
    if a.hash:
        write_hash(out)
        return
    out.mkdir(parents=True, exist_ok=True)

    # "2050:1,1302:0.5" keeps both submissions and weights their rows; a bare
    # "2050" is the old single-submission cache, with every row at weight 1
    subs = {}
    for part in str(a.submission).split(","):
        sid, _, w = part.partition(":")
        subs[int(sid)] = float(w) if w else 1.0

    index = {r["game"]: r for r in map(json.loads, (data / "index.jsonl").read_text().splitlines())}
    files = sorted(data.glob("*.npz"), key=lambda f: int(f.stem))
    sub_of = {f: index[int(f.stem)]["submission"] for f in files}
    held = held_out(files, sub_of, a.holdout)
    games = [f for f in files if sub_of[f] in subs]
    n = sum(index[int(f.stem)]["samples"] for f in games)
    for sid, w in subs.items():
        g = [f for f in games if sub_of[f] == sid]
        print(f"  submission {sid} at weight {w}: {len(g)} games, "
              f"{sum(index[int(f.stem)]['samples'] for f in g):,} samples", flush=True)
    print(f"{len(games)} games, {n:,} samples before filtering", flush=True)

    spec = {"local": (np.uint8, (23, 7, 7)), "scalar": (np.float32, (14,)),
            # uint64: sonar payloads are 64 bits wide. Four slots is still
            # plenty here -- msg_rows only decodes the first two, and this cache
            # exists to imitate scraped bots, whose scheme fits in the low 32.
            "msgs": (np.uint64, (4,)), "mask": (np.uint8, (48,)),
            "action": (np.int16, ()), "alt": (np.int16, ()), "dragon": (np.int32, ()),
            "round": (np.int16, ()), "game": (np.int32, ()), "won": (np.float32, ()),
            "keep": (np.bool_, ()), "sub_weight": (np.float32, ())}
    mm = {k: np.lib.format.open_memmap(out / f"{k}.npy", "w+", dt, (n,) + sh)
          for k, (dt, sh) in spec.items()}
    at = 0
    for gi, f in enumerate(games):
        d = np.load(f)
        m = len(d["action"])
        s = slice(at, at + m)
        loc = d["local"].astype(np.float32)
        loc[:, LC_PEARL_TIME] *= 99
        loc[:, LC_SELF_INDEX] *= 255
        mm["local"][s] = np.rint(loc).astype(np.uint8)
        for k in ("scalar", "msgs", "mask", "action", "alt", "dragon", "round"):
            mm[k][s] = d[k]
        mm["game"][s] = int(f.stem)
        mm["won"][s] = float(d["won"])
        mm["sub_weight"][s] = subs[sub_of[f]]
        # imitate.py's rule: drop what our action space cannot learn, keep a
        # dragon with no legal action at all
        act, mask = d["action"].astype(np.int64), d["mask"]
        ok = act >= 0
        legal = np.zeros(m, bool)
        legal[ok] = mask[np.flatnonzero(ok), act[ok]] > 0
        mm["keep"][s] = ok & (legal | (mask.sum(1) == 0))
        at += m
        if gi % 100 == 0:
            print(f"{gi}/{len(games)}", flush=True)
    assert at == n
    for v in mm.values():
        v.flush()
    val_games = sorted(int(f.stem) for f in games if int(f.stem) in held)
    (out / "meta.json").write_text(json.dumps(
        {"submission": max(subs, key=subs.get), "submissions": {str(k): v for k, v in subs.items()},
         "samples": n, "games": [int(f.stem) for f in games], "val_games": val_games}))
    write_hash(out)
    print(f"done: {n:,} samples, {len(val_games)} held-out games", flush=True)


if __name__ == "__main__":
    main()
