"""Gathers the night's numbers into one JSON for the write-up.

    python -m train.collect_results --runs ../runs > ../runs/evals/summary.json

Reads the imitate2 logs, the round-robin passes (rr*.jsonl), the anchor evals
(*_anchors.jsonl) and the label indexes, so the report quotes one source.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import contextlib, io                                # noqa: E402

from train.round_robin import table                  # noqa: E402


def last_epoch(log: pathlib.Path) -> dict:
    rows = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    best = min(rows, key=lambda r: r.get("val_nll", 9)) if rows else {}
    return {"epochs": len(rows), "best": {k: round(v, 4) for k, v in best.items()
                                          if k in ("epoch", "val_nll", "val_acc", "val_acc_novel",
                                                   "train_acc")}}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="../runs")
    a = p.parse_args()
    runs = pathlib.Path(a.runs)
    out: dict = {"training": {}, "round_robin": {}, "anchors": {}, "labels": {}}

    for d in sorted((runs / "i2").glob("*")):
        if (d / "log.jsonl").exists():
            out["training"][d.name] = last_epoch(d / "log.jsonl")

    names = [c.split("=")[0] for c in (runs / "rr_args.txt").read_text().split()]
    done = [json.loads(l) for f in ("rr.jsonl", "rr2.jsonl")
            for l in (runs / "evals" / f).read_text().splitlines()
            if (runs / "evals" / f).exists()]
    with contextlib.redirect_stdout(io.StringIO()):   # table() prints; we want the JSON only
        m, games = table(names, done)
    out["round_robin"] = {
        "names": names,
        "matrix": [[None if np.isnan(x) else round(float(x), 4) for x in row] for row in m],
        "games": games.tolist(),
        "mean": {n: round(float(np.nansum(m[i] * games[i]) / games[i].sum()), 4)
                 if games[i].sum() else None for i, n in enumerate(names)},
        "se": {n: round(float(np.sqrt(0.25 / games[i].sum())), 4) if games[i].sum() else None
               for i, n in enumerate(names)}}

    for f in sorted((runs / "evals").glob("*_anchors.jsonl")):
        rows = [json.loads(l) for l in f.read_text().splitlines()]
        out["anchors"][f.stem.replace("_anchors", "")] = {
            k: v["score"] for k, v in rows[-1]["summary"].items()}
    v10 = runs / "evals" / "v10.jsonl"
    if v10.exists():
        rows = [json.loads(l) for l in v10.read_text().splitlines()]
        out["anchors"]["v10"] = {k: v["score"] for k, v in rows[-1]["summary"].items()}
    for name, s in out["anchors"].items():
        s["mean"] = round(float(np.mean(list(s.values()))), 4)

    for kind in ("sprint", "selftrap"):
        idx = runs / f"replays/dev_test_1_p/{kind}/index.jsonl"
        if idx.exists():
            rows = [json.loads(l) for l in idx.read_text().splitlines()]
            keys = [k for k in rows[0] if k not in ("game", "error")]
            out["labels"][kind] = {k: sum(r.get(k, 0) for r in rows) for k in keys}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
