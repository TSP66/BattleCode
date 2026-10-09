"""The fixed panel: one yardstick for every checkpoint of every run (user, 2026-10-01: before
a week-long PPO run, a measurement precise enough to show whether it keeps improving).

Each checkpoint plays every panel member on every gate map, greedy (as it would be submitted;
--temp T samples it at temperature T instead, to measure the greedy-vs-sampled gap),
--games per (member, map) cell, half on each side -- ratchet.py's gate with the panel as its
league and no anchor. The simulator's grid is the largest any player needs; a player trained on
a smaller one reads a crop of it (distill_lstm.LSTMGreedy.grid_model), so 14x14 and 15x15
policies meet in one game. Rows go to --out (one per checkpoint) with the panel mean and its
standard error. The panel file is name=path lines; change it and the rows stop being comparable,
so a changed panel should get its own --out.

    python -m train.panel --ckpt ../runs/ratchet_ff3/anchors/gen0.pt ../runs/ratchet_ff3/anchors/gen1.pt
    python -m train.panel --show
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bcsim"))
PANEL = ROOT / "runs/panel/panel_v1.txt"
OUT = ROOT / "runs/panel/panel_v1.jsonl"
LIBS = {14: "libbcvec_s2.so", 15: "libbcvec_s2_g15.so", 17: "libbcvec_s2_g17.so"}


def members(path: pathlib.Path) -> list[tuple[str, str]]:
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            n, p = line.split("=", 1)
            out.append((n.strip(), p.strip()))
    return out


def show(out: pathlib.Path) -> None:
    rows = [json.loads(x) for x in out.read_text().splitlines()] if out.exists() else []
    if not rows:
        print("no rows yet")
        return
    names = list(rows[-1]["summary"])
    print(f"{'checkpoint':44s} {'panel':>13s} " + " ".join(f"{n[:11]:>11s}" for n in names))
    for r in rows:
        cells = " ".join(f"{r['summary'][n]['score']:11.3f}" if n in r["summary"] else f"{'-':>11s}" for n in names)
        print(f"{r['label'][:44]:44s} {r['panel']:.3f}±{r['panel_se']:.3f} {cells}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="*", default=[])
    p.add_argument("--label", nargs="*", default=[], help="one per --ckpt (default: run/file)")
    p.add_argument("--panel", default=str(PANEL))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--maps", default=str(ROOT / "maps-gate"))
    p.add_argument("--games", type=int, default=10, help="per (member, map) cell")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--temp", type=float, default=0.0,
                   help="the checkpoint samples softmax(logits / temp); 0 = greedy, as submitted. The panel "
                        "members stay greedy. Labels get a @T suffix")
    p.add_argument("--show", action="store_true")
    a = p.parse_args()
    out = pathlib.Path(a.out)
    if a.show:
        show(out)
        return
    import torch
    panel = members(pathlib.Path(a.panel))
    grids = [int(torch.load(x, map_location="cpu", weights_only=False)["args"].get("grid", 14))
             for x in a.ckpt + [p_ for _, p_ in panel]]
    G = max(grids)
    os.environ["BCSIM_LIB"] = str(ROOT / "bcsim/bcsim" / LIBS[G])        # before bcsim is imported
    from train.ratchet import gate
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, ck in enumerate(a.ckpt):
        label = a.label[i] if i < len(a.label) else f"{pathlib.Path(ck).parent.parent.name}/{pathlib.Path(ck).parent.name}/{pathlib.Path(ck).stem}"
        if a.temp > 0:
            label += f"@T{a.temp:g}"
        tmp = out.with_suffix(f".{os.getpid()}.json")
        t0 = time.time()
        gate(argparse.Namespace(cand=ck, anchor="", league=",".join(f"{n}={p_}" for n, p_ in panel), out=str(tmp),
                                maps=a.maps, exclude_maps="", games=a.games, anchor_reps=0, threads=16,
                                skip_done=False, fast=True, seed=a.seed, max_seconds=3600, memchan=False, s2=False,
                                cand_temp=a.temp))
        row = json.loads(tmp.read_text())
        tmp.unlink()
        sc = [row["summary"][n]["score"] for n, _ in panel]
        var = [s_ * (1 - s_) / max(row["summary"][n]["n"], 1) for s_, (n, _) in zip(sc, panel)]
        ck_turns = torch.load(ck, map_location="cpu", weights_only=False).get("total_turns")
        rec = {"label": label, "ckpt": ck, "total_turns": ck_turns, "time": time.time(), "grid": G, "temp": a.temp,
               "panel": round(sum(sc) / len(sc), 4), "panel_se": round(math.sqrt(sum(var)) / len(sc), 4),
               "games": sum(row["summary"][n]["n"] for n, _ in panel), "seconds": round(time.time() - t0),
               "summary": row["summary"], "cells": row["cells"], "panel_file": a.panel}
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{label}: panel {rec['panel']:.3f} ± {rec['panel_se']:.3f} over {rec['games']} games "
              f"({rec['seconds']} s)", flush=True)
    show(out)


if __name__ == "__main__":
    main()
