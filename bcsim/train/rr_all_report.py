"""Markdown tables from a roundrobin_all.py result, with the held-out map split out.

    python -m train.rr_all_report ../runs/night_0925/rr.json --holdout schooltime

Prints: the ranking (Bradley-Terry and mean score with its standard error), the
score matrix, and per agent its mean score on the held-out map(s) against its
mean on every other map -- a drop there is where it overfits. Note that
roundrobin_all.py's per-map cells are keyed "row|col" for the pair's first
agent, so both orientations are folded here.
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("result")
    p.add_argument("--holdout", default="schooltime")
    a = p.parse_args()
    d = json.load(open(a.result))
    names = [x["name"] for x in d["agents"]]
    archs = [x["arch"] for x in d["agents"]]
    W, G = np.array(d["wins"]), np.array(d["games"])
    S = np.array(d["strength"])
    held = set(a.holdout.split(","))
    order = np.argsort(-S)

    print(f"{int(G.sum() / 2)} games, {d['games_per_cell']} per (pair, map, side), maps {d['maps']}, "
          f"sonar {'on' if d.get('sonar') else 'off'}, {d['seconds'] / 60:.0f} min\n")
    print("| rank | agent | arch | Bradley-Terry | mean score | games |")
    print("|---|---|---|---|---|---|")
    for r, i in enumerate(order, 1):
        n = G[i].sum()
        m = W[i].sum() / max(n, 1)
        se = math.sqrt(max(m * (1 - m), 1e-9) / max(n, 1))
        print(f"| {r} | {names[i]} | {archs[i]} | {S[i]:+.3f} | {m:.3f} ± {se:.3f} | {int(n)} |")

    print("\nScore matrix (row's score against column; draws count half)\n")
    print("| | " + " | ".join(names[j] for j in order) + " |")
    print("|---" * (len(order) + 1) + "|")
    for i in order:
        cells = ["–" if i == j or G[i, j] == 0 else f"{W[i, j] / G[i, j]:.2f}" for j in order]
        print(f"| **{names[i]}** | " + " | ".join(cells) + " |")

    # per agent, held-out map vs the rest
    by = {n: {"held": [0.0, 0], "rest": [0.0, 0]} for n in names}
    for key, maps in d["per_map"].items():
        x, y = key.split("|")
        for mname, (sc, g) in maps.items():
            k = "held" if mname in held else "rest"
            by[x][k][0] += sc
            by[x][k][1] += g
            by[y][k][0] += g - sc
            by[y][k][1] += g
    print(f"\nHeld-out map ({', '.join(sorted(held))}) against the other maps\n")
    print("| agent | held-out score | other maps | difference |")
    print("|---|---|---|---|")
    for i in order:
        h, r = by[names[i]]["held"], by[names[i]]["rest"]
        hs, rs = h[0] / max(h[1], 1), r[0] / max(r[1], 1)
        se = math.sqrt(max(hs * (1 - hs), 1e-9) / max(h[1], 1))
        print(f"| {names[i]} | {hs:.3f} ({h[1]} games) | {rs:.3f} | {hs - rs:+.3f} ± {se:.3f} |")


if __name__ == "__main__":
    main()
