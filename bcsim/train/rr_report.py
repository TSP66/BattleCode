"""Round-robin table, split by map bucket.

round_robin.py stores evaluate()'s per-map `cells`, so one run on all 12 maps
gives the live-rotation score and the unseen-map score without replaying
anything. The buckets matter because they are not the same question:

  live    the 7 maps the ladder is played on now -> this is the ranking that counts
  retired arena, Colloseum: in the rotation until 2026-09-22 12:50, so older
          clones' teachers did play them
  never   help, small, queen_of_spades_but_she_ages: never played on the server,
          so no replay-trained clone has ever seen them. The honest test of
          whether a clone generalises beyond its teacher's map diet.

    python -m train.rr_report ../runs/evals/rr_20260923.jsonl
"""

import json
import pathlib
import sys
from collections import defaultdict

LIVE = {"big_empty", "default", "default_small", "devil",
        "queen_of_spades", "schooltime", "trophy"}
RETIRED = {"arena", "Colloseum"}
NEVER = {"help", "small", "queen_of_spades_but_she_ages"}
BUCKETS = [("live 7", LIVE), ("retired", RETIRED), ("never played", NEVER)]


def main() -> None:
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                        else "../runs/evals/rr_20260923.jsonl")
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    if not rows:
        print("no passes recorded yet")
        return

    # score[a][b] = a's score against b, and the mirror
    score = defaultdict(dict)
    bucket = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))   # name -> bucket -> [wins, n]
    per_map = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # name -> map -> [wins, n]
    for r in rows:
        me = r["learner"]
        for opp, s in r["summary"].items():
            score[me][opp] = s["score"]
            score[opp][me] = round(1 - s["score"], 4)
        for opp, by_map in r["cells"].items():
            for m, c in by_map.items():
                w, n = c["win"] + 0.5 * c["draw"], c["n"]
                for who, wins in ((me, w), (opp, n - w)):
                    per_map[who][m][0] += wins
                    per_map[who][m][1] += n
                    for bname, bset in BUCKETS:
                        if m in bset:
                            bucket[who][bname][0] += wins
                            bucket[who][bname][1] += n

    names = sorted(score, key=lambda n: -sum(score[n].values()) / max(len(score[n]), 1))
    w = max(len(n) for n in names) + 2

    print("Score matrix (row's score against column, draw = half a win)\n")
    print(" " * w + "".join(f"{n:>12s}" for n in names) + f"{'mean':>10s}")
    for a in names:
        cells = "".join("           -" if a == b else f"{score[a].get(b, float('nan')):>12.3f}"
                        for b in names)
        vals = [v for k, v in score[a].items()]
        print(f"{a:<{w}s}{cells}{sum(vals) / max(len(vals), 1):>10.3f}")

    print("\n\nBy map bucket (score across every game played on those maps)\n")
    print(f"{'':<{w}s}" + "".join(f"{b:>16s}" for b, _ in BUCKETS))
    for a in names:
        out = ""
        for bname, _ in BUCKETS:
            wins, n = bucket[a][bname]
            out += f"{(wins / n if n else float('nan')):>10.3f}{f'({n})':>6s}"
        print(f"{a:<{w}s}{out}")

    print("\n\nPer map\n")
    maps = sorted({m for a in per_map for m in per_map[a]},
                  key=lambda m: (m not in LIVE, m not in RETIRED, m))
    print(f"{'':<{w}s}" + "".join(f"{m[:11]:>12s}" for m in maps))
    for a in names:
        row = ""
        for m in maps:
            wins, n = per_map[a][m]
            row += f"{(wins / n if n else float('nan')):>12.3f}"
        print(f"{a:<{w}s}{row}")
    print("\nlive: " + " ".join(sorted(LIVE)))
    print("retired: " + " ".join(sorted(RETIRED)) + "   |   never played: " + " ".join(sorted(NEVER)))


if __name__ == "__main__":
    main()
