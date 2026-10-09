"""Differential stress: many random maps x policies x seeds against the engine.

Every game compares each dragon's init and round blocks byte for byte, every
death with its reason and round, and the final result. Prints the aggregate
rule coverage so a green run cannot hide an untested path.

    python stress.py [games] [--bundled DIR] [--jobs N] [--seed S]
"""

from __future__ import annotations

import concurrent.futures
import pathlib
import random
import sys
import time

import diffsim
import genmaps
import policies

POLICIES = ["smart_policy", "survivor_policy", "portal_policy", "splitter_policy",
            "random_policy", "queen_sprint_policy"]


def one(args) -> tuple[str, dict | str]:
    idx, map_text, pol_name, seed, label = args
    rng = random.Random(seed)
    factory = getattr(policies, pol_name, None) or getattr(diffsim, pol_name)
    try:
        return label, diffsim.compare(map_text, factory(rng), seed=rng.getrandbits(64), label=label)
    except diffsim.Mismatch as exc:
        return label, f"MISMATCH {exc}"
    except ValueError as exc:  # map the engine rewrote: not a fair comparison
        return label, f"skip: {exc}"


def build_jobs(games: int, bundled: pathlib.Path | None, seed: int = 20260921):
    jobs = []
    rng = random.Random(seed)
    pool = []
    if bundled:
        pool = [(p.name, p.read_text()) for p in sorted(bundled.glob("*.map"))]
    for i in range(games):
        pol = POLICIES[i % len(POLICIES)]
        if pool and i % 3 == 0:
            name, text = pool[(i // 3) % len(pool)]
            jobs.append((i, text, pol, rng.randrange(1 << 30), f"{name}/{pol}/{i}"))
        else:
            # deliberately awkward: tiny and huge boards, dense kelp, many portals
            extreme = i % 7 == 0
            text = genmaps.generate(
                rng,
                w=rng.choice([10, 64]) if extreme else None,
                h=rng.choice([10, 64]) if extreme else None,
                kelp_rate=rng.uniform(0.15, 0.35) if extreme else None,
                portal_pairs=rng.randint(4, 10) if extreme else None,
                teams_dragons=rng.choice([1, 2, 3]),
                dragon_len=rng.choice([2, 3, 5]),
            )
            jobs.append((i, text, pol, rng.randrange(1 << 30), f"gen{i}/{pol}"))
    return jobs


def main() -> int:
    games = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 200
    bundled = None
    if "--bundled" in sys.argv:
        bundled = pathlib.Path(sys.argv[sys.argv.index("--bundled") + 1])
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 8
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 20260921

    work = build_jobs(games, bundled, seed)
    totals: dict[str, int] = {}
    bad: list[str] = []
    skipped = 0
    turns = 0
    decided: dict[str, int] = {}
    t0 = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
        for n, (label, res) in enumerate(pool.map(one, work), 1):
            if isinstance(res, str):
                if res.startswith("skip"):
                    skipped += 1
                else:
                    bad.append(f"{label}: {res}")
            else:
                turns += res["turns"]
                decided[res["decided_by"]] = decided.get(res["decided_by"], 0) + 1
                for k, v in res["stats"].items():
                    totals[k] = totals.get(k, 0) + v
            if n % 25 == 0 or n == len(work):
                print(f"  {n}/{len(work)} games, {turns:,} turns, {len(bad)} mismatches",
                      flush=True)

    dt = time.perf_counter() - t0
    print(f"\n{len(work) - len(bad) - skipped} games matched the engine exactly "
          f"({turns:,} dragon turns, {skipped} skipped, {dt:.0f}s)")
    print("decided by: " + ", ".join(f"{k} {v}" for k, v in sorted(decided.items())))
    print("\nrule coverage across the run:")
    for k in sorted(totals):
        print(f"  {k:16s} {totals[k]:>12,}")
    missing = [k for k, v in totals.items() if v == 0]
    if missing:
        print(f"\nNEVER EXERCISED: {', '.join(missing)}")
    if bad:
        print(f"\n{len(bad)} MISMATCHES")
        for b in bad[:5]:
            print(b[:2000])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
