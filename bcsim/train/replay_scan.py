"""Finds a team's recent games by walking battle ids backwards.

replay_fetch.py lists a team's battles through the site's `?team=` filter,
which stopped filtering on 2026-09-25 (every team returns the same 31,602
battles, the whole site). What still works: GET /battles/:id returns any
game, ids are sequential, and each answer lists the ids of its whole series
(~9 games). So one request covers a series and the walk jumps to the one
before it -- roughly 9 game ids a request.

Keeps every game the teams played, whatever the submission (the match record
names it, so imitate.py can weight the newest -- a team's newest submission
may be only hours old), and writes the layout replay_dataset.py reads:
<out>/games/<game id>.json (the game's own match record) and .replay.

    python -m train.replay_scan --team 91=../runs/replays/sss_0925 \\
        --team 545=../runs/replays/devtest_0925 --per-team 450 --minutes 100
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time
import urllib.error

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from train.replay_fetch import api, api_key, download_replay  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--team", action="append", required=True, help="TEAM_ID=OUT_DIR, repeatable")
    p.add_argument("--start", type=int, default=0, help="highest id to try (default: probe for the newest)")
    p.add_argument("--span", type=int, default=45000, help="how many ids back to walk at most")
    p.add_argument("--per-team", type=int, default=700)
    p.add_argument("--minutes", type=float, default=70)
    p.add_argument("--workers", type=int, default=3, help="each walks its own slice of the ids")
    a = p.parse_args()
    key = api_key()
    teams = {int(t.split("=", 1)[0]): pathlib.Path(t.split("=", 1)[1]) for t in a.team}
    for d in teams.values():
        (d / "games").mkdir(parents=True, exist_ok=True)
    kept = {t: 0 for t in teams}
    subs: dict[int, dict[int, int]] = {t: {} for t in teams}
    lock = threading.Lock()
    pace = {"next": 0.0}
    t0 = time.time()

    def get(i):
        # one shared pace for all workers: 0.55 s apart keeps us under 120/min
        with lock:
            wait = pace["next"] - time.monotonic()
            pace["next"] = max(pace["next"], time.monotonic()) + 0.55
        if wait > 0:
            time.sleep(wait)
        try:
            return json.load(api(f"battles/{i}", key))
        except (RuntimeError, urllib.error.URLError, OSError, ValueError):
            return None

    def enough():
        return min(kept.values()) >= a.per_team or time.time() - t0 > a.minutes * 60

    hi = a.start
    if not hi:
        hi = 155000
        while get(hi + 300) is not None:
            hi += 300
        hi += 300
    lo = hi - a.span
    print(f"walking {hi} -> {lo} with {a.workers} workers, teams {sorted(teams)}", flush=True)

    def worker(top: int, bottom: int, w: int) -> None:
        i, series = top, 0
        while i > bottom and not enough():
            b = get(i)
            if b is None:
                i -= 1
                continue
            series += 1
            ids = [g["id"] for g in b["games"]] or [i]
            m = b["match"]
            for tid, out in teams.items():
                if tid not in (m["teamAId"], m["teamBId"]):
                    continue
                sub = m["submissionAId"] if m["teamAId"] == tid else m["submissionBId"]
                for g in b["games"]:
                    if g["status"] != "completed" or not g.get("hasReplay", True):
                        continue
                    gid = g["id"]
                    meta, rep = out / "games" / f"{gid}.json", out / "games" / f"{gid}.replay"
                    if rep.exists() and meta.exists():
                        continue
                    gm = get(gid)
                    if gm is None:
                        continue
                    try:
                        if download_replay(gid, key, rep):
                            meta.write_text(json.dumps(gm["match"]))
                            with lock:
                                kept[tid] += 1
                                subs[tid][sub] = subs[tid].get(sub, 0) + 1
                    except (RuntimeError, urllib.error.URLError, OSError) as e:
                        print(f"  game {gid}: {e}", flush=True)
            if series % 100 == 0:
                print(f"  worker {w} at id {i}: kept {kept}  by submission {subs}  "
                      f"{(time.time() - t0) / 60:.0f} min", flush=True)
            i = min(ids) - 1

    step = (hi - lo) // a.workers
    th = [threading.Thread(target=worker, args=(hi - k * step, hi - (k + 1) * step, k)) for k in range(a.workers)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    print(f"done: kept {kept}, by submission {subs}, {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
