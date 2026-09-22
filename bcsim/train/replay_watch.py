"""Keeps the replay datasets growing: every --every minutes, fetches new games
for the tracked teams and converts them (replay_fetch + replay_dataset).

Tracked: every team named in --teams, plus any team that enters the ladder's
top --top (its newest submission only). Each team gets runs/replays/<slug>/.

    python -m train.replay_watch --teams "Vibing++:all,SSS:latest" --top 3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from train.replay_fetch import API, api_key   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
BCSIM = ROOT / "bcsim"


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return {"vibing": "vibing"}.get(s, s) or "team"


def ladder(key: str) -> list[dict]:
    req = urllib.request.Request(f"{API}/ratings", headers={
        "Authorization": f"Bearer {key}", "User-Agent": "bc-replay-watch"})
    d = json.load(urllib.request.urlopen(req, timeout=60))
    return d if isinstance(d, list) else next(v for v in d.values() if isinstance(v, list))


def run(cmd: list[str], log) -> int:
    log.write(f"$ {' '.join(cmd)}\n")
    log.flush()
    return subprocess.call(["nice", "-n", "10", *cmd], cwd=BCSIM, stdout=log, stderr=log)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teams", default="Vibing++:all,SSS:latest",
                   help="name:all|latest pairs, comma separated")
    p.add_argument("--top", type=int, default=3, help="also track any team in this top N")
    p.add_argument("--every", type=float, default=30.0, help="minutes between passes")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--log", default=str(ROOT / "runs/replay_watch.log"))
    a = p.parse_args()
    key = api_key()
    tracked: dict[str, str] = {}
    for kv in filter(None, a.teams.split(",")):
        name, mode = kv.rsplit(":", 1)
        tracked[name] = mode
    # Vibing++ already lives in runs/replays/vibing
    dirs = {"Vibing++": "vibing"}
    py = sys.executable
    with open(a.log, "a") as log:
        while True:
            t0 = time.time()
            stamp = time.strftime("%Y-%m-%d %H:%M")
            try:
                top = ladder(key)[:a.top]
                ids = {r["name"]: r["id"] for r in ladder(key)}
                for r in top:
                    if r["name"] not in tracked:
                        tracked[r["name"]] = "latest"
                        log.write(f"[{stamp}] new top-{a.top} team: {r['name']} (#{r['rank']}, "
                                  f"{r['elo']}), recording its newest submission\n")
                log.write(f"[{stamp}] top {a.top}: "
                          + ", ".join(f"{r['name']} {r['elo']}" for r in top) + "\n")
            except Exception as e:                  # a bad ladder read skips a pass
                log.write(f"[{stamp}] ladder read failed: {e!r}\n")
                ids = {}
            log.flush()
            for name, mode in tracked.items():
                if name not in ids:
                    log.write(f"[{stamp}] {name}: not on the ladder read, skipped\n")
                    continue
                out = ROOT / "runs/replays" / dirs.get(name, slug(name))
                cmd = [py, "-u", "-m", "train.replay_fetch", "--team", name, "--out", str(out)]
                if mode == "latest":
                    cmd.append("--latest-only")
                run(cmd, log)
                run([py, "-m", "train.replay_dataset", "--games", str(out),
                     "--team-id", str(ids[name]), "--workers", str(a.workers)], log)
                idx = out / "dataset" / "index.jsonl"
                if idx.exists():
                    rows = [json.loads(x) for x in idx.read_text().splitlines()]
                    log.write(f"[{time.strftime('%H:%M')}] {name}: {len(rows)} games, "
                              f"{sum(r.get('samples', 0) for r in rows):,} samples, "
                              f"{sum(1 for r in rows if r.get('diverged') or r.get('error'))} bad\n")
                    log.flush()
            time.sleep(max(60.0, a.every * 60 - (time.time() - t0)))


if __name__ == "__main__":
    main()
