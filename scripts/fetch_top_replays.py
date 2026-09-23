"""Scrape a ladder team's replays, newest submission first, up to a cap.

`train/replay_fetch.py` downloads *every* game a team has played, in ascending
game id -- so the oldest submissions arrive first, and a full history is hours
of requests. For a new league anchor we want the team's *current* bot, and
DISTILL_DEVTEST.md's clone was trained on ~1200 games of one submission, so
this wrapper:

  * builds/extends `<out>/series.jsonl` in exactly replay_fetch's format,
  * orders the games by submission (newest first), then newest battle first,
  * stops after `--max-games` replays exist on disk,
  * runs `--workers` requests concurrently behind one shared rate limiter
    (`--rate` requests/minute, the documented ceiling is 120; /leaderboard and
    /ratings are the 30/min ones and are not used here),
  * writes `<out>/team.json` (team name + id + games per submission) so
    `replay_dataset --team-id` never has to be guessed later.

The file layout, the retry/backoff behaviour and the replay download are
replay_fetch's, imported rather than copied, so a later plain
`python -m train.replay_fetch --team ... --out ...` extends the same folder
incrementally.

Per-game metadata is fetched per game on purpose: `mapId` varies *within* a
battle (78% of dev-test games disagree with their series record), so it cannot
be synthesised from the battle listing.

    python scripts/fetch_top_replays.py --team "forgot to mention" \
        --team-id 264 --out runs/replays/forgot_to_mention --max-games 1400
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pathlib
import sys
import threading
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bcsim"))

from train.replay_fetch import (  # noqa: E402
    API, _no_redirect, _open, api_key, list_series,
)

_lock = threading.Lock()
_next = 0.0
_gap = 0.6


def api(path: str, key: str, opener=None):
    """replay_fetch.api, but with a lock so concurrent workers share the limit."""
    global _next
    with _lock:
        now = time.monotonic()
        start = max(now, _next)
        _next = start + _gap
    if start > now:
        time.sleep(start - now)
    req = urllib.request.Request(f"{API}/{path}", headers={
        "Authorization": f"Bearer {key}", "User-Agent": "bc-replay-fetch"})
    return _open(req, opener)


def download_replay(gid: int, key: str, dest: pathlib.Path) -> bool:
    reply = api(f"battles/{gid}/replay", key, opener=_no_redirect)
    loc = reply.headers.get("Location")
    if not loc:
        return False
    # the signed R2 link must be fetched without our key, and is not metered
    blob = _open(urllib.request.Request(loc, headers={"User-Agent": "bc-replay-fetch"})).read()
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(blob)
    tmp.rename(dest)
    return True


def team_sub(b: dict, team: str) -> int:
    m = b["match"]
    return m["submissionAId"] if b["teamAName"] == team else m["submissionBId"]


def main() -> int:
    global _gap
    p = argparse.ArgumentParser()
    p.add_argument("--team", required=True, help="team name exactly as the site shows it")
    p.add_argument("--team-id", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-games", type=int, default=1400,
                   help="stop once this many replays are on disk (0 = all)")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--rate", type=float, default=100.0, help="API requests per minute")
    p.add_argument("--list-only", action="store_true")
    a = p.parse_args()
    _gap = 60.0 / a.rate

    out = pathlib.Path(a.out)
    (out / "games").mkdir(parents=True, exist_ok=True)
    key = api_key()

    series = list_series(a.team)
    print(f"{a.team}: {len(series)} battles listed", flush=True)

    known: dict[int, dict] = {}
    path = out / "series.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            b = json.loads(line)
            known[b["match"]["id"]] = b
    print(f"{len(known)} battles already on file", flush=True)

    todo_s = [s["id"] for s in series if s["id"] not in known and s["outcome"] != "live"]
    write_lock = threading.Lock()
    with path.open("a") as f:
        def one_series(sid):
            try:
                b = json.load(api(f"battles/{sid}", key))
            except (RuntimeError, urllib.error.URLError, OSError) as e:
                print(f"  battle {sid}: skipped ({e})", flush=True)
                return
            if any(g["status"] not in ("completed", "failed", "errored")
                   for g in b["games"]):
                return                         # still playing; next run gets it
            with write_lock:
                known[sid] = b
                f.write(json.dumps(b) + "\n")
                f.flush()
        if todo_s:
            print(f"{len(todo_s)} battle records to fetch "
                  f"(~{len(todo_s) * _gap / 60:.0f} min)", flush=True)
            with cf.ThreadPoolExecutor(a.workers) as ex:
                list(ex.map(one_series, todo_s))
    print(f"{len(known)} battles on file", flush=True)

    subs: dict[int, int] = {}
    for b in known.values():
        s = team_sub(b, a.team)
        subs[s] = subs.get(s, 0) + sum(1 for g in b["games"] if g["status"] == "completed")
    print("games by submission (newest first): "
          + ", ".join(f"{s}:{n}" for s, n in sorted(subs.items(), reverse=True)[:8]),
          flush=True)

    (out / "team.json").write_text(json.dumps({
        "team": a.team, "team_id": a.team_id,
        "battles": len(known), "games_by_submission": subs,
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, indent=1, ensure_ascii=False) + "\n")

    # newest submission first, and inside a submission the newest battle first
    order = sorted(known.items(), key=lambda kv: (-team_sub(kv[1], a.team), -kv[0]))
    have = {int(g.stem) for g in (out / "games").glob("*.replay")}
    todo: list[int] = []
    budget = (a.max_games or 10 ** 9) - len(have)
    for _sid, b in order:
        if len(todo) >= budget:
            break
        for g in b["games"]:
            if g["status"] != "completed" or g["id"] in have:
                continue
            if len(todo) >= budget:
                break
            todo.append(g["id"])
    print(f"{len(have)} replays on disk, {len(todo)} to download "
          f"(~{len(todo) * 2 * _gap / 60:.0f} min)", flush=True)
    if a.list_only:
        return 0

    done = [0]

    def one_game(gid: int):
        meta = out / "games" / f"{gid}.json"
        try:
            if not meta.exists():
                meta.write_text(json.dumps(json.load(api(f"battles/{gid}", key))["match"]))
            if not download_replay(gid, key, out / "games" / f"{gid}.replay"):
                print(f"  game {gid}: no replay", flush=True)
        except (RuntimeError, urllib.error.URLError, OSError, KeyError) as e:
            print(f"  game {gid}: skipped ({e}); a rerun retries it", flush=True)
        with write_lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  {done[0]}/{len(todo)}", flush=True)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one_game, todo))
    print(f"done: {len(list((out / 'games').glob('*.replay')))} replays in {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
