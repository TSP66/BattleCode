"""Downloads games between the top teams of the leaderboard, with their pearl seeds.

Every game's match JSON is saved next to its replay. Since unswbc 1.1.0 (25 Sep) it
carries `seed`, and the engine's pearls are std::mt19937_64(seed) drawn as
rng() % (max - min + 1) + min over the map's TILE ranges -- verified 2026-09-28 on
five server games (Schooltime, 500 rounds: 1934 of 1934 round-start spawns). The
replay's embedded map has every TILE zeroed, so re-simulating also needs our copy of
the map (maps-live / maps-all).

The site's `?team=<name>` filter is broken (it returns every battle); `?teams=<id>`
works.

    python -m train.top_fetch --out ../runs/replays/top_0928 --top 17 --exclude 470,801 --max-games 10000

--exclude drops every game a listed team played in (2026-09-28: the user excluded 龙虎豹 470 and
中国必须人能飞 801, widely suspected on the Discord of hard-coding). --max-games keeps the newest.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from train.replay_fetch import SITE, _open, api, api_key, devalue, download_replay  # noqa: E402


def list_series(team_id: int, since: str) -> list[dict]:
    """The team's battles, newest first, down to `since` (YYYY-MM-DD)."""
    out, page = [], 1
    while True:
        q = urllib.parse.urlencode({"teams": team_id, "page": page})
        req = urllib.request.Request(f"{SITE}/battles/__data.json?{q}",
                                     headers={"User-Agent": "bc-replay-fetch"})
        d = devalue(json.load(_open(req))["nodes"][1]["data"])
        rows = d["battles"]
        out += [b for b in rows if b["at"][:10] >= since]
        if not rows or rows[-1]["at"][:10] < since or page * d["perPage"] >= d["total"]:
            return out
        page += 1
        time.sleep(0.3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--since", default="2026-09-25")
    p.add_argument("--exclude", default="", help="team ids, comma separated: skip their games")
    p.add_argument("--max-games", type=int, default=0, help="newest series first, up to this many games")
    p.add_argument("--workers", type=int, default=8,
                   help="downloads in flight; each API call takes ~3 s, the shared limiter keeps "
                        "all of them together under the API's 120 a minute")
    p.add_argument("--reuse-series", action="store_true",
                   help="skip the listing and use the series already in series.jsonl")
    p.add_argument("--teams", default="", help="team ids, comma separated: fetch exactly these teams "
                   "instead of the ladder's --top (names come from the ladder read)")
    p.add_argument("--any-opponent", action="store_true",
                   help="keep a top team's games against anyone, not only other top teams")
    a = p.parse_args()
    out = pathlib.Path(a.out)
    (out / "games").mkdir(parents=True, exist_ok=True)
    key = api_key()

    board = json.load(api("leaderboard", key))
    board = board if isinstance(board, list) else board.get("leaderboard", board)
    excluded = {int(x) for x in a.exclude.split(",") if x}
    top = {t["id"]: t["name"] for t in board[: a.top] if t["id"] not in excluded}
    if a.teams:
        want = [int(x) for x in a.teams.split(",") if x]
        names = {t["id"]: t["name"] for t in board}
        top = {i: names.get(i, str(i)) for i in want}
    (out / "top.json").write_text(json.dumps(board[: a.top], ensure_ascii=False, indent=1))
    print("top:", ", ".join(f"{n} ({i})" for i, n in top.items()), flush=True)

    series: dict[int, dict] = {}
    if a.reuse_series:
        for line in (out / "series.jsonl").read_text().splitlines():
            b = json.loads(line)
            series[b["match"]["id"]] = b
        print(f"reusing {len(series)} series from series.jsonl", flush=True)
    for tid, name in ({} if a.reuse_series else top).items():
        rows = list_series(tid, a.since)
        keep = [b for b in rows if b["outcome"] in ("a", "b", "draw")
                and (a.any_opponent or (b["a"]["id"] in top and b["b"]["id"] in top))]
        for b in keep:
            series[b["id"]] = b
        print(f"  {name}: {len(rows)} battles since {a.since}, {len(keep)} kept", flush=True)
    print(f"{len(series)} series", flush=True)

    path = out / "series.jsonl"
    known = {}
    if path.exists():
        for line in path.read_text().splitlines():
            b = json.loads(line)
            known[b["match"]["id"]] = b
    with path.open("a") as f:
        n_games = sum(len(known[s_]["games"]) for s_ in series if s_ in known)
        for i, sid in enumerate(sorted(series, reverse=True)):
            if a.max_games and n_games >= a.max_games:
                break
            if sid in known:
                continue
            try:
                b = json.load(api(f"battles/{sid}", key))
            except (RuntimeError, urllib.error.URLError, OSError) as e:
                print(f"  series {sid}: skipped ({e})", flush=True)
                continue
            known[sid] = b
            n_games += sum(g["status"] == "completed" for g in b["games"])
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"  series {i}/{len(series)}", flush=True)

    todo = [g["id"] for sid in sorted(series, reverse=True) if sid in known
            for g in known[sid]["games"] if g["status"] == "completed" and g.get("hasReplay", True)]
    if a.max_games:
        todo = todo[: a.max_games]
    todo = [g for g in todo if not (out / "games" / f"{g}.replay").exists()]
    print(f"{len(todo)} replays to download, {a.workers} at a time", flush=True)

    def one(gid):
        meta = out / "games" / f"{gid}.json"
        try:
            if not meta.exists():
                m = json.load(api(f"battles/{gid}", key))
                tmp = meta.with_suffix(".jsontmp")
                tmp.write_text(json.dumps({**m["match"], "mapName": m.get("mapName"),
                                           "teamAName": m.get("teamAName"),
                                           "teamBName": m.get("teamBName")}, ensure_ascii=False))
                tmp.rename(meta)                  # never a half-written meta
            if not download_replay(gid, key, out / "games" / f"{gid}.replay"):
                return f"  game {gid}: no replay"
        except (RuntimeError, urllib.error.URLError, OSError, ValueError) as e:
            return f"  game {gid}: skipped ({e}); a rerun retries it"
        return None

    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as pool:
        for i, msg in enumerate(pool.map(one, todo), 1):
            if msg:
                print(msg, flush=True)
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} {i / max(time.time() - t0, 1) * 60:.0f} a minute", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
