"""Downloads every game a team has played on game.battlecode.au.

Other teams' battles are readable: the site's battle list filters by team
name, `GET /battles/:id` returns any battle with its games, and each game has
its own replay. Everything is cached, so a rerun only fetches what is new.

    python -m train.replay_fetch --team "Vibing++" --out ../runs/replays/vibing

Writes <out>/series.jsonl (one battle per line, as the API returns it) and
<out>/games/<game id>.replay (gzipped Cap'n Proto, as the server serves it)
and <out>/games/<game id>.json (that game's own match record: sides,
submissions, map, winner).
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

SITE = "https://game.battlecode.au"
API = f"{SITE}/api/v1"
# the API allows 120 requests a minute per key; stay under it
API_GAP = 0.6
_last_api = 0.0


def api_key() -> str:
    d = json.loads((pathlib.Path.home() / ".unswbc/keys.json").read_text())
    return d.get(SITE) or next(iter(d.values()))


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_no_redirect = urllib.request.build_opener(NoRedirect)


def _open(req: urllib.request.Request, opener=None):
    """urlopen with retries on 429 (honouring Retry-After), 5xx and dropped connections."""
    for attempt in range(6):
        try:
            return (opener.open if opener else urllib.request.urlopen)(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                return e                      # the caller follows it by hand
            if e.code == 429 or e.code >= 500:
                wait = float(e.headers.get("Retry-After") or 5 * (attempt + 1))
                print(f"  {e.code} on {req.full_url}, waiting {wait:.0f}s", flush=True)
                time.sleep(wait)
                continue
            raise
        except (urllib.error.URLError, OSError) as e:   # resets, TLS timeouts
            print(f"  {e} on {req.full_url}, retrying", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gave up on {req.full_url}")


def api(path: str, key: str, opener=None):
    global _last_api
    gap = API_GAP - (time.monotonic() - _last_api)
    if gap > 0:
        time.sleep(gap)
    _last_api = time.monotonic()
    req = urllib.request.Request(f"{API}/{path}", headers={
        "Authorization": f"Bearer {key}", "User-Agent": "bc-replay-fetch"})
    return _open(req, opener)


def devalue(arr: list):
    """SvelteKit's __data.json packs values as indices into one flat array."""
    memo: dict[int, object] = {}

    def h(i):
        if i == -1:
            return None
        if i in memo:
            return memo[i]
        v = arr[i]
        if isinstance(v, dict):
            out: dict = {}
            memo[i] = out
            out.update({k: h(x) for k, x in v.items()})
            return out
        if isinstance(v, list):
            if v and isinstance(v[0], str) and v[0] in ("Date", "BigInt", "RegExp"):
                return v[1]
            lst: list = []
            memo[i] = lst
            lst.extend(h(x) for x in v)
            return lst
        return v
    return h(0)


def list_series(team: str) -> list[dict]:
    """Every battle (series) the team is in, from the site's filtered list."""
    out, page = [], 1
    while True:
        q = urllib.parse.urlencode({"team": team, "page": page})
        req = urllib.request.Request(f"{SITE}/battles/__data.json?{q}",
                                     headers={"User-Agent": "bc-replay-fetch"})
        data = json.load(_open(req))
        d = devalue(data["nodes"][1]["data"])
        out += d["battles"]
        if page * d["perPage"] >= d["total"] or not d["battles"]:
            return out
        page += 1
        time.sleep(0.5)


def download_replay(game_id: int, key: str, dest: pathlib.Path) -> bool:
    reply = api(f"battles/{game_id}/replay", key, opener=_no_redirect)
    loc = reply.headers.get("Location")
    if not loc:
        return False
    # the signed link must be fetched without our key
    req = urllib.request.Request(loc, headers={"User-Agent": "bc-replay-fetch"})
    blob = _open(req).read()
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(blob)
    tmp.rename(dest)
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--team", required=True, help="team name exactly as the site shows it")
    p.add_argument("--out", required=True)
    p.add_argument("--latest-only", action="store_true",
                   help="only download games the team played with its newest submission")
    a = p.parse_args()
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

    fresh = 0
    with path.open("a") as f:
        for s in series:
            sid = s["id"]
            if sid in known or s["outcome"] == "live":
                continue
            b = json.load(api(f"battles/{sid}", key))
            if any(g["status"] not in ("completed", "failed", "errored") for g in b["games"]):
                continue                     # still playing; pick it up next time
            known[sid] = b
            f.write(json.dumps(b) + "\n")
            fresh += 1
    print(f"{len(known)} battles on file ({fresh} new)", flush=True)

    def team_sub(b: dict) -> int:
        m = b["match"]
        return m["submissionAId"] if b["teamAName"] == a.team else m["submissionBId"]

    keep = dict(known)
    if a.latest_only:
        newest = max(team_sub(b) for b in known.values())
        keep = {sid: b for sid, b in known.items() if team_sub(b) == newest}
        print(f"newest submission {newest}: {len(keep)} of {len(known)} battles", flush=True)
    todo = [(sid, g["id"]) for sid, b in sorted(keep.items()) for g in b["games"]
            if g["status"] == "completed" and not (out / "games" / f"{g['id']}.replay").exists()]
    print(f"{len(todo)} replays to download (~{len(todo) * 2 * API_GAP / 60:.0f} min)", flush=True)
    for i, (sid, gid) in enumerate(todo):
        meta = out / "games" / f"{gid}.json"
        try:
            if not meta.exists():
                meta.write_text(json.dumps(json.load(api(f"battles/{gid}", key))["match"]))
            ok = download_replay(gid, key, out / "games" / f"{gid}.replay")
        except (RuntimeError, urllib.error.URLError, OSError) as e:
            print(f"  game {gid}: skipped ({e}); a rerun retries it", flush=True)
            continue
        if not ok:
            print(f"  game {gid}: no replay", flush=True)
        if i % 50 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
