"""Downloads the server's current map rotation.

`GET /api/v1/maps` returns every active public map with its full text, so the
rotation can be re-pulled whenever the organisers change it (they swapped
Arena and Colloseum out for Devil on 2026-09-22).

    python -m train.maps_fetch                  # writes ../maps-live, reports the diff
    python -m train.maps_fetch --check          # report only, write nothing

Private maps are not listed and their replays carry a placeholder map
("INTERNAL_PRIVATE_TESTING_MAP"), so they cannot be mirrored.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import urllib.request

SITE = "https://game.battlecode.au"
API = f"{SITE}/api/v1"
ROOT = pathlib.Path(__file__).resolve().parents[2]


def api_key() -> str:
    d = json.loads((pathlib.Path.home() / ".unswbc/keys.json").read_text())
    return d.get(SITE) or next(iter(d.values()))


def fetch() -> list[dict]:
    req = urllib.request.Request(f"{API}/maps",
                                 headers={"Authorization": f"Bearer {api_key()}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "maps-live"))
    p.add_argument("--check", action="store_true", help="report, write nothing")
    a = p.parse_args()

    maps = fetch()
    out = pathlib.Path(a.out)
    have = {f.name for f in out.glob("*.map")} if out.exists() else set()
    want = {}
    for m in maps:
        text = m["mapText"]
        want[slug(m["name"]) + ".map"] = text if text.endswith("\n") else text + "\n"

    for fn in sorted(want):
        old = (out / fn).read_text() if fn in have else None
        state = "new" if old is None else ("unchanged" if old == want[fn] else "CHANGED")
        w, h = want[fn].split("\n", 1)[0].split()[1:3]
        print(f"{fn:<22} {w:>2}x{h:<2} {state}")
    for fn in sorted(have - set(want)):
        print(f"{fn:<22} {'':<6} DROPPED from the rotation")

    if a.check:
        return
    out.mkdir(parents=True, exist_ok=True)
    for fn, text in want.items():
        (out / fn).write_text(text)
    for fn in have - set(want):
        (out / fn).unlink()
    print(f"\n{len(want)} maps in {out}")


if __name__ == "__main__":
    main()
