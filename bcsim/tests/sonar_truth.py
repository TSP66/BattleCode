"""Derives the engine's sonar targeting rule from its own replay, and checks it.

`replay.py` gives per-ray ground truth (origin, end, hitId, hitKind). This file
reconstructs the board at the moment of each ray from the same replay, predicts
what the ray should do under a candidate rule, and reports agreement. Nothing
here depends on our simulator being right, so a disagreement is unambiguous.

The rule this found, which is NOT the straight line our simulator cast:

    A ray whose first step enters the sender's OWN BODY is dragged along the
    body to the tail, and continues from the tail in the direction of the last
    body link -- not in the direction it was cast.

So a dragon curled into an L can cast west and have the ray leave southward.
That single rule accounts for the wrong-dragon hits on an empty torus, the
messages a dragon receives from itself, and the rays that stop short on kelp
maps. The old straight-line model got the exit tile right only when the body
happened to be straight behind the head.

The requested direction is recovered from the payload, because the event records
the direction the ray *left* in, which is the thing under test.

    python bcsim/tests/sonar_truth.py [map ...]
"""

from __future__ import annotations

import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import replay as R                               # noqa: E402
from oracle import OracleGame                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIRS = "NESW"
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}

KELP, PORTAL, OPEN = 1, 2, 0


# ------------------------------------------------------------------- the map

class Map:
    """Just enough of the map for a ray: kelp edges and portals.

    Edge indexing follows the simulator's loader (`bc_text.hpp`): rows alternate
    horizontal (a tile row's north side) and vertical (its west side), each row
    is width + 1 long, the duplicated east column and the final south row are
    dropped.
    """

    def __init__(self, text: str):
        self.w = self.h = 0
        portals: list[tuple[bool, int, int, int]] = []
        self.h_kind: dict[tuple[int, int], int] = {}
        self.v_kind: dict[tuple[int, int], int] = {}
        self.h_target: dict[tuple[int, int], tuple[bool, int, int]] = {}
        self.v_target: dict[tuple[int, int], tuple[bool, int, int]] = {}
        self.spawns: list[tuple[int, list[tuple[int, int]]]] = []
        for line in text.split("\n"):
            f = line.split()
            if not f:
                continue
            if f[0] == "MAP":
                self.w, self.h = int(f[1]), int(f[2])
            elif f[0] == "EDGE":
                index, kind = int(f[1]), int(f[2])
                pid = int(f[3]) if len(f) > 3 else -1
                stride = self.w + 1
                row, rem = divmod(index, stride)
                if rem == self.w or row > 2 * self.h:
                    continue
                vertical = bool(row & 1)
                if not vertical and row == 2 * self.h:
                    continue
                x = rem
                y = (row - 1) // 2 if vertical else row // 2
                k = KELP if kind == 1 else (PORTAL if kind == 2 else OPEN)
                (self.v_kind if vertical else self.h_kind)[(x, y)] = k
                if k == PORTAL:
                    portals.append((vertical, x, y, pid))
            elif f[0] == "DRAGON":
                team, count = int(f[1]), int(f[2])
                body = [(int(f[3 + 2 * i]), int(f[4 + 2 * i])) for i in range(count)]
                self.spawns.append((team, body))
        for i, (vi, xi, yi, pi) in enumerate(portals):
            for j, (vj, xj, yj, pj) in enumerate(portals):
                if i != j and pi == pj:
                    tgt = (self.v_target if vi else self.h_target)
                    tgt[(xi, yi)] = (vj, xj, yj)

    def wrap(self, x: int, y: int) -> tuple[int, int]:
        return x % self.w, y % self.h

    def edge_on_side(self, x: int, y: int, d: str):
        x, y = self.wrap(x, y)
        if d == "N":
            return False, x, y
        if d == "S":
            return False, x, (y + 1) % self.h
        if d == "W":
            return True, x, y
        return True, (x + 1) % self.w, y

    def step(self, x: int, y: int, d: str):
        """Where a step lands, or None when kelp blocks it. Mirrors TileAfterStep."""
        vertical, ex, ey = self.edge_on_side(x, y, d)
        kinds = self.v_kind if vertical else self.h_kind
        kind = kinds.get((ex, ey), OPEN)
        if kind == KELP:
            return None
        if kind == PORTAL:
            tv, tx, ty = (self.v_target if vertical else self.h_target)[(ex, ey)]
            if not tv:
                ty -= 1 if d != "S" else 0
            else:
                tx -= 1 if d != "E" else 0
            return self.wrap(tx, ty)
        dx = (d == "E") - (d == "W")
        dy = (d == "S") - (d == "N")
        return self.wrap(x + dx, y + dy)

    def direction_between(self, a: tuple[int, int], b: tuple[int, int]) -> str | None:
        for d in DIRS:
            if self.step(a[0], a[1], d) == b:
                return d
        return None


# ----------------------------------------------------------------- the rule

def predict(mp: Map, bodies: dict[int, list[tuple[int, int]]],
            teams: dict[int, int], sender: int, req: str, limit: int | None = None):
    """(exit direction, origin, end, hit id or None, echo kind) for one ray."""
    body = bodies[sender]
    head = body[0]
    own = {t: i for i, t in enumerate(body)}
    if limit is None:
        limit = mp.w + mp.h

    # The first step decides whether the ray is dragged along the body.
    nxt = mp.step(head[0], head[1], req)
    if nxt is None:
        return req, head, head, None, "kelp"

    direction, origin = req, head
    if len(body) >= 2 and nxt == body[1]:
        # Only the segment immediately behind the head is transparent, and
        # entering it drags the ray the whole length of the body: it re-emerges
        # from the TAIL, heading along the last body link rather than the way it
        # was cast. Entering any deeper segment is an ordinary hit on yourself,
        # which is why a curled dragon hears its own sonar.
        origin = body[-1]
        d = mp.direction_between(body[-2], body[-1])
        direction = d if d else req

    x, y = origin
    for _ in range(limit):
        nxt = mp.step(x, y, direction)
        if nxt is None:
            return direction, origin, (x, y), None, "kelp"
        x, y = nxt
        for did, b in bodies.items():
            if (x, y) in b:
                is_head = b[0] == (x, y)
                ally = teams[did] == teams[sender]
                kind = ("ally_head" if is_head else "ally") if ally else \
                       ("enemy_head" if is_head else "enemy")
                return direction, origin, (x, y), did, kind
    return direction, origin, (x, y), None, "empty"


# --------------------------------------------------- state from the replay

def reconstruct(blob: bytes):
    """Walks the replay, yielding (state, ping) at every ray.

    State is rebuilt from the events alone: initial bodies from the map, moves
    from dragonAction, growth resolved by the tail that dragonUpdate reports,
    and full bodies taken from dragonSplit.
    """
    if R.is_packed(blob):
        blob = R.unpack(blob)
    root = R.Message(blob).root()
    mp = Map(root.text(0))

    bodies: dict[int, list[tuple[int, int]]] = {}
    teams: dict[int, int] = {}
    for i, (team, body) in enumerate(mp.spawns):
        bodies[i] = list(body)
        teams[i] = team

    rnd, mismatches = 0, 0
    for ev in root.structs(3):
        which = ev.u16(0)
        b = ev.struct(0)
        if b is None:
            continue
        if which == R.EVENT_ROUND_START:
            rnd = b.i32(0)
        elif which == R.EVENT_DRAGON_ACTION:
            act = b.struct(0)
            did = b.i32(0)
            if act is not None and act.u16(0) == 0 and did in bodies:
                for d in act.u16_list(0):
                    if d >= 4:
                        continue
                    head = bodies[did][0]
                    nxt = mp.step(head[0], head[1], DIRS[d])
                    if nxt is None:
                        break
                    bodies[did].insert(0, nxt)
        elif which == R.EVENT_DRAGON_UPDATE:
            did = b.i32(0)
            hp, tp = b.struct(0), b.struct(1)
            if did in bodies and hp is not None and tp is not None:
                head = (hp.i32(0), hp.i32(4))
                tail = (tp.i32(0), tp.i32(4))
                body = bodies[did]
                while len(body) > 1 and body[-1] != tail:
                    body.pop()
                if body[0] != head or body[-1] != tail:
                    mismatches += 1
        elif which == R.EVENT_DRAGON_SPLIT:
            pid, cid = b.i32(0), b.i32(4)
            team = b.u16(8)
            pb = [(p.i32(0), p.i32(4)) for p in b.structs(0)]
            cb = [(p.i32(0), p.i32(4)) for p in b.structs(1)]
            if pb:
                bodies[pid] = pb
            bodies[cid] = cb
            teams[cid] = team
            teams.setdefault(pid, team)
        elif which == R.EVENT_DRAGON_DEATH:
            bodies.pop(b.i32(0), None)
        elif which == R.EVENT_SONAR_PING:
            tag = b.u64(16) or b.u32(8)
            o, e = b.struct(0), b.struct(1)
            yield mp, bodies, teams, rnd, {
                "sender": b.i32(0),
                "exit": R.DIRECTIONS[b.u16(4)] if b.u16(4) < 4 else "?",
                "origin": (o.i32(0), o.i32(4)) if o else None,
                "end": (e.i32(0), e.i32(4)) if e else None,
                "hit": b.i32(12) if b.u16(6) == R.SONAR_HIT_ID else None,
                "kind": R.HIT_KINDS[b.u16(24)] if b.u16(24) < len(R.HIT_KINDS) else "?",
                "tag": tag,
            }, mismatches


def play(map_text: str, seed: int = 0):
    """One game, every dragon broadcasting in all four directions every turn.

    The payload carries the requested direction, which the event does not. The
    policy plays to survive, because a game that ends in two rounds casts almost
    no rays and the interesting cases need curled-up bodies.
    """
    import random

    import probe_sonar as PS

    rng = random.Random(seed)

    def policy(did: int, block: str) -> str:
        out = []
        for d in range(4):
            out.append(f"SONAR {DIRS[d]} {(1 << 62) | (did & 0xFFFF) << 8 | d}")
        out.append("PROTOCOL 3")
        try:
            b = PS.parse(block)
            safe = PS.safe_dirs(b) or list(DIRS)
            # Prefer carrying straight on, so bodies stay long and straight
            # sometimes and curl at other times -- both cases matter here.
            weights = [3.0 if d == b["dir"] else 1.0 for d in safe]
            step = rng.choices(safe, weights)[0]
            if rng.random() < 0.02 and b["length"] >= 6:
                out.append(f"SPLIT {rng.randint(2, b['length'] - 2)}")
            else:
                out.append("MOVE " + step)
        except Exception:
            out.append("MOVE " + DIRS[rng.randrange(4)])
        out.append("ENDTURN")
        return "\n".join(out) + "\n"

    g = OracleGame(map_text, policy)
    g.run()
    return g._engine.replay("A", "B")


def check(map_path: pathlib.Path, seed: int = 0) -> tuple[int, int]:
    blob = play(map_path.read_text(), seed)
    agree = total = 0
    why: collections.Counter = collections.Counter()
    worst = []
    state_bad = 0
    for mp, bodies, teams, rnd, ping, mism in reconstruct(blob):
        state_bad = mism
        req = DIRS[ping["tag"] & 3]
        sender = ping["sender"]
        if sender not in bodies:
            why["sender not tracked"] += 1
            total += 1
            continue
        pd, po, pe, ph, pk = predict(mp, bodies, teams, sender, req)
        total += 1
        ok = (pd == ping["exit"] and po == ping["origin"] and pe == ping["end"]
              and ph == ping["hit"] and pk == ping["kind"])
        if ok:
            agree += 1
        else:
            tags = []
            if pd != ping["exit"]:
                tags.append("dir")
            if po != ping["origin"]:
                tags.append("origin")
            if pe != ping["end"]:
                tags.append("end")
            if ph != ping["hit"]:
                tags.append("hit")
            if pk != ping["kind"]:
                tags.append("kind")
            why["+".join(tags)] += 1
            if len(worst) < 6:
                worst.append((rnd, sender, req, bodies[sender][:6],
                              (pd, po, pe, ph, pk),
                              (ping["exit"], ping["origin"], ping["end"],
                               ping["hit"], ping["kind"])))
    pct = 100.0 * agree / total if total else 0.0
    print(f"{map_path.stem:16s} {agree:6d}/{total:6d} rays agree  {pct:6.2f}%"
          f"   (state rebuild mismatches: {state_bad})")
    for k, v in why.most_common(6):
        print(f"      {k:24s} {v}")
    for rnd, s, req, body, got, want in worst:
        print(f"      r{rnd} d{s} cast {req} body{body}")
        print(f"        ours {got}")
        print(f"        engine {want}")
    return agree, total


def main(argv: list[str]) -> int:
    maps = [pathlib.Path(a) for a in argv[1:]]
    if not maps:
        names = ["default_small", "big_empty", "arena", "default"]
        maps = [ROOT / "maps-official" / f"{n}.map" for n in names]
        maps = [m for m in maps if m.is_file()]
    a = t = 0
    for m in maps:
        da, dt = check(m)
        a += da
        t += dt
    print(f"\ntotal {a}/{t} = {100.0 * a / t if t else 0:.2f}%")
    return 0 if a == t else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
