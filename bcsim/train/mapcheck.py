"""Checks that a map is fair and loadable, independently of whatever wrote it.

    python -m train.mapcheck ../maps-loong/*.map          # prints the failures, exits 1 on any
    python -m train.mapcheck --official ../maps-live/*.map

A map passes when:
  * it is more than 10 and at most 64 tiles in each dimension;
  * it has a mirror symmetry (the declared one, if it declares one) that swaps the
    two teams: every tile's pearl range, every edge's kind, every portal pairing and
    every starting dragon (team flipped, body in order) map onto their mirror images;
  * portals come in pairs, no two dragons overlap, each body is connected;
  * every dragon has a legal first move and the teams can reach each other
    (official maps, --official: each team has a dragon that can move);
  * bcsim parses it, and the official engine (the local unswbc) loads and starts it.

The mirror maths is augment.Map's, which was written against the official maps, not
the generator's own mirror_tile/mirror_edge, so a generator bug cannot pass itself.
"""

from __future__ import annotations

import pathlib
import sys
from collections import deque

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from train.augment import (KELP, PORTAL, SYM_FLIP_X, SYM_FLIP_Y, SYM_NONE,  # noqa: E402
                           SYM_ROT180, WORD_SYM, Map, parse)

_ENGINE = None


def _engine():
    global _ENGINE
    if _ENGINE is None:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tests"))
        from oracle import EngineModule
        _ENGINE = EngineModule()
    return _ENGINE


def symmetric_under(m: Map, s: int, pearls: bool = True) -> str | None:
    """None if `m` is team-symmetric under `s`, else what breaks it.

    pearls=False skips the pearl ranges, for official maps that declare a SYMMETRY:
    there the engine drives each mirror pair from its lower-index tile (pearl_tick
    skips the other) and its own loader rejects beds that do not match (ring.map)."""
    saved, m.aug_sym = m.aug_sym, s
    try:
        for y in range(m.h if pearls else 0):
            for x in range(m.w):
                mx, my = m.mirror_tile(x, y)
                if (m.min_gap[y, x], m.max_gap[y, x]) != (m.min_gap[my, mx], m.max_gap[my, mx]):
                    return f"pearl range at {(x, y)} differs from its mirror {(mx, my)}"
        # edges: kinds match, and a portal's partner mirrors to its mirror's partner
        partner = {}
        for vertical, ks, ps in ((False, m.hk, m.hp), (True, m.vk, m.vp)):
            ys, xs = np.nonzero(ks == PORTAL)
            for y, x in zip(ys.tolist(), xs.tolist()):
                partner.setdefault(int(ps[y, x]), []).append((vertical, x, y))
        pair = {}
        for pid, ends in partner.items():
            if len(ends) != 2:
                return f"portal {pid} has {len(ends)} ends"
            pair[ends[0]], pair[ends[1]] = ends[1], ends[0]
        for vertical, ks, mirror in ((False, m.hk, m.mirror_hedge), (True, m.vk, m.mirror_vedge)):
            for y in range(m.h):
                for x in range(m.w):
                    mx, my = mirror(x, y)
                    if ks[y, x] != ks[my, mx]:
                        return f"{'west' if vertical else 'north'} edge of {(x, y)} is not its mirror's kind"
        for (vertical, x, y), (pv, px, py) in pair.items():
            mirror = m.mirror_vedge if vertical else m.mirror_hedge
            mirror_p = m.mirror_vedge if pv else m.mirror_hedge
            me = (vertical, *mirror(x, y))
            mp = (pv, *mirror_p(px, py))
            if pair.get(me) != mp:
                return f"portal at {(vertical, x, y)}: the mirror of its partner is not its mirror's partner"
        a = sorted((t, tuple(b)) for t, b in m.dragons)
        b = sorted((1 - t, tuple(m.mirror_tile(x, y) for x, y in body)) for t, body in m.dragons)
        if a != b:
            return "starting dragons are not mirror images across teams"
        return None
    finally:
        m.aug_sym = saved


def check(text: str, engine: bool = True, strict: bool = True) -> list[str]:
    """Every reason `text` fails; empty when it passes.

    strict=False is for official maps: help and slithery_fight box some dragons in
    on purpose, and on portals the teams never meet, so there it is enough that
    each team has a dragon that can move."""
    try:
        m = parse(text)
    except Exception as e:                          # noqa: BLE001
        return [f"parse: {e}"]
    bad = []
    if not (10 < m.w <= 64 and 10 < m.h <= 64):
        bad.append(f"size: {m.w}x{m.h} is not >10 and <=64 in each dimension")
    teams = {t for t, _ in m.dragons}
    if teams != {0, 1}:
        bad.append(f"dragons: teams present {sorted(teams)}")
    syms = [m.sym] if m.sym != SYM_NONE else [SYM_ROT180, SYM_FLIP_X, SYM_FLIP_Y]
    why = [symmetric_under(m, s, pearls=strict or m.sym == SYM_NONE) for s in syms]
    if all(w is not None for w in why):
        declared = WORD_SYM.get(m.sym, "none")
        bad.append(f"symmetry: declared {declared}; " + " / ".join(why))
    else:
        m.aug_sym = syms[[w is None for w in why].index(True)]
    occ = {}
    for i, (_, body) in enumerate(m.dragons):
        for c in body:
            if c in occ:
                bad.append(f"dragons: overlap at {c}")
            occ[c] = i
        for p, q in zip(body, body[1:]):
            if not any(m.step(p[0], p[1], d) == q for d in "NESW"):
                bad.append(f"dragons: body not connected at {p}-{q}")
                break
    def can_move(body):
        hx, hy = body[0]
        return any((c := m.step(hx, hy, d)) is not None and c not in occ for d in "NESW")
    for t, body in m.dragons:
        if strict and not can_move(body):
            bad.append(f"dragons: team {t} dragon at {body[0]} has no legal first move")
    for t in (0, 1):
        if not any(can_move(b) for tt, b in m.dragons if tt == t):
            bad.append(f"dragons: no team {t} dragon can move")
    start = [b[0] for t, b in m.dragons if t == 0]
    seen, q = set(start), deque(start)
    while q:
        x, y = q.popleft()
        for d in "NESW":
            c = m.step(x, y, d)
            if c is not None and c not in seen:
                seen.add(c)
                q.append(c)
    if strict and not {b[0] for t, b in m.dragons if t == 1} <= seen:
        bad.append("reach: team 0 cannot reach team 1")
    if bad:
        return bad
    try:
        import bcsim
        bcsim.BattlecodeVecEnv([text], num_envs=1, num_threads=1, seed=0)
    except Exception as e:                          # noqa: BLE001
        bad.append(f"bcsim: {e}")
    if engine:
        try:
            _engine().run(text.encode(), lambda d, b: b"", on_notice=lambda s: None)
        except Exception as e:                      # noqa: BLE001
            bad.append(f"engine: {e}")
    return bad


def main() -> None:
    failed = 0
    files = [f for f in sys.argv[1:] if f != "--official"]
    for f in files:
        bad = check(pathlib.Path(f).read_text(), strict="--official" not in sys.argv)
        if bad:
            failed += 1
            print(f"{f}: " + "; ".join(bad))
    print(f"{len(files) - failed} of {len(files)} pass")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
