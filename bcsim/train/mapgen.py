"""Writes the maps in MAPS_PROPOSAL.md, each aimed at a gap the official set
never exercises.

Distilled clones only know the maps their teacher played, which is why they are
helpless on Devil (PPO gen4 takes 0.929 there against v12, against 0.520
overall). PPO can train on anything we can write down, so these fill the holes:
UNIT_LIMIT is never set on any official map, dragons per team is only 2-4 on
the live seven, portals are 0/2/4/24 with no map built around them, pearl
coverage has nothing below 14%, every live map is mirror-symmetric though the
engine does not require it, aspect never exceeds 2:1, and there is no ring,
chokepoint or maze.

    python -m train.mapgen --out ../maps-gen

Hard rule from the user (2026-09-23): official maps always exceed 10 in every
dimension -- arena at 11x11 is the smallest, and the live seven bottom out at
16 -- so nothing here is thinner than 16.

These are for TRAINING only. Grade on maps-live/: the ladder is played on the
official seven, and --live-share keeps them at 60% of sampling.
"""

from __future__ import annotations

import argparse
import pathlib

# EDGE index: rows alternate horizontal (a tile row's north side) and vertical
# (its west side), each row width+1 long (see bc_text.hpp).
KELP, PORTAL = 1, 2


class Map:
    def __init__(self, name: str, w: int, h: int, sym: str | None, unit_limit: int | None = None):
        if min(w, h) <= 10:
            raise ValueError(f"{name}: {w}x{h} breaks the >10 rule")
        self.name, self.w, self.h, self.sym = name, w, h, sym
        self.unit_limit = unit_limit
        self.tiles = {}                      # (x, y) -> (min_gap, max_gap)
        self.edges = {}                      # index -> (kind, pid)
        self.dragons = []                    # (team, [(x, y), ...]) head first

    # -- geometry
    def north(self, x: int, y: int) -> int: return (2 * y) * (self.w + 1) + x
    def west(self, x: int, y: int) -> int: return (2 * y + 1) * (self.w + 1) + x

    def mirror(self, x: int, y: int) -> tuple[int, int]:
        if self.sym == "x":  return x, self.h - 1 - y
        if self.sym == "y":  return self.w - 1 - x, y
        if self.sym == "xy": return self.w - 1 - x, self.h - 1 - y
        return x, y

    # -- content
    def pearl(self, x: int, y: int, lo: int, hi: int) -> None:
        self.tiles[(x % self.w, y % self.h)] = (lo, hi)

    def kelp_n(self, x: int, y: int) -> None:
        self.edges[self.north(x % self.w, y % self.h)] = (KELP, -1)

    def kelp_w(self, x: int, y: int) -> None:
        self.edges[self.west(x % self.w, y % self.h)] = (KELP, -1)

    def portal(self, a: int, b: int, pid: int) -> None:
        """Two edges of the SAME orientation, exactly two per id (bc_text.hpp)."""
        self.edges[a] = (PORTAL, pid)
        self.edges[b] = (PORTAL, pid)

    def dragon(self, team: int, body: list[tuple[int, int]]) -> None:
        self.dragons.append((team, [(x % self.w, y % self.h) for x, y in body]))

    def pair(self, body: list[tuple[int, int]]) -> None:
        """Team 0 here, team 1 at the mirrored cells, so both sides match."""
        self.dragon(0, body)
        self.dragon(1, [self.mirror(x, y) for x, y in body])

    def render(self) -> str:
        out = [f"MAP {self.w} {self.h}"]
        if self.sym:
            out.append(f"SYMMETRY {self.sym}")
        out.append(f"MAP_NAME {self.name}")
        if self.unit_limit is not None:
            out.append(f"UNIT_LIMIT {self.unit_limit}")
        out.append(f"TILE_COUNT {self.w * self.h}")
        for y in range(self.h):
            for x in range(self.w):
                lo, hi = self.tiles.get((x, y), (0, 0))
                out.append(f"TILE {x} {y} {lo} {hi}")
        # every edge, as the official maps do, most of them open
        n_edges = 2 * self.w * self.h
        out.append(f"EDGE_COUNT {n_edges}")
        for y in range(self.h):
            for x in range(self.w):
                for idx in (self.north(x, y), self.west(x, y)):
                    kind, pid = self.edges.get(idx, (0, -1))
                    out.append(f"EDGE {idx} {kind} {pid}")
        out.append(f"DRAGON_COUNT {len(self.dragons)}")
        for team, body in self.dragons:
            cells = " ".join(f"{x} {y}" for x, y in body)
            out.append(f"DRAGON {team} {len(body)} {cells}")
        return "\n".join(out) + "\n"


def row(m: Map, y: int, x0: int, x1: int) -> list[tuple[int, int]]:
    """A body laid along a row, head at x0."""
    step = 1 if x1 >= x0 else -1
    return [(x, y) for x in range(x0, x1 + step, step)]


# ---------------------------------------------------------------- the maps
def famine() -> Map:
    """Pearl coverage ~5% and slow respawn: nothing official is below 14%."""
    m = Map("Famine", 40, 40, "xy")
    for y in range(m.h):
        for x in range(m.w):
            if (x * 7 + y * 11) % 97 < 5:
                m.pearl(x, y, 200, 400)
    for y in range(4, 36, 8):                    # sparse cover to break sightlines
        for x in range(6, 34):
            if (x + y) % 5:
                m.kelp_n(x, y)
    m.pair(row(m, 8, 3, 0))
    m.pair(row(m, 20, 3, 0))
    m.pair(row(m, 32, 3, 0))
    return m


def glut() -> Map:
    """Every tile a pearl, respawning almost at once: the opposite extreme."""
    m = Map("Glut", 24, 24, "xy")
    for y in range(m.h):
        for x in range(m.w):
            m.pearl(x, y, 1, 5)
    m.pair(row(m, 6, 2, 0))
    m.pair(row(m, 17, 2, 0))
    return m


def duel() -> Map:
    """One dragon a side: the live seven only ever have 2, 3 or 4."""
    m = Map("Duel", 20, 20, "xy")
    for y in range(m.h):
        for x in range(m.w):
            if (x + y) % 2 == 0:
                m.pearl(x, y, 1, 60)
    for y in range(3, 17):                        # two pillars to break the open box
        if y % 4:
            m.kelp_w(6, y)
            m.kelp_w(14, y)
    m.pair([(5, 10), (4, 10), (3, 10), (2, 10), (1, 10), (0, 10)])
    return m


def hive() -> Map:
    """Six dragons a side, at the unit cap: only help (not live) has six."""
    m = Map("Hive", 48, 48, "xy")
    for y in range(m.h):
        for x in range(m.w):
            if (x * 3 + y * 5) % 11 < 4:
                m.pearl(x, y, 20, 90)
    for y in range(0, m.h, 6):                    # open cells, kelp on the seams
        for x in range(0, m.w):
            if x % 6:
                m.kelp_n(x, y)
    for i, y in enumerate(range(4, 46, 8)):
        m.pair(row(m, y, 2, 0))
    return m


def corridor() -> Map:
    """4:1 aspect, double the current maximum (devil is 2:1)."""
    m = Map("Corridor", 64, 16, "y")
    for y in range(m.h):
        for x in range(m.w):
            if x % 3 == 0:
                m.pearl(x, y, 10, 120)
    for x in range(8, 56, 8):                     # rungs, so it is not one open tube
        for y in range(m.h):
            if y % 5:
                m.kelp_w(x, y)
    m.pair(row(m, 3, 3, 0))
    m.pair(row(m, 8, 3, 0))
    m.pair(row(m, 12, 3, 0))
    return m


def wormhole() -> Map:
    """DISABLED -- not in BUILDERS. Portals as structure is a real gap, but no
    version of this map has worked.

    At one portal per row and no kelp, random play ended games in 25 rounds
    against 131-422 on every other map: dragons were flung into each other
    constantly. Adding kelp along the two portal walls to make a portal a route
    rather than one opening among many made it worse in a new way -- it boxes
    each team into an 8-wide strip whose only exit is a portal, and the portal
    lands you in the ENEMY's strip, so a dragon that leaves home dies at once.
    Under trained sampled play that is 5-round games and ~1 recorded position
    each (2026-09-23).

    A working version needs portals that are a shortcut rather than a delivery
    into enemy territory: pair edges within one side of the map, or across the
    corners, so crossing one keeps you on your own half.
    """
    m = Map("Wormhole", 32, 32, "xy")
    for y in range(m.h):
        for x in range(m.w):
            if (x * 5 + y * 3) % 7 < 3:
                m.pearl(x, y, 5, 80)
    pid = 0
    # Portals are the structure here, but sparsely: at one per row (84 edges)
    # random play ended games in 25 rounds against 131-422 on every other map,
    # because dragons were flung across the board into each other constantly.
    # A map that ends that fast teaches almost nothing under result-only reward.
    for y in range(3, 30, 4):
        m.portal(m.west(8, y), m.west(24, y), pid); pid += 1
    for x in range(3, 30, 8):
        m.portal(m.north(x, 8), m.north(x, 24), pid); pid += 1
    # kelp along the same two walls, so a portal is a way through rather than
    # one opening among many
    for y in range(m.h):
        if (y - 3) % 4:
            m.kelp_w(8, y)
            m.kelp_w(24, y)
    m.pair(row(m, 4, 2, 0))
    m.pair(row(m, 27, 2, 0))
    return m


def choke() -> Map:
    """One wall, two gaps: forced contention, which no official map has."""
    m = Map("Choke", 40, 20, "y")
    for y in range(m.h):
        for x in range(m.w):
            if (x + 2 * y) % 4 == 0:
                m.pearl(x, y, 15, 100)
    for y in range(m.h):
        if y not in (5, 6, 13, 14):               # the two gaps
            m.kelp_w(20, y)
    m.pair(row(m, 4, 3, 0))
    m.pair(row(m, 10, 3, 0))
    m.pair(row(m, 16, 3, 0))
    return m


def ring() -> Map:
    """A solid kelp donut: play has to circulate. No ring topology exists."""
    m = Map("Ring", 36, 36, "xy")
    for y in range(m.h):
        for x in range(m.w):
            if (x + y) % 3 == 0:
                m.pearl(x, y, 10, 110)
    lo, hi = 12, 24
    for y in range(lo, hi):                       # box the middle off entirely
        m.kelp_w(lo, y)
        m.kelp_w(hi, y)
    for x in range(lo, hi):
        m.kelp_n(x, lo)
        m.kelp_n(x, hi)
    for x in range(lo, hi):                       # and clear it, so nothing spawns inside
        for y in range(lo, hi):
            m.tiles.pop((x, y), None)
    m.pair(row(m, 6, 3, 0))
    m.pair(row(m, 18, 3, 0))
    m.pair(row(m, 30, 3, 0))
    return m


def drift() -> Map:
    """No SYMMETRY line at all: every live map is mirror-symmetric, which lets
    a policy lean on mirror priors. Fair in aggregate because every eval plays
    both sides of every map."""
    m = Map("Drift", 30, 30, None)
    for y in range(m.h):
        for x in range(m.w):
            if (x * x + y * 3) % 13 < 5:          # deliberately not mirror-equal
                m.pearl(x, y, 8, 130)
    for y in range(2, 28, 3):
        for x in range(y % 7, m.w, 6):
            m.kelp_n(x, y)
    m.dragon(0, row(m, 5, 3, 0))
    m.dragon(0, row(m, 15, 3, 0))
    m.dragon(0, row(m, 25, 3, 0))
    m.dragon(1, row(m, 8, 26, 29))
    m.dragon(1, row(m, 20, 26, 29))
    m.dragon(1, row(m, 27, 26, 29))
    return m


def capped() -> Map:
    """UNIT_LIMIT 8. The rule exists and no official map ever sets it."""
    m = Map("Capped", 32, 32, "xy", unit_limit=8)
    for y in range(m.h):
        for x in range(m.w):
            if (x * 2 + y) % 5 < 2:
                m.pearl(x, y, 6, 70)
    for x in range(4, 28, 6):
        for y in range(4, 28):
            if y % 7:
                m.kelp_w(x, y)
    m.pair(row(m, 6, 3, 0))
    m.pair(row(m, 14, 3, 0))
    m.pair(row(m, 22, 3, 0))
    m.pair(row(m, 29, 3, 0))
    return m


# wormhole is deliberately absent: see its docstring. Nine maps, not ten.
BUILDERS = [famine, glut, duel, hive, corridor, choke, ring, drift, capped]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(pathlib.Path(__file__).resolve().parents[2] / "maps-gen"))
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for build in BUILDERS:
        m = build()
        text = m.render()
        (out / f"{m.name.lower()}.map").write_text(text)
        per = {}
        for team, _ in m.dragons:
            per[team] = per.get(team, 0) + 1
        print(f"  {m.name.lower():<12} {m.w}x{m.h:<3} sym={str(m.sym):<5} "
              f"dragons/team={per.get(0, 0)}  pearls={len(m.tiles):>5} "
              f"({100 * len(m.tiles) / (m.w * m.h):4.1f}%)  "
              f"kelp={sum(1 for k, _ in m.edges.values() if k == KELP):>4} "
              f"portals={sum(1 for k, _ in m.edges.values() if k == PORTAL):>3}"
              f"{'  UNIT_LIMIT ' + str(m.unit_limit) if m.unit_limit else ''}")
    print(f"\n{len(BUILDERS)} maps in {out}")


if __name__ == "__main__":
    main()
