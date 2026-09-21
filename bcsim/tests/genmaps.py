"""Random but legal maps, for stress testing against the reference engine.

Deliberately biased towards the awkward cases: heavy portal use, kelp mazes,
every symmetry, and the smallest and largest boards the rules allow.
"""

from __future__ import annotations

import random


def h_edge_index(w: int, x: int, y: int) -> int:
    return 2 * y * (w + 1) + x


def v_edge_index(w: int, x: int, y: int) -> int:
    return (2 * y + 1) * (w + 1) + x


def generate(rng: random.Random, w: int | None = None, h: int | None = None,
             sym: str | None = None, kelp_rate: float | None = None,
             portal_pairs: int | None = None, teams_dragons: int = 2,
             dragon_len: int = 3) -> str:
    w = w or rng.randint(7, 24)
    h = h or rng.randint(7, 24)
    if sym is None:
        sym = rng.choice([None, "x", "y", "xy"])
    kelp_rate = rng.uniform(0.0, 0.25) if kelp_rate is None else kelp_rate
    portal_pairs = rng.randint(0, 6) if portal_pairs is None else portal_pairs

    def mirror(x, y):
        if sym == "x":
            return x, h - 1 - y
        if sym == "y":
            return w - 1 - x, y
        if sym == "xy":
            return w - 1 - x, h - 1 - y
        return x, y

    lines = [f"MAP {w} {h}"]
    if sym:
        lines.append(f"SYMMETRY {sym}")
    lines.append(f"MAP_NAME rnd{w}x{h}")

    # tiles: mirrored tiles share their spawn range
    tiles = {}
    for y in range(h):
        for x in range(w):
            if (x, y) in tiles:
                continue
            if rng.random() < 0.15:
                mn = mx = 0
            else:
                mn = rng.randint(1, 30)
                mx = mn + rng.randint(0, 60)
            tiles[(x, y)] = (mn, mx)
            tiles[mirror(x, y)] = (mn, mx)
    lines.append(f"TILE_COUNT {len(tiles)}")
    for (x, y), (mn, mx) in sorted(tiles.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        lines.append(f"TILE {x} {y} {mn} {mx}")

    # edges: index -> (kind, portal id); mirrored edges get the same kind
    edges: dict[int, tuple[int, int]] = {}
    used: set[tuple[str, int, int]] = set()

    def claim(vertical: bool, x: int, y: int) -> bool:
        key = ("v" if vertical else "h", x % w, y % h)
        if key in used:
            return False
        used.add(key)
        return True

    for y in range(h):
        for x in range(w):
            for vertical in (False, True):
                if rng.random() >= kelp_rate:
                    continue
                if not claim(vertical, x, y):
                    continue
                idx = v_edge_index(w, x, y) if vertical else h_edge_index(w, x, y)
                edges[idx] = (1, -1)
                mx_, my_ = mirror(x, y)
                # the mirror of an edge lies on the mirrored tile's matching side
                if sym == "x" and not vertical:
                    my_ = (my_ + 1) % h
                if sym == "y" and vertical:
                    mx_ = (mx_ + 1) % w
                if sym == "xy":
                    if vertical:
                        mx_ = (mx_ + 1) % w
                    else:
                        my_ = (my_ + 1) % h
                if claim(vertical, mx_, my_):
                    midx = v_edge_index(w, mx_, my_) if vertical else h_edge_index(w, mx_, my_)
                    edges[midx] = (1, -1)

    pid = 0
    for _ in range(portal_pairs):
        vertical = rng.random() < 0.5
        spots = []
        for _ in range(12):
            x, y = rng.randrange(w), rng.randrange(h)
            if claim(vertical, x, y):
                spots.append((x, y))
            if len(spots) == 2:
                break
        if len(spots) < 2:
            continue
        for (x, y) in spots:
            idx = v_edge_index(w, x, y) if vertical else h_edge_index(w, x, y)
            edges[idx] = (2, pid)
        pid += 1

    lines.append(f"EDGE_COUNT {len(edges)}")
    for idx in sorted(edges):
        kind, portal = edges[idx]
        lines.append(f"EDGE {idx} {kind} {portal}")

    # dragons: straight horizontal runs on free rows, mirrored per team
    taken: set[tuple[int, int]] = set()
    dragons = []
    attempts = 0
    while len(dragons) < 2 * teams_dragons and attempts < 400:
        attempts += 1
        x, y = rng.randrange(w), rng.randrange(h)
        body = [((x + i) % w, y) for i in range(dragon_len)]
        mbody = [mirror(px, py) for (px, py) in body]
        cells = set(body) | set(mbody)
        if len(cells) != 2 * dragon_len or cells & taken:
            continue
        # every step must be passable, or the dragon is not connected
        if any(("v", (px + 1) % w, py) in used or ("v", px, py) in used for px, py in body):
            continue
        if any(("v", (px + 1) % w, py) in used or ("v", px, py) in used for px, py in mbody):
            continue
        taken |= cells
        dragons.append((0, body))
        dragons.append((1, mbody))

    lines.append(f"DRAGON_COUNT {len(dragons)}")
    for team, body in dragons:
        coords = " ".join(f"{x} {y}" for x, y in body)
        lines.append(f"DRAGON {team} {len(body)} {coords}")
    return "\n".join(lines) + "\n"
