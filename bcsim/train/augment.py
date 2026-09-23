"""Map augmentation: symmetric variants of the maps we have.

The maps in the elimination rounds are not the ones in maps/, so training on
exactly those invites overfitting to their layouts. Each variant is built from
a base map by some of:

  * crop / pad: delete a row or column, or insert a blank one (a wall running
    across the insertion point is carried through it);
  * kelp noise: open some kelp edges and add kelp to some open ones;
  * pearl noise: rescale the spawn gaps and switch a few spawn tiles on or off;
  * start noise: slide each starting dragon a few tiles.

Every change is applied to a whole symmetry orbit, so a variant keeps its base
map's symmetry and neither team gets a different map. Maps with no SYMMETRY
line get the symmetry their dragons have, for augmentation only; the file
still says none, as the original did.

A variant is only kept if it parses, every dragon has a move, and the two
teams can reach each other.

    python -m train.augment --maps maps --out /tmp/aug --per-map 4
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import pathlib
from collections import deque

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root

OPEN, KELP, PORTAL = 0, 1, 2
SYM_NONE, SYM_FLIP_Y, SYM_FLIP_X, SYM_ROT180 = 0, 1, 2, 3
SYM_WORD = {"x": SYM_FLIP_Y, "y": SYM_FLIP_X, "xy": SYM_ROT180}
WORD_SYM = {v: k for k, v in SYM_WORD.items()}
DIRS = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}


@dataclasses.dataclass
class Map:
    w: int
    h: int
    sym: int                     # as written in the file
    name: str
    min_gap: np.ndarray          # (h, w)
    max_gap: np.ndarray          # (h, w), 0 = never spawns
    hk: np.ndarray               # (h, w) kind of the edge on each tile's north side
    vk: np.ndarray               # (h, w) kind of the edge on each tile's west side
    hp: np.ndarray               # (h, w) portal id, -1 if none
    vp: np.ndarray
    dragons: list                # [(team, [(x, y), ...head first])]
    unit_limit: int | None = None
    aug_sym: int = SYM_NONE      # symmetry augmentation keeps

    def copy(self) -> "Map":
        return Map(self.w, self.h, self.sym, self.name, self.min_gap.copy(),
                   self.max_gap.copy(), self.hk.copy(), self.vk.copy(), self.hp.copy(),
                   self.vp.copy(), [(t, list(b)) for t, b in self.dragons],
                   self.unit_limit, self.aug_sym)

    # ---- symmetry
    def mirror_tile(self, x, y, sym=None):
        s = self.aug_sym if sym is None else sym
        if s == SYM_FLIP_Y:
            return x, self.h - 1 - y
        if s == SYM_FLIP_X:
            return self.w - 1 - x, y
        if s == SYM_ROT180:
            return self.w - 1 - x, self.h - 1 - y
        return x, y

    def mirror_hedge(self, x, y):
        """North side of (x, y) mirrors to the north side of the returned tile."""
        s = self.aug_sym
        if s == SYM_FLIP_Y:
            return x, (self.h - y) % self.h
        if s == SYM_FLIP_X:
            return self.w - 1 - x, y
        if s == SYM_ROT180:
            return self.w - 1 - x, (self.h - y) % self.h
        return x, y

    def mirror_vedge(self, x, y):
        s = self.aug_sym
        if s == SYM_FLIP_Y:
            return x, self.h - 1 - y
        if s == SYM_FLIP_X:
            return (self.w - x) % self.w, y
        if s == SYM_ROT180:
            return (self.w - x) % self.w, self.h - 1 - y
        return x, y

    # ---- movement, as tile_after_step does it
    def portal_exit(self, vertical: bool, pid: int, ex: int, ey: int):
        ks, ps = (self.vk, self.vp) if vertical else (self.hk, self.hp)
        ys, xs = np.nonzero((ks == PORTAL) & (ps == pid))
        for y, x in zip(ys, xs):
            if (x, y) != (ex, ey):
                return int(x), int(y)
        return None

    def step(self, x, y, d):
        """Tile reached by stepping `d` from (x, y), or None for kelp."""
        if d in "NS":
            ex, ey = x, (y if d == "N" else (y + 1) % self.h)
            kind, pid, vertical = self.hk[ey, ex], self.hp[ey, ex], False
        else:
            ex, ey = (x if d == "W" else (x + 1) % self.w), y
            kind, pid, vertical = self.vk[ey, ex], self.vp[ey, ex], True
        if kind == KELP:
            return None
        if kind == PORTAL:
            t = self.portal_exit(vertical, int(pid), ex, ey)
            if t is None:
                return None
            tx, ty = t
            if not vertical:
                ty -= 1 if d != "S" else 0
            else:
                tx -= 1 if d != "E" else 0
            return tx % self.w, ty % self.h
        dx, dy = DIRS[d]
        return (x + dx) % self.w, (y + dy) % self.h


# ------------------------------------------------------------------ io
def parse(text: str) -> Map:
    w = h = 0
    sym, name, unit_limit = SYM_NONE, "", None
    tiles, edges, dragons = [], [], []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tok, *rest = line.split()
        if tok == "MAP":
            w, h = int(rest[0]), int(rest[1])
        elif tok == "MAP_NAME":
            name = line[len("MAP_NAME"):].strip()
        elif tok == "SYMMETRY":
            sym = SYM_WORD[rest[0]]
        elif tok == "UNIT_LIMIT":
            unit_limit = int(rest[0])
        elif tok == "TILE":
            tiles.append(tuple(int(v) for v in rest[:4]))
        elif tok == "EDGE":
            edges.append((int(rest[0]), int(rest[1]), int(rest[2]) if len(rest) > 2 else -1))
        elif tok == "DRAGON":
            team, n = int(rest[0]), int(rest[1])
            xy = [int(v) for v in rest[2:2 + 2 * n]]
            dragons.append((team, [(xy[2 * i], xy[2 * i + 1]) for i in range(n)]))
    z = lambda v=0: np.full((h, w), v, np.int32)
    m = Map(w, h, sym, name, z(), z(), z(), z(), z(-1), z(-1), dragons, unit_limit)
    for x, y, mn, mx in tiles:
        m.min_gap[y, x], m.max_gap[y, x] = mn, mx
    stride = w + 1
    for index, kind, pid in edges:
        row, rem = divmod(index, stride)
        if rem == w:
            continue
        vertical = row % 2 == 1
        if not vertical and row == 2 * h:
            continue
        y = (row - 1) // 2 if vertical else row // 2
        k = KELP if kind == 1 else PORTAL if kind == 2 else OPEN
        if vertical:
            m.vk[y, rem], m.vp[y, rem] = k, (pid if k == PORTAL else -1)
        else:
            m.hk[y, rem], m.hp[y, rem] = k, (pid if k == PORTAL else -1)
    m.aug_sym = sym if sym != SYM_NONE else infer_symmetry(m)
    return m


def infer_symmetry(m: Map) -> int:
    """The symmetry that swaps the two teams' starting dragons, preferring one
    the kelp also has."""
    fits = []
    for s in (SYM_ROT180, SYM_FLIP_X, SYM_FLIP_Y):
        a = sorted((t, tuple(b)) for t, b in m.dragons)
        b = sorted((1 - t, tuple(m.mirror_tile(x, y, s) for x, y in body))
                   for t, body in m.dragons)
        if a == b:
            fits.append(s)
    saved = m.aug_sym
    for s in fits:
        m.aug_sym = s
        ok = all(kind[y, x] == kind[my, mx]
                 for kind, mirror in ((m.hk, m.mirror_hedge), (m.vk, m.mirror_vedge))
                 for y in range(m.h) for x in range(m.w)
                 for mx, my in [mirror(x, y)])
        m.aug_sym = saved
        if ok:
            return s
    return fits[0] if fits else SYM_NONE


def render(m: Map) -> str:
    out = [f"MAP {m.w} {m.h}"]
    if m.sym != SYM_NONE:
        out.append(f"SYMMETRY {WORD_SYM[m.sym]}")
    out.append(f"MAP_NAME {m.name}")
    if m.unit_limit is not None:
        out.append(f"UNIT_LIMIT {m.unit_limit}")
    out.append(f"TILE_COUNT {m.w * m.h}")
    for y in range(m.h):
        for x in range(m.w):
            out.append(f"TILE {x} {y} {m.min_gap[y, x]} {m.max_gap[y, x]}")
    # the same layout the shipped maps use: horizontal rows without their
    # wrap column, vertical rows with it, and the south border row at the end
    stride = m.w + 1
    edges = []

    def emit(index, kind, pid):
        code = 1 if kind == KELP else 2 if kind == PORTAL else 0
        edges.append(f"EDGE {index} {code} {pid if kind == PORTAL else -1}")

    for y in range(m.h + 1):
        yy = y % m.h
        for x in range(m.w):
            emit(2 * y * stride + x, m.hk[yy, x], m.hp[yy, x])
        if y < m.h:
            for x in range(m.w + 1):
                xx = x % m.w
                emit((2 * y + 1) * stride + x, m.vk[y, xx], m.vp[y, xx])
    out.append(f"EDGE_COUNT {len(edges)}")
    out += edges
    out.append(f"DRAGON_COUNT {len(m.dragons)}")
    for team, body in m.dragons:
        out.append(f"DRAGON {team} {len(body)} " + " ".join(f"{x} {y}" for x, y in body))
    out.append("END")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ checks
def valid(m: Map, min_reach: float = 0.35) -> bool:
    if not (8 <= m.w <= 64 and 8 <= m.h <= 64):
        return False
    # portals come in pairs
    for ks, ps in ((m.hk, m.hp), (m.vk, m.vp)):
        ids, counts = np.unique(ps[ks == PORTAL], return_counts=True)
        if (counts != 2).any():
            return False
    occ = {}
    for i, (_, body) in enumerate(m.dragons):
        for c in body:
            if c in occ:
                return False
            occ[c] = i
    # bodies hang together through open edges
    for _, body in m.dragons:
        for a, b in zip(body, body[1:]):
            if not any(m.step(a[0], a[1], d) == b for d in "NESW"):
                return False
    # each team has a dragon that can move (help.map boxes some in on purpose),
    # and the teams can meet
    for team in (0, 1):
        if not any(any((c := m.step(b[0][0], b[0][1], d)) is not None and c not in occ
                       for d in "NESW") for t, b in m.dragons if t == team):
            return False
    start = [body[0] for t, body in m.dragons if t == 0]
    goal = {body[0] for t, body in m.dragons if t == 1}
    seen = set(start)
    q = deque(start)
    while q:
        x, y = q.popleft()
        for d in "NESW":
            t = m.step(x, y, d)
            if t is not None and t not in seen:
                seen.add(t)
                q.append(t)
    return goal <= seen and len(seen) >= min_reach * m.w * m.h


# ------------------------------------------------------------------ transforms
def _axis_ops(n: int, mirrored: bool, rng, avoid: set[int], op: str) -> list | None:
    """New index -> old index (None = blank) for one axis."""
    idx: list = list(range(n))
    if op == "del":
        cands = [r for r in range(1, n - 1) if r not in avoid and
                 (not mirrored or (n - 1 - r) not in avoid)]
        if not cands:
            return None
        r = int(rng.choice(cands))
        drop = {r, n - 1 - r} if mirrored else {r}
        return [i for i in idx if i not in drop]
    # insert a blank before position r (and, mirrored, after n-1-r), never
    # between two busy lines, which could cut a dragon in half
    cands = [r for r in range(1, n) if not ({r - 1, r} <= avoid) and
             (not mirrored or not ({n - r - 1, n - r} <= avoid))]
    if not cands:
        return None
    r = int(rng.choice(cands))
    out = []
    for i in range(n):
        if i == r:
            out.append(None)
        if mirrored and i == n - r and n - r != r:
            out.append(None)
        out.append(i)
    return out


def remap(m: Map, rows: list, cols: list) -> Map | None:
    """Rebuilds the map on a new grid: rows[y'] / cols[x'] name the old row /
    column behind each new one, None for a blank."""
    H, W = len(rows), len(cols)
    z = lambda v=0: np.full((H, W), v, np.int32)
    n = Map(W, H, m.sym, m.name, z(), z(), z(), z(), z(-1), z(-1), [], m.unit_limit, m.aug_sym)
    spawn = m.max_gap > 0
    typical = (int(np.median(m.min_gap[spawn])), int(np.median(m.max_gap[spawn]))) \
        if spawn.any() else (0, 0)

    def between(a, b, size):
        """Old edge rows/cols crossed going from old a to old b (a before b)."""
        out, i = [], (a + 1) % size
        while True:
            out.append(i)
            if i == b:
                return out
            i = (i + 1) % size

    for y, oy in enumerate(rows):
        for x, ox in enumerate(cols):
            if oy is not None and ox is not None:
                n.min_gap[y, x], n.max_gap[y, x] = m.min_gap[oy, ox], m.max_gap[oy, ox]
            else:
                n.min_gap[y, x], n.max_gap[y, x] = typical

            # north edge of (x, y)
            py = rows[(y - 1) % H]
            if ox is not None and oy is not None and py is not None:
                crossed = between(py, oy, m.h)
                kinds = [m.hk[r, ox] for r in crossed]
                if PORTAL in kinds:
                    if len(crossed) > 1:
                        return None
                    n.hk[y, x], n.hp[y, x] = PORTAL, m.hp[oy, ox]
                else:
                    n.hk[y, x] = KELP if KELP in kinds else OPEN
            # west edge of (x, y)
            px = cols[(x - 1) % W]
            if ox is not None and oy is not None and px is not None:
                crossed = between(px, ox, m.w)
                kinds = [m.vk[oy, c] for c in crossed]
                if PORTAL in kinds:
                    if len(crossed) > 1:
                        return None
                    n.vk[y, x], n.vp[y, x] = PORTAL, m.vp[oy, ox]
                else:
                    n.vk[y, x] = KELP if KELP in kinds else OPEN

    # walls carry straight through blank rows and columns
    def nearest(seq, i, step):
        j = (i + step) % len(seq)
        while seq[j] is None and j != i:
            j = (j + step) % len(seq)
        return j

    for y, oy in enumerate(rows):
        if oy is None:
            up, dn = nearest(rows, y, -1), nearest(rows, y, 1)
            for x in range(W):
                if n.vk[up, x] == KELP and n.vk[dn, x] == KELP:
                    n.vk[y, x] = KELP
    for x, ox in enumerate(cols):
        if ox is None:
            lf, rt = nearest(cols, x, -1), nearest(cols, x, 1)
            for y in range(H):
                if n.hk[y, lf] == KELP and n.hk[y, rt] == KELP:
                    n.hk[y, x] = KELP

    inv_r = {o: i for i, o in enumerate(rows) if o is not None}
    inv_c = {o: i for i, o in enumerate(cols) if o is not None}
    for team, body in m.dragons:
        if any(y not in inv_r or x not in inv_c for x, y in body):
            return None
        n.dragons.append((team, [(inv_c[x], inv_r[y]) for x, y in body]))
    return n


def structural(m: Map, rng, n_ops: int) -> Map:
    rows_mirrored = m.aug_sym in (SYM_FLIP_Y, SYM_ROT180)
    cols_mirrored = m.aug_sym in (SYM_FLIP_X, SYM_ROT180)
    # rows / columns holding a dragon or a portal stay
    busy_r = {y for _, b in m.dragons for _, y in b}
    busy_c = {x for _, b in m.dragons for x, _ in b}
    ys, xs = np.nonzero((m.hk == PORTAL) | (m.vk == PORTAL))
    busy_r |= set(ys.tolist()) | {(y - 1) % m.h for y in ys.tolist()}
    busy_c |= set(xs.tolist()) | {(x - 1) % m.w for x in xs.tolist()}
    for _ in range(n_ops):
        op = rng.choice(["del", "ins"])
        on_rows = rng.random() < 0.5
        if on_rows:
            rows = _axis_ops(m.h, rows_mirrored, rng, busy_r, op)
            cols = list(range(m.w))
        else:
            rows = list(range(m.h))
            cols = _axis_ops(m.w, cols_mirrored, rng, busy_c, op)
        if rows is None or cols is None:
            continue
        n = remap(m, rows, cols)
        if n is None or not (8 <= n.w <= 64 and 8 <= n.h <= 64):
            continue
        m = n
        busy_r = {y for _, b in m.dragons for _, y in b}
        busy_c = {x for _, b in m.dragons for x, _ in b}
        ys, xs = np.nonzero((m.hk == PORTAL) | (m.vk == PORTAL))
        busy_r |= set(ys.tolist()) | {(y - 1) % m.h for y in ys.tolist()}
        busy_c |= set(xs.tolist()) | {(x - 1) % m.w for x in xs.tolist()}
    return m


def kelp_noise(m: Map, rng, p_add: float, p_del: float) -> None:
    for kind, mirror in ((m.hk, m.mirror_hedge), (m.vk, m.mirror_vedge)):
        u = rng.random(kind.shape)
        new = kind.copy()
        for y in range(m.h):
            for x in range(m.w):
                mx, my = mirror(x, y)
                if (my, mx) < (y, x):
                    continue              # the orbit's first member decides
                k = kind[y, x]
                if k == PORTAL or kind[my, mx] == PORTAL:
                    continue
                if k == KELP and u[y, x] < p_del:
                    new[y, x] = new[my, mx] = OPEN
                elif k == OPEN and u[y, x] < p_add:
                    new[y, x] = new[my, mx] = KELP
        kind[...] = new


def pearl_noise(m: Map, rng, scale_range: tuple[float, float], p_flip: float) -> None:
    spawn = m.max_gap > 0
    if spawn.any():
        typical = (int(np.median(m.min_gap[spawn])), int(np.median(m.max_gap[spawn])))
    else:
        typical = (20, 60)
    s = math.exp(rng.uniform(math.log(scale_range[0]), math.log(scale_range[1])))
    for y in range(m.h):
        for x in range(m.w):
            mx, my = m.mirror_tile(x, y)
            if (my, mx) < (y, x):
                continue
            if rng.random() < p_flip:
                if m.max_gap[y, x] > 0:
                    mn, mxg = 0, 0
                else:
                    mn, mxg = typical
            else:
                mn, mxg = int(m.min_gap[y, x]), int(m.max_gap[y, x])
            if mxg > 0:
                mn = max(1, int(round(mn * s)))
                mxg = max(mn, int(round(mxg * s)))
            m.min_gap[y, x] = m.min_gap[my, mx] = mn
            m.max_gap[y, x] = m.max_gap[my, mx] = mxg


def start_noise(m: Map, rng, max_shift: int) -> None:
    """Slides each team-A dragon, and its mirror image, by a small offset."""
    if m.aug_sym == SYM_NONE or max_shift <= 0:
        return
    bodies = [list(b) for _, b in m.dragons]
    teams = [t for t, _ in m.dragons]
    for i, (team, body) in enumerate(m.dragons):
        if team != 0:
            continue
        mirrored = [m.mirror_tile(x, y) for x, y in body]
        try:
            j = next(k for k, b in enumerate(bodies) if teams[k] == 1 and b == mirrored)
        except StopIteration:
            continue
        for _ in range(8):
            dx, dy = (int(v) for v in rng.integers(-max_shift, max_shift + 1, 2))
            nb = [((x + dx) % m.w, (y + dy) % m.h) for x, y in bodies[i]]
            nm = [m.mirror_tile(x, y) for x, y in nb]
            others = {c for k, b in enumerate(bodies) if k not in (i, j) for c in b}
            cells = nb + nm
            if len(set(cells)) != len(cells) or others & set(cells):
                continue
            if not all(any(m.step(a[0], a[1], d) == b for d in "NESW")
                       for body2 in (nb, nm) for a, b in zip(body2, body2[1:])):
                continue
            bodies[i], bodies[j] = nb, nm
            break
    m.dragons = list(zip(teams, bodies))


@dataclasses.dataclass
class AugConfig:
    p_struct: float = 0.5        # chance of any crop / pad at all
    max_struct_ops: int = 3
    p_kelp: float = 0.8          # chance of kelp noise
    kelp_add: float = 0.03       # per open edge
    kelp_del: float = 0.15       # per kelp edge
    p_pearl: float = 0.8
    pearl_scale: tuple = (0.6, 1.6)
    pearl_flip: float = 0.05
    p_start: float = 0.7
    start_shift: int = 3


def augment(base: Map, rng, cfg: AugConfig = AugConfig(), tries: int = 30) -> Map | None:
    for _ in range(tries):
        m = base.copy()
        if rng.random() < cfg.p_struct:
            m = structural(m, rng, int(rng.integers(1, cfg.max_struct_ops + 1)))
        if rng.random() < cfg.p_kelp:
            kelp_noise(m, rng, cfg.kelp_add * rng.random() * 2, cfg.kelp_del * rng.random() * 2)
        if rng.random() < cfg.p_pearl:
            pearl_noise(m, rng, cfg.pearl_scale, cfg.pearl_flip)
        if rng.random() < cfg.p_start:
            start_noise(m, rng, cfg.start_shift)
        if valid(m):
            return m
    return None


def size_weight(w: int, h: int, alpha: float) -> float:
    """Relative sampling weight: 1 at 16x16, smaller maps more, larger less."""
    return (256.0 / (w * h)) ** alpha


def build_pool(map_dir: str, per_map: int, seed: int, alpha: float,
               original_share: float = 0.25, cfg: AugConfig = AugConfig()):
    """Returns (texts, weights, base_names, areas): every base map plus
    `per_map` variants of it, weighted so each base map's total is its size
    weight and the untouched original keeps `original_share` of that. `areas`
    is each entry's base map area, so the size weighting can be redone with
    another alpha (see `reweight`)."""
    rng = np.random.default_rng(seed)
    texts, weights, names, areas = [], [], [], []
    for f in sorted(pathlib.Path(map_dir).glob("*.map")):
        text = f.read_text()
        base = parse(text)
        total = size_weight(base.w, base.h, alpha)
        variants = []
        for k in range(per_map):
            v = augment(base, rng, cfg)
            if v is not None:
                v.name = f"{base.name} aug{k}"
                variants.append(render(v))
        share = original_share if variants else 1.0
        texts.append(text)
        weights.append(total * share)
        names.append(f.stem)
        areas.append(base.w * base.h)
        for v in variants:
            texts.append(v)
            weights.append(total * (1 - share) / len(variants))
            names.append(f.stem)
            areas.append(base.w * base.h)
    return texts, np.array(weights), names, np.array(areas, dtype=np.float64)


def reweight(weights: np.ndarray, areas: np.ndarray, alpha_from: float,
             alpha_to: float) -> np.ndarray:
    """Pool weights built with `alpha_from`, redone for `alpha_to`."""
    return weights * (256.0 / areas) ** (alpha_to - alpha_from)


def set_group_share(weights: np.ndarray, names: list[str], group: set[str],
                    share: float) -> np.ndarray:
    """Rescale so the base maps in `group` take `share` of the pool together.

    build_pool weights every base map equally, so adding maps of our own dilutes
    the ones the ladder is actually played on: twelve maps put the live seven at
    7/12, and ten invented maps on top would put them at 7/22. This keeps the
    relative weights inside each side untouched and only moves the split between
    them, so `share` is exactly the probability of drawing a group map.

    Returns a copy, normalised to sum to 1. A group that is empty or that is the
    whole pool is returned normalised and otherwise untouched, since there is no
    split to set.
    """
    if not 0.0 <= share <= 1.0:
        raise ValueError(f"share must be in [0, 1], got {share}")
    w = np.asarray(weights, dtype=np.float64).copy()
    total = w.sum()
    if total <= 0:
        raise ValueError("pool weights sum to zero")
    w /= total
    inside = np.array([n in group for n in names])
    win, wout = w[inside].sum(), w[~inside].sum()
    if win <= 0 or wout <= 0:          # nothing to split
        return w
    w[inside] *= share / win
    w[~inside] *= (1.0 - share) / wout
    return w


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--maps", default=str(ROOT / "maps"))
    p.add_argument("--out", required=True)
    p.add_argument("--per-map", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    for f in sorted(pathlib.Path(a.maps).glob("*.map")):
        base = parse(f.read_text())
        # a round trip through render must not change the map
        assert parse(render(base)).hk.tolist() == base.hk.tolist(), f
        for k in range(a.per_map):
            v = augment(base, rng)
            if v is None:
                print(f"{f.stem}: no valid variant")
                continue
            (out / f"{f.stem}_aug{k}.map").write_text(render(v))
            print(f"{f.stem}_aug{k}: {v.w}x{v.h}, kelp {int((v.hk == KELP).sum() + (v.vk == KELP).sum())}"
                  f" (base {int((base.hk == KELP).sum() + (base.vk == KELP).sum())})")


if __name__ == "__main__":
    main()
