"""Extra inputs for cloning, computed from a clone_cache.py cache.

The simulator's observation is memoryless: one 7x7 window and 14 scalars. A
deployed dragon is a long-lived process, though, so it can remember its own
earlier turns, and a team whose bot does is only partly predictable without
that memory. Each feature here uses only what this dragon saw or did on its
own earlier turns, so a bot can compute it as it plays.

    python -m train.clone_features --cache ../runs/clone_cache/devtest_1302 --build hist3

writes <cache>/feat_<name>.npy once; load_features() then slices the rows a
trainer wants. A feature is either local planes (C, 7, 7) or scalars (K,).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

FEATURES: dict[str, tuple[str, callable]] = {}   # name -> ("local" | "scalar", builder)


def feature(name: str, kind: str):
    def wrap(fn):
        FEATURES[name] = (kind, fn)
        return fn
    return wrap


SC_LENGTH_RAW, SC_FACE_N, SC_HEAD_X, SC_HEAD_Y, SC_MAP_W, SC_MAP_H = 2, 4, 8, 9, 10, 11


def action_class(a: np.ndarray) -> np.ndarray:
    """Codec id -> 0 straight, 1 left, 2 right, 3 sprint, 4 split, 5 other."""
    c = np.full(a.shape, 5, np.int64)
    c[(a >= 0) & (a < 3)] = a[(a >= 0) & (a < 3)]
    c[(a >= 3) & (a < 39)] = 3
    c[a >= 39] = 4
    return c


def prev_index(cache: pathlib.Path) -> np.ndarray:
    """For each row, the row of the same dragon's previous turn in the same
    game, or -1. Rows are in turn order within a game, so a stable sort by
    (game, dragon) lines each dragon's turns up in order."""
    game = np.load(cache / "game.npy").astype(np.int64)
    dragon = np.load(cache / "dragon.npy").astype(np.int64)
    key = game * 100_000 + dragon
    order = np.argsort(key, kind="stable")
    prev = np.full(len(key), -1, np.int64)
    same = key[order[1:]] == key[order[:-1]]
    prev[order[1:][same]] = order[:-1][same]
    return prev


def _hist(cache: pathlib.Path, k: int) -> np.ndarray:
    """The last k moves (6 classes + "none" each), turns alive (/64, capped),
    and the length change since the previous turn (/3, clipped)."""
    prev = prev_index(cache)
    act = np.load(cache / "action.npy").astype(np.int64)
    length = np.load(cache / "scalar.npy", mmap_mode="r")[:, SC_LENGTH_RAW].astype(np.float32)
    cls = action_class(act)
    out = np.zeros((len(act), 7 * k + 2), np.float16)
    p = prev.copy()
    rows = np.arange(len(act))
    for j in range(k):
        has = p >= 0
        c = np.where(has, cls[np.maximum(p, 0)], 6)
        out[rows, 7 * j + c] = 1
        p = np.where(has, prev[np.maximum(p, 0)], -1)
    age = np.zeros(len(act), np.int64)
    for i in np.argsort(np.load(cache / "round.npy"), kind="stable"):
        if prev[i] >= 0:
            age[i] = age[prev[i]] + 1
    out[:, 7 * k] = np.minimum(age, 64) / 64
    has = prev >= 0
    dl = np.where(has, length - length[np.maximum(prev, 0)], 0)
    out[:, 7 * k + 1] = np.clip(dl, -3, 3) / 3
    return out


@feature("hist3", "scalar")
def hist3(cache):
    return _hist(cache, 3)


@feature("hist8", "scalar")
def hist8(cache):
    return _hist(cache, 8)


MEM_R = 6                       # memory map radius: 13x13 around the head
LC_PEARL, LC_PEARL_TIME, LC_NEVER_SPAWN = 0, 1, 2
LC_KELP_N = 13


def _ego_offsets(r: int):
    """World (dx, dy) of every ego cell (row, col) of a (2r+1)^2 grid, per
    facing, as bc_obs.hpp's ego_to_world does it."""
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]
    return [(ox, oy), (-oy, ox), (-ox, -oy), (oy, -ox)]


EGO7, EGOM = _ego_offsets(3), _ego_offsets(MEM_R)


def _mem_game(args):
    """mem planes for one game's rows (see mem())."""
    local, scalar, dragon, rnd = args
    n = len(dragon)
    out = np.zeros((n, 4, 2 * MEM_R + 1, 2 * MEM_R + 1), np.uint8)
    w = int(round(scalar[0, SC_MAP_W] * 64))
    h = int(round(scalar[0, SC_MAP_H] * 64))
    state = {}
    for i in range(n):
        d = int(dragon[i])
        if d not in state:
            # a dragon starts with no memory: a split child is a new process
            state[d] = dict(seen=np.full((h, w), -10_000, np.int32), pearl=np.zeros((h, w), bool),
                            cd=np.full((h, w), -1, np.int32), kelp=np.zeros((h, w), bool),
                            visit=np.full((h, w), -10_000, np.int32))
        s = state[d]
        sc = scalar[i]
        f = int(np.argmax(sc[SC_FACE_N:SC_FACE_N + 4]))
        hx, hy = int(round(sc[SC_HEAD_X] * w)) % w, int(round(sc[SC_HEAD_Y] * h)) % h
        r = int(rnd[i])
        dx, dy = EGO7[f]
        wx, wy = (hx + dx) % w, (hy + dy) % h
        lo = local[i].astype(np.int32)
        s["seen"][wy, wx] = r
        s["pearl"][wy, wx] = lo[LC_PEARL] > 0
        s["cd"][wy, wx] = np.where(lo[LC_NEVER_SPAWN] > 0, -1, lo[LC_PEARL_TIME])
        s["kelp"][wy, wx] |= lo[LC_KELP_N:LC_KELP_N + 4].max(0) > 0
        s["visit"][hy, hx] = r
        dx, dy = EGOM[f]
        wx, wy = (hx + dx) % w, (hy + dy) % h
        age = r - s["seen"][wy, wx]
        cd = s["cd"][wy, wx]
        expect = s["pearl"][wy, wx] | ((cd >= 0) & (cd <= age))
        known = age < 10_000
        out[i, 0] = expect & known
        out[i, 1] = np.where(known, np.rint(255 * np.exp(-age / 32)), 0)
        out[i, 2] = np.rint(255 * np.exp(-(r - s["visit"][wy, wx]) / 16))
        out[i, 3] = s["kelp"][wy, wx]
    out[:, 0] *= 255
    out[:, 3] *= 255
    return out.reshape(n, -1)


@feature("mem", "scalar")
def mem(cache):
    """What this dragon remembers of the 13x13 around its head, in its own
    frame: pearls it expects there (seen, or a respawn timer that has run
    out), how recently it saw each cell, where its own head has been, and
    kelp it has seen. uint8, x255."""
    import multiprocessing as mp
    game = np.load(cache / "game.npy")
    cut = np.flatnonzero(np.diff(game)) + 1
    bounds = list(zip(np.r_[0, cut], np.r_[cut, len(game)]))
    local = np.load(cache / "local.npy", mmap_mode="r")
    scalar = np.load(cache / "scalar.npy", mmap_mode="r")
    dragon = np.load(cache / "dragon.npy", mmap_mode="r")
    rnd = np.load(cache / "round.npy", mmap_mode="r")
    out = np.lib.format.open_memmap(cache / "feat_mem.npy", "w+", np.uint8,
                                    (len(game), 4 * (2 * MEM_R + 1) ** 2))
    jobs = ((np.asarray(local[a:b]), np.asarray(scalar[a:b]), np.asarray(dragon[a:b]),
             np.asarray(rnd[a:b])) for a, b in bounds)
    with mp.Pool(10) as pool:
        for (a, b), arr in zip(bounds, pool.imap(_mem_game, jobs, chunksize=2)):
            out[a:b] = arr
    return out


MEMFAR_K = 3


def _memfar_game(args):
    """memfar scalars for one game's rows (see memfar())."""
    local, scalar, dragon, rnd = args
    n = len(dragon)
    out = np.zeros((n, 4 * MEMFAR_K + 6), np.float16)
    w = int(round(scalar[0, SC_MAP_W] * 64))
    h = int(round(scalar[0, SC_MAP_H] * 64))
    ys, xs = np.mgrid[0:h, 0:w]
    state = {}
    for i in range(n):
        d = int(dragon[i])
        if d not in state:
            state[d] = dict(seen=np.full((h, w), -10_000, np.int32), pearl=np.zeros((h, w), bool),
                            cd=np.full((h, w), -1, np.int32))
        s = state[d]
        sc = scalar[i]
        f = int(np.argmax(sc[SC_FACE_N:SC_FACE_N + 4]))
        hx, hy = int(round(sc[SC_HEAD_X] * w)) % w, int(round(sc[SC_HEAD_Y] * h)) % h
        r = int(rnd[i])
        dx, dy = EGO7[f]
        wx, wy = (hx + dx) % w, (hy + dy) % h
        lo = local[i].astype(np.int32)
        s["seen"][wy, wx] = r
        s["pearl"][wy, wx] = lo[LC_PEARL] > 0
        s["cd"][wy, wx] = np.where(lo[LC_NEVER_SPAWN] > 0, -1, lo[LC_PEARL_TIME])
        known = s["seen"] > -10_000
        age = r - s["seen"]
        expect = known & (s["pearl"] | ((s["cd"] >= 0) & (s["cd"] <= age)))
        py, px = np.nonzero(expect)
        o = out[i]
        o[4 * MEMFAR_K + 4] = known.mean()
        o[4 * MEMFAR_K + 5] = min(len(py), 32) / 32
        if not len(py):
            continue
        # torus offsets to the head, then into the dragon's own frame
        ddx = (px - hx + w // 2) % w - w // 2
        ddy = (py - hy + h // 2) % h - h // 2
        if f == 0:
            ox, oy = ddx, ddy
        elif f == 1:
            ox, oy = ddy, -ddx
        elif f == 2:
            ox, oy = -ddx, -ddy
        else:
            ox, oy = -ddy, ddx
        dist = np.abs(ox) + np.abs(oy)
        near = np.argsort(dist, kind="stable")[:MEMFAR_K]
        for j, q in enumerate(near):
            o[4 * j:4 * j + 4] = (1, ox[q] / 16, oy[q] / 16, dist[q] / 32)
        wgt = 1 / (1 + dist)
        ahead, back = oy < 0, oy > 0
        cone_f = (-oy >= np.abs(ox)) & ahead
        cone_b = (oy >= np.abs(ox)) & back
        cone_l = (-ox > np.abs(oy))
        cone_r = (ox > np.abs(oy))
        base = 4 * MEMFAR_K
        for j, cone in enumerate((cone_f, cone_l, cone_r, cone_b)):
            o[base + j] = min(wgt[cone].sum(), 4) / 4
    return out


@feature("memfar", "scalar")
def memfar(cache):
    """The whole map as this dragon remembers it, summarised: the 3 nearest
    pearls it expects anywhere (present flag, ego dx, dy, distance), the
    1/(1+d)-weighted count of expected pearls ahead, left, right and behind,
    the share of the map it has seen and how many pearls it expects."""
    import multiprocessing as mp
    game = np.load(cache / "game.npy")
    cut = np.flatnonzero(np.diff(game)) + 1
    bounds = list(zip(np.r_[0, cut], np.r_[cut, len(game)]))
    local = np.load(cache / "local.npy", mmap_mode="r")
    scalar = np.load(cache / "scalar.npy", mmap_mode="r")
    dragon = np.load(cache / "dragon.npy", mmap_mode="r")
    rnd = np.load(cache / "round.npy", mmap_mode="r")
    out = np.zeros((len(game), 4 * MEMFAR_K + 6), np.float16)
    jobs = ((np.asarray(local[a:b]), np.asarray(scalar[a:b]), np.asarray(dragon[a:b]),
             np.asarray(rnd[a:b])) for a, b in bounds)
    with mp.Pool(9) as pool:
        for (a, b), arr in zip(bounds, pool.imap(_memfar_game, jobs, chunksize=2)):
            out[a:b] = arr
    return out


class MemoryTracker:
    """mem and memfar computed online, for many dragons at once (evaluation).

    Keys are any hashable per dragon, e.g. (env index, dragon id); forget()
    drops a finished game's keys. The result matches the offline builders
    above row for row (checked by feeding cached games through it one turn at
    a time: mem exactly, memfar to float16 rounding), which is what makes a
    checkpoint trained on them playable.
    """

    B = 64

    def __init__(self):
        self.slot: dict = {}
        self.free: list[int] = []
        cap = 256
        self._alloc(cap)

    def _alloc(self, cap):
        B = self.B
        old = getattr(self, "seen", None)
        new = dict(seen=np.full((cap, B, B), -10_000, np.int32), pearl=np.zeros((cap, B, B), bool),
                   cd=np.full((cap, B, B), -1, np.int32), kelp=np.zeros((cap, B, B), bool),
                   visit=np.full((cap, B, B), -10_000, np.int32))
        if old is not None:
            n = len(old)
            for k, v in new.items():
                v[:n] = getattr(self, k)
            self.free += list(range(n, cap))
        else:
            self.free = list(range(cap))
        for k, v in new.items():
            setattr(self, k, v)

    def _slot(self, key):
        s = self.slot.get(key)
        if s is None:
            if not self.free:
                self._alloc(2 * len(self.seen))
            s = self.free.pop()
            self.seen[s] = -10_000
            self.pearl[s] = False
            self.cd[s] = -1
            self.kelp[s] = False
            self.visit[s] = -10_000
            self.slot[key] = s
        return s

    def forget(self, pred):
        """Drops every key for which pred(key) is true."""
        for k in [k for k in self.slot if pred(k)]:
            self.free.append(self.slot.pop(k))

    def step(self, keys, local, scalar, rnd, want=("mem", "memfar")):
        """local: (n, 23, 7, 7) as the env gives it (floats) or the cache
        (uint8); returns {name: (n, k) float32} for the wanted features."""
        n = len(keys)
        s = np.array([self._slot(k) for k in keys])
        loc = np.asarray(local)
        if loc.dtype != np.uint8:
            pt = np.rint(loc[:, LC_PEARL_TIME] * 99)
        else:
            pt = loc[:, LC_PEARL_TIME].astype(np.float64)
        w = np.rint(scalar[:, SC_MAP_W] * 64).astype(np.int64)
        h = np.rint(scalar[:, SC_MAP_H] * 64).astype(np.int64)
        f = np.argmax(scalar[:, SC_FACE_N:SC_FACE_N + 4], 1)
        hx = np.rint(scalar[:, SC_HEAD_X] * w).astype(np.int64) % w
        hy = np.rint(scalar[:, SC_HEAD_Y] * h).astype(np.int64) % h
        r = np.asarray(rnd).astype(np.int64)
        e7x = np.stack([EGO7[k][0] for k in range(4)])[f]      # (n, 7, 7)
        e7y = np.stack([EGO7[k][1] for k in range(4)])[f]
        wx = (hx[:, None, None] + e7x) % w[:, None, None]
        wy = (hy[:, None, None] + e7y) % h[:, None, None]
        S = np.broadcast_to(s[:, None, None], wx.shape)
        self.seen[S, wy, wx] = r[:, None, None]
        self.pearl[S, wy, wx] = loc[:, LC_PEARL] > 0
        self.cd[S, wy, wx] = np.where(loc[:, LC_NEVER_SPAWN] > 0, -1, pt).astype(np.int32)
        self.kelp[S, wy, wx] |= loc[:, LC_KELP_N:LC_KELP_N + 4].max(1) > 0
        self.visit[s, hy, hx] = r
        out = {}
        if "mem" in want:
            emx = np.stack([EGOM[k][0] for k in range(4)])[f]
            emy = np.stack([EGOM[k][1] for k in range(4)])[f]
            mx = (hx[:, None, None] + emx) % w[:, None, None]
            my = (hy[:, None, None] + emy) % h[:, None, None]
            S = np.broadcast_to(s[:, None, None], mx.shape)
            age = r[:, None, None] - self.seen[S, my, mx]
            cd = self.cd[S, my, mx]
            known = age < 10_000
            expect = (self.pearl[S, my, mx] | ((cd >= 0) & (cd <= age))) & known
            m = np.zeros((n, 4) + mx.shape[1:], np.float32)
            m[:, 0] = expect
            m[:, 1] = np.where(known, np.rint(255 * np.exp(-age / 32)), 0) / 255
            m[:, 2] = np.rint(255 * np.exp(-(r[:, None, None] - self.visit[S, my, mx]) / 16)) / 255
            m[:, 3] = self.kelp[S, my, mx]
            out["mem"] = m.reshape(n, -1)
        if "memfar" in want:
            out["memfar"] = self._memfar(s, w, h, f, hx, hy, r)
        return out

    def _memfar(self, s, w, h, f, hx, hy, r):
        B = self.B
        n = len(s)
        seen, pearl, cd = self.seen[s], self.pearl[s], self.cd[s]
        yy, xx = np.mgrid[0:B, 0:B]
        inside = (xx[None] < w[:, None, None]) & (yy[None] < h[:, None, None])
        known = (seen > -10_000) & inside
        expect = known & (pearl | ((cd >= 0) & (cd <= r[:, None, None] - seen)))
        o = np.zeros((n, 4 * MEMFAR_K + 6), np.float32)
        o[:, 4 * MEMFAR_K + 4] = known.sum((1, 2)) / (w * h)
        # the rest only looks at expected pearls, a few per row: np.nonzero
        # lists them row-major over (y, x), the order the offline builder has
        row, py, px = np.nonzero(expect)
        cnt = np.bincount(row, minlength=n)
        o[:, 4 * MEMFAR_K + 5] = np.minimum(cnt, 32) / 32
        if len(row):
            W, H = w[row], h[row]
            ddx = (px - hx[row] + W // 2) % W - W // 2
            ddy = (py - hy[row] + H // 2) % H - H // 2
            F = f[row]
            ox = np.select([F == 0, F == 1, F == 2], [ddx, ddy, -ddx], -ddy)
            oy = np.select([F == 0, F == 1, F == 2], [ddy, -ddx, -ddy], ddx)
            dist = np.abs(ox) + np.abs(oy)
            order = np.lexsort((np.arange(len(row)), dist, row))
            start = np.r_[0, np.cumsum(cnt)[:-1]]
            rank = np.empty(len(row), np.int64)
            rank[order] = np.arange(len(row)) - start[row[order]]
            for j in range(MEMFAR_K):
                sel = rank == j
                rr = row[sel]
                o[rr, 4 * j] = 1
                o[rr, 4 * j + 1] = ox[sel] / 16
                o[rr, 4 * j + 2] = oy[sel] / 16
                o[rr, 4 * j + 3] = dist[sel] / 32
            wgt = 1 / (1 + dist)
            cones = [(-oy >= np.abs(ox)) & (oy < 0), (-ox > np.abs(oy)), (ox > np.abs(oy)),
                     (oy >= np.abs(ox)) & (oy > 0)]
            for j, cone in enumerate(cones):
                acc = np.zeros(n)
                np.add.at(acc, row[cone], wgt[cone])
                o[:, 4 * MEMFAR_K + j] = np.minimum(acc, 4) / 4
        return o.astype(np.float16).astype(np.float32)


@feature("msg", "scalar")
def msg(cache):
    """The first two sonar messages this turn. Most carry a 0xbca tag in the
    top 12 bits and a position (x = bits 14..19, y = bits 8..13, within 11
    cells of the receiver at the median, against 30 at random), so each gives:
    present, tagged, the position's torus offset in the dragon's own frame
    (/16), and the low byte (/64)."""
    return msg_rows(np.load(cache / "scalar.npy"), np.load(cache / "msgs.npy"))


def msg_rows(sc: np.ndarray, msgs: np.ndarray) -> np.ndarray:
    """msg for rows of scalars and raw sonar values (also used online)."""
    m = msgs.astype(np.int64)
    k = sc[:, 12].astype(np.int64)
    w, h = np.rint(sc[:, SC_MAP_W] * 64), np.rint(sc[:, SC_MAP_H] * 64)
    hx, hy = np.rint(sc[:, SC_HEAD_X] * w), np.rint(sc[:, SC_HEAD_Y] * h)
    f = np.argmax(sc[:, SC_FACE_N:SC_FACE_N + 4], 1)
    out = np.zeros((len(sc), 10), np.float16)
    for j in range(2):
        v = m[:, j]
        has = k > j
        tag = has & ((v >> 20) == 0xbca)
        x, y = (v >> 14) & 0x3f, (v >> 8) & 0x3f
        dx = (x - hx + w // 2) % w - w // 2
        dy = (y - hy + h // 2) % h - h // 2
        ox = np.select([f == 0, f == 1, f == 2], [dx, dy, -dx], -dy)
        oy = np.select([f == 0, f == 1, f == 2], [dy, -dx, -dy], dx)
        out[:, 5 * j:5 * j + 5] = np.stack(
            [has, tag, np.where(tag, ox / 16, 0), np.where(tag, oy / 16, 0),
             np.where(has, (v & 0xff) / 64, 0)], 1)
    return out


LC_SELF_HEAD, LC_SELF_BODY, LC_ALLY_HEAD, LC_ALLY_BODY, LC_ENEMY_HEAD, LC_ENEMY_BODY = 3, 4, 5, 6, 7, 8
LC_PORTAL_N, LC_SELF_TAIL = 17, 22
# ego steps: forward (up), right, back, left, as (drow, dcol); LC_KELP_N + k
# marks kelp on side k of a cell
STEP = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MOVES = [0, 3, 1]                # straight, left, right as STEP indices
BFS_MAX = 12


def _shift(x, dr, dc):
    """out[r, c] = x[r - dr, c - dc] (zero-filled), over the last two axes."""
    out = np.zeros_like(x)
    H = x.shape[-1]
    rs, rd = (slice(0, H - dr), slice(dr, H)) if dr >= 0 else (slice(-dr, H), slice(0, H + dr))
    cs, cd = (slice(0, H - dc), slice(dc, H)) if dc >= 0 else (slice(-dc, H), slice(0, H + dc))
    out[..., rd, cd] = x[..., rs, cs]
    return out


def _moves_chunk(loc: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """See moves(). loc: (n, 23, 7, 7) uint8 from the cache, mask its (n, 48)."""
    n = len(loc)
    pearl = loc[:, LC_PEARL] > 0
    occ = loc[:, [LC_SELF_BODY, LC_SELF_HEAD, LC_ALLY_HEAD, LC_ALLY_BODY,
                  LC_ENEMY_HEAD, LC_ENEMY_BODY]].max(1) > 0
    # the tail moves off its cell this turn (unless the move eats a pearl)
    free = ~occ | (loc[:, LC_SELF_TAIL] > 0)
    kelp = loc[:, LC_KELP_N:LC_KELP_N + 4] > 0
    portal = loc[:, LC_PORTAL_N:LC_PORTAL_N + 4] > 0
    # passable[k][r, c]: a step in direction k from (r, c) is allowed
    passable = [~kelp[:, k] & ~portal[:, k] for k in range(4)]
    enemy_head = loc[:, LC_ENEMY_HEAD] > 0
    ally_head = loc[:, LC_ALLY_HEAD] > 0
    cheb = lambda m: sum(_shift(m.astype(np.int8), dr, dc) for dr, dc in STEP)
    enemy_adj, ally_adj = cheb(enemy_head), cheb(ally_head)
    free_nb = sum(_shift(free.astype(np.int8), dr, dc) for dr, dc in STEP)
    # pearl timers: a cell whose pearl comes back within 10 turns
    soon = (loc[:, LC_PEARL_TIME] <= 10) & (loc[:, LC_NEVER_SPAWN] == 0) & ~pearl

    feats = []
    for m, k in enumerate(MOVES):
        dr, dc = STEP[k]
        r, c = 3 + dr, 3 + dc
        # the first step is allowed exactly when the mask allows it
        ok = mask[:, m] > 0
        reach = np.zeros((n, 7, 7), bool)
        reach[:, r, c] = ok
        # the head's old cell is now neck: never walk back through it
        blocked_head = np.zeros((7, 7), bool)
        blocked_head[3, 3] = True
        dist_pearl = np.full(n, BFS_MAX + 1, np.int32)
        dist_soon = np.full(n, BFS_MAX + 1, np.int32)
        dist_pearl[ok & pearl[:, r, c]] = 0
        dist_soon[ok & soon[:, r, c]] = 0
        area_at = {}
        for step in range(1, BFS_MAX + 1):
            grow = reach.copy()
            for kk, (sr, sc_) in enumerate(STEP):
                grow |= _shift(reach & passable[kk], sr, sc_)
            grow &= free & ~blocked_head
            reach = grow
            hitp = (reach & pearl).any((1, 2)) & (dist_pearl > BFS_MAX)
            dist_pearl[hitp] = step
            hits = (reach & soon).any((1, 2)) & (dist_soon > BFS_MAX)
            dist_soon[hits] = step
            if step in (3, 6):
                area_at[step] = reach.sum((1, 2))
        area = reach.sum((1, 2))
        # nearest pearl ignoring obstacles (Manhattan within the window)
        pr, pc = np.nonzero(np.ones((7, 7)))
        md = np.abs(pr - r) + np.abs(pc - c)
        man = np.where(pearl.reshape(n, -1), md[None], 99).min(1)
        f = [ok, ok & pearl[:, r, c],
             np.where(dist_pearl <= BFS_MAX, 1 - dist_pearl / (BFS_MAX + 1), 0),
             dist_pearl > BFS_MAX,
             np.where(dist_soon <= BFS_MAX, 1 - dist_soon / (BFS_MAX + 1), 0),
             area / 49, area_at[3] / 49, area_at[6] / 49,
             np.where(man < 99, 1 - man / 13, 0),
             enemy_adj[:, r, c] / 2, ally_adj[:, r, c] / 2, free_nb[:, r, c] / 4]
        feats += f
    return np.stack(feats, 1).astype(np.float16)


@feature("moves", "scalar")
def moves(cache):
    """Per candidate first step (straight, left, right), from the window
    alone: allowed, pearl on the cell, BFS distance to the nearest pearl and
    to a pearl respawning within 10 turns (through free cells, no kelp or
    portal edges, not back through the neck), whether none is reachable, free
    area reachable (in 12, 3 and 6 steps), Manhattan distance to the nearest
    pearl, enemy and ally heads next to the cell, and its free neighbours."""
    local = np.load(cache / "local.npy", mmap_mode="r")
    mask = np.load(cache / "mask.npy", mmap_mode="r")
    out = np.zeros((len(local), 36), np.float16)
    for s in range(0, len(local), 200_000):
        out[s:s + 200_000] = _moves_chunk(np.asarray(local[s:s + 200_000]),
                                          np.asarray(mask[s:s + 200_000]))
    return out


def load_features(cache: pathlib.Path, names: list[str], rows: np.ndarray):
    """(extra local planes, extra scalars): lists of (array, scale) for these
    rows, kept in their stored dtype; the trainer converts per batch."""
    loc, sc = [], []
    for n in names:
        kind, _ = FEATURES[n]
        arr = np.load(cache / f"feat_{n}.npy", mmap_mode="r")[rows]
        scale = 1 / 255 if arr.dtype == np.uint8 else 1.0
        (loc if kind == "local" else sc).append((arr, scale))
    return loc, sc


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--build", required=True, help="comma list of feature names")
    a = p.parse_args()
    cache = pathlib.Path(a.cache)
    for n in a.build.split(","):
        arr = FEATURES[n][1](cache)
        if not isinstance(arr, np.memmap):      # big features write their own file
            np.save(cache / f"feat_{n}.npy", arr)
        print(f"{n}: {arr.shape} {arr.dtype}", flush=True)


if __name__ == "__main__":
    main()
