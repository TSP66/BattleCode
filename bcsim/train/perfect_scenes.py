"""Synthetic perfect-play positions (supervised_learning.md; user 2026-10-02: there is a finite set
of situations, so generate them -- and never on the training maps, the point is generalisation).

Every pass draws fresh random maps (random torus size, kelp density, kelp walls, portal pairs,
pearl timers) and builds each situation on them directly: snake-shaped bodies by random walk, the
acting dragon's surroundings placed in its own frame. The simulator then judges every position
(VecEnv::perfect, cpp/bc_obs.hpp, checked by playing in tests/test_perfect.py): a scene counts as
whatever it labels, so a construction that misses (a wall with a hole, a path blocked by kelp) is
simply not kept. Nothing here decides a label.

    scenes = SceneMaker(rng)
    texts, widths = scenes.maps(n)                       # the maps, one env pinned to each
    spec = scenes.scene(map_index, kind)                 # -> (round, dragons, pearls, acting) or None
    env.set_scenario(i, *spec)
"""

from __future__ import annotations

import numpy as np

from train import augment as A

DV = [(0, -1), (1, 0), (0, 1), (-1, 0)]          # world N E S W
# "blank" (straight on) is no longer trained (user, 2026-10-05); queen_deadend added the same day
KINDS = ["queen_kill", "late_suicide", "trapped_queen", "pearl", "keep_wall", "queen_deadend"]
LATE = (481, 498)


def ego_to_world(f: int, ox: int, oy: int):
    """cpp/bc_obs.hpp ego_to_world: ox right, oy back, the dragon facing f (0 N .. 3 W)."""
    return [(ox, oy), (-oy, ox), (-ox, -oy), (oy, -ox)][f]


class SceneMap:
    def __init__(self, m: A.Map):
        self.m, self.w, self.h = m, m.w, m.h

    def wrap(self, x, y):
        return x % self.w, y % self.h

    def open(self, x, y, d) -> bool:
        """The edge on side d of (x, y) is open (no kelp, no portal)."""
        m = self.m
        if d == 0:
            k = m.hk[y, x]
        elif d == 2:
            k = m.hk[(y + 1) % self.h, x]
        elif d == 3:
            k = m.vk[y, x]
        else:
            k = m.vk[y, (x + 1) % self.w]
        return k == A.OPEN

    def step(self, c, d):
        return self.wrap(c[0] + DV[d][0], c[1] + DV[d][1])


class SceneMaker:
    def __init__(self, rng):
        self.rng = rng
        self.smaps: list[SceneMap] = []
        self._tubes: dict = {}                       # map name -> carve_tubes()
        self.tubes: list = []                        # per map index, set by maps()

    # ------------------------------------------------------------ maps
    def random_map(self, name: str) -> A.Map:
        rng = self.rng
        w, h = int(rng.integers(12, 41)), int(rng.integers(12, 41))
        hk = np.zeros((h, w), np.int64)
        vk = np.zeros((h, w), np.int64)
        dens = float(rng.choice([0.0, 0.02, 0.05, 0.1, 0.18, 0.28]))
        hk[rng.random((h, w)) < dens] = A.KELP
        vk[rng.random((h, w)) < dens] = A.KELP
        for _ in range(int(rng.integers(0, 8))):          # straight kelp walls
            x, y, n = int(rng.integers(w)), int(rng.integers(h)), int(rng.integers(2, 10))
            if rng.random() < 0.5:
                for i in range(n):
                    hk[y, (x + i) % w] = A.KELP
            else:
                for i in range(n):
                    vk[(y + i) % h, x] = A.KELP
        hp = np.full((h, w), -1, np.int64)
        vp = np.full((h, w), -1, np.int64)
        if rng.random() < 0.4:                            # portal pairs, one orientation each
            pid = 0
            for _ in range(int(rng.integers(1, 4))):
                ks, ps = (hk, hp) if rng.random() < 0.5 else (vk, vp)
                cells = [(int(rng.integers(h)), int(rng.integers(w))) for _ in range(2)]
                if cells[0] == cells[1] or any(ks[c] != A.OPEN for c in cells):
                    continue
                for c in cells:
                    ks[c], ps[c] = A.PORTAL, pid
                pid += 1
        spawn = rng.random((h, w)) < float(rng.uniform(0.2, 1.0))
        lo = int(rng.integers(3, 30))
        min_gap = np.where(spawn, lo, 0).astype(np.int64)
        max_gap = np.where(spawn, lo + int(rng.integers(0, 40)), 0).astype(np.int64)
        tubes = self.carve_tubes(w, h, hk, vk, hp, vp) if rng.random() < 0.8 else []
        m = A.Map(w, h, A.SYM_NONE, name, min_gap, max_gap, hk, vk, hp, vp, [])
        m.dragons = self.spawns(SceneMap(m))
        self._tubes[name] = tubes
        return m

    def carve_tubes(self, w, h, hk, vk, hp, vp):
        """Dead-end tubes (queen_deadend, user 2026-10-05): from a mouth tile, depth 1-4 tiles in a
        straight line with kelp on both sides and at the far end, the edges along it open.
        Returns [(mouth (x, y), direction into the tube, depth)]."""
        rng = self.rng

        def edge(x, y, d):
            x, y = x % w, y % h
            return {0: (hk, y, x), 2: (hk, (y + 1) % h, x), 3: (vk, y, x), 1: (vk, y, (x + 1) % w)}[d]

        def set_edge(x, y, d, kind):
            x, y = x % w, y % h
            if d == 0:
                hk[y, x], hp[y, x] = kind, -1
            elif d == 2:
                hk[(y + 1) % h, x], hp[(y + 1) % h, x] = kind, -1
            elif d == 3:
                vk[y, x], vp[y, x] = kind, -1
            else:
                vk[y, (x + 1) % w], vp[y, (x + 1) % w] = kind, -1
        out = []
        for _ in range(int(rng.integers(1, 6))):
            mx, my = int(rng.integers(w)), int(rng.integers(h))
            d = int(rng.integers(4))
            k = int(rng.integers(1, 5))
            tiles = [((mx + DV[d][0] * i) % w, (my + DV[d][1] * i) % h) for i in range(1, k + 1)]
            touched = [(mx, my, d)] + [(tx, ty, dd) for tx, ty in tiles for dd in range(4)]
            if any(ka[yy, xx] == A.PORTAL for ka, yy, xx in (edge(*t) for t in touched)):
                continue                                      # never break a portal pair
            # half the time an approach corridor (stripes, maze): 1-3 tiles behind the mouth walled on
            # both sides, and the mouth itself walled on one side, so the only way on is the other side
            approach = int(rng.integers(1, 4)) if rng.random() < 0.5 else 0
            back = (d + 2) % 4
            app = [((mx + DV[back][0] * i) % w, (my + DV[back][1] * i) % h) for i in range(1, approach + 1)]
            touched += [(ax, ay, dd) for ax, ay in app for dd in range(4)] + [(mx, my, dd) for dd in range(4)]
            if any(ka[yy, xx] == A.PORTAL for ka, yy, xx in (edge(*t) for t in touched)):
                continue
            if approach:
                for ax, ay in app:
                    set_edge(ax, ay, (d + 1) % 4, A.KELP)
                    set_edge(ax, ay, (d + 3) % 4, A.KELP)
                    set_edge(ax, ay, d, A.OPEN)
                side = (d + 1) % 4 if rng.random() < 0.5 else (d + 3) % 4
                set_edge(mx, my, side, A.OPEN)
                set_edge(mx, my, (side + 2) % 4, A.KELP)
            set_edge(mx, my, d, A.OPEN)                       # the mouth
            for i, (tx, ty) in enumerate(tiles):
                set_edge(tx, ty, (d + 1) % 4, A.KELP)
                set_edge(tx, ty, (d + 3) % 4, A.KELP)
                set_edge(tx, ty, d, A.KELP if i == k - 1 else A.OPEN)
            out.append(((mx, my), d, k, approach))
        return out

    def spawns(self, sm: SceneMap):
        """Starting dragons for ordinary games on the map (the KL set and memory donors; a scene
        replaces them): 1-6 a team, 2-5 long, alternating teams so ids 0 and 1 are the two queens."""
        rng = self.rng
        n = int(rng.integers(1, 7))
        used: set = set()
        out = []
        for i in range(2 * n):
            for _ in range(20):
                b = self.walk(sm, self.free_cell(sm, used) or (0, 0), int(rng.integers(2, 6)), set(used))
                if b and len(b) >= 2:
                    used |= set(b)
                    out.append((i % 2, b))
                    break
            else:
                if i < 2:                                   # the two queens must exist
                    raise RuntimeError("no room for a queen")
                break
        if len(out) % 2:                                    # equal teams
            out.pop()
        return out

    def maps(self, n: int):
        """n fresh maps: (texts, widths). Scene j must use map j's index."""
        ms = []
        while len(ms) < n:
            try:
                ms.append(self.random_map(f"synthetic_{len(ms)}"))
            except RuntimeError:
                continue
        self.smaps = [SceneMap(m) for m in ms]
        self.tubes = [self._tubes.get(m.name, []) for m in ms]
        return [A.render(m) for m in ms], [m.w for m in ms]

    # ------------------------------------------------------------ bodies
    def walk(self, sm: SceneMap, head, length: int, used: set, first=None):
        """A body of up to `length` cells, head first, each next cell one open step from the last."""
        rng = self.rng
        if head in used:
            return None
        body = [head]
        used.add(head)
        d_prev = None
        for i in range(1, length):
            c = body[-1]
            ds = [d for d in range(4) if sm.open(c[0], c[1], d) and sm.step(c, d) not in used]
            if i == 1 and first is not None:
                ds = [d for d in ds if d == first]
            if not ds:
                break
            if d_prev in ds and rng.random() < 0.5:
                d = d_prev
            else:
                d = int(rng.choice(ds))
            n = sm.step(c, d)
            body.append(n)
            used.add(n)
            d_prev = d
        return body

    def facing(self, sm: SceneMap, body):
        """The way a body faces: its neck to its head (random for one cell: the simulator draws it)."""
        if len(body) < 2:
            return None
        for d in range(4):
            if sm.step(body[1], d) == body[0]:
                return d
        return None

    def far_cell(self, sm: SceneMap, centre, used):
        """A free cell outside the 7x7 window about centre (if the map leaves room)."""
        for _ in range(50):
            c = (int(self.rng.integers(sm.w)), int(self.rng.integers(sm.h)))
            dx = min((c[0] - centre[0]) % sm.w, (centre[0] - c[0]) % sm.w)
            dy = min((c[1] - centre[1]) % sm.h, (centre[1] - c[1]) % sm.h)
            if max(dx, dy) > 3 and c not in used:
                return c
        return None

    def free_cell(self, sm: SceneMap, used):
        for _ in range(50):
            c = (int(self.rng.integers(sm.w)), int(self.rng.integers(sm.h)))
            if c not in used:
                return c
        return None

    # ------------------------------------------------------------ scenes
    def scene(self, mi: int, kind: str):
        """(round, dragons [(team, body)], pearls [(x, y)], acting index) or None."""
        rng, sm = self.rng, self.smaps[mi]
        used: set = set()
        us = int(rng.integers(2))                           # our team
        bodies: dict = {}                                   # role -> (team, body)
        pearls: list = []
        rnd = int(rng.integers(0, 500))
        L = lambda lo, hi: int(rng.integers(lo, hi + 1))    # noqa: E731

        def actor(length, used_):
            c = self.free_cell(sm, used_)
            return None if c is None else self.walk(sm, c, length, used_)

        if kind in ("blank", "pearl"):
            a_is_queen = rng.random() < 0.25
            body = actor(L(1, 24), used)
            if body is None:
                return None
            bodies["actor"] = (us, body)
            if kind == "pearl":
                f = self.facing(sm, body)
                f = int(rng.integers(4)) if f is None else f
                for d in rng.permutation([f, (f + 3) % 4, (f + 1) % 4])[:L(1, 3)]:
                    c = sm.step(body[0], int(d))
                    if c not in used and sm.open(body[0][0], body[0][1], int(d)):
                        pearls.append(c)
                for _ in range(L(0, 3)):                    # more pearls about
                    ox, oy = L(-3, 3), L(-3, 3)
                    c = sm.wrap(body[0][0] + ox, body[0][1] + oy)
                    if c not in used and c not in pearls:
                        pearls.append(c)
            role_q = "actor" if a_is_queen else None
            far_others = True
        elif kind == "queen_deadend":
            # our queen at (or 1-2 tiles before) a tube's mouth, facing in; often a pearl inside
            tubes = self.tubes[mi] if mi < len(self.tubes) else []
            if not tubes:
                return None
            (mx, my), d, k, approach = tubes[int(rng.integers(len(tubes)))]
            back = (d + 2) % 4
            head = (mx, my)
            for _ in range(L(0, max(2, approach))):
                if not sm.open(head[0], head[1], back):
                    break
                head = sm.step(head, back)
            tube = [sm.wrap(mx + DV[d][0] * i, my + DV[d][1] * i) for i in range(1, k + 1)]
            if head in tube:
                return None
            used |= set(tube)                                 # nobody in the tube
            used.discard(head)
            body = self.walk(sm, head, L(2, 24), used, first=back)
            if body is None or len(body) < 2:
                return None
            bodies["actor"] = (us, body)
            if rng.random() < 0.85:
                pearls.append(tube[int(rng.integers(k))] if rng.random() < 0.5 else tube[-1])
            for c in tube:
                used.discard(c)
            used |= {c for c in tube}                         # keep the tube empty of other dragons
            role_q = "actor"
            far_others = rng.random() < 0.5
        elif kind == "queen_kill":
            body = actor(L(2, 24), used)
            if body is None:
                return None
            bodies["actor"] = (us, body)
            f = self.facing(sm, body)
            f = int(rng.integers(4)) if f is None else f
            for _ in range(20):
                ox, oy = L(-3, 3), L(-3, 3)
                if 1 <= abs(ox) + abs(oy) <= 6:
                    break
            wx, wy = ego_to_world(f, ox, oy)
            qb = self.walk(sm, sm.wrap(body[0][0] + wx, body[0][1] + wy), L(1, 24), used)
            if qb is None:
                return None
            bodies["their_q"] = (1 - us, qb)
            role_q = None
            far_others = False
        elif kind == "late_suicide":
            rnd = L(*LATE)
            body = actor(L(2, 24), used)
            if body is None:
                return None
            bodies["actor"] = (us, body)
            ox, oy = L(-2, 2), L(-2, 2)
            qb = self.walk(sm, sm.wrap(body[0][0] + ox, body[0][1] + oy), L(1, 30), used)
            if qb is None:
                return None
            bodies["our_q"] = (us, qb)
            role_q = None
            far_others = True                                # no enemy in sight
        elif kind in ("trapped_queen", "keep_wall"):
            q_team = us if kind == "trapped_queen" else 1 - us
            role = "our_q" if q_team == us else "their_q"
            c = self.free_cell(sm, used)
            if c is None:
                return None
            # the queen: head c, neck n, the rest of her body away from the wall
            nd = int(rng.integers(4))
            if not sm.open(c[0], c[1], nd):
                return None
            n = sm.step(c, nd)
            qf = (nd + 2) % 4                                # she faces away from her neck
            lf, rf = (qf + 3) % 4, (qf + 1) % 4
            # the U around her head: left, front-left, front, front-right, right
            cl, cf, cr = sm.step(c, lf), sm.step(c, qf), sm.step(c, rf)
            chain = [cl, sm.step(cl, qf), cf, sm.step(cr, qf), cr]
            if len(set(chain + [c, n])) != 7:
                return None
            for a_, b_ in zip(chain, chain[1:]):             # the wall must be one body: open links
                if not any(sm.step(a_, d) == b_ and sm.open(a_[0], a_[1], d) for d in range(4)):
                    return None
            if rng.random() < 0.5:
                chain = chain[::-1]
            if any(x in used for x in chain + [c, n]):
                return None
            used |= {c} | set(chain)
            qb = [c] + (self.walk(sm, n, L(1, 20), used) or [])
            if len(qb) < 2:
                return None
            bodies[role] = (q_team, qb)
            # the wall dragon: a short way on past the chain's first cell (its head), a longer tail
            # behind its last
            used.discard(chain[0])
            head_ext = (self.walk(sm, chain[0], L(0, 2) + 1, used) or [chain[0]])[1:]
            used.add(chain[0])
            used.discard(chain[-1])
            tail_ext = (self.walk(sm, chain[-1], L(2, 12) + 1, used) or [chain[-1]])[1:]
            used.add(chain[-1])
            wall = head_ext[::-1] + chain + tail_ext
            a_team = us
            bodies["actor"] = (a_team, wall)
            # keep_wall: sometimes our own queen is the wall
            role_q = "actor" if (kind == "keep_wall" and rng.random() < 0.2) else None
            far_others = False
        else:
            raise ValueError(kind)

        # the queens: ids 0 (team 0) and 1 (team 1); a missing one is placed (or dead)
        queens = {}
        for team in (0, 1):
            mine = team == us
            key = "our_q" if mine else "their_q"
            if mine and role_q == "actor":
                queens[team] = "actor"
                continue
            if key in bodies:
                queens[team] = key
                continue
            if rng.random() < 0.15:
                bodies[key] = (team, [])                     # already dead
            else:
                head = self.far_cell(sm, bodies["actor"][1][0], used) if far_others or not mine \
                    else self.free_cell(sm, used)
                if head is None:
                    return None
                bodies[key] = (team, self.walk(sm, head, L(1, 24), used))
            queens[team] = key
        # a few more dragons; out of the window where the scene wants nobody else in sight
        extra = []
        for _ in range(L(0, 4)):
            team = int(rng.integers(2))
            head = self.far_cell(sm, bodies["actor"][1][0], used) if far_others else self.free_cell(sm, used)
            if head is None:
                break
            b = self.walk(sm, head, L(1, 16), used)
            if b:
                extra.append((team, b))
        rest = [bodies["actor"]] if queens[0] != "actor" and queens[1] != "actor" else []
        rest += extra
        order = list(rng.permutation(len(rest)))
        dragons = [bodies[queens[0]], bodies[queens[1]]] + [rest[i] for i in order]
        if queens[0] == "actor":
            acting = 0
        elif queens[1] == "actor":
            acting = 1
        else:
            acting = 2 + order.index(0)
        for _ in range(L(0, 6)):                            # pearls scattered over the map
            c = self.free_cell(sm, used | set(pearls))
            if c is None or (kind == "blank" and not self._outside(sm, bodies["actor"][1][0], c)):
                continue
            pearls.append(c)
        pearls = [p for p in pearls if p not in used]
        return rnd, dragons, pearls, acting

    @staticmethod
    def _outside(sm: SceneMap, centre, c) -> bool:
        dx = min((c[0] - centre[0]) % sm.w, (centre[0] - c[0]) % sm.w)
        dy = min((c[1] - centre[1]) % sm.h, (centre[1] - c[1]) % sm.h)
        return max(dx, dy) > 3
