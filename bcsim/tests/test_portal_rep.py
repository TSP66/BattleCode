"""The portal report (BC_PORTALREP, cpp/bc_sonar2.hpp) against an independent model of the map.

    BCSIM_LIB=bcsim/libbcvec_priv_s2_g15p.so python3 tests/test_portal_rep.py

Random legal play on portal maps, both teams speaking (odd envs: team 1 silent). Every turn:
  1. trip: python walks each dragon's move on its own parse of the map file and keeps that
     dragon's last single-portal trip; the packet the dragon would send now (s2_preview) must
     name exactly that entrance and direction, or no portal when it has none;
  2. received packets: the entrance is truly a portal side; a non-zero box field matches the
     true region behind it (closed by kelp and portal edges) bucket for bucket;
  3. planes: the dragon's own trip is painted on the tile across its entrance edge whenever
     that tile is inside the grid, every painted tile is across some portal edge, and a silent
     team draws nothing.
"""
import ctypes
import os
import pathlib
import sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("BCSIM_LIB", str(ROOT / "bcsim" / "libbcvec_priv_s2_g15p.so"))
sys.path.insert(0, str(ROOT))

import numpy as np                  # noqa: E402
import bcsim                        # noqa: E402
from bcsim import env as E          # noqa: E402
from train import sonar2 as S2      # noqa: E402

PT = slice(39, 43)                  # PT_KNOWN, PT_CLOSED, PT_ROOM, PT_PEARLS
UX, UY = (0, 1, 0, -1), (-1, 0, 1, 0)
BOX_MAX = 100


def bucket(n):
    return 0 if n <= 0 or n > BOX_MAX else (1 if n <= 4 else 2 if n <= 25 else 3)


class Map:
    """The map file, parsed here the way the engine's text format defines it (bc_text.hpp)."""

    def __init__(self, path):
        self.hk, self.vk, pid = {}, {}, {}
        for ln in open(path):
            t = ln.split()
            if not t:
                continue
            if t[0] == "MAP":
                self.w, self.h = int(t[1]), int(t[2])
            elif t[0] == "EDGE":
                idx, kind = int(t[1]), int(t[2])
                s = self.w + 1
                row, rem = divmod(idx, s)
                if rem == self.w or row > 2 * self.h:
                    continue
                vert = row & 1
                if not vert and row == 2 * self.h:
                    continue
                x, y = rem, ((row - 1) // 2 if vert else row // 2)
                (self.vk if vert else self.hk)[(x, y)] = kind
                if kind == 2:
                    pid.setdefault(int(t[3]), []).append((vert, x, y))
        self.partner = {}
        for ends in pid.values():
            assert len(ends) == 2
            self.partner[ends[0]] = ends[1]
            self.partner[ends[1]] = ends[0]

    def edge(self, x, y, d):           # (vertical, ex, ey) of tile (x, y)'s side d
        x, y = x % self.w, y % self.h
        return [(0, x, y), (1, (x + 1) % self.w, y), (0, x, (y + 1) % self.h), (1, x, y)][d]

    def kind(self, x, y, d):
        v, ex, ey = self.edge(x, y, d)
        return (self.vk if v else self.hk).get((ex, ey), 0)

    def step(self, x, y, d):           # -> (nx, ny, crossed a portal) or None for kelp
        k = self.kind(x, y, d)
        if k == 1:
            return None
        if k == 2:
            v, tx, ty = self.partner[self.edge(x, y, d)]
            if v == 0:
                ty -= 1 if d != 2 else 0
            else:
                tx -= 1 if d != 1 else 0
            return tx % self.w, ty % self.h, True
        return (x + UX[d]) % self.w, (y + UY[d]) % self.h, False

    def region(self, x, y):
        """How many tiles are reachable from (x, y) without crossing kelp or a portal (to BOX_MAX + 1)."""
        return len(self.tiles(x, y))

    def tiles(self, x, y):
        """The tiles reachable from (x, y) without crossing kelp or a portal, up to BOX_MAX + 1."""
        seen, q = {(x, y)}, [(x, y)]
        while q and len(seen) <= BOX_MAX:
            cx, cy = q.pop()
            for d in range(4):
                if self.kind(cx, cy, d) != 0:
                    continue
                n = ((cx + UX[d]) % self.w, (cy + UY[d]) % self.h)
                if n not in seen:
                    seen.add(n)
                    q.append(n)
        return seen


def dirs_of(action, facing):
    """Codec move id -> absolute step directions (bc_vec.hpp decode_for); [] for a non-move."""
    if action < 3:
        n, rest = 1, action
    elif action < 12:
        n, rest = 2, action - 3
    elif action < 39:
        n, rest = 3, action - 12
    else:
        return []
    out, f = [], facing
    for _ in range(n):
        turn = rest % 3
        rest //= 3
        f = f if turn == 0 else ((f + 3) % 4 if turn == 1 else (f + 1) % 4)
        out.append(f)
    return out


def ego(facing, wx, wy):                # world offset -> ego (bc_obs.hpp world_to_ego)
    return [(wx, wy), (wy, -wx), (-wx, -wy), (-wy, wx)][facing]


def wrap_off(d, m):
    d %= m
    return d - m if d > m // 2 else d


def main():
    names = ["portals", "maze", "stronghold", "trauma", "default", "schooltime", "help", "stripes",
             "autarky", "prisoners_dilemma"]
    paths = [ROOT.parent / "maps-train-official" / f"{n}.map" for n in names]
    loong = sorted((ROOT.parent / "maps-loong").glob("*.map"))
    paths += [p for p in loong if "EDGE" in p.read_text() and " 2 " in p.read_text()][:10]
    maps = bcsim.load_maps([str(p) for p in paths])
    models = [Map(p) for p in paths]
    N = 2 * len(paths)
    env = bcsim.BattlecodeVecEnv(maps, num_envs=N, num_threads=4, seed=5, board=True, grid=True, sonar=True)
    assert env.sonar2 and env.portal and env.grid.shape[1] in (54, 57), env.grid.shape
    E._lib.bcv_s2_preview.restype = ctypes.c_ulonglong
    E._lib.bcv_s2_preview.argtypes = [ctypes.c_void_p, ctypes.c_int]
    G = env.grid.shape[-1]
    HALF = G // 2
    mi = [i % len(paths) for i in range(N)]
    for i in range(N):
        env.set_opponent(i, team=-1, bot=0, map_index=mi[i])
        env.set_sonar2(i, (0, 1) if i % 2 == 0 else (0,))
    obs = env.reset()
    rng = np.random.default_rng(0)
    trip = {}                 # (env, uid) -> (ax, ay, dir) | None; absent = unknown
    pending = {}              # (env, uid) -> its last move's trip, or "keep" (none, or two portals)
    st = defaultdict(int)
    for step in range(12000):
        for e in range(N):
            M = models[mi[e]]
            W, H = M.w, M.h
            team, rnd = int(obs.team[e]), int(obs.round[e])
            uid = int(obs.uid[e])
            sc = obs.scalar[e]
            hx = int(round(sc[SCALAR("head_x")] * W)) % W
            hy = int(round(sc[SCALAR("head_y")] * H)) % H
            facing = int(np.argmax(sc[SCALAR("face_n"):SCALAR("face_n") + 4]))
            speaks = e % 2 == 0 or team == 0
            # a dragon's first observation comes before its first move: no trip yet
            key = (e, uid)
            if key in pending:
                if pending[key] != "keep":
                    trip[key] = pending[key]
                del pending[key]
            if key not in trip:
                trip[key] = None
            # 1. the packet it would send now names its last portal
            pkt = S2.decode(int(E._lib.bcv_s2_preview(ctypes.c_void_p(env._h), e)), team, rnd, portal=True)
            if speaks:
                assert pkt is not None
                want = trip[key]
                if want is None:
                    assert not pkt["portal"], (e, uid, pkt)
                    st["no_trip_ok"] += 1
                else:
                    assert pkt["portal"] and (pkt["px"], pkt["py"], pkt["pdir"]) == want, (e, uid, pkt, want)
                    st["trip_ok"] += 1
                    ax, ay, d = want
                    nx, ny, crossed = M.step(ax, ay, d)
                    assert crossed
                    if pkt["pbox"]:
                        assert bucket(M.region(nx, ny)) == pkt["pbox"], (pkt, M.region(nx, ny))
                        st["own_box"] += 1
                        st[f"own_box_b{pkt['pbox']}"] += 1
                        # pearls: memory's expectation against the board now (not exact: pearls
                        # eaten or spawned out of its sight since)
                        truth = min(3, sum(int(env.board[e, 4, ty, tx]) for tx, ty in M.tiles(nx, ny)))
                        st["own_pearls_agree"] += int(truth == pkt["ppearls"])
                    elif bucket(M.region(nx, ny)):
                        st["own_box_unproven"] += 1
                    # 3. painted on the tile across the entrance edge, when the grid covers it
                    bx, by = (ax + UX[d]) % W, (ay + UY[d]) % H
                    ox, oy = ego(facing, wrap_off(bx - hx, W), wrap_off(by - hy, H))
                    if -HALF <= ox < G - HALF and -HALF <= oy < G - HALF:
                        g = env.grid[e]
                        assert g[39, oy + HALF, ox + HALF] == 1.0, "own trip not painted"
                        assert g[40, oy + HALF, ox + HALF] >= (1.0 if pkt["pbox"] else 0.0)
                        st["own_painted"] += 1
            # every painted tile lies across a portal edge from one of its neighbours
            g = env.grid[e]
            if not speaks:
                assert np.abs(g[PT]).sum() == 0, "a silent team drew portal planes"
            for r, c in np.argwhere(g[39] > 0):
                wx, wy = ego((4 - facing) % 4, c - HALF, r - HALF)   # ego -> world: the inverse rotation
                bx, by = (hx + wx) % W, (hy + wy) % H
                assert any(M.kind(bx - UX[d], by - UY[d], d) == 2 for d in range(4)), (e, bx, by)
                st["painted"] += 1
            # 2. what arrived
            n = min(int(obs.num_msgs[e]), obs.msgs.shape[1])
            for v in obs.msgs[e, :n].tolist():
                p = S2.decode(v, team, rnd, portal=True)
                if p is None or not p["portal"]:
                    continue
                st["heard"] += 1
                if M.kind(p["px"], p["py"], p["pdir"]) != 2:
                    st["heard_not_portal"] += 1          # only a tag collision can do this
                    continue
                nx, ny, _ = M.step(p["px"], p["py"], p["pdir"])
                if p["pbox"]:
                    st["heard_box"] += 1
                    st["heard_box_bad"] += int(bucket(M.region(nx, ny)) != p["pbox"])
            st["turns"] += 1
        # random legal actions; the move each acting dragon makes, walked on the map model
        acts = np.zeros(N, np.int32)
        for e in range(N):
            legal = np.flatnonzero(obs.mask[e])
            acts[e] = rng.choice(legal) if len(legal) else 0
            M = models[mi[e]]
            sc = obs.scalar[e]
            x = int(round(sc[SCALAR("head_x")] * M.w)) % M.w
            y = int(round(sc[SCALAR("head_y")] * M.h)) % M.h
            facing = int(np.argmax(sc[SCALAR("face_n"):SCALAR("face_n") + 4]))
            ds = dirs_of(int(acts[e]), facing)
            crossings, last = 0, None
            for d in ds:
                s = M.step(x, y, d)
                if s is None:
                    break
                if s[2]:
                    crossings += 1
                    last = (x, y, d)
                x, y = s[0], s[1]
            pending[(e, int(obs.uid[e]))] = last if crossings == 1 else "keep"
            st["cross1"] += int(crossings == 1)
            st["cross2"] += int(crossings >= 2)
        obs, _, _ = env.step(acts)
        if step % 2000 == 0:
            print(step, dict(st), flush=True)
    print(dict(st))
    assert st["trip_ok"] > 1000 and st["no_trip_ok"] > 1000
    assert st["own_painted"] > 500 and st["own_box"] > 50, "the test never saw a box"
    assert st["heard"] > 1000
    assert st["heard_not_portal"] <= 0.01 * st["heard"]
    assert st["heard_box_bad"] <= 0.01 * max(1, st["heard_box"])
    assert st["own_pearls_agree"] >= 0.8 * st["own_box"], "box pearl counts disagree with the board"
    print("OK")


def SCALAR(name):
    return bcsim.SCALARS.index(name)


if __name__ == "__main__":
    main()
