"""Is the shared-map channel exact, and does a newborn actually get a pearl?

train/memfeat.py claims three things, and each is checked here against the
simulator rather than against itself:

  1. THE FRAME IS RIGHT. The packet carries absolute board cells, recovered from
     memfar's ego offsets and the dragon's own normalised head position. If any
     part of that -- the head recovery, the rotation by facing, the torus wrap --
     is wrong, cells come out somewhere else. The check does not trust any
     rederivation: `never_spawn` is a STATIC property of a tile, so every dragon
     that ever sees that tile, from any pose, in any frame, must report the same
     value for the same absolute cell. One conflict means the transform is wrong.
     (This is the test that would have caught an inverted rotation, which reading
     the code twice did not: see the note in KNOWN_ISSUES about direction_between.)

  2. IT IS A NO-OP WHEN NOTHING ARRIVES. With an empty inbox the merged scalar
     row must be BIT-IDENTICAL to the one the simulator wrote. 43% of children
     hear nothing, so "no message" is the common case and it has to leave the
     policy's input alone rather than nearly alone.

  3. IT REACHES THE CHILD. What a newborn knows before the merge and after it.
     A child is NOT blind at birth even with no channel: bc_obs.hpp feeds its own
     7x7 view into its fresh memory before building features, so it already knows
     the pearls it can see -- 86% of children have one. The number that matters is
     therefore not "has a pearl" but "has a pearl it CANNOT SEE", and how often a
     child with nothing at all is given something.

Every invariant of the merged block is asserted too: distance equals the
Manhattan norm of the offsets it was written with, slots are nearest-first,
valid slots are a prefix, no slot names a cell off the board, and the pearl count
is consistent with the slots.

    python3 tests/test_memfeat.py [maps_dir]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                    # noqa: E402
from train import memfeat as mf                 # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
NEVER_SPAWN = bcsim.CHANNELS.index("never_spawn")
VISION = bcsim.WINDOW // 2


def random_actions(obs, rng):
    legal = obs.mask.astype(np.float64) + 1e-12
    return np.array([rng.choice(len(r), p=r / r.sum()) for r in legal], np.int32)


# ------------------------------------------------------------------ 1. the bits
def bits() -> bool:
    """encode/decode on synthetic rows: every field survives, nothing spills."""
    rng = np.random.default_rng(7)
    n = 4096
    sc = np.zeros((n, bcsim.N_SCALARS), np.float32)
    w = rng.integers(11, 65, n)
    h = rng.integers(11, 65, n)
    hx = (rng.random(n) * w).astype(np.int64)
    hy = (rng.random(n) * h).astype(np.int64)
    facing = rng.integers(0, 4, n)
    team = rng.integers(0, 2, n)
    sc[:, mf.SC_MAP_W] = w / 64.0
    sc[:, mf.SC_MAP_H] = h / 64.0
    sc[:, mf.SC_HEAD_X] = hx / w
    sc[:, mf.SC_HEAD_Y] = hy / h
    sc[np.arange(n), mf.SC_FACE_N + facing] = 1.0
    sc[:, mf.SC_TEAM_B] = team

    # three pearls per row at known absolute cells, written into memfar the way
    # the simulator writes them
    px = (rng.random((n, 3)) * w[:, None]).astype(np.int64)
    py = (rng.random((n, 3)) * h[:, None]).astype(np.int64)
    valid = rng.random((n, 3)) < 0.7
    ex, ey, dist = mf.abs_to_ego(px, py, hx[:, None], hy[:, None], w[:, None], h[:, None],
                                 facing[:, None])
    for j in range(3):
        sc[:, mf.FAR_AT + 4 * j] = valid[:, j]
        sc[:, mf.FAR_AT + 4 * j + 1] = np.where(valid[:, j], ex[:, j] / 16.0, 0.0)
        sc[:, mf.FAR_AT + 4 * j + 2] = np.where(valid[:, j], ey[:, j] / 16.0, 0.0)
        sc[:, mf.FAR_AT + 4 * j + 3] = np.where(valid[:, j], dist[:, j] / 32.0, 0.0)

    pay = mf.encode(sc)
    ok, s_team, dv, dx, dy = mf.decode(pay)
    bad = 0
    if not ok.all():
        bad += 1
        print("  FAIL every payload should carry the tag")
    if not (s_team == team).all():
        bad += 1
        print("  FAIL the sender's team did not survive")
    if mf.reserved(pay).any():
        bad += 1
        print("  FAIL reserved bits are not zero")
    if not (dv == valid).all():
        bad += 1
        print("  FAIL a valid flag changed")
    if not ((dx == px) | ~valid).all() or not ((dy == py) | ~valid).all():
        bad += 1
        print("  FAIL an absolute cell did not survive the round trip")
    # and back into a DIFFERENT dragon's frame: the cell is the same cell
    ox, oy, _ = mf.abs_to_ego(dx, dy, hx[:, None], hy[:, None], w[:, None], h[:, None],
                              facing[:, None])
    if not ((ox == ex) | ~valid).all() or not ((oy == ey) | ~valid).all():
        bad += 1
        print("  FAIL ego -> absolute -> ego is not the identity")

    # the rotation on its own, over every facing and both signs, including cells
    # that only reach by wrapping
    f = np.arange(4)
    for d in (-31, -7, -1, 0, 1, 5, 30):
        for e in (-30, -2, 0, 3, 29):
            a, b = mf._rot_from_ego(f, np.full(4, d), np.full(4, e))
            c, g = mf._rot_to_ego(f, a, b)
            if not ((c == d).all() and (g == e).all()):
                bad += 1
                print(f"  FAIL rotation is not invertible at ({d}, {e})")
    # cones: a sender's forward/left/right/behind must land on the receiver's
    # forward/left/right/behind, turned by the difference in facing. Sent as a
    # compass bearing, so this is the one place the rotation is done twice by
    # different dragons and has to cancel.
    cone = np.rint(rng.random((n, 4)) * mf.CONE_LEVELS) / mf.CONE_LEVELS
    sc[:, mf.FAR_CONES:mf.FAR_CONES + 4] = cone
    pay = mf.encode(sc)
    rf = rng.integers(0, 4, n)                  # the receiver's facing
    rx = np.zeros((n, bcsim.N_SCALARS), np.float32)
    rx[:, mf.SC_MAP_W], rx[:, mf.SC_MAP_H] = w / 64.0, h / 64.0
    rx[np.arange(n), mf.SC_FACE_N + rf] = 1.0
    rx[:, mf.SC_TEAM_B] = team
    mf.merge(rx, pay[:, None], np.ones(n, np.int32))
    got = rx[:, mf.FAR_CONES:mf.FAR_CONES + 4]
    # sender ego jj sits at bearing (facing + R[jj]); receiver ego j reads
    # bearing (rf + R[j]); so j takes jj with R[jj] = (rf - facing + R[j]) % 4
    want = np.zeros_like(got)
    for j in range(4):
        r = (rf - facing + mf.R_OF_EGO[j]) % 4
        want[:, j] = np.take_along_axis(cone, mf.EGO_OF_R[r][:, None], axis=1)[:, 0]
    if not np.allclose(got, want, atol=1e-3):
        bad += 1
        print(f"  FAIL the cones arrive turned the wrong way "
              f"(max error {np.abs(got - want).max():.3f})")
    print(f"  {n:,} synthetic packets: {'ok' if not bad else f'{bad} failures'}")
    return bad == 0


# ----------------------------------------------------- 2. the frame, vs the sim
def frame(maps, steps: int = 900, envs: int = 32, seed: int = 3) -> bool:
    """Static terrain must read the same from every pose. Also checks the
    window: a cell the dragon can see maps to an absolute cell, and two dragons
    in one game that both see it have to agree."""
    env = bcsim.BattlecodeVecEnv(maps, num_envs=envs, num_threads=4, seed=seed)
    rng = np.random.default_rng(seed)
    obs = env.reset()
    # per env, per absolute cell: the never_spawn value first reported there.
    # -1 is "nobody has looked yet"; an episode boundary resets the env's map.
    table = [{} for _ in range(envs)]
    rounds = np.zeros(envs, np.int64)
    checked = conflicts = 0
    for _ in range(steps):
        hx, hy, w, h, facing, _ = mf.pose(obs.scalar)
        # a new episode means a new map, so forget what was learned about it
        r = obs.round.astype(np.int64)
        for i in np.flatnonzero(r < rounds):
            table[i] = {}
        rounds = r
        oy, ox = np.meshgrid(np.arange(-VISION, VISION + 1), np.arange(-VISION, VISION + 1),
                             indexing="ij")
        for i in range(envs):
            if obs.mask[i].sum() == 0:
                continue
            ax, ay = mf.ego_to_abs(ox.ravel(), oy.ravel(), hx[i], hy[i], w[i], h[i], facing[i])
            val = obs.local[i, NEVER_SPAWN].ravel()
            t = table[i]
            for cell, v in zip((ax * 64 + ay).tolist(), val.tolist()):
                checked += 1
                prev = t.get(cell)
                if prev is None:
                    t[cell] = v
                elif prev != v:
                    conflicts += 1
                    t[cell] = v
        obs, _, _ = env.step(random_actions(obs, rng))
    env.close()
    print(f"  {checked:,} window cells mapped to absolute coords, "
          f"{conflicts:,} disagreements")
    if conflicts:
        print("  FAIL two poses reported different static terrain for one cell, so "
              "the head recovery, the rotation or the wrap is wrong")
    return conflicts == 0


# ------------------------------------------------- 3. no message, no difference
def identity(maps, steps: int = 400, envs: int = 64, seed: int = 11) -> bool:
    """With nothing in the inbox the merge must not move a single bit."""
    env = bcsim.BattlecodeVecEnv(maps, num_envs=envs, num_threads=4, seed=seed)
    rng = np.random.default_rng(seed)
    obs = env.reset()
    diffs = 0
    for _ in range(steps):
        if obs.num_msgs.any():
            raise SystemExit("this env should be silent; something cast a sonar")
        before = obs.scalar.copy()
        mf.merge(obs.scalar, obs.msgs, obs.num_msgs)
        diffs += int((obs.scalar.view(np.uint32) != before.view(np.uint32)).sum())
        obs, _, _ = env.step(random_actions(obs, rng))
    env.close()
    print(f"  {steps * envs:,} silent turns merged: {diffs} changed words")
    if diffs:
        print("  FAIL the merge is not a no-op on an empty inbox")
    return diffs == 0


# ------------------------------------------------------- 4. does the child hear
def knows(sc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(nearest pearl distance or -1, how many slots are filled)."""
    valid, ex, ey = mf.own_pearls(sc)
    d = np.abs(ex) + np.abs(ey)
    near = np.where(valid[:, 0], d[:, 0], -1)
    return near, valid.sum(axis=1)


def delivery(maps, steps: int = 1200, envs: int = 128, seed: int = 5) -> dict:
    """What the merge adds, measured on the same rows before and after it.

    Comparing pre-merge with post-merge is exact: it is the same dragon on the
    same turn, so nothing but the channel can account for a difference. Children
    come from the engine's own split events, so a dragon that merely spawned is
    not counted as one.
    """
    env = bcsim.BattlecodeVecEnv(maps, num_envs=envs, num_threads=4, seed=seed)
    chan = mf.MemChannel(envs)
    rng = np.random.default_rng(seed)
    obs = env.reset()
    pending = [{} for _ in range(envs)]         # dragon id -> waiting to be seen
    cone_sum = {"before": 0.0, "after": 0.0, "c_before": 0.0, "c_after": 0.0}
    acc = {k: 0 for k in ("rows", "blind", "blind_heard", "rescued", "unseen_before",
                          "unseen_after", "closer", "children", "c_blind",
                          "c_blind_heard", "c_rescued", "c_unseen_before", "c_heard",
                          "c_unseen_after", "heard", "heard_pearl", "bad",
                          "cone_up", "c_cone_up")}
    for _ in range(steps):
        before = obs.scalar.copy()
        # did a packet arrive at all, and did it name a pearl? read before the
        # merge so the inbox is untouched, though the merge does not alter it
        team = mf.pose(obs.scalar)[5]
        k = np.minimum(obs.num_msgs, obs.msgs.shape[1])
        heard = np.zeros(envs, bool)
        heard_pearl = np.zeros(envs, bool)
        for col in range(min(mf.MAX_READ, obs.msgs.shape[1])):
            tagged, s_team, mv, _x, _y = mf.decode(obs.msgs[:, col])
            ok = (col < k) & tagged & (s_team == team)
            heard |= ok
            heard_pearl |= ok & mv.any(axis=1)
        chan.receive(obs)
        acc["heard_pearl"] += int(heard_pearl.sum())
        acc["bad"] += check_block(obs.scalar)
        d0, _ = knows(before)
        d1, _ = knows(obs.scalar)
        c0 = before[:, mf.FAR_CONES:mf.FAR_CONES + 4]
        c1 = obs.scalar[:, mf.FAR_CONES:mf.FAR_CONES + 4]
        cone_up = (c1 > c0 + 1e-6).any(axis=1)
        newborn = np.zeros(envs, bool)
        for i in range(envs):
            if pending[i].pop(int(obs.dragon_id[i]), None) is not None:
                newborn[i] = True
        live = obs.mask.sum(axis=1) > 0
        for tag, rows in (("", live), ("c_", live & newborn)):
            n = int(rows.sum())
            acc["rows" if not tag else "children"] += n
            acc[tag + "blind"] += int((rows & (d0 < 0)).sum())
            acc[tag + "heard"] += int((rows & heard).sum())
            acc[tag + "blind_heard"] += int((rows & (d0 < 0) & heard_pearl).sum())
            acc[tag + "rescued"] += int((rows & (d0 < 0) & (d1 >= 0)).sum())
            # a pearl beyond the 7x7 window is one the dragon could not have
            # found by looking, which is the only thing the channel can add
            acc[tag + "unseen_before"] += int((rows & (d0 > 2 * VISION)).sum())
            acc[tag + "unseen_after"] += int((rows & (d1 > 2 * VISION)).sum())
        acc["cone_up"] += int((live & cone_up).sum())
        acc["c_cone_up"] += int((live & newborn & cone_up).sum())
        cone_sum["before"] += float(c0[live].sum())
        cone_sum["after"] += float(c1[live].sum())
        cone_sum["c_before"] += float(c0[live & newborn].sum())
        cone_sum["c_after"] += float(c1[live & newborn].sum())
        acc["closer"] += int((live & (d1 >= 0) & ((d0 < 0) | (d1 < d0))).sum())
        obs, _, _ = env.step(random_actions(obs, rng), *chan.send(obs))
        for i in range(envs):
            for _p, child, _k in env.last_splits(i):
                pending[i][int(child)] = True
    env.close()
    return {**acc, **cone_sum, **chan.rates()}


def check_block(sc: np.ndarray) -> int:
    """Invariants of the memfar pearl slots, however they were filled."""
    hx, hy, w, h, facing, _ = mf.pose(sc)
    valid, ex, ey = mf.own_pearls(sc)
    dist = np.stack([np.rint(sc[:, mf.FAR_AT + 4 * j + 3] * 32.0) for j in range(3)], 1)
    bad = 0
    bad += int((valid & (dist != np.abs(ex) + np.abs(ey))).sum())
    # nearest first, and valid slots form a prefix
    for j in range(2):
        both = valid[:, j] & valid[:, j + 1]
        bad += int((both & (dist[:, j] > dist[:, j + 1])).sum())
        bad += int((~valid[:, j] & valid[:, j + 1]).sum())
    # every named cell is within a torus half-width. The bound is over BOTH axes:
    # ex is a rotated delta, so on a 20x60 map a facing of east puts a 30-cell
    # north-south offset into ex.
    reach = np.maximum(w, h)[:, None] // 2 + 1
    bad += int((valid & (np.abs(ex) > reach)).sum())
    bad += int((valid & (np.abs(ey) > reach)).sum())
    # the count cannot claim fewer pearls than the slots name
    count = np.rint(sc[:, mf.FAR_PEARLS] * 32.0)
    bad += int((count < valid.sum(axis=1)).sum())
    return bad


def main() -> int:
    maps_dir = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "maps")
    maps = bcsim.load_maps(maps_dir)
    print(f"{len(maps)} maps from {maps_dir}\n")

    print("the packet, on synthetic rows")
    ok_bits = bits()
    print("\nthe frame, against the simulator's own windows")
    ok_frame = frame(maps)
    print("\nan empty inbox")
    ok_id = identity(maps)
    print("\nwhat the channel adds, before the merge against after it")
    d = delivery(maps)
    pct = lambda x, n: f"{x / max(n, 1):.1%}"
    n, c = d["rows"], d["children"]
    print(f"  {n:,} turns, {c:,} of them a child's first")
    print(f"  heard a packet          {pct(d['heard'], n)} of turns, "
          f"one naming a pearl {pct(d['heard_pearl'], n)}, "
          f"{d['mem_added']:.2f} pearls merged in per turn")
    print(f"  knew no pearl at all    {pct(d['blind'], n)} of turns; "
          f"{pct(d['blind_heard'], d['blind'])} of those heard a pearl and "
          f"{pct(d['rescued'], d['blind'])} were given one")
    print(f"  a blind turn that heard one was rescued "
          f"{pct(d['rescued'], d['blind_heard'])} of the time")
    print(f"  knew one out of sight   {pct(d['unseen_before'], n)} -> "
          f"{pct(d['unseen_after'], n)}")
    print(f"  nearest pearl got closer {pct(d['closer'], n)} of turns")
    print(f"  a child heard a packet  {pct(d['c_heard'], c)} of the time")
    print(f"  a child, blind at birth {pct(d['c_blind'], c)}; "
          f"{pct(d['c_blind_heard'], d['c_blind'])} of those heard a pearl, "
          f"{pct(d['c_rescued'], d['c_blind'])} were given one")
    print(f"  a child knowing one out of sight  {pct(d['c_unseen_before'], c)} -> "
          f"{pct(d['c_unseen_after'], c)}")
    print(f"  a cone weight rose      {pct(d['cone_up'], n)} of turns, "
          f"{pct(d['c_cone_up'], c)} of children's first turns")
    print(f"  total cone weight       {d['before'] / max(n, 1):.3f} -> "
          f"{d['after'] / max(n, 1):.3f} per turn; for a child "
          f"{d['c_before'] / max(c, 1):.3f} -> {d['c_after'] / max(c, 1):.3f}")
    print(f"  {d['bad']} broken feature blocks")

    ok_deliver = (c > 100 and d["rescued"] > 0
                  and d["unseen_after"] > d["unseen_before"]
                  and d["c_cone_up"] > 0 and d["c_after"] > 1.15 * d["c_before"])
    ok_blocks = d["bad"] == 0
    if not ok_deliver:
        print("\nFAIL the channel did not add anything a dragon could not see")
    if not ok_blocks:
        print("\nFAIL a merged feature block broke its own invariants")
    good = all((ok_bits, ok_frame, ok_id, ok_deliver, ok_blocks))
    print("\nthe shared-map channel is exact and reaches children" if good
          else "\nthe channel is not right yet")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
