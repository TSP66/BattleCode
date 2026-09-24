"""The team's shared map, as 64 bits of sonar, and as features the nets already read.

A split makes a child with a blank memory standing where its parent stood, and
1.0.0 lets a dragon put 64 bits into a sonar ray. This is the channel that
carries what one dragon knows to another, and -- the part that matters -- it
lands in a feature block every existing checkpoint is already trained on, so a
policy benefits from it without being retrained for it.

WHY A HAND-WRITTEN CODEC AND NOT A LEARNED ONE
----------------------------------------------
train/memcodec.py compresses a dragon's remembered-map planes into 56 bits by
reconstruction, and its measurements are real (62% of the variance the mean
leaves). It cannot be deployed as it stands, for a reason that is geometric
rather than statistical: `wide` planes are centred on the sender's HEAD and
rotated to the sender's FACING. A child's head is the parent's old segment
`len - k`, several cells away and usually facing elsewhere, so planes decoded
from a parent's code describe the world around the PARENT. Injecting them into
the child's frame is an offset-plus-rotation error of several cells at a scale
where one cell matters. Fixing that needs the parent to re-render its memory at
the child's pose, which needs the child's pose, which the receiver cannot check.

An absolute-coordinate codec has no frame at all to get wrong. Board
coordinates are shared by both teams (bc_obs.hpp writes SC_HEAD_X as hx / w,
not mirrored), so a payload means the same thing to every dragon that hears it
and the geometry is exact arithmetic instead of something a decoder has to
learn. What it carries is therefore chosen, not discovered -- which is also
what the measurements say to carry: terrain and pearls survive compression
(0.86-0.93 explained variance) and enemy sightings do not (0.09-0.20).

WHAT IS SENT
------------
Up to three expected-pearl cells, in absolute board coordinates. That is not an
arbitrary choice of field: memfar (the 18 scalars at index 690) ALREADY carries
"the three nearest expected pearls I know about" as ego offsets, and
cpp/bc_memory.hpp computes them from exactly the set we want to share. So the
sender reads its own three out of its own observation, converts to absolute
coordinates, and packs them; the receiver converts back into ITS ego frame and
unions them with its own three, keeping the nearest three overall.

The receiver's feature block ends up holding the same quantity it always held
-- the nearest expected pearls this dragon knows about -- computed over a
larger set of knowledge. No feature changes meaning, no column moves, no
checkpoint is invalidated, and both PyramidActorCritic (which reads memfar as
part of its 32 scalars) and the flat 708-scalar nets see it.

    [63:56] tag        0xB4, so a receiver can tell a memory packet from a
                       payload some other bot chose to send. Not a signature:
                       it cannot prove who sent it, and nothing here needs it to.
    [55]    team       the sender's team (SC_TEAM_B). A packet from the other
                       team is ignored by default -- its pearls are TRUE, since
                       the coordinates are absolute, but "what my team knows" is
                       the quantity the feature is supposed to mean.
    [54:16] pearls     three slots of 13 bits: valid(1) | x(6) | y(6).
                       6 bits each because bc_memory.hpp caps the board at 64.
    [15:4]  cones      memfar's four directional pearl weights, 3 bits each, in
                       ABSOLUTE compass order rather than the sender's own
                       forward/left/right/behind. Coarse on purpose: the cone is
                       a hint about where pearls lie, not a coordinate.
    [3:0]   reserved   zero, and tests assert it.

WHY THE CONES ARE IN THERE, WHICH IS A MEASUREMENT AND NOT A GUESS
------------------------------------------------------------------
The pearl slots alone did almost nothing for the case this work exists to fix.
Measured over 153,532 turns of random play (tests/test_memfeat.py):

    a child hears a packet                           77.8% of turns
    a child born knowing no pearl at all             14.1%
      ... of those, given one by the pearl slots      2.5%
    a child knowing a pearl it cannot see       0.0% -> 0.2%

So delivery to children is not the problem. The problem is that a parent is
ADJACENT to the child it just made, so the parent's three nearest pearls are
mostly the ones the child can already see out of its own window -- a child is
not blind at birth, because bc_obs.hpp feeds its first look into its fresh
memory before building features, and 86% of children already know a pearl. What
a child lacks is not a nearby pearl, it is everything further out.

The cones are exactly that: a weighted density over the sender's WHOLE
remembered map, which a blank-memory child has no way to build. With them:

    a cone weight rose        64.3% of children's first turns (39.6% of all turns)
    total cone weight, child  1.473 -> 2.782     nearly doubled
    total cone weight, any    1.006 -> 1.259

(The other case, a dragon that knows no pearl anywhere, is delivery-bound and
nothing in the payload can fix it: such a dragon heard a pearl on only 7.8% of
its blind turns, and on 100% of those it was given one. It is alone, and a sonar
ray stops at the first dragon it meets, so no one is in range to tell it
anything. Those figures are from random play, where dragons stay short; a ray is
dragged the length of a body before it flies, so a trained policy's longer
dragons reach further and should do better.)

WHAT IT DOES NOT DO
-------------------
It does not identify the sender. The engine hands a receiver `NUM_MSGS n` and n
payloads and never says who spoke (see the note in cpp/bc_vec.hpp), so a child
hearing three packets on its first turn cannot tell which came from its parent.
It does not need to: every packet is a set of true pearl cells, so the union is
correct whichever teammate sent it, and the parent's is in there when it arrives.

It does not assume arrival. Measured delivery is 57% of children at birth,
rising to 89% by their sixth turn (tests/test_msg_transfer.py). A dragon that
hears nothing keeps exactly the memfar block the simulator built for it -- with
no messages the merge is bit-identical to the input, which is asserted by
tests/test_memfeat.py -- so the policy degrades to its old behaviour rather than
to a broken one.

    python -m train.memfeat            # measure it on real play
"""

from __future__ import annotations

import numpy as np

# The eight bits that say "this is a memory packet". One byte with four
# transitions, which no run of zeros or ones can be mistaken for; the same value
# train/memcodec.py reserved.
SONAR_TAG = 0xB4

TAG_SHIFT = 56
TEAM_SHIFT = 55
PEARL_SLOTS = 3
SLOT_BITS = 13                      # valid(1) | x(6) | y(6)
SLOT0_SHIFT = 42                    # slot j occupies [SLOT0_SHIFT - 13j + 12 : ...]
COORD_BITS = 6
COORD_MAX = 1 << COORD_BITS         # boards are at most 64 x 64

N_CONES = 4
CONE_BITS = 3
CONE_LEVELS = (1 << CONE_BITS) - 1                             # 7
CONE0_SHIFT = SLOT0_SHIFT - (PEARL_SLOTS - 1) * SLOT_BITS - CONE_BITS   # 13
RESERVED_BITS = CONE0_SHIFT - (N_CONES - 1) * CONE_BITS        # 4

# memfar's cone order is the dragon's own forward, left, right, behind. Turning
# that into a compass bearing is a rotation by the sender's facing, and the
# receiver rotates it back by its own: R_OF_EGO[j] is how many clockwise quarter
# turns from `facing` the j-th cone points, with facing 0 = N, 1 = E, 2 = S, 3 = W.
R_OF_EGO = np.array([0, 3, 1, 2])   # forward, left, right, behind
EGO_OF_R = np.array([0, 2, 3, 1])   # and the inverse

# Where memfar sits in the scalar row, and its layout (cpp/bc_memory.hpp
# memfar): PEARL_SLOTS slots of (valid, ex/16, ey/16, dist/32), four cone
# weights, the known fraction, then the pearl count over 32.
N_BASE = 14
N_MEM = 676
FAR_AT = N_BASE + N_MEM             # 690
FAR_CONES = FAR_AT + 4 * PEARL_SLOTS                # the four cone weights, 702
FAR_PEARLS = FAR_CONES + N_CONES + 1                # the count field, 707

# How many of a dragon's payloads to look at. The inbox is unbounded and reached
# 24 deep when every dragon broadcast four ways, so this is a cost/benefit choice
# and it was measured rather than picked (read depth against what the merge adds,
# 76,800 turns; nothing beyond 8 ever changed a single number):
#
#     read     child cone weight      pearl rescues     pearls merged per turn
#        1       1.174 -> 2.105                 698                      0.146
#        2                2.149                 752                      0.149
#        4                2.163               1,036                      0.168
#        8                2.165               1,102                      0.173
#       16                2.165               1,102                      0.173
#
# Four gets 99.9% of the cone gain and 94% of the rescues for 54% of the work,
# which at 1,024 envs is 1.6ms a step against the simulator's own 1.4ms. It also
# happens to be where SC_NUM_MSGS saturates, so the count the network reads and
# the depth the merge reads agree. Reading the FIRST four is not arbitrary in the
# case that matters: dragons act in id order and a child's id is higher than its
# parent's, so the parent's ray lands before the child's turn and early in its
# inbox.
MAX_READ = 4

# Scalar indices used here, from bc_vec.hpp's SC_* enum.
SC_FACE_N, SC_HEAD_X, SC_HEAD_Y, SC_MAP_W, SC_MAP_H, SC_TEAM_B = 4, 8, 9, 10, 11, 13


def _half(x: np.ndarray) -> np.ndarray:
    """cpp/bc_memory.hpp mem_half: the value as the offline builder stored it.

    Every field written here is k/16 or k/32 for a small integer k, which float16
    holds exactly, so this is the identity -- it is applied anyway so that the
    quantisation lives in one place if a field ever stops being exact.
    """
    return x.astype(np.float16).astype(np.float32)


def pose(sc: np.ndarray):
    """(head x, head y, map w, map h, facing, team) from the base scalars.

    The simulator writes hx / w and w / 64, so this inverts the normalisation.
    Both are exact for integers this small, and an all-zero row (an env with no
    acting dragon) comes back as a 1x1 board at the origin, which then matches
    nothing.
    """
    w = np.maximum(np.rint(sc[:, SC_MAP_W] * 64.0).astype(np.int64), 1)
    h = np.maximum(np.rint(sc[:, SC_MAP_H] * 64.0).astype(np.int64), 1)
    hx = np.rint(sc[:, SC_HEAD_X] * w).astype(np.int64)
    hy = np.rint(sc[:, SC_HEAD_Y] * h).astype(np.int64)
    facing = np.argmax(sc[:, SC_FACE_N:SC_FACE_N + 4], axis=1).astype(np.int64)
    team = (sc[:, SC_TEAM_B] > 0.5).astype(np.int64)
    return hx, hy, w, h, facing, team


def _rot_from_ego(f, ex, ey):
    """Ego offset -> world delta: the inverse of memfar's rotation by facing."""
    ddx = np.where(f == 0, ex, np.where(f == 1, -ey, np.where(f == 2, -ex, ey)))
    ddy = np.where(f == 0, ey, np.where(f == 1, ex, np.where(f == 2, -ey, -ex)))
    return ddx, ddy


def _rot_to_ego(f, ddx, ddy):
    """World delta -> ego offset, exactly as cpp/bc_memory.hpp memfar rotates."""
    ex = np.where(f == 0, ddx, np.where(f == 1, ddy, np.where(f == 2, -ddx, -ddy)))
    ey = np.where(f == 0, ddy, np.where(f == 1, -ddx, np.where(f == 2, -ddy, ddx)))
    return ex, ey


def ego_to_abs(ex, ey, hx, hy, w, h, facing):
    ddx, ddy = _rot_from_ego(facing, ex, ey)
    return (hx + ddx) % w, (hy + ddy) % h


def abs_to_ego(px, py, hx, hy, w, h, facing):
    """Absolute cell -> (ex, ey, Manhattan distance) in the dragon's own frame.

    The wrap is mem_wrap's: the shortest signed delta on the torus, which is
    what memfar measures, so a pearl two cells away around the edge reads as two
    and not as sixty-two.
    """
    ddx = (px - hx + w // 2) % w - w // 2
    ddy = (py - hy + h // 2) % h - h // 2
    ex, ey = _rot_to_ego(facing, ddx, ddy)
    return ex, ey, np.abs(ex) + np.abs(ey)


def own_pearls(sc: np.ndarray):
    """The dragon's own three memfar slots as (valid, ex, ey), de-quantised.

    ex and ey were stored as k/16 for integer k, so this recovers k exactly.
    """
    valid = np.stack([sc[:, FAR_AT + 4 * j] > 0.5 for j in range(PEARL_SLOTS)], axis=1)
    ex = np.stack([np.rint(sc[:, FAR_AT + 4 * j + 1] * 16.0) for j in range(PEARL_SLOTS)],
                  axis=1).astype(np.int64)
    ey = np.stack([np.rint(sc[:, FAR_AT + 4 * j + 2] * 16.0) for j in range(PEARL_SLOTS)],
                  axis=1).astype(np.int64)
    return valid, ex, ey


# ------------------------------------------------------------------ the packet
def encode(sc: np.ndarray) -> np.ndarray:
    """One payload per row: the dragon's own nearest pearls in absolute coords.

    Always returns a packet, even with no pearls to report: casting is free, the
    ray's echo is sensing the dragon wants anyway, and an empty packet merges to
    nothing on the other side.
    """
    hx, hy, w, h, facing, team = pose(sc)
    valid, ex, ey = own_pearls(sc)
    out = np.full(len(sc), (np.uint64(SONAR_TAG) << np.uint64(TAG_SHIFT)), np.uint64)
    out |= team.astype(np.uint64) << np.uint64(TEAM_SHIFT)
    for j in range(PEARL_SLOTS):
        px, py = ego_to_abs(ex[:, j], ey[:, j], hx, hy, w, h, facing)
        # a cell off a small board cannot be addressed in six bits; the slot is
        # simply not sent, which is safer than sending a wrong cell
        ok = valid[:, j] & (px < COORD_MAX) & (py < COORD_MAX)
        field = (np.uint64(1) << np.uint64(12)) | (px.astype(np.uint64) << np.uint64(6)) \
            | py.astype(np.uint64)
        shift = np.uint64(SLOT0_SHIFT - j * SLOT_BITS)
        out |= np.where(ok, field, np.uint64(0)).astype(np.uint64) << shift
    for j in range(N_CONES):
        # the cone points R_OF_EGO[j] quarter turns from the sender's facing;
        # sent as the compass bearing, so a receiver needs nothing but its own
        q = np.rint(np.clip(sc[:, FAR_CONES + j], 0.0, 1.0) * CONE_LEVELS).astype(np.uint64)
        d = (facing + R_OF_EGO[j]) % 4
        out |= q << (np.uint64(CONE0_SHIFT) - np.uint64(CONE_BITS) * d.astype(np.uint64))
    return out


def decode(payload: np.ndarray):
    """(is a packet, sender team, valid, x, y) for payloads of any shape.

    Shape-agnostic on purpose: `merge` hands it a whole (rows, messages) block at
    once, which is several times faster than a loop over messages, and the tests
    hand it a flat array. One implementation of the layout either way.
    """
    p = np.asarray(payload, np.uint64)
    tagged = (p >> np.uint64(TAG_SHIFT)) == np.uint64(SONAR_TAG)
    team = ((p >> np.uint64(TEAM_SHIFT)) & np.uint64(1)).astype(np.int64)
    shifts = np.array([SLOT0_SHIFT - j * SLOT_BITS for j in range(PEARL_SLOTS)], np.uint64)
    f = (p[..., None] >> shifts) & np.uint64((1 << SLOT_BITS) - 1)
    valid = ((f >> np.uint64(12)) & np.uint64(1)).astype(bool) & tagged[..., None]
    px = ((f >> np.uint64(COORD_BITS)) & np.uint64(COORD_MAX - 1)).astype(np.int64)
    py = (f & np.uint64(COORD_MAX - 1)).astype(np.int64)
    return tagged, team, valid, px, py


def decode_cones(payload: np.ndarray) -> np.ndarray:
    """The four cone weights by COMPASS bearing (N, E, S, W), back in [0, 1].

    Appends the bearing axis, so a (rows, messages) block comes back as
    (rows, messages, 4).
    """
    p = np.asarray(payload, np.uint64)
    shifts = np.array([CONE0_SHIFT - d * CONE_BITS for d in range(N_CONES)], np.uint64)
    q = (p[..., None] >> shifts) & np.uint64(CONE_LEVELS)
    return q.astype(np.float32) / CONE_LEVELS


def reserved(payload: np.ndarray) -> np.ndarray:
    """The bits this layout leaves unused, which must be zero."""
    return np.asarray(payload, np.uint64) & np.uint64((1 << RESERVED_BITS) - 1)


# ------------------------------------------------------------------- the merge
def merge(sc: np.ndarray, msgs: np.ndarray, num_msgs: np.ndarray,
          max_read: int = MAX_READ, cross_team: bool = False,
          cones: bool = True) -> dict:
    """Unions a dragon's inbox into its own memfar block, in place.

    Both fields keep the meaning they already had, computed over the team's
    knowledge instead of one dragon's:

      * the three pearl slots become the nearest expected pearls ANY of us knows
        about, deduplicated by cell and re-sorted by this dragon's own distance;
      * each cone becomes the largest weight any of us reports that way. A cone
        is a sum over the sender's whole remembered map, so a maximum is a lower
        bound on the union rather than the union itself -- and for the case that
        matters, a newborn whose own cones are all zero, it is the sender's value
        exactly.

    The known fraction is left alone: two dragons' explored areas overlap by an
    unknown amount, so neither the max nor the sum is a bound on the union, and a
    feature that says "how much of the board have I seen" is not improved by
    replacing it with a number that means neither one thing nor the other.

    Returns counters for logging.
    """
    n = len(sc)
    hx, hy, w, h, facing, team = pose(sc)
    ovalid, oex, oey = own_pearls(sc)
    k = np.minimum(num_msgs.astype(np.int64), msgs.shape[1])
    read = min(max_read, msgs.shape[1])

    # the dragon's own three, back in absolute coordinates, then everything it was
    # told -- all of the inbox in one pass rather than a loop over messages
    own_x, own_y = ego_to_abs(oex, oey, hx[:, None], hy[:, None], w[:, None], h[:, None],
                              facing[:, None])
    block = msgs[:, :read]
    tagged, s_team, mv, mx, my = decode(block)
    ok = (np.arange(read) < k[:, None]) & tagged & (cross_team | (s_team == team[:, None]))
    heard = ok.any(axis=1)

    valid = np.concatenate([ovalid, (mv & ok[:, :, None]).reshape(n, -1)], axis=1)
    px = np.concatenate([own_x, mx.reshape(n, -1)], axis=1)
    py = np.concatenate([own_y, my.reshape(n, -1)], axis=1)
    told = (np.where(ok[:, :, None], decode_cones(block), 0.0).max(axis=1)
            if cones and read else np.zeros((n, N_CONES), np.float32))
    # a cell outside this board cannot be one of ours; a packet from a dragon on
    # a differently sized map (impossible today, but free to rule out) or a
    # payload that only looks like a packet is dropped here
    valid &= (px < w[:, None]) & (py < h[:, None])
    # Distance needs no rotation: turning the board only permutes and negates the
    # two offsets, and |ex| + |ey| is invariant under that. So the candidates are
    # ranked on the wrapped deltas alone and only the three that win get rotated.
    ddx = (px - hx[:, None] + w[:, None] // 2) % w[:, None] - w[:, None] // 2
    ddy = (py - hy[:, None] + h[:, None] // 2) % h[:, None] - h[:, None] // 2
    dist = np.abs(ddx) + np.abs(ddy)

    # One cell, one slot: drop a candidate an earlier one already named. Done by
    # sorting rather than by comparing every pair, which is what this cost before
    # -- (rows, 27, 27) booleans per step is 3x the simulator's own work. A stable
    # sort puts duplicates next to each other with the earliest candidate first,
    # so "equal to my predecessor" is exactly "already named".
    c = valid.shape[1]
    key = np.where(valid, px * COORD_MAX + py, (1 << 20) + np.arange(c))
    by_cell = np.argsort(key, axis=1, kind="stable")
    sorted_key = np.take_along_axis(key, by_cell, axis=1)
    dup_sorted = np.zeros_like(valid)
    dup_sorted[:, 1:] = sorted_key[:, 1:] == sorted_key[:, :-1]
    dup = np.empty_like(dup_sorted)
    np.put_along_axis(dup, by_cell, dup_sorted, axis=1)
    valid &= ~dup

    # nearest first, ties by candidate order, which puts the dragon's own three
    # ahead of anything it was told
    order = np.argsort(np.where(valid, dist * c + np.arange(c), 1 << 40), axis=1)[:, :PEARL_SLOTS]
    rows = np.arange(n)[:, None]
    sv, sd = valid[rows, order], dist[rows, order]
    sex, sey = _rot_to_ego(facing[:, None], ddx[rows, order], ddy[rows, order])

    added = int((sv & (order >= PEARL_SLOTS)).sum())
    for j in range(PEARL_SLOTS):
        on = sv[:, j]
        sc[:, FAR_AT + 4 * j] = on
        sc[:, FAR_AT + 4 * j + 1] = np.where(on, _half(sex[:, j] / 16.0), 0.0)
        sc[:, FAR_AT + 4 * j + 2] = np.where(on, _half(sey[:, j] / 16.0), 0.0)
        sc[:, FAR_AT + 4 * j + 3] = np.where(on, _half(sd[:, j] / 32.0), 0.0)
    # keep the count consistent with the slots: it can only have grown
    count = np.maximum(np.rint(sc[:, FAR_PEARLS] * 32.0), sv.sum(axis=1))
    sc[:, FAR_PEARLS] = _half(np.minimum(count, 32.0) / 32.0)

    lifted = 0
    if cones:
        # compass bearing -> this dragon's own forward/left/right/behind
        bearing = (facing[:, None] + R_OF_EGO[None, :]) % 4
        mine = sc[:, FAR_CONES:FAR_CONES + N_CONES]
        got = _half(np.take_along_axis(told, bearing, axis=1))
        lifted = int((got > mine).any(axis=1).sum())
        sc[:, FAR_CONES:FAR_CONES + N_CONES] = np.maximum(mine, got)
    return {"heard": int(heard.sum()), "added": added, "lifted": lifted,
            "with_pearl": int(sv[:, 0].sum()), "rows": n}


class MemChannel:
    """Both ends of the channel, for one vec env.

    Per step: `receive` before the observation is used, so the policy sees the
    team's pearls; then `send`, whose payload is built from the MERGED block, so
    a dragon relays what it was told and knowledge spreads past one hop.

    Every dragon is put on protocol 3 and broadcasts in all four directions. The
    protocol is not optional: the engine DROPS a payload above 2^32 addressed to
    a dragon still on protocol 2 rather than truncating it (cpp/bc_core.hpp), and
    every packet here is above 2^32 because of the tag. A split child inherits
    its parent's protocol, so it can receive on its very first turn.

    Turning this on changes what the opponent sees too, through SC_NUM_MSGS and
    the echo scalars, so a league built without it is not strictly comparable.
    """

    ALL_DIRS = 0b1111

    def __init__(self, num_envs: int, n_dirs: int = 4, max_read: int = MAX_READ,
                 cross_team: bool = False, cones: bool = True):
        self.num_envs = num_envs
        self.max_read = max_read
        self.cross_team = cross_team
        self.cones = cones
        self._send = np.full(num_envs, self.ALL_DIRS, np.uint8)
        self._proto = np.full(num_envs, 3, np.int8)
        self._sonar = np.zeros((num_envs, n_dirs), np.uint64)
        self.stats = {"heard": 0, "added": 0, "lifted": 0, "with_pearl": 0, "rows": 0}

    def receive(self, obs) -> dict:
        """Merges the inbox into obs.scalar in place and accumulates counters."""
        st = merge(obs.scalar, obs.msgs, obs.num_msgs, self.max_read, self.cross_team,
                   self.cones)
        for key, v in st.items():
            self.stats[key] += v
        return st

    def send(self, obs):
        """(send_dirs, sonar, protocol) to hand to env.step."""
        self._sonar[:] = encode(obs.scalar)[:, None]
        return self._send, self._sonar, self._proto

    def rates(self) -> dict:
        """Shares rather than counts, for a log line."""
        n = max(self.stats["rows"], 1)
        return {"mem_heard": round(self.stats["heard"] / n, 4),
                "mem_added": round(self.stats["added"] / n, 4),
                "mem_lifted": round(self.stats["lifted"] / n, 4),
                "mem_pearl": round(self.stats["with_pearl"] / n, 4)}

    def reset_stats(self) -> None:
        self.stats = dict.fromkeys(self.stats, 0)


# --------------------------------------------------------------------- measure
def _measure(argv=None) -> None:
    """What the channel does to real play: how many dragons hear a packet, how
    often a NEWBORN has a pearl to go to, and what it costs in throughput."""
    import argparse
    import pathlib
    import sys
    import time

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    import bcsim

    p = argparse.ArgumentParser()
    p.add_argument("--maps", default=str(pathlib.Path(__file__).resolve().parents[2] / "maps"))
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)

    maps = bcsim.load_maps(a.maps)
    rng = np.random.default_rng(a.seed)

    def play(on: bool) -> dict:
        env = bcsim.BattlecodeVecEnv(maps, num_envs=a.envs, num_threads=a.threads,
                                     seed=a.seed)
        chan = MemChannel(a.envs) if on else None
        obs = env.reset()
        # a dragon is newborn on the first turn we see its uid
        seen = set()
        born, born_pearl, rows, rows_pearl = 0, 0, 0, 0
        t0 = time.perf_counter()
        for _ in range(a.steps):
            if chan is not None:
                chan.receive(obs)
            fresh = np.array([u not in seen for u in obs.uid])
            seen.update(int(u) for u in obs.uid)
            has = obs.scalar[:, FAR_AT] > 0.5
            born += int(fresh.sum())
            born_pearl += int((fresh & has).sum())
            rows += len(has)
            rows_pearl += int(has.sum())
            legal = obs.mask.astype(np.float64) + 1e-9
            act = np.array([rng.choice(len(r), p=r / r.sum()) for r in legal], np.int32)
            if chan is not None:
                obs, _, _ = env.step(act, *chan.send(obs))
            else:
                obs, _, _ = env.step(act)
        dt = time.perf_counter() - t0
        env.close()
        out = {"newborn_with_pearl": born_pearl / max(born, 1),
               "any_with_pearl": rows_pearl / max(rows, 1),
               "turns_per_sec": a.steps * a.envs / dt}
        if chan is not None:
            out.update(chan.rates())
        return out

    for on in (False, True):
        r = play(on)
        print(f"channel {'on ' if on else 'off'}: " +
              ", ".join(f"{k} {v:,.4f}" if v < 100 else f"{k} {v:,.0f}"
                        for k, v in r.items()), flush=True)


if __name__ == "__main__":
    _measure()
