"""Builds the policy's observation and action mask from the protocol block.

This is a port of bcsim/cpp/bc_obs.hpp. It has to agree with that file
exactly: the simulator's version is what the policy was trained on, and any
disagreement here means the deployed bot is answering a different question
from the one it learned. tests/parity.py checks the two against each other.

Everything is in the dragon's own frame -- the window is rotated so the
dragon always faces up, and the four directional channel groups are rolled to
match -- so the policy never has to learn four copies of the same tactic.
"""

import numpy as np

import helper as unswbc
from helper import Direction, EdgeType

VISION = 3
WINDOW = 7
CELLS = WINDOW * WINDOW
N_CHANNELS = 23
N_SCALARS = 14
MAX_MSGS = 4

N_MOVES = 3 + 9 + 27
N_SPLITS = 9
N_ACTIONS = N_MOVES + N_SPLITS
SPLIT_K = (2, 3, 4, 5, 6, 8, 12, 16, -1)      # -1 means half our length

# channel offsets, in the order bc_vec.hpp's LocalChannel declares them
C_PEARL, C_PEARL_TIME, C_NEVER_SPAWN = 0, 1, 2
C_SELF_HEAD, C_SELF_BODY = 3, 4
C_ALLY_HEAD, C_ALLY_BODY = 5, 6
C_ENEMY_HEAD, C_ENEMY_BODY = 7, 8
C_FACE, C_KELP, C_PORTAL = 9, 13, 17          # four channels each
C_SELF_INDEX, C_SELF_TAIL = 21, 22

DIRS = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)
DIR_INDEX = {d: i for i, d in enumerate(DIRS)}
OFFSETS = tuple(d.get_offset() for d in DIRS)


def _spatial_perm():
    """For each facing, output cell -> source cell in the protocol's window.

    ego_to_world from bc_obs.hpp, applied to every cell up front so the turn
    itself is one fancy-index rather than 49 rotations.
    """
    perm = np.zeros((4, CELLS), dtype=np.intp)
    for facing in range(4):
        for row in range(WINDOW):
            for col in range(WINDOW):
                ox, oy = col - VISION, row - VISION
                if facing == 0:
                    wx, wy = ox, oy
                elif facing == 1:
                    wx, wy = -oy, ox
                elif facing == 2:
                    wx, wy = -ox, -oy
                else:
                    wx, wy = oy, -ox
                perm[facing, row * WINDOW + col] = (wy + VISION) * WINDOW + (wx + VISION)
    return perm


PERM = _spatial_perm()


def _move_table():
    """Action id -> tuple of turns, as bc_vec.hpp's decode_for reads them.

    The id is a base-3 number read low digit first: 0 straight on, 1 left,
    2 right, each turn relative to where the previous step left us pointing.
    """
    table = []
    for action_id in range(N_MOVES):
        if action_id < 3:
            n, rest = 1, action_id
        elif action_id < 12:
            n, rest = 2, action_id - 3
        else:
            n, rest = 3, action_id - 12
        turns = []
        for _ in range(n):
            turns.append(rest % 3)
            rest //= 3
        table.append(tuple(turns))
    return tuple(table)


MOVES = _move_table()


def decode_move(action_id, facing):
    """The world directions a move action walks, given where we point now."""
    dirs = []
    f = facing
    for turn in MOVES[action_id]:
        if turn == 1:
            f = (f + 3) % 4
        elif turn == 2:
            f = (f + 1) % 4
        dirs.append(f)
    return dirs


class Snapshot:
    """One turn's protocol block, indexed the way the observation needs it."""

    __slots__ = ("ct", "game", "w", "h", "hx", "hy", "facing", "length",
                 "my_id", "tiles", "by_pos", "parts", "body_index", "body_pos")

    def __init__(self, ct, game):
        self.ct = ct
        self.game = game
        self.w, self.h = game.width, game.height
        pos = ct.get_position()
        self.hx, self.hy = pos.x, pos.y
        self.facing = DIR_INDEX[ct.get_dir()]
        self.length = ct.get_length()
        self.my_id = ct.get_id()

        self.tiles = ct.get_tiles()
        self.by_pos = {}
        self.parts = {}
        for tile in self.tiles:
            p = tile.position
            key = (p.x, p.y)
            self.by_pos[key] = tile
            part = tile.dragon_part
            if part is not None:
                self.parts[key] = part
        self.body_index, self.body_pos = self._trace_body()

    def _trace_body(self):
        """Walks our own body outward from the head, newest segment first.

        The protocol never numbers our segments, but every body segment points
        at the segment ahead of it (bc_core.hpp sets seg_dir that way), so the
        chain can be followed backwards from the head. A segment that points
        through a portal lands somewhere we cannot see, which ends the walk --
        the segments past it stay unnumbered, which only costs us two channels
        on tiles we can see anyway.
        """
        back = {}
        me = self.my_id
        for key, part in self.parts.items():
            if part.dragon_id != me or part.is_dragon_head:
                continue
            tile = self.by_pos[key]
            edge = tile.get_edge(part.dir)
            kind = edge.edge_type
            if kind != EdgeType.EMPTY:
                continue                       # kelp cannot happen, a portal hides the target
            dx, dy = part.dir._offset
            back[((key[0] + dx) % self.w, (key[1] + dy) % self.h)] = key

        index = {}
        pos = {}
        cur = (self.hx, self.hy)
        index[cur] = 0
        pos[0] = cur
        i = 0
        while True:
            nxt = back.get(cur)
            if nxt is None or nxt in index:
                break
            i += 1
            index[nxt] = i
            pos[i] = nxt
            cur = nxt
        return index, pos

    # ---- observation

    def local(self):
        """The 23 x 7 x 7 window, already rotated into the dragon's frame."""
        world = np.zeros((N_CHANNELS, CELLS), dtype=np.float32)
        me = self.my_id
        my_team = self.ct.get_team()
        denom = float(max(1, self.length - 1))

        for i, tile in enumerate(self.tiles):
            if tile.pearl:
                world[C_PEARL, i] = 1.0
            pt = tile.pearl_time
            if pt < 0:
                world[C_NEVER_SPAWN, i] = 1.0
            else:
                world[C_PEARL_TIME, i] = (pt if pt < 99 else 99) / 99.0

            part = tile.dragon_part
            if part is not None:
                head = part.is_dragon_head
                if part.dragon_id == me:
                    world[C_SELF_HEAD if head else C_SELF_BODY, i] = 1.0
                elif part.team == my_team:
                    world[C_ALLY_HEAD if head else C_ALLY_BODY, i] = 1.0
                else:
                    world[C_ENEMY_HEAD if head else C_ENEMY_BODY, i] = 1.0
                world[C_FACE + DIR_INDEX[part.dir], i] = 1.0

            edges = tile._edges                 # north, east, south, west
            for d in range(4):
                kind = edges[d].edge_type
                if kind == EdgeType.KELP:
                    world[C_KELP + d, i] = 1.0
                elif kind == EdgeType.PORTAL:
                    world[C_PORTAL + d, i] = 1.0

        # our own segments, numbered head-first, only where the window shows them
        last = self.length - 1
        for idx, key in self.body_pos.items():
            tile = self.by_pos.get(key)
            if tile is None:
                continue
            cell = self._cell_of(key)
            if cell < 0:
                continue
            world[C_SELF_INDEX, cell] = idx / denom
            if idx == last:
                world[C_SELF_TAIL, cell] = 1.0

        facing = self.facing
        ego = world[:, PERM[facing]]
        if facing:
            # a channel group is ordered N, E, S, W in world terms; rolling it
            # by the facing renames them forward, right, back, left
            for base in (C_FACE, C_KELP, C_PORTAL):
                ego[base:base + 4] = np.roll(ego[base:base + 4], -facing, axis=0)
        return ego.reshape(N_CHANNELS, WINDOW, WINDOW)

    def _cell_of(self, key):
        dx = (key[0] - self.hx + VISION) % self.w
        dy = (key[1] - self.hy + VISION) % self.h
        if dx >= WINDOW or dy >= WINDOW:
            return -1
        return dy * WINDOW + dx

    def scalars(self):
        s = np.zeros(N_SCALARS, dtype=np.float32)
        s[0] = self.game.round_num / 500.0
        s[1] = min(self.length, 64) / 64.0
        s[2] = float(self.length)
        s[3] = self.ct.get_unit_count() / float(self.game.unit_limit)
        s[4 + self.facing] = 1.0
        s[8] = self.hx / float(self.w)
        s[9] = self.hy / float(self.h)
        s[10] = self.w / 64.0
        s[11] = self.h / 64.0
        s[12] = min(len(self.ct.sonar_messages), MAX_MSGS)
        s[13] = 1.0 if self.ct.get_team() is unswbc.Team.B else 0.0
        return s

    # ---- action mask

    def mask(self):
        """Legality as far as the dragon can tell, port of VecEnv::fill_mask.

        A portal ends the walk and counts as allowed, because the far side is
        not visible and guessing would teach the policy something false.
        """
        m = np.zeros(N_ACTIONS, dtype=np.float32)
        w, h, me = self.w, self.h, self.my_id
        body_pos = self.body_pos
        last = self.length - 1

        for action_id in range(N_MOVES):
            x, y = self.hx, self.hy
            length, dropped = self.length, 0
            added = []
            legal = True
            for s, d in enumerate(decode_move(action_id, self.facing)):
                if s > 0 and length <= 2:
                    legal = False
                    break
                tile = self.by_pos.get((x, y))
                if tile is None:
                    legal = False
                    break
                kind = tile._edges[d].edge_type
                if kind == EdgeType.KELP:
                    legal = False
                    break
                if kind == EdgeType.PORTAL:
                    break                       # cannot see where it lets out
                dx, dy = OFFSETS[d]
                nx, ny = (x + dx) % w, (y + dy) % h
                t = (nx, ny)

                blocked = t in added
                if not blocked:
                    part = self.parts.get(t)
                    if part is not None and part.dragon_id == me:
                        blocked = True
                        for j in range(dropped):
                            if body_pos.get(last - j) == t:
                                blocked = False   # that tail has already moved on
                                break
                    elif part is not None and not part.is_dragon_head:
                        blocked = True            # another dragon's body is certain death
                if blocked:
                    legal = False
                    break

                if len(added) < 3:
                    added.append(t)
                nt = self.by_pos.get(t)
                if nt is not None and nt.pearl:
                    length += 1
                else:
                    dropped += 1
                if s > 0:
                    dropped += 1
                    length -= 1
                x, y = nx, ny
            if legal:
                m[action_id] = 1.0

        room = self.ct.get_unit_count() < self.game.unit_limit
        if room:
            for i, k0 in enumerate(SPLIT_K):
                k = self.length // 2 if k0 < 0 else k0
                if 2 <= k <= self.length - 2:
                    m[N_MOVES + i] = 1.0
        return m
