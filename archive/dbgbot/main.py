import numpy as np
import helper as unswbc
from helper import Direction, EdgeType
import obs as O
from policy import Policy

DIRS = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)
NAMES = "NESW"


def execute_turn():
    snap = O.Snapshot(ct, game)
    mask = snap.mask()
    a = policy.act(snap.local(), snap.scalars(), mask)
    if a < O.N_MOVES:
        ds = O.decode_move(a, snap.facing)
        path = "".join(NAMES[d] for d in ds)
        # what we think is in the way, step by step
        x, y = snap.hx, snap.hy
        trace = []
        for d in ds:
            t = snap.by_pos.get((x, y))
            if t is None:
                trace.append("?")
                break
            k = t._edges[d].edge_type
            trace.append(k.name[0])
            dx, dy = O.OFFSETS[d]
            x, y = (x + dx) % snap.w, (y + dy) % snap.h
            p = snap.parts.get((x, y))
            trace.append("." if p is None else ("H" if p.is_dragon_head else "B"))
        ct.output_log("T", ct.get_id(), "r", game.round_num, "at", snap.hx, snap.hy,
                      "f", NAMES[snap.facing], "L", snap.length, "a", a, path,
                      "".join(trace), "legal", int(mask.sum()))
        ct.make_moves([DIRS[d] for d in ds])
    else:
        k = O.SPLIT_K[a - O.N_MOVES]
        ct.output_log("T", ct.get_id(), "r", game.round_num, "SPLIT", k)
        ct.do_split(snap.length // 2 if k < 0 else k)


def main():
    global ct, game, policy
    ct, game = unswbc.init()
    policy = Policy()
    while unswbc.update(ct, game):
        execute_turn()
        unswbc.end_turn()


main()
