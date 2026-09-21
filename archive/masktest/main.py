"""Picks uniformly among the actions obs.py calls legal.

Any death by kelp or by running into a body is then a mask bug, not a policy
mistake, so the engine's own death reasons score the port.
"""
import random

import helper as unswbc
from helper import Direction

import obs as O

random.seed(1)
DIRS = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)


def execute_turn():
    snap = O.Snapshot(ct, game)
    mask = snap.mask()
    legal = [i for i in range(O.N_ACTIONS) if mask[i]]
    if not legal:
        ct.make_moves([Direction.NORTH])
        return
    a = random.choice(legal)
    if a < O.N_MOVES:
        ct.make_moves([DIRS[d] for d in O.decode_move(a, snap.facing)])
    else:
        k = O.SPLIT_K[a - O.N_MOVES]
        ct.do_split(snap.length // 2 if k < 0 else k)


def main():
    global ct, game
    ct, game = unswbc.init()
    while unswbc.update(ct, game):
        execute_turn()
        unswbc.end_turn()


main()
