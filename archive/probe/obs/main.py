import helper as unswbc
from helper import Direction
import obs as O
DIRS = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)
ct, game = unswbc.init()
while unswbc.update(ct, game):
    s = O.Snapshot(ct, game)
    m = s.mask(); l = s.local(); sc = s.scalars()
    ct.make_moves([Direction.NORTH])
    unswbc.end_turn()
