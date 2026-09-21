import helper as unswbc
from helper import Direction
ct, game = unswbc.init()
while unswbc.update(ct, game):
    ct.get_tiles()
    ct.make_moves([Direction.NORTH])
    unswbc.end_turn()
