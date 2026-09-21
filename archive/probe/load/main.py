import helper as unswbc
from helper import Direction
from policy import Policy
ct, game = unswbc.init()
policy = Policy()
while unswbc.update(ct, game):
    ct.make_moves([Direction.NORTH])
    unswbc.end_turn()
