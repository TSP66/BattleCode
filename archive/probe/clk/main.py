"""Checks that the sandbox clock is denominated in CPU points."""
import time
import numpy as np
import helper as unswbc
from helper import Direction

a = np.zeros((64, 512), np.float32)
b = np.zeros((512, 49), np.float32)   # 6,422,528 MACs

ct, game = unswbc.init()
while unswbc.update(ct, game):
    t0 = time.time_ns()
    a @ b
    t1 = time.time_ns()
    ct.output_log("selfmeter", t1 - t0)
    ct.make_moves([Direction.NORTH])
    unswbc.end_turn()
