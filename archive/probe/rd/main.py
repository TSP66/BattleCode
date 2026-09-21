"""Prices reading a file inside the sandbox, by size and by chunking."""
import os
import time
import helper as unswbc
from helper import Direction

HERE = os.path.dirname(os.path.abspath(__file__))
ct, game = unswbc.init()
round_no = 0
while unswbc.update(ct, game):
    round_no += 1
    if round_no == 1:
        t0 = time.time_ns(); open(os.path.join(HERE, "a.dat"), "rb").read(); t1 = time.time_ns()
        ct.output_log("read 100KB", t1 - t0)
    elif round_no == 2:
        t0 = time.time_ns(); open(os.path.join(HERE, "b.dat"), "rb").read(); t1 = time.time_ns()
        ct.output_log("read 1MB", t1 - t0)
    elif round_no == 3:
        t0 = time.time_ns()
        with open(os.path.join(HERE, "c.dat"), "rb") as f:
            while f.read(65536):
                pass
        t1 = time.time_ns()
        ct.output_log("read 3.73MB chunked", t1 - t0)
    ct.make_moves([Direction.NORTH])
    unswbc.end_turn()
