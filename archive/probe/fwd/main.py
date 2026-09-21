"""Forward pass on zeroed weights: measures compute with no load cost."""
import numpy as np
import helper as unswbc
from helper import Direction
import policy as P

W, B = 48, 4


class Fake(P.Policy):
    def __init__(self):
        w = {}
        w["stem_c"] = np.zeros((W, 23 * 9), np.float32)
        w["stem_n_w"] = np.ones(W, np.float32); w["stem_n_b"] = np.zeros(W, np.float32)
        for i in range(B):
            for j in (1, 2):
                w[f"b{i}_c{j}"] = np.zeros((W, W * 9), np.float32)
                w[f"b{i}_n{j}_w"] = np.ones(W, np.float32)
                w[f"b{i}_n{j}_b"] = np.zeros(W, np.float32)
        w["flat_c"] = np.zeros((32, W), np.float32)
        w["flat_n_w"] = np.ones(32, np.float32); w["flat_n_b"] = np.zeros(32, np.float32)
        for n, o, i in (("sc0", 128, 14), ("sc2", 128, 128), ("fu0", 512, 1696),
                        ("fu2", 512, 512), ("pi", 48, 512)):
            w[n + "_w"] = np.zeros((o, i), np.float32)
            w[n + "_b"] = np.zeros(o, np.float32)
        self.w = w
        self.blocks, self.width = B, W
        self._pad = np.zeros((W, 81), np.float32)


ct, game = unswbc.init()
pol = Fake()
local = np.zeros((23, 7, 7), np.float32)
sc = np.zeros(14, np.float32)
mask = np.ones(48, np.float32)
while unswbc.update(ct, game):
    pol.act(local, sc, mask)
    ct.make_moves([Direction.NORTH])
    unswbc.end_turn()
