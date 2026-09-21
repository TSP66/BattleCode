"""Cost of one matmul of a known MAC count, by dtype and shape."""
import numpy as np
import helper as unswbc
from helper import Direction

PART = "mm32"

a32 = np.zeros((256, 1024), np.float32); b32 = np.zeros((1024, 49), np.float32)
a64 = a32.astype(np.float64); b64 = b32.astype(np.float64)
m32 = np.zeros((512, 1024), np.float32); v32 = np.zeros(1024, np.float32)
s32 = np.zeros((32, 288), np.float32); c32 = np.zeros((288, 49), np.float32)
relu_x = np.zeros((96, 49), np.float32)


def work():
    if PART == "mm32":      a32 @ b32          # 12,845,056 MACs
    elif PART == "mm64":    a64 @ b64
    elif PART == "matvec":  m32 @ v32          #    524,288 MACs
    elif PART == "small":   s32 @ c32          #    451,584 MACs
    elif PART == "relu":
        for _ in range(8): np.maximum(relu_x, 0.0)
    elif PART == "tanh":
        for _ in range(8): np.tanh(relu_x)


ct, game = unswbc.init()
while unswbc.update(ct, game):
    work()
    ct.make_moves([Direction.NORTH])
    unswbc.end_turn()
