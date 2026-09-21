import helper as unswbc
from helper import Direction
import obs as O

DIRS = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)
ct, game = unswbc.init()
while unswbc.update(ct, game):
    snap = O.Snapshot(ct, game)
    mask = snap.mask()
    local = snap.local().reshape(-1)
    sc = snap.scalars()

    lsum = sum(int(v * 1000.0 + 0.5) * (i + 1) for i, v in enumerate(local.tolist()))
    ssum = sum(int(v * 1000.0 + 0.5) * (i + 1) for i, v in enumerate(sc.tolist()))
    bits = "".join("1" if m else "0" for m in mask)

    act = next((i for i in range(O.N_ACTIONS) if mask[i]), 0)
    ct.output_log("P", ct.get_id(), game.round_num, bits, lsum, ssum, act)
    if game.round_num <= 1:
        w2 = snap.local()
        for ch in range(O.N_CHANNELS):
            row = w2[ch].reshape(-1).tolist()
            s = sum(int(v * 1000.0 + 0.5) * (j + 1) for j, v in enumerate(row))
            ct.output_log("C", ct.get_id(), game.round_num, ch, s)

    if act < O.N_MOVES:
        ct.make_moves([DIRS[d] for d in O.decode_move(act, snap.facing)])
    else:
        k = O.SPLIT_K[act - O.N_MOVES]
        ct.do_split(snap.length // 2 if k < 0 else k)
    unswbc.end_turn()
