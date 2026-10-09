"""The queen channels of the policy's grid (cpp/bc_memory.hpp grid_cfg IS_QUEEN..EQ_FRESH) and
the queen bits of the sonar packet, against the true board (2026-10-01, the queen rules).

Self-play with both teams speaking sonar v2, random legal moves. Per row, from the board
(planes 8-9 the acting dragon, 18-19 our queen and theirs):
  * IS_QUEEN is 1 exactly when the acting dragon is our queen;
  * ALLY_QUEEN / ENEMY_QUEEN are exactly the queens' cells in the live 7x7 window (not the
    acting dragon's own body);
  * a queen in the window is known this round (FRESH 1) at one of its cells -- its head if the
    head shows -- and a queen knows where it is itself (offset 0);
  * the vector and the memory plane agree, and everything is in range;
  * a verified packet that says "queen" came from near our queen, and one that names the enemy
    queen names a cell near it -- within 4, plus 4 for each round of a RELAYED report's age;
    and relays (age 1-3, a position heard or seen earlier and passed on) do happen.
And it counts how often a dragon knows where a queen is without seeing it: only sonar and
memory can do that, so a zero means the reports never arrive.

    python3 tests/test_queens.py
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("BCSIM_LIB", str(ROOT / "bcsim" / "libbcvec_priv_s2_g15.so"))
sys.path.insert(0, str(ROOT))

import numpy as np                  # noqa: E402
import bcsim                        # noqa: E402
from train import sonar2 as S2      # noqa: E402

IS_QUEEN, ALLY_QUEEN, ENEMY_QUEEN, ALLY_MEM, ENEMY_MEM = 43, 44, 45, 46, 47
VEC = {0: 48, 1: 51}                # side -> DX; DY and FRESH follow
VISION = 3


def ego_to_world(f: int, ox: int, oy: int) -> tuple[int, int]:
    return [(ox, oy), (-oy, ox), (-ox, -oy), (oy, -ox)][f]


def main() -> None:
    maps = bcsim.load_maps([str(p) for p in sorted((ROOT.parent / "maps-gate").glob("*.map"))])
    N = 22
    env = bcsim.BattlecodeVecEnv(maps, num_envs=N, num_threads=4, seed=5, privileged=True, board=True,
                                 grid=True, sonar=True)
    G = env.grid.shape[2]
    HALF = G // 2
    assert env.sonar2 and env.grid.shape[1] in (54, 57), env.grid.shape
    for i in range(N):
        env.set_opponent(i, team=-1, bot=0, map_index=i % len(maps))
        env.set_sonar2(i, (0, 1))
    obs = env.reset()
    rng = np.random.default_rng(0)
    sc = {k: bcsim.SCALARS.index(k) for k in ("map_w", "map_h", "face_n")}
    st = dict(rows=0, queen_rows=0, ally_in_view=0, enemy_in_view=0, ally_known_unseen=0,
              enemy_known_unseen=0, pkt_queen=0, pkt_queen_ok=0, pkt_eq=0, pkt_eq_ok=0,
              pkt_queen_dead=0, pkt_eq_dead=0, pkt_queen_far=0, pkt_eq_far=0, pkt_eq_relayed=0)
    for step in range(3000):
        B = env.board
        for e in range(N):
            g = env.grid[e]
            W = int(round(obs.scalar[e, sc["map_w"]] * 64))
            H = int(round(obs.scalar[e, sc["map_h"]] * 64))
            f = int(np.argmax(obs.scalar[e, sc["face_n"]:sc["face_n"] + 4]))
            hy, hx = np.argwhere(B[e, 9])[0]
            me = (B[e, 8] | B[e, 9]) > 0
            queen = bool(B[e, 18, hy, hx])
            assert (g[IS_QUEEN] == float(queen)).all(), f"IS_QUEEN wrong at step {step} env {e}"
            st["rows"] += 1
            st["queen_rows"] += queen
            seen = {0: [], 1: []}
            for row in range(7):
                for col in range(7):
                    wx, wy = ego_to_world(f, col - VISION, row - VISION)
                    x, y = (hx + wx) % W, (hy + wy) % H
                    want = {0: bool(B[e, 18, y, x]) and not me[y, x], 1: bool(B[e, 19, y, x])}
                    for side, ch in ((0, ALLY_QUEEN), (1, ENEMY_QUEEN)):
                        got = g[ch, row + HALF - VISION, col + HALF - VISION]
                        assert got == float(want[side]), f"{'ALLY' if side == 0 else 'ENEMY'}_QUEEN step {step} env {e}"
                        if want[side]:
                            seen[side].append((x, y, bool(B[e, 1 if side == 0 else 3, y, x])))
            for side in (0, 1):
                dx, dy, fresh = (float(g[VEC[side] + k, 0, 0]) for k in range(3))
                for k in range(3):
                    assert (g[VEC[side] + k] == g[VEC[side] + k, 0, 0]).all(), "a vector plane is not constant"
                assert 0.0 <= fresh <= 1.0 and -1.0 <= dx <= 1.0 and -1.0 <= dy <= 1.0
                mem = g[ALLY_MEM if side == 0 else ENEMY_MEM]
                ox, oy = int(round(dx * 32)), int(round(dy * 32))
                if fresh > 0 and -HALF <= ox < G - HALF and -HALF <= oy < G - HALF:
                    assert mem[oy + HALF, ox + HALF] == fresh and (mem > 0).sum() == 1, "memory plane off its vector"
                else:
                    assert (mem == 0).all(), "a memory plane with nothing known"
                if side == 0 and queen:
                    assert fresh == 1.0 and ox == 0 and oy == 0, "a queen does not know where it is"
                if seen[side]:
                    st["ally_in_view" if side == 0 else "enemy_in_view"] += 1
                    assert fresh == 1.0, "a queen in view is not known this round"
                    wx, wy = ego_to_world(f, ox, oy)
                    at = ((hx + wx) % W, (hy + wy) % H)
                    cells = {(x, y) for x, y, _ in seen[side]}
                    heads = {(x, y) for x, y, h in seen[side] if h}
                    assert at in (heads or cells), f"queen {side} placed at {at}, not {heads or cells}"
                elif fresh > 0 and not (side == 0 and queen):
                    st["ally_known_unseen" if side == 0 else "enemy_known_unseen"] += 1
            # the packets' queen bits against the board: senders and sightings are a move or so stale
            team, rnd = int(obs.team[e]), int(obs.round[e])
            for v in obs.msgs[e, :min(int(obs.num_msgs[e]), obs.msgs.shape[1])].tolist():
                p = S2.decode(v, team, rnd) if v else None
                if p is None:
                    continue
                for flag, plane, x, y, key in (("queen", 18, p["hx"], p["hy"], "pkt_queen"),
                                               ("enemy_queen", 19, p["ex"], p["ey"], "pkt_eq")):
                    if not p[flag]:
                        continue
                    cells = np.argwhere(B[e, plane, :H, :W] > 0)
                    if not len(cells):
                        st[key + "_dead"] += 1      # the queen died after the report was sent
                        continue
                    st[key] += 1
                    age = p["queen_age"] if flag == "enemy_queen" else 0
                    st["pkt_eq_relayed"] += int(age > 0)
                    ddx = np.abs(cells[:, 1] - x); ddy = np.abs(cells[:, 0] - y)
                    d = np.maximum(np.minimum(ddx, W - ddx), np.minimum(ddy, H - ddy)).min()
                    st[key + "_ok"] += int(d <= 4 + 4 * age)
                    if d > 4 + 4 * age and st[key + "_far"] < 3:
                        st[key + "_far"] += 1
                        print(f"  far {key}: step {step} env {e} round {rnd} sent {p['round']} at {(x, y)}, "
                              f"nearest queen cell {d} away")
        acts = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in obs.mask], np.int32)
        for e in range(N):
            env.intent[e] = rng.dirichlet(np.ones(4)) * 0.5
        obs, _, _ = env.step(acts)
    print(st)
    assert st["queen_rows"] > 0 and st["enemy_in_view"] > 0 and st["ally_in_view"] > 0, "a case never came up"
    assert st["ally_known_unseen"] > 0 and st["enemy_known_unseen"] > 0, "no queen known out of sight: reports never land"
    assert st["pkt_queen"] > 0 and st["pkt_eq"] > 0, "no queen packet was ever verified"
    assert st["pkt_queen_ok"] / st["pkt_queen"] > 0.99, "a 'queen' packet came from elsewhere"
    assert st["pkt_eq_ok"] / st["pkt_eq"] > 0.99, "an enemy-queen report named the wrong place"
    assert st["pkt_eq_relayed"] > 0, "no enemy-queen position was ever relayed"
    print("OK")


if __name__ == "__main__":
    main()
