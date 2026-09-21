"""The egocentric window must be exactly the world window, rotated, with the
direction channels permuted to match. Anything else silently teaches the policy
a frame it will not see at submission time."""
import pathlib, sys, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import bcsim

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root

DIRS = "NESW"
ROT = {name: i for i, name in enumerate(bcsim.CHANNELS)}
REL = ["fwd", "right", "back", "left"]


def run(maps, ego, actions):
    env = bcsim.BattlecodeVecEnv(maps, num_envs=8, num_threads=2, seed=3,
                                 egocentric=ego, random_pearl_seed=False)
    obs = env.reset()
    frames = []
    for a in actions:
        frames.append((obs.local.copy(), obs.scalar.copy()))
        obs, _, _ = env.step(a)
    env.close()
    return frames


def main():
    maps = bcsim.load_maps(str(ROOT / "maps"))
    rng = np.random.default_rng(0)
    actions = [rng.integers(0, 3, size=8).astype(np.int32) for _ in range(120)]
    world = run(maps, False, actions)
    ego = run(maps, True, actions)

    checked = 0
    for (w_local, w_scalar), (e_local, e_scalar) in zip(world, ego):
        assert np.array_equal(w_scalar, e_scalar), "scalars must not depend on the frame"
        for i in range(w_local.shape[0]):
            facing = int(np.argmax(w_scalar[i, 4:8]))       # face_n .. face_w
            # cell (row, col) of the ego window is the board cell you reach by
            # walking (col - 3) to the dragon's right and (row - 3) behind it
            expect = np.zeros_like(w_local[i])
            for row in range(7):
                for col in range(7):
                    ox, oy = col - 3, row - 3
                    wx, wy = [(ox, oy), (-oy, ox), (-ox, -oy), (oy, -ox)][facing]
                    for c, name in enumerate(bcsim.CHANNELS):
                        src = c
                        for group in ("face_", "kelp_", "portal_"):
                            if name.startswith(group):
                                rel = REL.index(name[len(group):])
                                src = ROT[group + REL[(rel + facing) % 4]]
                        expect[c, row, col] = w_local[i][src, wy + 3, wx + 3]
            assert np.allclose(e_local[i], expect), f"ego frame wrong for facing {DIRS[facing]}"
            checked += 1
    print(f"egocentric rotation verified on {checked} observations")


main()
