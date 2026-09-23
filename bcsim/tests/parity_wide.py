"""The wide planes agree with the flat `mem` feature where the two overlap.

`wide`'s near scale is radius 7 and `mem` is radius 6, both in the head's own
frame, and `wide`'s first four channels are built from the same four quantities.
So the inner 13x13 of wide[0:4] must equal mem reshaped to (4, 13, 13) exactly
-- if it does not, the rotation, the wrap or the channel order is wrong.

The far scale is checked separately: every far cell must be the mean of the
POOL x POOL block of near-scale quantities around its centre, which is only
testable where that whole block lands inside the near crop (|offset| <= 1).

    /usr/bin/python3 tests/parity_wide.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import bcsim  # noqa: E402

R_MEM, R_WIDE, POOL = 6, 7, 4
SIDE_MEM, SIDE_WIDE = 2 * R_MEM + 1, 2 * R_WIDE + 1
N_BASE = 14          # scalars before mem
N_MEM = 4 * SIDE_MEM * SIDE_MEM


def check(map_path: pathlib.Path, turns: int, seed: int) -> tuple[int, float, float]:
    env = bcsim.BattlecodeVecEnv([map_path.read_text()], num_envs=16, num_threads=4,
                                 seed=seed, wide=True)
    obs = env.reset()
    rng = np.random.default_rng(seed)
    worst_near, worst_far, compared = 0.0, 0.0, 0

    for _ in range(turns):
        mem = obs.scalar[:, N_BASE:N_BASE + N_MEM]
        mem = mem.reshape(-1, 4, SIDE_MEM, SIDE_MEM)
        near = env.wide[:, :4]                       # (n, 4, 15, 15)
        inner = near[:, :, 1:1 + SIDE_MEM, 1:1 + SIDE_MEM]
        worst_near = max(worst_near, float(np.abs(inner - mem).max()))

        # far cell (row, col) averages the POOL x POOL ego block centred on
        # POOL * offset; only |offset| <= 1 keeps that block inside the near crop
        full = env.wide[:, :6]
        far = env.wide[:, 6:]
        for orow in (-1, 0, 1):
            for ocol in (-1, 0, 1):
                rows = [POOL * orow + s - POOL // 2 + R_WIDE for s in range(POOL)]
                cols = [POOL * ocol + s - POOL // 2 + R_WIDE for s in range(POOL)]
                block = full[:, :, rows][:, :, :, cols]
                want = block.mean(axis=(2, 3))
                got = far[:, :, orow + R_WIDE, ocol + R_WIDE]
                worst_far = max(worst_far, float(np.abs(want - got).max()))
        compared += obs.scalar.shape[0]

        legal = obs.mask.astype(np.float64)
        legal /= np.maximum(legal.sum(1, keepdims=True), 1e-9)
        acts = np.array([rng.choice(len(p), p=p) if p.sum() > 0 else 0 for p in legal],
                        dtype=np.int32)
        obs, _, _ = env.step(acts)
    return compared, worst_near, worst_far


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[2]
    maps = sorted((root / "maps-live").glob("*.map"))
    if not maps:
        print("no maps in maps-live", file=sys.stderr)
        return 1
    total, near, far = 0, 0.0, 0.0
    for path in maps:
        n, wn, wf = check(path, turns=40, seed=abs(hash(path.name)) % 1000)
        print(f"  {path.stem:<18} near {wn:.3g}  far {wf:.3g}")
        total += n
        near, far = max(near, wn), max(far, wf)
    ok = near == 0.0 and far < 1e-6
    print(f"\n{'parity holds' if ok else 'PARITY BROKEN'}: {total:,} turns, "
          f"max |diff| near {near:.3g}, far {far:.3g}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
