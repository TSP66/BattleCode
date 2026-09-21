"""Observation parity where the map wraps: the 7x7 window crosses the edge of
the torus and another dragon is inside it.

parity_obs.py follows a single dragon, which seldom meets an enemy right at
the seam. This plays many random games, keeps only the turns whose window
wraps around the map edge with some other dragon's body in view, and feeds
each one to the bot as a one-turn transcript.

    python parity_seam.py path/to/bot_native [n_turns]
"""

import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT / "bcsim"))
sys.path.insert(0, str(ROOT / "wasmprobe"))

import bcsim                                  # noqa: E402
from parity_obs import block, C, S, A         # noqa: E402
from train import augment                     # noqa: E402

CH = {n: i for i, n in enumerate(bcsim.CHANNELS)}
OTHER = [CH[k] for k in ("ally_head", "ally_body", "enemy_head", "enemy_body")]


def main() -> None:
    bot = sys.argv[1]
    want_n = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    total = bad = 0
    for f in sorted(ROOT / "maps".glob("*.map")):
        text = f.read_text()
        m = augment.parse(text)
        env = bcsim.BattlecodeVecEnv([text], num_envs=1, num_threads=1, seed=11)
        obs = env.reset()
        rng = np.random.default_rng(3)
        cases = []
        for _ in range(60000):
            sc = obs.scalar[0]
            hx = int(round(sc[bcsim.SCALARS.index("head_x")] * m.w))
            hy = int(round(sc[bcsim.SCALARS.index("head_y")] * m.h))
            wraps = hx < 3 or hx > m.w - 4 or hy < 3 or hy > m.h - 4
            if wraps and obs.local[0, OTHER].sum() > 0:
                header = (f"ID {int(obs.dragon_id[0])}\nTEAM {'AB'[int(obs.team[0])]}\n"
                          f"MAP {m.w} {m.h}\nUNIT_LIMIT {m.unit_limit or 64}\n")
                cases.append((header + block(env), obs.local[0].copy(), obs.scalar[0].copy(),
                              obs.mask[0].copy()))
                if len(cases) >= want_n // 11 + 1:
                    break
            legal = np.nonzero(obs.mask[0])[0]
            a = int(rng.choice(legal)) if len(legal) else 0
            obs, _, _ = env.step(np.array([a], np.int32))
        errs = []
        for tr, wl, ws, wm in cases:
            res = subprocess.run([bot], input=tr.encode(), capture_output=True, timeout=60)
            r = next((np.array(l.split()[1:], np.float64)
                      for l in res.stderr.decode().splitlines() if l.startswith("DUMP")), None)
            if r is None:
                errs.append(("no answer", [], [], []))
                continue
            loc = r[2:2 + C * 49].reshape(C, 7, 7)
            sc = r[2 + C * 49:2 + C * 49 + S]
            mask = r[2 + C * 49 + S:2 + C * 49 + S + A]
            dl = np.abs(loc - wl) > 1e-4
            dl[[CH["self_index"], CH["self_tail"]]] = False     # known gap, KNOWN_ISSUES.md
            ds = np.abs(sc - ws) > 1e-4
            dm = (mask > 0) & (wm == 0)                          # bot allows what sim forbids
            if dl.any() or ds.any() or dm.any():
                errs.append(("", [bcsim.CHANNELS[c] for c in sorted(set(np.nonzero(dl)[0]))],
                             [bcsim.SCALARS[i] for i in np.nonzero(ds)[0]],
                             np.nonzero(dm)[0].tolist()))
        total += len(cases)
        bad += len(errs)
        print(f"{f.stem:32s} {len(cases):3d} seam turns with another dragon in view, "
              f"{len(errs)} differ")
        for e in errs[:3]:
            print(f"    {e}")
    print(f"\n{total} seam turns, {bad} differ")
    sys.exit(1 if bad or total == 0 else 0)


if __name__ == "__main__":
    main()
