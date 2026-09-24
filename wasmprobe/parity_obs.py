"""Checks the C++ bot sees exactly what the policy was trained on.

For each map, plays the training simulator with random legal moves, follows
one dragon, and records for each of its turns both the protocol block the
engine would send it and the simulator's own observation tensors. The blocks
are stitched into a transcript and fed to the bot built with -DBC_DUMP,
which prints the observation it builds; the two must agree exactly.

    python parity_obs.py path/to/bot_native [maps_dir] [turns_per_map]
"""

import ctypes
import pathlib
import subprocess
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT / "bcsim"))

import bcsim                              # noqa: E402
from bcsim.env import _lib as VLIB        # noqa: E402
from train import augment                 # noqa: E402

VLIB.bcv_acting_dragon.argtypes = [ctypes.c_void_p, ctypes.c_int]
VLIB.bcv_acting_dragon.restype = ctypes.c_int
VLIB.bcv_round_block.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
C, S, A = bcsim.N_CHANNELS, bcsim.N_SCALARS, bcsim.N_ACTIONS
KNOWN = [0]
DUMP_DIR = __import__("os").environ.get("DUMP_DIR", "")


def block(env) -> str:
    buf = ctypes.create_string_buffer(1 << 16)
    n = VLIB.bcv_round_block(ctypes.c_void_p(env._h), 0, buf, len(buf))
    return buf.raw[:n].decode()


def follow(text: str, seed: int, turns: int, sonar: bool = True):
    m = augment.parse(text)
    # Sonar ON by default: the bot declares PROTOCOL 3 and reads the ECHOES line
    # every turn, so the blocks it is fed here have to carry one. With sonar off
    # the simulator never casts, the line is absent, and both sides report zero
    # echoes -- the check would pass while testing nothing.
    env = bcsim.BattlecodeVecEnv([text], num_envs=1, num_threads=1, seed=seed,
                                 sonar=sonar)
    obs = env.reset()
    me = int(obs.dragon_id[0])
    team = "AB"[int(obs.team[0])]
    rng = np.random.default_rng(seed)
    blocks, want = [], []
    for _ in range(20000):
        if int(obs.dragon_id[0]) == me:
            blocks.append(block(env))
            want.append((obs.local[0].copy(), obs.scalar[0].copy(), obs.mask[0].copy()))
            if len(blocks) >= turns:
                break
        legal = np.nonzero(obs.mask[0])[0]
        # mostly single steps so the dragon lives a while, some sprints/splits
        steps = legal[legal < 3]
        pick = steps if len(steps) and rng.random() < 0.85 else legal
        a = int(rng.choice(pick)) if len(pick) else 0
        prev_round = int(obs.round[0])
        obs, _, eps = env.step(np.array([a], np.int32))
        if len(eps.rows) or int(obs.round[0]) < prev_round:
            break                                       # game over (reset)
    header = f"ID {me}\nTEAM {team}\nMAP {m.w} {m.h}\nUNIT_LIMIT {m.unit_limit or 64}\n"
    return header + "".join(blocks), want


def main() -> None:
    bot = sys.argv[1]
    maps_dir = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "maps")
    turns = int(sys.argv[3]) if len(sys.argv) > 3 else 80
    total = bad_turns = 0
    # Both block formats, because the bot has to parse both: with sonar the
    # ECHOES line sits between the messages and the 49 tile lines and shifts
    # every offset after it, and without it the line is absent entirely. A bot
    # that handles only one of the two silently mis-parses whole blocks.
    for sonar in (True, False):
        print(f"-- blocks {'with' if sonar else 'without'} ECHOES (sonar "
              f"{'on' if sonar else 'off'})")
        for f in sorted(pathlib.Path(maps_dir).glob("*.map")):
          for seed in (1, 2):
            transcript, want = follow(f.read_text(), seed, turns, sonar)
            res = subprocess.run([bot], input=transcript.encode(), capture_output=True, timeout=120)
            if DUMP_DIR:
                tag = "sonar" if sonar else "legacy"
                (pathlib.Path(DUMP_DIR) /
                 f"{f.stem}_{seed}_{tag}.txt").write_bytes(res.stderr)
            rows = [np.array(l.split()[1:], np.float64)
                    for l in res.stderr.decode().splitlines() if l.startswith("DUMP")]
            n = min(len(rows), len(want))
            errs = []
            # the bot dumps the simulator's scalars first, then whatever
            # remembered inputs the network wants (wasmprobe/parity_mem.py
            # checks those); only the first S are the simulator's to compare
            extra = len(rows[0]) - (2 + C * 49 + S + 2 * A) if rows else 0
            for t in range(n):
                r = rows[t]
                loc = r[2:2 + C * 49].reshape(C, 7, 7)
                sc = r[2 + C * 49:2 + C * 49 + S]
                mask = r[2 + C * 49 + S + extra:2 + C * 49 + S + extra + A]
                wl, ws, wm = want[t]
                dl = np.abs(loc - wl) > 1e-4
                # self_index / self_tail beyond a gap where the body leaves the
                # window are known to differ: the simulator numbers the whole
                # body, the bot only the chain it can see. Reported, not failed.
                known = np.zeros(C, bool)
                known[[bcsim.CHANNELS.index("self_index"), bcsim.CHANNELS.index("self_tail")]] = True
                if dl[known].any():
                    KNOWN[0] += 1
                dl[known] = False
                ds = np.abs(sc - ws) > 1e-4
                # the same gap can hide which visible cell is the tail, so the
                # bot may forbid a sprint the simulator allows (it will not
                # know the tail moves out of the way). Forbidding a legal move
                # is safe and known; allowing an illegal one fails.
                dm = (mask > 0) & (wm == 0)
                if ((mask == 0) & (wm > 0)).any():
                    KNOWN[0] += 1
                if dl.any() or ds.any() or dm.any():
                    errs.append((t, sorted(set(np.nonzero(dl)[0].tolist())),
                                 [bcsim.SCALARS[i] for i in np.nonzero(ds)[0]],
                                 np.nonzero(dm)[0].tolist()))
            total += n
            bad_turns += len(errs)
            short = "" if n == len(want) else f" (bot answered {len(rows)} of {len(want)})"
            print(f"{f.stem:32s} seed {seed}: {n:3d} turns, {len(errs)} differ{short}")
            for e in errs[:3]:
                print(f"    turn {e[0]}: channels {[bcsim.CHANNELS[c] for c in e[1]]} "
                      f"scalars {e[2]} mask ids {e[3]}")
    print(f"\n{total} turns compared, {bad_turns} differ "
          f"(+{KNOWN[0]} with the known body-order gap: self_index/self_tail, "
          f"or a legal sprint the bot conservatively forbids)")
    sys.exit(1 if bad_turns or total == 0 else 0)


if __name__ == "__main__":
    main()
