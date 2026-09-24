"""Reward bookkeeping invariants.

  1. every kill whose killer survives is paid, and no other: the kills
     credited in closures equal the kills recorded minus head-on kills (where
     the killer died too, which reward v3 deliberately does not pay);
  2. every death is paid once, to the dragon that died;
except for credit held by a dragon that never took a turn, which has no
transition to be paid into.

    python tests/test_rewards.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root

COMP = {n: i for i, n in enumerate(bcsim.REWARD_COMPS)}
COLS = bcsim.EpisodeStats.COLUMNS


def run(gamma: float, steps: int = 4000, n: int = 64, seed: int = 0, masked: bool = True):
    maps = bcsim.load_maps(str(ROOT / "maps"))
    env = bcsim.BattlecodeVecEnv(maps, num_envs=n, num_threads=8, seed=seed)
    env.set_potential_gamma(gamma)
    obs = env.reset()
    rng = np.random.default_rng(seed)
    kills_paid = deaths_paid = 0.0
    kills_seen = deaths_seen = 0
    finished = set()
    per_env_kills = np.zeros(n)
    per_env_deaths = np.zeros(n)
    for _ in range(steps):
        if masked:
            m = obs.mask.astype(np.float64)
            m[m.sum(1) == 0] = 1
            # aggressive random play: sprints and head-ons happen often
            acts = np.array([rng.choice(np.nonzero(r)[0]) for r in m], np.int32)
        else:
            # Masked play cannot test premise 1 at all. A step is one tile, so
            # the 7x7 window always contains the target and the mask always
            # forbids moving onto a body -- measured over 396 masked games: 372
            # kills, ALL 372 of them head-ons, zero non-head-on kills. The only
            # way a cross-team body collision happens is to ignore the mask and
            # let the illegal move kill the dragon, which reaches 296 non-head-on
            # kills over 18,910 games.
            acts = rng.integers(0, bcsim.N_ACTIONS, size=obs.mask.shape[0]).astype(np.int32)
        obs, cl, eps = env.step(acts)
        if len(cl.uid):
            np.add.at(per_env_kills, cl.env, cl.comps[:, COMP["kills"]])
            np.add.at(per_env_deaths, cl.env, cl.comps[:, COMP["died"]])
        for r in eps.rows:
            e = int(r[COLS.index("env")])
            # closures for a finished game are all emitted by the step that
            # ended it, so the tallies for that env are complete now
            kills_seen += int(r[COLS.index("a_kills")] + r[COLS.index("b_kills")]
                              - r[COLS.index("a_headon")] - r[COLS.index("b_headon")])
            deaths_seen += int(r[COLS.index("a_deaths")] + r[COLS.index("b_deaths")])
            kills_paid += per_env_kills[e]
            deaths_paid += per_env_deaths[e]
            per_env_kills[e] = per_env_deaths[e] = 0
            finished.add(e)
    return kills_seen, kills_paid, deaths_seen, deaths_paid, len(finished)


def main() -> int:
    """Runs itself in a subprocess against an instrumented build, which reports
    the credit held by dragons that never took a turn: they made no decision,
    so no transition exists to pay, and that credit is dropped by design."""
    import os
    import subprocess
    import tempfile
    if os.environ.get("BCSIM_LIB") is None:
        root = pathlib.Path(__file__).resolve().parents[1]
        lib = pathlib.Path(tempfile.mkdtemp()) / "libbcvec_dbg.so"
        subprocess.run(["g++", "-O2", "-std=c++17", "-shared", "-fPIC", "-pthread",
                        "-DBC_DEBUG_REWARD", str(root / "cpp/bc_capi_vec.cpp"), "-o", str(lib)],
                       check=True)
        res = subprocess.run([sys.executable, __file__], env={**os.environ, "BCSIM_LIB": str(lib)},
                             capture_output=True, text=True)
        print(res.stdout, end="")
        unpaid = [l.split() for l in res.stderr.splitlines() if l.startswith("UNPAID")]
        if any(u[4] != "0" for u in unpaid):
            print("FAIL: a dragon that had acted went unpaid")
            return 1
        ud = sum(float(u[6]) for u in unpaid)
        uk = sum(float(u[8]) for u in unpaid)
        nums = [float(x) for x in res.stdout.split("RESULT")[1].split()]
        ks, kp, ds, dp = nums
        print(f"never-acted dragons held {ud:.0f} deaths and {uk:.0f} kills")
        ok = ks > 0 and abs(kp + uk - ks) < 0.5 and abs(dp + ud - ds) < 0.5
        print("PASS: every kill and death is paid, or belongs to a dragon that never acted"
              if ok else "FAIL: paid + never-acted != recorded")
        return 0 if ok else 1
    # Both regimes: masked is the realistic one and is where the death
    # accounting gets its volume; unmasked is the only one that produces a
    # non-head-on kill, which is the only thing that exercises premise 1.
    a = run(0.997, masked=True)
    print(f"masked   {a[4]} games: kills recorded {a[0]}, paid {a[1]:.0f}; "
          f"deaths recorded {a[2]}, paid {a[3]:.0f}")
    b = run(0.997, steps=2500, masked=False)
    print(f"unmasked {b[4]} games: kills recorded {b[0]}, paid {b[1]:.0f}; "
          f"deaths recorded {b[2]}, paid {b[3]:.0f}")
    print(f"RESULT {a[0] + b[0]} {a[1] + b[1]} {a[2] + b[2]} {a[3] + b[3]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
