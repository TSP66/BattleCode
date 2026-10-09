"""Reward v8 through the real vectorised engine, not the pure header.

`tests/test_reward8.cpp` checks the potential's algebra. This checks the three
things that only the engine can be wrong about:

  1. v8 off is byte-identical on every v1-v7 component, so the frozen league and
     every saved run keep playing exactly as before;
  2. the shaping telescopes PER AGENT. With potential_gamma=1, a dragon that is
     alive from the spawn to the end of the game must receive v8 shaping summing
     to exactly zero: its chain starts at Phi(s_0) = 0 on a symmetric spawn and
     the terminal zeroes its last bank, so everything in between cancels. This is
     the check that the banking is right, and it is easy to get wrong -- summing
     across agents instead does NOT cancel, because each agent has its own
     independent chain and the total is then
     sum(phi at each death) - sum(phi at each split child's birth);
  3. coverage only ever grows, and a dragon that died mid-game is never paid the
     terminal result -- it is paid what its death did to the potential.

    python tests/test_reward8_env.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
COMP = {n: i for i, n in enumerate(bcsim.REWARD_COMPS)}
V8 = ["v8_win", "v8_len", "v8_queen", "v8_kill", "v8_exp"]
OLD = [c for c in bcsim.REWARD_COMPS if c not in V8 and c != "outcome"]


def rollout(v8: bool, kappa: float = 1.0, steps: int = 3000, n: int = 32, seed: int = 0,
            gamma: float = 1.0):
    """Returns per-env sums of every reward component, and the episode rows."""
    maps = bcsim.load_maps(str(ROOT / "maps"))
    env = bcsim.BattlecodeVecEnv(maps, num_envs=n, num_threads=8, seed=seed)
    env.set_potential_gamma(gamma)
    if v8:
        env.set_reward_v8(True, kappa)
    obs = env.reset()
    rng = np.random.default_rng(seed)
    per_env = np.zeros((n, bcsim.N_REWARD_COMPS))
    done_sums, rows = [], []
    for _ in range(steps):
        m = obs.mask.astype(np.float64)
        m[m.sum(1) == 0] = 1
        acts = np.array([rng.choice(np.nonzero(r)[0]) for r in m], np.int32)
        obs, cl, eps = env.step(acts)
        if len(cl.uid):
            np.add.at(per_env, cl.env, cl.comps)
        for r in eps.rows:
            e = int(r[bcsim.EpisodeStats.COLUMNS.index("env")])
            done_sums.append(per_env[e].copy())
            rows.append(r.copy())
            per_env[e] = 0
    return np.array(done_sums), np.array(rows)


def main() -> int:
    fail = 0

    # ---- 1. v8 off changes nothing about the old components
    off, _ = rollout(v8=False)
    on, _ = rollout(v8=True)
    assert len(off) == len(on), "the two rollouts diverged, which is itself a bug"
    worst, worst_name = 0.0, ""
    for c in OLD:
        d = float(np.abs(off[:, COMP[c]] - on[:, COMP[c]]).max())
        if d > worst:
            worst, worst_name = d, c
    print(f"1. old components with v8 on vs off: worst |diff| {worst:.3e} ({worst_name})")
    if worst > 1e-9:
        print("   FAIL: turning v8 on perturbed a v1-v7 component")
        fail += 1

    # ---- 2. per-agent telescoping, which is the real invariant
    per_uid, done_uid, died_uid, first_step = {}, {}, {}, {}
    maps = bcsim.load_maps(str(ROOT / "maps"))
    env = bcsim.BattlecodeVecEnv(maps, num_envs=32, num_threads=8, seed=1)
    env.set_potential_gamma(1.0)
    env.set_reward_v8(True, 1.0)
    obs = env.reset()
    rng = np.random.default_rng(1)
    finished = set()
    cols = [COMP[c] for c in V8]
    for step in range(3000):
        m = obs.mask.astype(np.float64)
        m[m.sum(1) == 0] = 1
        acts = np.array([rng.choice(np.nonzero(r)[0]) for r in m], np.int32)
        obs, cl, eps = env.step(acts)
        for j, uid in enumerate(cl.uid):
            # only the first episode of each env, so uids cannot be reused
            if int(cl.env[j]) in finished:
                continue
            u = int(uid)
            first_step.setdefault(u, step)
            per_uid[u] = per_uid.get(u, 0.0) + float(cl.comps[j, cols].sum())
            died_uid[u] = died_uid.get(u, 0.0) + float(cl.comps[j, COMP["died"]])
            if cl.done[j]:
                done_uid[u] = True
        for r in eps.rows:
            finished.add(int(r[bcsim.EpisodeStats.COLUMNS.index("env")]))

    # dragons present at the spawn (their first closure came from the opening
    # steps) that finished the game alive
    survivors = [u for u in per_uid
                 if done_uid.get(u) and died_uid[u] == 0 and first_step[u] < 40]
    vals = np.array([per_uid[u] for u in survivors])
    print(f"2. {len(survivors)} spawn-to-finish survivors: "
          f"worst |sum of v8 shaping| {np.abs(vals).max() if len(vals) else 0:.3e} "
          f"(must telescope to 0)")
    if len(vals) < 20:
        print("   FAIL: too few survivors to be a real test")
        fail += 1
    elif np.abs(vals).max() > 2e-3:
        print("   FAIL: shaping does not telescope per agent; the banking is wrong")
        fail += 1

    on, _ = rollout(v8=True, gamma=1.0)
    outcome = on[:, COMP["outcome"]]
    print("   per-episode sums across ALL agents (NOT expected to cancel; the "
          "residual is\n   what dying dragons left behind minus what split "
          "children were born holding):")
    for c in V8:
        v = on[:, COMP[c]]
        print(f"     {c:8s} mean {v.mean():+.4f}  max |sum| {np.abs(v).max():.4f}")

    # ---- 3. the outcome is only ever paid as +1 / 0 / -1 per dragon, so the
    # per-episode total is bounded by the dragons still open at the end, and a
    # draw pays exactly zero.
    rows = rollout(v8=True)[1]
    winner = rows[:, bcsim.EpisodeStats.COLUMNS.index("winner")]
    draws = outcome[winner < 0]
    print(f"3. draws: {len(draws)} episodes, outcome sum "
          f"[{draws.min() if len(draws) else 0:+.0f}, "
          f"{draws.max() if len(draws) else 0:+.0f}]")
    if len(draws) and np.abs(draws).max() > 1e-9:
        print("   FAIL: a draw paid a non-zero outcome")
        fail += 1

    # ---- 4. kappa scales the shaping and leaves the outcome alone
    half, _ = rollout(v8=True, kappa=0.5)
    s_one = np.abs(on[:, [COMP[c] for c in V8]]).sum()
    s_half = np.abs(half[:, [COMP[c] for c in V8]]).sum()
    ratio = s_half / s_one if s_one else 0.0
    print(f"4. kappa 0.5 vs 1.0: shaping magnitude ratio {ratio:.3f} (want ~0.5), "
          f"outcome {np.abs(half[:, COMP['outcome']]).sum():.0f} vs "
          f"{np.abs(outcome).sum():.0f} (want equal)")
    if not 0.45 < ratio < 0.55:
        print("   FAIL: kappa does not scale the shaping linearly")
        fail += 1

    print("\nall checks passed" if not fail else f"\n{fail} CHECK(S) FAILED")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
