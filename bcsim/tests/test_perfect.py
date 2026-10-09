"""Perfect-play labels (VecEnv::perfect, cpp/bc_obs.hpp) checked by playing them.

Random legal play on the official and generated maps. Wherever a label fires, the env plays
one of its correct actions (chosen at random among them) and the step must show what the label
promised:

  queen_kill     the enemy queen is among the deaths of that step
  trapped_queen  our dragon died (the self-kill) and our queen did not
  late_suicide   our dragon died, our queen did not, round in 481..498
  pearl          the dragon is alive and longer after its move
  keep_wall      the dragon is alive, and at the walled queen's next turn her own mask offers
                 no move except into a head
  queen_deadend  (2026-10-05) the queen is alive after the correct action; and a negative control:
                 on some labelled turns she plays one of the dives instead (a legal move the label
                 leaves out), and must then die within 12 of her own turns (random legal play)
  blank          the dragon is alive after its step straight on, and nothing but itself and
                 kelp was in its window (checked from the obs: no pearl/ally/enemy planes)

Then train/perfect_play.py's memory augmenter on the labelled rows (check_augmenter).

Negative control: queen_kill labels checked against a WRONG legal move (another move id) must
not all kill the queen -- otherwise the check could not fail.

    BCSIM_LIB=bcsim/libbcvec_s2_g15p.so python3 tests/test_perfect.py [--steps 4000]
"""

from __future__ import annotations

import argparse
import ctypes
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("BCSIM_LIB", str(ROOT / "bcsim" / "libbcvec_s2_g15p.so"))
sys.path.insert(0, str(ROOT))
import bcsim  # noqa: E402

KINDS = bcsim.BattlecodeVecEnv.PERFECT_KINDS
SUICIDE = 48
G = bcsim.GRID_SIDE
H = G // 2
W0, W1 = H - 3, H + 4                     # the live 7x7 window inside the grid
LATE_LO, LATE_HI = 481, 498


def check_augmenter(rows, donors, rng) -> bool:
    """train/perfect_play.py's Augmenter (a fresh memory replaced by an ordinary turn's, outside the
    window). Returns True on failure: it must never touch the window, our own body planes, the
    round, length, is-queen, or a queen's planes while she is in sight; it must change something."""
    from train import perfect_play as pp
    aug = pp.Augmenter(G, rng, 1.0)
    keep = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 28, 29, 43, 44, 45]   # never touched anywhere
    win = slice(W0, W1)
    bad, changed, n = [], 0, 0
    for k, g0, m0, acts in rows:
        g = g0.astype(np.float32).copy()
        aug(g, donors[rng.integers(len(donors))][0].astype(np.float32))
        n += 1
        if not np.array_equal(g[keep], g0[keep]):
            bad.append((k, "a kept plane changed"))
        cells = [c for c in range(g.shape[0]) if c not in pp.CONST_SWAP and c not in range(48, 54)]
        if not np.array_equal(g[cells][:, win, win], g0[cells][:, win, win]):   # constant planes aside
            bad.append((k, "the window changed"))
        for live, mem, vec in ((44, 46, [48, 49, 50]), (45, 47, [51, 52, 53])):
            if g0[live, win, win].max() > 0 and not (np.array_equal(g[mem], g0[mem]) and
                                                      np.array_equal(g[vec], g0[vec])):
                bad.append((k, "a queen in sight lost her planes"))
        changed += not np.array_equal(g, g0)
    print(f"  augmenter: {n} rows, {changed} changed, {len(bad)} violations" + (f"  e.g. {bad[:3]}" if bad else ""))
    return bool(bad) or changed == 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--scenes", action="store_true",
                   help="positions from train/perfect_scenes.py on fresh random maps (every env reseeded every "
                        "8 steps, random play in between), not games on the training maps")
    a = p.parse_args()
    if a.scenes:
        from train.perfect_scenes import KINDS as SCENE_KINDS, SceneMaker
        maker = SceneMaker(np.random.default_rng(a.seed + 99))
        texts, widths = maker.maps(a.envs)
        env = bcsim.BattlecodeVecEnv(texts, num_envs=a.envs, num_threads=8, seed=a.seed, sonar=True, grid=True)
        for i in range(a.envs):
            env.set_opponent(i, -1, 0, map_index=i)
            env.set_sonar2(i, (0, 1))
        env.set_scenario_maps(widths)
    else:
        files = sorted((ROOT.parent / "maps-train-official").glob("*.map"))
        files += sorted((ROOT.parent / "maps-loong").glob("*.map"))[:60]
        env = bcsim.BattlecodeVecEnv(bcsim.load_maps([str(f) for f in files]), num_envs=a.envs, num_threads=8,
                                     seed=a.seed, sonar=True, grid=True)
        for i in range(a.envs):
            env.set_sonar2(i, (0, 1))
    rng = np.random.default_rng(a.seed)
    obs = env.reset()
    n = {k: 0 for k in KINDS}
    ok = {k: 0 for k in KINDS}
    bad = {k: [] for k in KINDS}
    neg_tried = neg_killed = 0
    n_ended = [0]
    n_last = [0]
    rounds_late = []
    aug_rows, donors = [], []
    wall_wait = {}
    dive_wait = {}                 # env -> [queen id, her turns left] after a deliberate dive
    dive = {"tried": 0, "died": 0, "lived": 0, "ended": 0}
    n_scenes = [0]
    n_wall_death = [0]
    lib = bcsim.env._lib
    lib.bcv_dragon_len.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.bcv_far_path.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    h = ctypes.c_void_p(env._h)
    buf = np.zeros(16, np.int32)
    FAR0 = bcsim.BattlecodeVecEnv.far_targets()[0]
    for step in range(a.steps):
        if a.scenes and step % 8 == 0:
            n_scene = 0
            for i in range(a.envs):
                spec = maker.scene(i, SCENE_KINDS[(i + step // 8) % len(SCENE_KINDS)])
                if spec is not None and env.set_scenario(i, *spec):
                    n_scene += 1
                    wall_wait.pop(i, None)
                    if dive_wait.pop(i, None) is not None:     # a new scene: that dive is not checkable
                        dive["ended"] += 1
            n_scenes[0] += n_scene
            obs = env.observation()
        kind, acts = env.perfect()
        mask = obs.mask.astype(bool)
        # labels must be legal moves
        assert not (acts.astype(bool) & ~mask).any(), "a correct action the mask forbids"
        # random legal play, biased away from the self-kill so games last
        logits = np.where(mask, rng.gumbel(size=mask.shape), -np.inf)
        logits[:, SUICIDE] -= 3.0
        action = logits.argmax(1).astype(np.int32)
        neg = np.zeros(a.envs, bool)
        for e in np.flatnonzero(kind):
            k = KINDS[kind[e]]
            n[k] += 1
            good = np.flatnonzero(acts[e])
            if k == "queen_kill" and rng.random() < 0.3:
                wrong = np.flatnonzero(mask[e] & ~acts[e].astype(bool))
                wrong = wrong[(wrong < 39) | (wrong >= 51)]
                if len(wrong):
                    action[e] = rng.choice(wrong)
                    neg[e] = True
                    continue
            if k == "queen_deadend" and e not in dive_wait and rng.random() < 0.4:
                wrong = np.flatnonzero(mask[e] & ~acts[e].astype(bool))
                if len(wrong):
                    action[e] = rng.choice(wrong)
                    neg[e] = True
                    continue
            action[e] = rng.choice(good)
        before = {e: (KINDS[kind[e]], int(obs.dragon_id[e]), int(obs.team[e]), int(obs.round[e]),
                      env.grid[e].copy()) for e in np.flatnonzero(kind)}
        len_before = {e: lib.bcv_dragon_len(h, int(e), int(obs.dragon_id[e])) for e in np.flatnonzero(kind)}
        for e in np.flatnonzero(kind):
            if len(aug_rows) < 4000 and (KINDS[kind[e]] != "blank" or rng.random() < 0.02):
                aug_rows.append((KINDS[kind[e]], env.grid[e].copy(), obs.mask[e].copy(), acts[e].copy()))
        for e in np.flatnonzero((kind == 0) & (rng.random(a.envs) < 0.01)):
            if len(donors) < 2000:
                donors.append((env.grid[e].copy(), obs.mask[e].copy()))
        obs, _, eps = env.step(action)
        # a game that ended on this step has restarted: its deaths are gone, its winner is in eps
        ended = {int(r[0]): int(r[2]) for r in eps.rows}
        # dives: does the queen die within 12 of her own turns?
        for e in list(dive_wait):
            if e in ended:
                dive_wait.pop(e)
                dive["ended"] += 1
                continue
            qid = dive_wait[e][0]
            if qid in {int(r[0]) for r in env.last_deaths(e)}:
                dive_wait.pop(e)
                dive["died"] += 1
            elif int(obs.dragon_id[e]) == qid:
                dive_wait[e][1] -= 1
                if dive_wait[e][1] <= 0:
                    dive_wait.pop(e)
                    dive["lived"] += 1
        for e, (k, did, team, rnd, grid) in before.items():
            if k == "queen_deadend" and e not in ended:
                dead_now = did in {int(r[0]) for r in env.last_deaths(e)}
                if neg[e]:
                    dive["tried"] += 1
                    if dead_now:
                        dive["died"] += 1
                    else:
                        dive_wait[e] = [did, 12]
                elif dead_now:
                    bad[k].append((step, e, did, "queen died on the correct action"))
                else:
                    ok[k] += 1
                continue
            if e in ended:
                if k == "queen_kill" and not neg[e]:
                    n_ended[0] += 1
                    if ended[e] == team:          # she was their last dragon: we won outright
                        ok[k] += 1
                    elif rnd == 499:              # the last turn: decided on the tiebreak, not checkable
                        n_last[0] += 1
                    else:
                        bad[k].append((step, e, did, "ended", ended[e], rnd))
                continue
            deaths = env.last_deaths(e)
            dead = {int(r[0]) for r in deaths}
            # queens are ids 0 and 1, one a team: whose is whose comes from the death rows' team
            dead_team = {int(r[0]): int(r[3]) for r in deaths}
            if k == "queen_kill":
                killed_q = any(d < 2 and dead_team[d] != team for d in dead)
                if neg[e]:
                    neg_tried += 1
                    neg_killed += killed_q
                    continue
                if killed_q:
                    ok[k] += 1
                else:
                    bad[k].append((step, e, did))
            elif k in ("trapped_queen", "late_suicide"):
                fine = did in dead and not any(d < 2 and dead_team[d] == team for d in dead)
                if k == "late_suicide":
                    fine = fine and 481 <= rnd <= 498
                    rounds_late.append(rnd)
                if fine:
                    ok[k] += 1
                else:
                    bad[k].append((step, e, did, sorted(dead)))
            elif k == "pearl":
                grew = lib.bcv_dragon_len(h, int(e), did) > len_before[e]
                if did not in dead and grew:
                    ok[k] += 1
                else:
                    bad[k].append((step, e, did, "dead" if did in dead else "did not grow"))
            elif k == "keep_wall":
                if did in dead:
                    bad[k].append((step, e, did, "our dragon died"))
                elif e not in wall_wait:                  # one pending check per env
                    wall_wait[e] = (1 - team, step)
            elif k == "blank":
                win = grid[:, W0:W1, W0:W1]
                # pearl 9, ally head/body 11/12, enemy head/body 13/14
                clear = win[[9, 11, 12, 13, 14]].max() == 0
                fine = did not in dead and clear
                if fine:
                    ok[k] += 1
                else:
                    bad[k].append((step, e, did, sorted(dead), bool(clear)))
        # keep_wall (after this step's labels are in, so a queen moving next is caught): at the walled queen's next turn her own mask may offer no move except into a head
        for e in list(wall_wait):
            if e in ended:
                wall_wait.pop(e)
                continue
            # a death in between (a head-on can kill a wall dragon after our move) opens walls the
            # label could not foresee: not checkable
            if wall_wait[e][1] != step and len(env.last_deaths(int(e))):
                wall_wait.pop(e)
                n_wall_death[0] += 1
                continue
            q_team = wall_wait[e][0]
            if int(obs.dragon_id[e]) < 2 and int(obs.team[e]) == q_team:
                wall_wait.pop(e)
                g_ = env.grid[e]
                heads = (g_[11] > 0) | (g_[13] > 0)
                out = []
                for id_ in np.flatnonzero(obs.mask[e]):
                    if id_ < 39:
                        t_ = id_ if id_ < 3 else ((id_ - 3) % 3 if id_ < 12 else (id_ - 12) % 3)
                        ox, oy = [(0, -1), (-1, 0), (1, 0)][t_]
                    elif FAR0 <= id_ < FAR0 + 24:
                        nn = lib.bcv_far_path(h, int(e), int(id_ - FAR0), buf.ctypes.data)
                        face = int(np.argmax(obs.scalar[e, 4:8]))
                        ox, oy = [(0, -1), (1, 0), (0, 1), (-1, 0)][(int(buf[0]) - face) & 3]
                    else:
                        continue                          # splits, the self-kill: she does not move
                    if not heads[H + oy, H + ox]:
                        out.append(int(id_))
                if out:
                    bad["keep_wall"].append(("queen could move", step, e, out))
                else:
                    ok["keep_wall"] += 1
    print(f"{a.steps} steps x {a.envs} envs" + (f", {n_scenes[0]} synthetic scenes" if a.scenes else ""))
    for k in KINDS[1:]:
        print(f"  {k:14s} labelled {n[k]:7d}   checked ok {ok[k]:7d}   bad {len(bad[k])}"
              + (f"  e.g. {bad[k][:3]}" if bad[k] else ""))
    print(f"  queen_kill checks where the kill ended the game (winner checked): {n_ended[0]}, "
          f"of them on the game's last turn (not checkable): {n_last[0]}")
    print(f"  keep_wall checks dropped for a death in between: {n_wall_death[0]}")
    if rounds_late:
        print(f"  late_suicide rounds {min(rounds_late)}..{max(rounds_late)}")
    print(f"  negative control: {neg_killed}/{neg_tried} wrong moves killed the queen")
    print(f"  queen_deadend dives: {dive['tried']} tried, {dive['died']} died, {dive['lived']} lived 12 turns, "
          f"{dive['ended']} games ended first")
    if dive["lived"]:
        print(f"FAIL: {dive['lived']} queens survived a labelled dive")
    fail = any(bad[k] for k in KINDS[1:]) or dive["lived"] > 0
    fail |= check_augmenter(aug_rows, donors, rng)
    if neg_tried and neg_killed == neg_tried:
        print("FAIL: every wrong move also killed the queen -- the check cannot tell")
        fail = True
    print("FAIL" if fail else "OK")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
