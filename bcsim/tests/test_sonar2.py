"""Sonar v2 in the simulator (BC_SONAR2 build): self-play with both teams speaking.

    BCSIM_LIB=bcsim/libbcvec_priv_s2.so python3 tests/test_sonar2.py
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("BCSIM_LIB", str(ROOT / "bcsim" / "libbcvec_priv_s2.so"))
sys.path.insert(0, str(ROOT))

import numpy as np                  # noqa: E402
import bcsim                        # noqa: E402
from train import sonar2 as S2      # noqa: E402

REP = slice(38, 43)   # the five report planes (the queen channels follow)


def intent_of(e, rnd):
    r = np.random.default_rng(1000 * e + rnd)
    return S2.intents_from_probs(r.dirichlet(np.ones(48))[None])[0]


def main():
    maps = bcsim.load_maps([str(p) for p in sorted((ROOT.parent / "maps-gate").glob("*.map"))])
    N = 22
    env = bcsim.BattlecodeVecEnv(maps, num_envs=N, num_threads=4, seed=3, board=True, grid=True,
                                 sonar=True)
    assert env.sonar2 and env.grid.shape[1] in (54, 57), env.grid.shape
    for i in range(N):
        env.set_opponent(i, team=-1, bot=0, map_index=i % len(maps))
        env.set_sonar2(i, (0, 1) if i % 2 == 0 else (0,))     # odd envs: team 1 silent
    obs = env.reset()
    rng = np.random.default_rng(0)
    st = dict(turns=0, verified=0, ally_ok=0, enemy_pass=0, enemy_pkts=0, rep_planes=0,
              silent_rep=0, far_ally_mem=0, intent_ok=0, intent_n=0)
    last_intent = {}
    for step in range(6000):
        B = env.board
        for e in range(N):
            team, rnd = int(obs.team[e]), int(obs.round[e])
            W = int(round(obs.scalar[e, bcsim.SCALARS.index("map_w")] * 64))
            H = int(round(obs.scalar[e, bcsim.SCALARS.index("map_h")] * 64))
            speaks = e % 2 == 0 or team == 0
            n = min(int(obs.num_msgs[e]), obs.msgs.shape[1])
            for v in obs.msgs[e, :n].tolist():
                if v == 0:
                    continue
                p = S2.decode(v, team, rnd)
                q = S2.decode(v, 1 - team, rnd)
                if q is not None and p is None:
                    st["enemy_pkts"] += 1
                if p is None:
                    continue
                if q is not None:
                    st["enemy_pass"] += 1          # verifies for both teams: a tag collision
                st["verified"] += 1
                # the sender's head is one of this dragon's team's segments now or moved
                # at most 3 since; check it is an ally cell or within 3 of one
                own = np.argwhere((B[e, 0, :H, :W] > 0) | (B[e, 1, :H, :W] > 0))
                if len(own):
                    dy = np.abs(own[:, 0] - p["hy"]); dx = np.abs(own[:, 1] - p["hx"])
                    d = np.maximum(np.minimum(dx, W - dx), np.minimum(dy, H - dy)).min()
                    st["ally_ok"] += int(d <= 3)
                want = intent_of(e, p["round"])
                st["intent_n"] += 1
                st["intent_ok"] += int(np.allclose(want, [p["p_split"], p["p_sprint"], p["p_left"], p["p_right"]],
                                                   atol=0.034))   # P(sprint) has 4 bits: steps of 1/15
            g = env.grid[e]
            if speaks:
                st["rep_planes"] += int(np.abs(g[REP]).sum() > 0)
            else:
                st["silent_rep"] += int(np.abs(g[REP]).sum() > 0)
            st["turns"] += 1
        # random legal actions, with random intents; remember what each sender said
        acts = np.zeros(N, np.int32)
        for e in range(N):
            legal = np.flatnonzero(obs.mask[e])
            acts[e] = rng.choice(legal) if len(legal) else 0
        # every dragon of env e says the same thing in a given round, so a receiver
        # can check what arrived against what was sent
        for e in range(N):
            env.intent[e] = intent_of(e, int(obs.round[e]))
        obs, _, _ = env.step(acts)
    print(st)
    assert st["verified"] > 1000
    assert st["ally_ok"] / st["verified"] > 0.99, "a verified packet named a non-ally head"
    assert st["enemy_pass"] / max(1, st["verified"]) < 0.02
    assert st["intent_ok"] / st["intent_n"] > 0.99, "intents did not survive the packet"
    assert st["rep_planes"] > 0 and st["silent_rep"] == 0, "a silent team drew report planes"
    print("OK")


if __name__ == "__main__":
    main()
