"""Server replays -> critic pretraining data (train/critic_data.py format), user 2026-10-01.

Every downloaded game (runs/replays/*/games, deduplicated by game id: a game appears under both
teams' downloads) is played again in the simulator with the critic view and the v8 potential on,
one dragon turn at a time, with the actions the replay records -- the same stepping and the
same checks as train/replay_dataset.py (turn order, head position after each move, final
winner). Only games that replay EXACTLY to the end are kept: the returns need the true final
Phi. Both teams are recorded; a real team plays greedy (T = 0, user) and the game is OBSERVED.
Sides are named "team:<contest id>"; critic_pretrain.py maps them (and the clones) to identities.

Server games are long (~17k turns), so a game keeps only the turns its returns need: each team's
first turn of every round, the game's last turn, and the sampled turns (--keep). The reward is
discounted per ROUND, so between two kept turns of a team in the same round nothing is lost:
critic_data.returns gives exactly the full-turn value at every kept turn (--full-turns keeps all, to check).

    python -m train.critic_replays --out ../runs/critic_pre/replays --workers 6

Writes <out>/part_NNN/{samples.npz, turns.npz, games.jsonl} (critic_data format, --part-games games
a part) and <out>/skipped.jsonl.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# real games sprint further than the training library's 8 steps (as replay_dataset.py)
os.environ.setdefault("BCSIM_LIB", str(pathlib.Path(__file__).resolve().parents[1] / "bcsim" /
                                       "libbcvec_replay_s2_g15.so"))

import bcsim                                       # noqa: E402
from bcsim.env import MAX_STEPS                    # noqa: E402
from train.replay_dataset import SC, game_map      # noqa: E402
from train.replay_read import read                 # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
CFG: dict = {}


def play(game_json: str) -> dict:
    """One game replayed with the critic view on: its turns and kept samples, or why it was skipped."""
    gj = pathlib.Path(game_json)
    gid = int(gj.stem)
    try:
        meta = json.loads(gj.read_text())
        if not meta.get("seed"):
            return {"game": gid, "skip": "no per-match seed (pre unswbc 1.1)"}
        rp = read(gj.with_suffix(".replay"))
        map_text = game_map(rp.map_text)
        if map_text is None:
            return {"game": gid, "skip": "no copy of this map with its pearl ranges"}
        from train.critic_v8 import GAMMA, KAPPA
        env = bcsim.BattlecodeVecEnv([map_text], num_envs=1, num_threads=1, seed=0, random_pearl_seed=False,
                                     max_rounds=500, privileged=True, cview=CFG["cview"])
        env.set_potential_gamma(GAMMA)
        env.set_reward_v8(True, KAPPA)
        env.set_pearl_seed64(0, int(meta["seed"], 16))
        obs = env.reset()
        w, h = [int(v) for v in map_text.split("\n", 1)[0].split()[1:3]]
        rng = np.random.default_rng(gid)
        T = {k: [] for k in ("team", "round", "step", "phi")}
        S = {k: [] for k in ("cview", "priv", "team", "round", "step")}
        expect_head: dict[int, tuple[int, int]] = {}
        z = lambda dt, *s: np.zeros((1,) + s, dt)   # noqa: E731
        kind, nst, dirs, split = z(np.int8), z(np.int8), z(np.int8, MAX_STEPS), z(np.int16)
        send, value = z(np.uint8), z(np.uint64, bcsim.SONAR_DIRS)
        why = None
        winner = None
        for i, t in enumerate(rp.turns):
            did, rnd = int(obs.dragon_id[0]), int(obs.round[0])
            if (did, rnd) != (t.dragon, t.round):
                why = f"turn {i}: turn order"
                break
            sc = obs.scalar[0]
            head = (round(sc[SC["head_x"]] * w), round(sc[SC["head_y"]] * h))
            if did in expect_head and expect_head[did] != head:
                why = f"turn {i}: head position"
                break
            if t.head_after is not None:
                expect_head[did] = t.head_after
            if len(t.dirs) > MAX_STEPS:
                why = f"turn {i}: {len(t.dirs)}-step move"
                break
            T["team"].append(int(obs.team[0]))
            T["round"].append(rnd)
            T["step"].append(i)
            T["phi"].append(float(obs.priv[0, bcsim.PRIV_BASE:].sum()))
            kept_s = rng.random() < CFG["keep"]
            T.setdefault("sampled", []).append(kept_s)
            if kept_s:
                S["cview"].append(env.cview[0].copy())
                S["priv"].append(obs.priv[0].astype(np.float32))
                S["team"].append(int(obs.team[0]))
                S["round"].append(rnd)
                S["step"].append(i)
            kind[0] = t.kind if t.kind >= 0 else 2
            nst[0] = len(t.dirs)
            dirs[0] = 0
            dirs[0, :nst[0]] = t.dirs[:nst[0]]
            split[0] = t.split_k
            send[0] = 0
            value[0] = 0
            for _d, _v in t.sonars:
                send[0] |= np.uint8(1 << _d)
                value[0, _d] = _v
            obs, _, eps = env.step_raw(kind, nst, dirs, split, send, value)
            if i == len(rp.turns) - 1:
                ep = eps.as_dicts()
                if not ep:
                    why = "the simulator did not finish the game on the replay's last turn"
                elif ep[0]["winner"] != rp.winner:
                    why = f"winner {ep[0]['winner']} vs the replay's {rp.winner}"
                else:
                    winner = int(ep[0]["winner"])
        env.close()
        if why is not None:
            return {"game": gid, "skip": why}
        tm, rd, smp = np.asarray(T["team"]), np.asarray(T["round"]), np.asarray(T.pop("sampled"), bool)
        if not CFG["full_turns"]:
            # each team's first turn of every round, the last turn, the sampled turns (see the docstring)
            first = np.ones(len(tm), bool)
            for team in (0, 1):
                it = np.flatnonzero(tm == team)
                first[it[1:]] = rd[it[1:]] != rd[it[:-1]]
            keep_t = first | smp
            keep_t[-1] = True
            T = {k: np.asarray(v)[keep_t] for k, v in T.items()}
        return {"game": gid, "meta": {"sides": [f"team:{meta['teamAId']}", f"team:{meta['teamBId']}"],
                                      "team_names": [meta.get("teamAName"), meta.get("teamBName")],
                                      "temps": [0.0, 0.0], "observed": 1, "map": meta.get("mapName"),
                                      "source": "replay", "winner": winner, "rounds": int(rp.rounds),
                                      "completed_at": meta.get("completedAt")},
                "T": {k: np.asarray(v) for k, v in T.items()},
                "S": {k: np.asarray(v) for k, v in S.items()}}
    except Exception:
        return {"game": gid, "skip": "error: " + traceback.format_exc(limit=2)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replays", default=str(ROOT / "runs/replays"))
    p.add_argument("--out", required=True)
    p.add_argument("--keep", type=float, default=0.004, help="share of turns sampled (~65 a server game)")
    p.add_argument("--cview", type=int, default=27)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0, help="first N games only (a smoke test)")
    p.add_argument("--part-games", type=int, default=2000, help="games a part directory")
    p.add_argument("--full-turns", action="store_true", help="keep every turn (to check the compact form)")
    a = p.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob("part_*")):
        raise SystemExit(f"{out} already holds parts")
    files: dict[int, str] = {}
    for f in sorted(pathlib.Path(a.replays).glob("*/games/*.json")):
        if f.with_suffix(".replay").exists():
            files.setdefault(int(f.stem), str(f))         # one copy of each game
    todo = [files[g] for g in sorted(files, reverse=True)]  # newest first
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo):,} unique games with a replay; {a.workers} workers", flush=True)
    CFG.update(keep=a.keep, cview=a.cview, full_turns=a.full_turns)
    stride = bcsim.cview_layout(a.cview)["stride"]
    sfile = open(out / "skipped.jsonl", "w")
    n_ok = n_skip = n_samples = 0
    part = {"k": 0}

    def new_part():
        d = out / f"part_{part['k']:03d}"
        d.mkdir()
        part.update(dir=d, n=0, gfile=open(d / "games.jsonl", "w"),
                    T={k: [] for k in ("game", "team", "round", "step", "phi")},
                    S={k: [] for k in ("cview", "priv", "game", "team", "round", "step")})

    def flush_part():
        part["gfile"].close()
        cat = lambda d, dt: {k: (np.concatenate(v).astype(dt.get(k, None)) if v else np.zeros(0))   # noqa: E731
                             for k, v in d.items()}
        np.savez(part["dir"] / "turns.npz",
                 **cat(part["T"], {"team": np.int8, "round": np.int16, "step": np.int32, "phi": np.float32}))
        Sx = cat(part["S"], {"team": np.int8, "round": np.int16, "step": np.int32, "cview": np.uint8,
                             "priv": np.float32})
        if not len(Sx["game"]):
            Sx["cview"] = np.zeros((0, stride), np.uint8)
        np.savez(part["dir"] / "samples.npz", cview_w=np.array(a.cview), **Sx)
        part["k"] += 1

    new_part()
    t0 = time.time()
    with mp.Pool(a.workers, initializer=CFG.update, initargs=(dict(CFG),)) as pool:
        for k, r in enumerate(pool.imap_unordered(play, todo, chunksize=4)):
            if "skip" in r:
                n_skip += 1
                sfile.write(json.dumps({"game": r["game"], "skip": r["skip"][:300]}) + "\n")
            else:
                n_ok += 1
                g = r["game"]
                T, S = part["T"], part["S"]
                part["gfile"].write(json.dumps({"game": g, **r["meta"], "finished": True}) + "\n")
                nt = len(r["T"]["team"])
                T["game"].append(np.full(nt, g, np.int64))
                for c in ("team", "round", "step", "phi"):
                    T[c].append(r["T"][c])
                ns = len(r["S"]["team"])
                n_samples += ns
                if ns:
                    S["cview"].append(r["S"]["cview"].reshape(ns, stride))
                    S["priv"].append(r["S"]["priv"])
                    S["game"].append(np.full(ns, g, np.int64))
                    for c in ("team", "round", "step"):
                        S[c].append(r["S"][c])
                part["n"] += 1
                if part["n"] >= a.part_games:
                    flush_part()
                    new_part()
            if (k + 1) % 500 == 0:
                el = time.time() - t0
                print(f"  {k + 1:,}/{len(todo):,}: {n_ok:,} exact, {n_skip:,} skipped, {el / 60:.1f} min, "
                      f"~{(len(todo) - k - 1) * el / (k + 1) / 60:.0f} min left", flush=True)
    sfile.close()
    flush_part()
    print(f"done: {n_ok:,} games exact, {n_skip:,} skipped, {n_samples:,} samples in {part['k']} parts, "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
