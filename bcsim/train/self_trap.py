"""Finds avoidable self-traps in a team's replays: a single step into a pocket
too small for the dragon, when another single step had clearly more room,
after which the dragon really did die.

Each game is replayed as replay_dataset.py does; on every turn of the named
team whose move was a single step, VecEnv::probe gives each single step's
reachable free area (PROBE_AREA: a flood fill from the new head with every
body a wall, a lower bound on the room). A turn counts when

  * the team's step leaves area < the dragon's length (it cannot fit),
  * another single step leaves area >= max(2 * length, that area + 10),
  * the dragon never acts again within area + 3 rounds while the game goes
    on (so it died, and did not split its way out: a split in between
    disqualifies the turn),
  * and the dragon killed no enemy between the move and one round after
    its death (head-on or an enemy running into it: not a sacrifice).

The roomiest single step becomes the label.

    python -m train.self_trap --games ../runs/replays/dev_test_1_p --team-id 545 --only 1302

writes <games>/selftrap/<game id>.npz (`label` per dataset sample, -1 none).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pathlib
import sys
import traceback

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
os.environ.setdefault("BCSIM_LIB", str(pathlib.Path(__file__).resolve().parents[1]
                                       / "bcsim" / "libbcvec_replay.so"))

import bcsim                                  # noqa: E402
from bcsim.env import MAX_STEPS               # noqa: E402
from train.replay_dataset import SC, encode   # noqa: E402
from train.replay_read import read            # noqa: E402

AREA = 7


def label_game(game_json: pathlib.Path, out_dir: pathlib.Path, team_id: int) -> dict:
    gid = int(game_json.stem)
    meta = json.loads(game_json.read_text())
    rp = read(game_json.with_suffix(".replay"))
    side = 0 if meta["teamAId"] == team_id else 1
    # when each dragon last acted, and the rounds it split in
    last, splits = {}, {}
    for t in rp.turns:
        last[t.dragon] = t.round
        if t.kind == 1:
            splits.setdefault(t.dragon, []).append(t.round)
    final = max(last.values())
    foe_team = 1 - side
    team_of = {}
    env = bcsim.BattlecodeVecEnv([rp.map_text], num_envs=1, num_threads=1, seed=0,
                                 random_pearl_seed=False, max_rounds=500)
    obs = env.reset()
    z = lambda dt, *s: np.zeros((1,) + s, dt)
    kind, nst, dirs, split = z(np.int8), z(np.int8), z(np.int8, MAX_STEPS), z(np.int16)
    send, value = z(np.int8), z(np.uint32)
    labels = []
    cand = {}                               # sample -> (dragon, round, death round)
    kills_by = {}                           # our dragon id -> rounds it killed an enemy
    sacrifices = 0
    trapped_any = 0
    for i, t in enumerate(rp.turns):
        did, rnd = int(obs.dragon_id[0]), int(obs.round[0])
        if (did, rnd) != (t.dragon, t.round) or len(t.dirs) > MAX_STEPS:
            break
        if int(obs.team[0]) == side and t.kind >= 0:
            sc = obs.scalar[0]
            length = int(sc[SC["length_raw"]])
            facing = int(np.argmax(sc[SC["face_n"]:SC["face_n"] + 4]))
            a, _ = encode(t, facing, length)
            lab = -1
            if 0 <= a < 3 and obs.mask[0][:3].sum() >= 2:
                _, per = env.probe(0)
                area = per[:3, AREA]
                legal = obs.mask[0][:3] > 0
                if legal[a] and area[a] < length:
                    trapped_any += 1
                    alt = [m for m in range(3) if m != a and legal[m]
                           and area[m] >= max(2 * length, area[a] + 10)]
                    died = last[did] <= rnd + area[a] + 3 and last[did] < final
                    split_between = any(rnd < r <= last[did] for r in splits.get(did, []))
                    if alt and died and not split_between:
                        lab = max(alt, key=lambda m: (area[m], -m))
                        cand[len(labels)] = (did, rnd, last[did])
            labels.append(lab)
        if did not in team_of:
            team_of[did] = int(obs.team[0])
        kind[0] = t.kind if t.kind >= 0 else 2
        nst[0] = len(t.dirs)
        dirs[0] = 0
        dirs[0, :nst[0]] = t.dirs[:nst[0]]
        split[0] = t.split_k
        send[0] = t.sonar is not None
        value[0] = t.sonar or 0
        obs, _, _ = env.step_raw(kind, nst, dirs, split, send, value)
        for dead, _, killer, team in env.last_deaths(0):
            if team == foe_team and killer >= 0:
                kills_by.setdefault(int(killer), []).append(rnd)
    env.close()
    # a trap that took an opponent with it is a sacrifice, not a mistake: drop
    # every candidate whose dragon killed an enemy (head-on, or an enemy ran
    # into its body; the simulator's death events name the killer) between
    # the move and one round after its own death
    for j, (d, r0, r1) in cand.items():
        if any(r0 <= r <= r1 + 1 for r in kills_by.get(d, [])):
            labels[j] = -1
            sacrifices += 1
    lab = np.array(labels, np.int16)
    np.savez_compressed(out_dir / f"{gid}.npz", label=lab)
    return {"game": gid, "samples": len(lab), "trapped_moves": trapped_any,
            "self_traps": int((lab >= 0).sum()), "sacrifices": sacrifices}


def _job(args):
    try:
        return label_game(*args)
    except Exception:
        return {"game": int(args[0].stem), "error": traceback.format_exc(limit=3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True)
    p.add_argument("--team-id", type=int, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--only", default="")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    root = pathlib.Path(a.games)
    out = root / "selftrap"
    out.mkdir(exist_ok=True)
    index = {r["game"]: r for r in map(json.loads, (root / "dataset/index.jsonl").read_text().splitlines())}
    subs = {int(s) for s in a.only.split(",") if s}
    jobs = sorted((g for g in (root / "games").glob("*.json")
                   if int(g.stem) in index and index[int(g.stem)].get("samples", 0)
                   and (not subs or index[int(g.stem)]["submission"] in subs)),
                  key=lambda g: int(g.stem))
    if a.limit:
        jobs = jobs[:a.limit]
    rows = []
    with mp.Pool(a.workers) as pool:
        for r in pool.imap_unordered(_job, [(g, out, a.team_id) for g in jobs]):
            rows.append(r)
            if r.get("error"):
                print(f"game {r['game']}: {r['error'].strip().splitlines()[-1]}", flush=True)
    rows.sort(key=lambda r: r["game"])
    (out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ok = [r for r in rows if not r.get("error")]
    tot = lambda k: sum(r[k] for r in ok)
    print(f"{len(ok)} games, {tot('samples'):,} samples; single steps into a pocket smaller than "
          f"the dragon: {tot('trapped_moves'):,}; avoidable and fatal: {tot('self_traps'):,} "
          f"(after dropping {tot('sacrifices'):,} where an enemy died too)")


if __name__ == "__main__":
    main()
