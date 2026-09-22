"""Labels "super-sprints" in a team's replays: turns where a 2 or 3 step
sprint would have ended the game or taken the enemy's longest dragon, and
the move the team actually made would not.

Each game is replayed exactly as replay_dataset.py does. On every turn of the
named team, VecEnv::probe tries every codec move on a copy of the game, so the
team's own move is scored too (`teacher_cat`, used for loss weights). Where a
sprint is allowed, a sprint qualifies when, right after it,

  * "last":    every enemy dragon is dead or trapped (no way forward that is
               not kelp or a body that cannot move away, too short to split),
               and our team still has a dragon; or
  * "longest": the enemy's longest dragon (length >= LONGEST_MIN) is dead or
               trapped, and our team still has a dragon;

and neither the team's own move nor any single step achieves the same. The
best qualifying sprint ("last" over "longest", then our dragon surviving,
then enemy length removed, then fewer steps) becomes the label.

    python -m train.super_sprint --games ../runs/replays/dev_test_1_p --team-id 545

writes <games>/sprint/<game id>.npz, one row per dataset sample (same order
as replay_dataset.py's npz): `label` (codec id or -1), `cat` (0 none,
1 longest, 2 last) and `teacher_cat` (what the team's own move achieved).
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

OUR_ALIVE, KILLED, LONGEST_KILLED, ENEMY_ALIVE, TRAPPED, LONGEST_TRAPPED, KILLED_LEN = range(7)
LONGEST_MIN = 4
N_MOVES = 39


def categorise(before, per):
    """(cat per move id, sort key per move id) from one probe."""
    foe_alive, foe_longest, our_alive = (int(x) for x in before)
    cats = np.zeros(N_MOVES, np.int8)
    for m in range(N_MOVES):
        o = per[m]
        legal = o.any()
        if not legal:
            continue
        our_left = our_alive - (1 - int(o[OUR_ALIVE]))
        if our_left <= 0:
            continue
        if foe_alive > 0 and o[ENEMY_ALIVE] - o[TRAPPED] <= 0:
            cats[m] = 2
        elif foe_longest >= LONGEST_MIN and (o[LONGEST_KILLED] or o[LONGEST_TRAPPED]):
            cats[m] = 1
    return cats


def label_game(game_json: pathlib.Path, out_dir: pathlib.Path, team_id: int) -> dict:
    gid = int(game_json.stem)
    meta = json.loads(game_json.read_text())
    rp = read(game_json.with_suffix(".replay"))
    side = 0 if meta["teamAId"] == team_id else 1
    env = bcsim.BattlecodeVecEnv([rp.map_text], num_envs=1, num_threads=1, seed=0,
                                 random_pearl_seed=False, max_rounds=500)
    obs = env.reset()
    z = lambda dt, *s: np.zeros((1,) + s, dt)
    kind, nst, dirs, split = z(np.int8), z(np.int8), z(np.int8, MAX_STEPS), z(np.int16)
    send, value = z(np.int8), z(np.uint32)
    labels, cats, tcats = [], [], []
    probed = 0
    for i, t in enumerate(rp.turns):
        did, rnd = int(obs.dragon_id[0]), int(obs.round[0])
        if (did, rnd) != (t.dragon, t.round) or len(t.dirs) > MAX_STEPS:
            break                               # replay_dataset.py cut the game here too
        if int(obs.team[0]) == side and t.kind >= 0:
            sc = obs.scalar[0]
            facing = int(np.argmax(sc[SC["face_n"]:SC["face_n"] + 4]))
            a, _ = encode(t, facing, int(sc[SC["length_raw"]]))
            lab, cat, tcat = -1, 0, 0
            # every turn with a legal move is probed: the team's own move is
            # scored too (teacher_cat), sprint or not
            if obs.mask[0][:N_MOVES].any():
                probed += 1
                before, per = env.probe(0)
                c = categorise(before, per)
                tcat = int(c[a]) if 0 <= a < N_MOVES else 0
                best_single = int(c[:3].max())
                sprint = c[3:]
                top = int(sprint.max())
                if obs.mask[0][3:N_MOVES].any() and top > max(tcat, best_single):
                    ids = [3 + j for j in np.flatnonzero(sprint == top)]
                    key = lambda m: (-int(per[m][OUR_ALIVE]), -int(per[m][KILLED_LEN]), m)
                    lab, cat = min(ids, key=key), top
            labels.append(lab)
            cats.append(cat)
            tcats.append(tcat)
        kind[0] = t.kind if t.kind >= 0 else 2
        nst[0] = len(t.dirs)
        dirs[0] = 0
        dirs[0, :nst[0]] = t.dirs[:nst[0]]
        split[0] = t.split_k
        send[0] = t.sonar is not None
        value[0] = t.sonar or 0
        obs, _, _ = env.step_raw(kind, nst, dirs, split, send, value)
    env.close()
    cats_a = np.array(cats, np.int8)
    np.savez_compressed(out_dir / f"{gid}.npz", label=np.array(labels, np.int16), cat=cats_a,
                        teacher_cat=np.array(tcats, np.int8))
    return {"game": gid, "samples": len(labels), "probed": probed,
            "last": int((cats_a == 2).sum()), "longest": int((cats_a == 1).sum()),
            "teacher_last": int((np.array(tcats) == 2).sum()),
            "teacher_longest": int((np.array(tcats) == 1).sum())}


def _job(args):
    try:
        return label_game(*args)
    except Exception:
        return {"game": int(args[0].stem), "error": traceback.format_exc(limit=3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True)
    p.add_argument("--team-id", type=int, required=True)
    p.add_argument("--workers", type=int, default=9)
    p.add_argument("--only", default="", help="dataset/index.jsonl submissions to keep, e.g. 1302")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    root = pathlib.Path(a.games)
    out = root / "sprint"
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
            elif len(rows) % 50 == 0:
                print(f"{len(rows)}/{len(jobs)} games", flush=True)
    rows.sort(key=lambda r: r["game"])
    (out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    ok = [r for r in rows if not r.get("error")]
    tot = lambda k: sum(r[k] for r in ok)
    print(f"{len(ok)} games, {tot('samples'):,} samples, {tot('probed'):,} probed; "
          f"super-sprints: {tot('last'):,} last-dragon, {tot('longest'):,} longest-dragon; "
          f"the team's own move did it {tot('teacher_last'):,} / {tot('teacher_longest'):,} times")


if __name__ == "__main__":
    main()
