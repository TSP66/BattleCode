"""Turns downloaded replays into (observation, mask, action) for imitation.

Each game is played again in the simulator, one dragon turn at a time, with
the actions the replay records. The simulator is an exact copy of the engine
(fixed pearl seed included), so the observations it produces are the ones the
training loop would produce in that position, which is what a policy trained
on them will see. Every turn is checked against the replay (turn order, head
position after the move, final result). A game is cut at the first turn that
disagrees, and only the turns before that are kept.

    python -m train.replay_dataset --games ../runs/replays/vibing --team-id 306

Writes <games>/dataset/<game id>.npz and <games>/dataset/index.jsonl.
Samples are the named team's turns only. `action` is the codec id (see
VecEnv::decode_for), -1 when the codec cannot express the move (sprints of
4+ steps, split sizes outside CODEC_SPLIT_K, suicide); `alt` is a second id
meaning the same thing (an explicit split k that is also len/2), else -1.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import traceback

import os

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# the training library caps a move at 8 steps; real games sprint further
os.environ.setdefault("BCSIM_LIB", str(pathlib.Path(__file__).resolve().parents[1]
                                       / "bcsim" / "libbcvec_replay.so"))

import bcsim                                  # noqa: E402
from bcsim.env import MAX_STEPS               # noqa: E402
from train.replay_read import read            # noqa: E402

SPLIT_K = [2, 3, 4, 5, 6, 8, 12, 16, -1]      # CODEC_SPLIT_K, -1 = len // 2
N_MOVES = 3 + 9 + 27
FACE = ["face_n", "face_e", "face_s", "face_w"]
SC = {n: i for i, n in enumerate(bcsim.SCALARS)}
# bcsim.SCALARS names the base scalars; N_SCALARS is the full row, which also
# carries mem and memfar (bc_memory.hpp)
N_BASE_SCALARS = len(bcsim.SCALARS)


def encode(turn, facing: int, length: int) -> tuple[int, int]:
    """Codec id(s) for a replayed action, given the dragon's facing (0 N .. 3 W)."""
    if turn.kind == 0:
        n = len(turn.dirs)
        if not 1 <= n <= 3:
            return -1, -1
        code, f = 0, facing
        for i, d in enumerate(turn.dirs):
            if d == f:
                t = 0
            elif d == (f + 3) % 4:
                t = 1                          # left
            elif d == (f + 1) % 4:
                t = 2                          # right
            else:
                return -1, -1                  # straight back into the neck
            code += t * 3 ** i
            f = d
        return (0, 3, 12)[n - 1] + code, -1
    if turn.kind == 1:
        ids = split_ids(turn.split_k, length)
        return (ids[0], ids[1] if len(ids) > 1 else -1) if ids else (-1, -1)
    return -1, -1


XSPLIT_ID, XSPLIT_KEEP = 49, (2, 3)   # BC_XSPLIT builds (51 actions): k = len - 2, len - 3


def split_ids(k: int, length: int) -> list[int]:
    """Every codec id that splits off k segments from a dragon of this length: the fixed
    sizes and half, then (in a 51-action build) the len-2 / len-3 ones."""
    ids = [N_MOVES + i for i, kk in enumerate(SPLIT_K) if (kk if kk > 0 else length // 2) == k]
    if bcsim.N_ACTIONS >= XSPLIT_ID + len(XSPLIT_KEEP):
        ids += [XSPLIT_ID + j for j, keep in enumerate(XSPLIT_KEEP) if length - keep == k]
    return ids


MAP_DIRS = ["maps-server-1003", "maps-live", "maps-train-official", "maps-all", "maps", "maps-official"]
_by_name: dict[str, list[str]] | None = None


def _geometry(text: str) -> list[str]:
    return [l for l in text.splitlines() if l.split()[:1] not in (["TILE"], ["MAP_NAME"], [])]


def game_map(replay_map: str) -> str | None:
    """The map to simulate a replay on. Server replays zero every TILE range (seen
    2026-09-28), so the pearls come from our copy of the map with the same name and,
    line for line, the same size, symmetry, edges and dragons. None if we have none."""
    tiles = [l.split() for l in replay_map.splitlines() if l.startswith("TILE ")]
    if any(t[3:5] != ["0", "0"] for t in tiles):
        return replay_map                    # an older replay: the ranges are there
    global _by_name
    if _by_name is None:
        _by_name = {}
        root = pathlib.Path(__file__).resolve().parents[2]
        for d in MAP_DIRS:
            for f in sorted((root / d).glob("*.map")):
                t = f.read_text()
                name = next((l[9:].strip() for l in t.splitlines() if l.startswith("MAP_NAME ")), "")
                _by_name.setdefault(name, []).append(t)
    name = next((l[9:].strip() for l in replay_map.splitlines() if l.startswith("MAP_NAME ")), "")
    geo = _geometry(replay_map)
    for t in _by_name.get(name, []):
        if _geometry(t) == geo:
            return t
    # Since 2026-09-28 the server plays variants of each map: the same terrain with
    # the DRAGON lines changed (team labels swapped, which also changes who moves
    # first in each pair; Prisoners Dilemma also with extra dragons). The pearl
    # ranges are the map's own, so take the replay's dragons and our TILE lines.
    terrain = [l for l in geo if not l.startswith("DRAGON")]
    for t in _by_name.get(name, []):
        if [l for l in _geometry(t) if not l.startswith("DRAGON")] == terrain:
            ours = {l.split()[1] + " " + l.split()[2]: l for l in t.splitlines() if l.startswith("TILE ")}
            return "\n".join(ours.get(l.split()[1] + " " + l.split()[2], l) if l.startswith("TILE ") else l
                             for l in replay_map.splitlines()) + "\n"
    return None


def convert(game_json: pathlib.Path, out_dir: pathlib.Path, team_id: int, write: bool = True) -> dict:
    gid = int(game_json.stem)
    meta = json.loads(game_json.read_text())
    rp = read(game_json.with_suffix(".replay"))
    side = 0 if meta["teamAId"] == team_id else 1
    row = {"game": gid, "side": side, "submission": meta.get("submissionAId" if side == 0 else "submissionBId"),
           "opponent": meta["teamBId" if side == 0 else "teamAId"], "map": meta["mapId"],
           "winner_meta": meta["winner"], "winner_replay": rp.winner, "rounds": rp.rounds,
           "turns": len(rp.turns)}

    map_text = game_map(rp.map_text)
    if map_text is None:
        row.update(samples=0, diverged="no copy of this map with its pearl ranges")
        return row
    env = bcsim.BattlecodeVecEnv([map_text], num_envs=1, num_threads=1, seed=0,
                                 random_pearl_seed=False, max_rounds=500)
    if meta.get("seed"):                      # unswbc 1.1+: per-match pearls
        env.set_pearl_seed64(0, int(meta["seed"], 16))
    obs = env.reset()
    w, h = [int(v) for v in map_text.split("\n", 1)[0].split()[1:3]]

    keep = {k: [] for k in ("local", "scalar", "msgs", "mask", "action", "alt",
                            "dragon", "round", "mask_ok")}
    expect_head: dict[int, tuple[int, int]] = {}
    diverged = None
    unrepresentable = 0
    z = lambda dt, *s: np.zeros((1,) + s, dt)
    kind, nst, dirs, split = z(np.int8), z(np.int8), z(np.int8, MAX_STEPS), z(np.int16)
    send, value = z(np.uint8), z(np.uint64, bcsim.SONAR_DIRS)

    for i, t in enumerate(rp.turns):
        did, rnd = int(obs.dragon_id[0]), int(obs.round[0])
        if (did, rnd) != (t.dragon, t.round):
            diverged = f"turn {i}: sim has dragon {did} round {rnd}, replay {t.dragon} round {t.round}"
            break
        sc = obs.scalar[0]
        head = (round(sc[SC["head_x"]] * w), round(sc[SC["head_y"]] * h))
        if did in expect_head and expect_head[did] != head:
            diverged = f"turn {i}: dragon {did} head {head}, replay put it at {expect_head[did]}"
            break
        if t.head_after is not None:
            expect_head[did] = t.head_after

        if int(obs.team[0]) == side and t.kind >= 0:
            facing = int(np.argmax(sc[SC["face_n"]:SC["face_n"] + 4]))
            a, alt = encode(t, facing, int(sc[SC["length_raw"]]))
            if a < 0:
                unrepresentable += 1
            keep["local"].append(obs.local[0].astype(np.float16))
            # only the base scalars: bcsim now appends mem and memfar
            # (bc_memory.hpp), but clone_cache.py's layout is the 14, and
            # clone_features.py rebuilds the remembered inputs from the planes
            keep["scalar"].append(obs.scalar[0, :N_BASE_SCALARS].copy())
            # the first four only: obs.msgs is MAX_MSGS wide now, and the
            # clone cache keeps four (clone_cache.py's spec)
            keep["msgs"].append(obs.msgs[0, :4].copy())
            keep["mask"].append(obs.mask[0].copy())
            keep["action"].append(a)
            keep["alt"].append(alt)
            keep["dragon"].append(did)
            keep["round"].append(rnd)
            keep["mask_ok"].append(a >= 0 and bool(obs.mask[0][a]))

        kind[0] = t.kind if t.kind >= 0 else 2   # no action given: the engine kills it
        if len(t.dirs) > MAX_STEPS:
            diverged = f"turn {i}: a {len(t.dirs)}-step move, over this build's MAX_STEPS {MAX_STEPS}"
            break
        nst[0] = len(t.dirs)
        dirs[0] = 0
        dirs[0, :nst[0]] = t.dirs[:nst[0]]
        split[0] = t.split_k
        # Every ray the dragon really cast, with its direction and full 64-bit
        # payload. Reading only one 32-bit value along the facing (which this
        # replaced) both truncated wide payloads and dropped every ray after the
        # first, so a replayed game diverged from the one the engine played.
        send[0] = 0
        value[0] = 0
        for _d, _v in t.sonars:
            send[0] |= np.uint8(1 << _d)
            value[0, _d] = _v
        obs, _, eps = env.step_raw(kind, nst, dirs, split, send, value)
        if i == len(rp.turns) - 1:
            ep = eps.as_dicts()
            if not ep:
                diverged = "sim did not finish the game on the replay's last turn"
            elif ep[0]["winner"] != rp.winner:
                diverged = f"sim winner {ep[0]['winner']}, replay winner {rp.winner}"
    env.close()

    n = len(keep["action"])
    row.update(samples=n, unrepresentable=unrepresentable, diverged=diverged,
               mask_violations=int(n - sum(keep["mask_ok"])) - unrepresentable)
    if n and write:
        # the value target: did this team win the game (1 / 0 / 0.5 draw)
        won = 0.5 if rp.winner < 0 else float(rp.winner == side)
        np.savez_compressed(
            out_dir / f"{gid}.npz",
            local=np.stack(keep["local"]), scalar=np.stack(keep["scalar"]),
            msgs=np.stack(keep["msgs"]), mask=np.stack(keep["mask"]),
            action=np.array(keep["action"], np.int16), alt=np.array(keep["alt"], np.int16),
            dragon=np.array(keep["dragon"], np.int32), round=np.array(keep["round"], np.int16),
            mask_ok=np.array(keep["mask_ok"], bool), won=np.float32(won),
            game=np.int32(gid), submission=np.int32(row["submission"] if row["submission"] is not None else -1))
    return row


def _job(args):
    try:
        return convert(*args)
    except Exception:
        return {"game": int(args[0].stem), "error": traceback.format_exc(limit=3)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", required=True, help="directory replay_fetch wrote")
    p.add_argument("--team-id", type=int, required=True)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--redo", action="store_true", help="convert every game again")
    p.add_argument("--index-only", action="store_true",
                   help="only check the games (index.jsonl), no .npz: clone_lstm re-renders every turn "
                        "with our sonar and reads nothing else")
    a = p.parse_args()
    root = pathlib.Path(a.games)
    out = root / "dataset"
    out.mkdir(exist_ok=True)
    jobs = sorted((g for g in (root / "games").glob("*.json")
                   if g.with_suffix(".replay").exists()), key=lambda g: int(g.stem))
    if a.limit:
        jobs = jobs[:a.limit]
    # games converted cleanly before are kept as they are
    rows = []
    idx_path = out / "index.jsonl"
    if idx_path.exists() and not a.redo:
        old = {r["game"]: r for r in map(json.loads, idx_path.read_text().splitlines())}
        clean = {g for g, r in old.items() if not r.get("error") and not r.get("diverged")
                 and (a.index_only or r.get("samples", 0) == 0 or (out / f"{g}.npz").exists())}
        rows = [old[g] for g in clean]
        jobs = [j for j in jobs if int(j.stem) not in clean]
        print(f"{len(clean)} games already converted, {len(jobs)} to go", flush=True)
    with mp.Pool(a.workers) as pool:
        for r in pool.imap_unordered(_job, [(g, out, a.team_id, not a.index_only) for g in jobs]):
            rows.append(r)
            flag = r.get("error") or r.get("diverged") or ""
            print(f"game {r['game']}: {r.get('samples', 0)} samples"
                  + (f"  <-- {flag.strip().splitlines()[-1]}" if flag else ""), flush=True)
    rows.sort(key=lambda r: r["game"])
    with (out / "index.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in rows if not r.get("error") and not r.get("diverged")]
    tot = sum(r.get("samples", 0) for r in rows)
    print(f"\n{len(rows)} games, {len(ok)} replayed exactly, "
          f"{sum(1 for r in rows if r.get('diverged'))} diverged, "
          f"{sum(1 for r in rows if r.get('error'))} errors; {tot:,} samples, "
          f"{sum(r.get('unrepresentable', 0) for r in rows):,} not expressible in the codec, "
          f"{sum(r.get('mask_violations', 0) for r in rows):,} outside the mask")


if __name__ == "__main__":
    main()
