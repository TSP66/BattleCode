"""Plays two checkpoints on one map in the simulator (exactly as a gate does), records
every dragon turn, then re-plays those moves through the official unswbc engine to get a
.replay file the VS Code replay viewer opens.

The simulator's parity with the engine is exact, so the k-th engine callback is the k-th
simulator turn of that game; every turn's dragon id and round are checked against it.

Sonar (2026-10-05): the simulator library is picked from the checkpoints' own args (--portal
-> libbcvec_s2_g15p, the run's lib), so both teams speak the packet they trained with. Each
turn's v2 packet (bcv_s2_preview, what the sim casts in all four directions) is re-sent to the
engine with PROTOCOL 3, exactly as lstmbot does, and every engine block's NUM_MSGS and payloads
are checked against what the simulator handed the policy on that turn. Run with the run's own
BC_QUEEN_GUARD (scratch_1002: 1); it is printed.

    BC_QUEEN_GUARD=1 BC_EVAL_QUEUE=0 /usr/bin/python3 -m train.watch_game --a CAND.pt --b ANCHOR.pt \
        --map ../maps-gate-1001/queen_of_spades.map --out ../runs/watch/
"""
from __future__ import annotations

import argparse
import ctypes
import os
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tests"))

import numpy as np  # noqa: E402
import torch  # noqa: E402


def sim_lib(paths: list[str]) -> str:
    """ratchet.gate's library choice, from the checkpoints' args. Both must agree: a
    policy played on another build reads a different packet / plane layout."""
    libs = set()
    for p in paths:
        a = torch.load(p, map_location="cpu", weights_only=False)["args"]
        if not a.get("s2", a.get("sonar2", False)):
            libs.add("libbcvec.so")
        elif a.get("portal", False):
            libs.add("libbcvec_s2_g15p.so")
        else:
            libs.add("libbcvec_s2_g15.so" if int(a.get("grid", 14)) == 15 else "libbcvec_s2.so")
    if len(libs) != 1:
        raise SystemExit(f"checkpoints need different simulators: {sorted(libs)}")
    return libs.pop()

DIRS = "NESW"
SPLIT_K = [2, 3, 4, 5, 6, 8, 12, 16, -1]          # bc_vec.hpp CODEC_SPLIT_K
XSPLIT_KEEP = [2, 3]


def reply_text(action_id: int, facing: str, length: int, far: str = "") -> str:
    """bc_vec.hpp decode_for, as protocol text. A far sprint (BC_FARSPRINT, ids 51+) moves along
    `far`, the path the simulator walked (bcv_far_path), as NESW letters."""
    if far:
        return f"MOVE {far}\nENDTURN\n"
    if action_id >= 49:
        return f"SPLIT {length - XSPLIT_KEEP[action_id - 49]}\nENDTURN\n"
    if action_id == 48:
        return "ENDTURN\n"                            # the self-kill: no action
    if action_id < 39:
        n, rest = (1, action_id) if action_id < 3 else (2, action_id - 3) if action_id < 12 \
            else (3, action_id - 12)
        f = DIRS.index(facing)
        path = ""
        for _ in range(n):
            turn = rest % 3
            rest //= 3
            f = f if turn == 0 else (f + 3) % 4 if turn == 1 else (f + 1) % 4
            path += DIRS[f]
        return f"MOVE {path}\nENDTURN\n"
    k = SPLIT_K[action_id - 39]
    return f"SPLIT {length // 2 if k < 0 else k}\nENDTURN\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="checkpoint playing team A in game 1")
    ap.add_argument("--b", required=True)
    ap.add_argument("--name-a", default="cand")
    ap.add_argument("--name-b", default="anchor")
    ap.add_argument("--map", required=True)
    ap.add_argument("--seed", type=int, default=None, help="match seed (pearls); random by default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    lib_name = sim_lib([a.a, a.b])
    if "BCSIM_LIB" in os.environ and pathlib.Path(os.environ["BCSIM_LIB"]).name != lib_name:
        raise SystemExit(f"BCSIM_LIB={os.environ['BCSIM_LIB']} but the checkpoints were trained on {lib_name}")
    os.environ["BCSIM_LIB"] = str(HERE / "bcsim" / lib_name)
    import bcsim
    print(f"sim {lib_name}, BC_QUEEN_GUARD={os.environ.get('BC_QUEEN_GUARD', '0')}", flush=True)

    from train.yardstick import evaluate, load_net
    seed = a.seed if a.seed is not None else random.getrandbits(63)
    dev = torch.device(a.device)
    mp = pathlib.Path(a.map)
    map_text = mp.read_text()

    # the gate's own player setup (ratchet.gate's player()), one copy per checkpoint
    nets, players = [], []
    for path in (a.a, a.b):
        net, ck = load_net(path, dev)
        nets.append(net)
        from train.distill_lstm import LSTMGreedy
        f = LSTMGreedy(net, dev, 2)
        f.n_act = int(ck["args"].get("n_actions", 48))
        f.grid_model = int(ck["args"].get("grid", 14))
        if getattr(net, "temp_in", False):
            net.temp_default.fill_(float(ck["args"].get("temp_min", ck["args"].get("temp", 0.4))))
        f.speaks = bool(ck["args"].get("sonar2", False))
        f.keep_probs = f.speaks
        players.append(f)

    # record every step of each env's first game
    turns = {0: [], 1: []}
    finished = {0: False, 1: False}
    real_step, real_reset = bcsim.BattlecodeVecEnv.step, bcsim.BattlecodeVecEnv.reset
    state = {}

    def reset(env):
        for e in range(env.num_envs):
            env.set_pearl_seed64(e, seed)
        state["obs"] = real_reset(env)
        return state["obs"]

    far0, _ = bcsim.BattlecodeVecEnv.far_targets()
    lib = bcsim.env._lib
    lib.bcv_far_path.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    buf = np.zeros(16, np.int32)
    lib.bcv_s2_preview.restype = ctypes.c_ulonglong
    lib.bcv_s2_preview.argtypes = [ctypes.c_void_p, ctypes.c_int]

    def step(env, acts, *rest):
        o = state["obs"]
        for e in (0, 1):
            if not finished[e]:
                act, far = int(acts[e]), ""
                if far0 >= 0 and act >= far0:          # the path depends on the window: ask the sim now
                    n = lib.bcv_far_path(ctypes.c_void_p(env._h), e, act - far0, buf.ctypes.data)
                    far = "".join(DIRS[d] for d in buf[:n])
                    if not far:
                        raise SystemExit(f"far sprint {act} has no path (env {e}, round {int(o.round[e])})")
                # the packet the sim casts this turn (before the move, as in step()), and what it handed the policy
                pkt = int(lib.bcv_s2_preview(ctypes.c_void_p(env._h), e)) if env.sonar2 else None
                nm = int(o.num_msgs[e])
                heard = [int(x) for x in o.msgs[e][:min(nm, o.msgs.shape[1])]]
                turns[e].append((int(o.dragon_id[e]), int(o.round[e]), int(o.team[e]), act, far, pkt, nm, heard))
        out = real_step(env, acts, *rest)
        state["obs"] = out[0]
        for row in out[2].rows:
            finished[int(row[0])] = True
        return out

    bcsim.BattlecodeVecEnv.reset, bcsim.BattlecodeVecEnv.step = reset, step
    # games=2: env 0 has player A on team A (side 0), env 1 has it on team B
    res = evaluate(players[0], [{"name": a.name_b, "act": players[1]}], bcsim.load_maps([str(mp)]),
                   [mp.stem], games=2, threads=1, seed=seed & 0x7fffffff, sonar=True,
                   return_games=True)
    bcsim.BattlecodeVecEnv.reset, bcsim.BattlecodeVecEnv.step = real_reset, real_step

    for e in (0, 1):
        speaking = [p.speaks for p in (players if res["layout"][e][2] == 0 else players[::-1])]
        print(f"env {e}: team A speaks {speaking[0]}, team B speaks {speaking[1]}; "
              f"{sum(t[5] is not None and t[5] != 0 for t in turns[e])}/{len(turns[e])} turns cast a packet", flush=True)

    from oracle import OracleGame
    from blockparse import Block
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for e, (_, _, side) in enumerate(res["layout"]):
        sim_row = dict(zip(__import__("train.yardstick", fromlist=["COLS"]).COLS, res["games"][e][0].tolist()))
        seq = iter(turns[e])
        chk = {"turns": 0, "msgs": 0, "bad": 0}
        speaks_side = [p.speaks for p in (players if side == 0 else players[::-1])]

        def policy(dragon_id: int, text: str) -> str:
            did, rnd, team, act, far, pkt, nm, heard = next(seq)
            lines = text.splitlines()
            facing, length = lines[1].split()[1], int(lines[2].split()[1])
            if did != dragon_id:
                raise SystemExit(f"desync: engine dragon {dragon_id}, sim dragon {did} (round {rnd})")
            # the engine must hand this dragon exactly what the sim handed the policy
            got = Block(text).msgs
            chk["turns"] += 1
            chk["msgs"] += nm
            if len(got) != nm or got[:len(heard)] != heard:
                chk["bad"] += 1
                if chk["bad"] <= 5:
                    print(f"  sonar mismatch: dragon {did} round {rnd}: engine {len(got)} msgs {got[:4]}, "
                          f"sim {nm} msgs {heard[:4]}", flush=True)
            reply = reply_text(act, facing, length, far)
            if speaks_side[team] and act != 48:
                # bc_vec step(): the v2 packet in all four directions, protocol 3 (lstmbot flush_turn)
                reply = reply[:-len("ENDTURN\n")] + "".join(f"SONAR {d} {pkt}\n" for d in DIRS) \
                    + "PROTOCOL 3\nENDTURN\n"
            return reply

        g = OracleGame(map_text, policy, debug=0, seed=seed)
        r = g.run()
        na, nb = (a.name_a, a.name_b) if side == 0 else (a.name_b, a.name_a)
        path = out / f"{na}-vs-{nb}-on-{mp.stem}-{seed:x}.replay"
        path.write_bytes(g._engine.replay(na, nb))
        left = sum(1 for _ in seq)
        print(f"{path}\n  engine: winner {r.winner} after {r.rounds} rounds "
              f"(A {r.a_dragons} dragons len {r.a_length}, B {r.b_dragons} len {r.b_length}); "
              f"sim: winner {'AB'[sim_row['winner']] if sim_row['winner'] >= 0 else 'draw'} "
              f"after {sim_row['rounds']} rounds; {left} sim turns unused\n"
              f"  sonar: {chk['msgs']} packets heard over {chk['turns']} turns, "
              f"{chk['bad']} turns where the engine's inbox differs from the sim's", flush=True)
        sim_w = "AB"[sim_row["winner"]] if sim_row["winner"] >= 0 else "draw"
        if chk["bad"] or left or str(r.winner) != sim_w:       # rounds: the engine counts from 0
            raise SystemExit("engine replay does not match the simulated game")


if __name__ == "__main__":
    main()
