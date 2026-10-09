"""Checks the FFL bot (lstmbot/, train/ff_net.py FFLPolicy checkpoints) against the simulator
the policy trained in and against the checkpoint itself, in closed loop.

One game per (map, seed, who): the run's own simulator library (libbcvec_s2_g15p, picked from the
checkpoint's args like ratchet.gate does, BC_QUEEN_GUARD=1), both teams speaking the v2 packet.
ONE dragon is played by the bot, built natively with -DBC_DUMP: each of its turns the simulator
renders the protocol block (bcv_round_block), the bot answers, and the simulator plays the bot's
own action (the "ACT <id>" line). Every other dragon is the checkpoint in PyTorch, sampled at
temperature --temp, so the bot sees real packets, real queens, real portal reports.

Per bot turn:
  grid     the bot's Q12 CNN input (54 channels) against the simulator's float grid; the self
           channels (tracked body) reported apart
  scalars  mlp0's 8 extra inputs (scal_in, ident_in) against the simulator's constant planes,
           and the 75 action-history inputs (ahist_in) against its plane 57
  mask     exact, against the simulator's (queen guard included)
  logits   the ORIGINAL checkpoint (temperature input = the gate's greedy temperature, fp32) on
           the simulator's own grid and the bot's previous action, against the bot's int16 logits
           over the legal actions; and whether the argmax agrees
  packet   the 64 bits the bot casts, against what the simulator would cast (bcv_s2_preview): equal
  far      a far sprint's MOVE path against the simulator's own (bcv_far_path): equal

    python parity_ffl.py bot_native ckpt.pt [maps_dir] [games_per_map] [turns_per_game]

Exits 1 unless: mask, packets and far paths always agree; no grid channel but the self ones
differs; scalars agree; the self channels differ on under 2% of turns; the bot answered every
block; the int32 accumulator stays under half of overflow; logits p99 max|diff| < LOGIT_P99;
the argmax differs on under 1% of turns.
"""

import ctypes
import os
import pathlib
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bcsim"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["BC_QUEEN_GUARD"] = "1"          # scratch_1002 trains, gates and plays with the queen guard
_ARGS = torch.load(sys.argv[2], map_location="cpu", weights_only=False)["args"]
if not (_ARGS.get("portal") and _ARGS.get("s2") and int(_ARGS.get("grid", 14)) == 15):
    raise SystemExit("parity_ffl checks portal-report (s2g15p) checkpoints only")
# the bot always plays the queen dead-end mask (user, 2026-10-09; lstmbot obs.hpp queen_deadend) at
# BC_QD_LEVEL (main.cpp, default 2), so its mask is checked against the simulator's with BC_QUEEN_DEADEND=2
# -- built into libbcvec_s2_g15p_qd2.so (a separate file, so a live run's library is untouched; with the
# option off it plays as the old one). A level-1 bot (-DBC_QD_LEVEL=1): BC_QUEEN_DEADEND=1.
os.environ.setdefault("BC_QUEEN_DEADEND", "2")
os.environ["BCSIM_LIB"] = os.environ.get("PARITY_LIB", str(ROOT / "bcsim/bcsim/libbcvec_s2_g15p_qd2.so"))

import bcsim                              # noqa: E402
from bcsim.env import _lib as VLIB        # noqa: E402
from train import augment                 # noqa: E402
from train.ff_net import build as build_net  # noqa: E402
from train.export_ffl import greedy_temp  # noqa: E402
from train.net import masked_logits       # noqa: E402

torch.set_num_threads(1)
VLIB.bcv_round_block.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
VLIB.bcv_s2_preview.restype = ctypes.c_ulonglong
VLIB.bcv_s2_preview.argtypes = [ctypes.c_void_p, ctypes.c_int]
VLIB.bcv_far_path.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
FAR0, _ = bcsim.BattlecodeVecEnv.far_targets()
NCH, G = int(_ARGS["in_ch"]), 15
GC = G * G
A = int(_ARGS["n_actions"])
AHIST = bool(_ARGS.get("ahist_in"))        # + n_actions history inputs (grid plane 57)
N_X = 8 + (A if AHIST else 0)              # mlp0's scalars: 5 scal_in + 3 ident_in (+ ahist_in)
N_X += (int(_ARGS["embed"]) + 48 + N_X) % 2  # export_ffl.mlp0_width: the zero pad to an even width
SELF = (19, 20, 21)                        # grid_cfg SELF_BODY, SELF_INDEX, SELF_TAIL
TOL = 1.5 / 4096                           # Q12 rounding of values the simulator keeps in float
LOGIT_P99 = 0.25
ACC_LIMIT = 1 << 30
DIRS = "NESW"
# PARITY_FAR_TEST=1: the bot (a BC_DUMP build) plays a far sprint whenever one is legal, so their
# paths are checked against the simulator's; the argmax is then not the policy's and is not compared
FAR_TEST = os.environ.get("PARITY_FAR_TEST") == "1"
if FAR_TEST:
    os.environ["BC_FAR_TEST"] = "1"
NAMES = ["kelp_n", "kelp_e", "kelp_s", "kelp_w", "portal_n", "portal_e", "portal_s", "portal_w", "never_spawns",
         "pearl", "pearl_timer", "ally_head", "ally_body", "enemy_head", "enemy_body", "seg_n", "seg_e", "seg_s",
         "seg_w", "self_body", "self_index", "self_tail", "seen", "pearl_expected", "pearl_timer_proj",
         "enemy_mem", "ally_mem", "visited", "round", "length", "units", "map_w", "map_h", "echo0", "echo1",
         "echo2", "echo3", "echo4", "rep_len", "pt_known", "pt_closed", "pt_room", "pt_pearls", "is_queen",
         "ally_queen", "enemy_queen", "ally_queen_mem", "enemy_queen_mem", "aq_dx", "aq_dy", "aq_fresh",
         "eq_dx", "eq_dy", "eq_fresh"]


def block(env) -> str:
    buf = ctypes.create_string_buffer(1 << 16)
    n = VLIB.bcv_round_block(ctypes.c_void_p(env._h), 0, buf, len(buf))
    return buf.raw[:n].decode()


def far_path(env, k: int) -> str:
    buf = np.zeros(16, np.int32)
    n = VLIB.bcv_far_path(ctypes.c_void_p(env._h), 0, k, buf.ctypes.data)
    return "".join(DIRS[d] for d in buf[:n])


class Bot:
    """The native bot as a subprocess, one block in, one reply out."""

    def __init__(self, exe: str, header: str, dump: str):
        self.p = subprocess.Popen([exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(os.environ["PARITY_STDERR"], "a") if os.environ.get("PARITY_STDERR")
                                  else subprocess.DEVNULL, env={**os.environ, "BC_DUMP_FILE": dump},
                                  text=True, bufsize=1)
        self.p.stdin.write(header)

    def turn(self, text: str) -> list[str]:
        self.p.stdin.write(text)
        self.p.stdin.flush()
        lines = []
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("the bot exited mid-game")
            line = line.rstrip("\n")
            lines.append(line)
            if line == "ENDTURN":
                return lines

    def close(self) -> int:
        try:
            self.p.stdin.write("ENDGAME\n")
            self.p.stdin.close()
        except BrokenPipeError:
            pass
        return self.p.wait(timeout=60)


def play(exe, net, text, seed, who, turns, temp):
    """One game with the bot as one dragon. Returns the per-turn records."""
    m = augment.parse(text)
    env = bcsim.BattlecodeVecEnv([text], num_envs=1, num_threads=1, seed=seed, sonar=True, grid=True)
    env.set_sonar2(0, (0, 1))
    obs = env.reset()
    rng = np.random.default_rng(seed)
    prev: dict[int, int] = {}
    target, bot, recs = None, None, []
    dump = tempfile.NamedTemporaryFile("w", suffix=".dump", delete=False).name
    last_round_seen = None
    t_temp = torch.tensor([temp])
    for _ in range(200000):
        did, rnd, uid = int(obs.dragon_id[0]), int(obs.round[0]), int(obs.uid[0])
        if target is None:
            # who: "queen" (the first queen to act), "spawn" (the first non-queen map dragon), "child"
            # (the first dragon born from a split after round 3)
            if (who == "queen" and did < 2) or (who == "spawn" and did >= 2 and rnd == 0) or \
                    (who == "child" and rnd > 3 and uid not in prev):
                target = did
                header = f"ID {did}\nTEAM {'AB'[int(obs.team[0])]}\nMAP {m.w} {m.h}\nUNIT_LIMIT {m.unit_limit or 64}\n"
                bot = Bot(exe, header, dump)
        if target is not None and did == target and bot is not None:
            lines = bot.turn(block(env))
            act = int(next(l.split()[1] for l in lines if l.startswith("ACT ")))
            sent = [int(l.split()[2]) for l in lines if l.startswith("SONAR ")]
            move = next((l.split()[1] for l in lines if l.startswith("MOVE ")), "")
            want_pkt = int(VLIB.bcv_s2_preview(ctypes.c_void_p(env._h), 0))
            rec = {"round": rnd, "grid": env.grid[0].copy(), "mask": obs.mask[0].astype(bool).copy(),
                   "act": act, "sent": sent, "want_pkt": want_pkt, "far_ok": True}
            if FAR0 >= 0 and act >= FAR0:
                rec["far_ok"] = move == far_path(env, act - FAR0)
            recs.append(rec)
            last_round_seen = rnd
            a = act
            if len(recs) >= turns or act == 48:
                break
        else:
            if target is not None and last_round_seen is not None and rnd > last_round_seen + 1:
                break                                  # a whole round without the bot's dragon: it died
            g = torch.from_numpy(env.grid[:1])
            p = torch.tensor([prev.get(uid, net.no_action)])
            with torch.no_grad():
                lg, _ = net(g, p, temp=t_temp)
            ml = masked_logits(lg.float(), torch.from_numpy(obs.mask[:1]).bool())[0]
            pr = torch.softmax(ml / temp, 0).numpy().astype(np.float64)
            a = int(rng.choice(len(pr), p=pr / pr.sum()))
        prev[uid] = a
        prev_round = rnd
        obs, _, eps = env.step(np.array([a], np.int32))
        if len(eps.rows) or int(obs.round[0]) < prev_round:
            break
    env.close()
    rc = bot.close() if bot is not None else 0
    rows, peak = [], 0
    for line in pathlib.Path(dump).read_text().splitlines():
        if line.startswith("DUMP"):
            rows.append(line.split()[1:])
        elif line.startswith("ACCPEAK"):
            peak = max(peak, int(line.split()[1]))
    os.unlink(dump)
    return recs, rows, peak, rc, target


def main() -> None:
    exe, ckpt = sys.argv[1], sys.argv[2]
    maps_dir = pathlib.Path(sys.argv[3] if len(sys.argv) > 3 else ROOT / "maps-gate-1001")
    per_map = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    turns = int(sys.argv[5]) if len(sys.argv) > 5 else 150
    temp = float(os.environ.get("PARITY_TEMP", "0.25"))
    maps = [maps_dir / x for x in os.environ.get("PARITY_MAPS", "").split(",") if x] or sorted(maps_dir.glob("*.map"))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    net = build_net(ck["args"])
    net.load_state_dict(ck["net"])
    net.eval()
    t_greedy = torch.tensor([greedy_temp(ck["args"])])

    n_turns = grid_bad = self_bad = scal_bad = mask_bad = argmax_diff = pk_bad = far_bad = short = 0
    far_n = 0
    cov = {"own portal report sent": 0, "portal planes lit": 0, "enemy queen in packet": 0,
           "queen relayed (age > 0)": 0, "team packets heard (rep_len lit)": 0}
    acc_peak = 0
    ch_bad = np.zeros(NCH, np.int64)
    dl_all = []
    miss = []          # argmax misses: (the checkpoint's top-two logit margin, self channels differed)
    whos = ("queen", "spawn", "child")
    for f in maps:
        for i in range(per_map):
            seed, who = 1000 + i, whos[i % len(whos)]
            recs, rows, peak, rc, target = play(exe, net, f.read_text(), seed, who, turns, temp)
            acc_peak = max(acc_peak, peak)
            n = min(len(recs), len(rows))
            short += n != len(recs) or rc != 0
            g_bad = s_bad = a_diff = p_bad = 0
            for t in range(n):
                r, row = recs[t], rows[t]
                prev = int(row[0])
                q = np.array(row[1:1 + NCH * GC], np.float64).reshape(NCH, G, G) / 4096.0
                k = 1 + NCH * GC
                x = np.array(row[k:k + N_X], np.float64)
                k += N_X
                mask = np.array(row[k:k + A], np.int64) > 0
                k += A
                logits_bot = np.array(row[k:k + A], np.float64)
                k += A
                act_bot, pkt_bot = int(row[k]), int(row[k + 1])
                sim = r["grid"]
                d = np.abs(q - sim[:NCH]) > TOL
                per = d.any(axis=(1, 2))
                ch_bad += per
                g_bad += per[[c for c in range(NCH) if c not in SELF]].any()
                s_bad += per[list(SELF)].any()
                c = G // 2
                want_x = np.array([sim[28, c, c], sim[28, c, c] ** 2, sim[29, c, c], sim[30, c, c] ** 2,
                                   sim[43, c, c], sim[54, c, c], sim[55, c, c], sim[56, c, c]]
                                  + (sim[57].reshape(-1)[:A].tolist() if AHIST else []) + [0.0] * (N_X - 8 - A * AHIST))
                if np.abs(x - want_x).max() > 1e-6:
                    scal_bad += 1
                    if scal_bad <= 3:
                        print(f"  scalars differ at turn {t}: bot {x.round(5).tolist()} sim {want_x.round(5).tolist()}")
                if not np.array_equal(mask, r["mask"]):
                    mask_bad += 1
                    if mask_bad <= 5:
                        print(f"  mask differs at turn {t} (round {r['round']}): bot {np.flatnonzero(mask).tolist()} "
                              f"sim {np.flatnonzero(r['mask']).tolist()}")
                legal = r["mask"]
                with torch.no_grad():
                    lg, _ = net(torch.from_numpy(sim)[None], torch.tensor([prev]), temp=t_greedy)
                lt = masked_logits(lg.float(), torch.from_numpy(legal)[None])[0].numpy()
                if legal.any():
                    dl_all.append(np.abs(lt[legal] - logits_bot[legal]).max())
                    if int(np.argmax(np.where(legal, lt, -1e30))) != act_bot and not FAR_TEST:
                        a_diff += 1
                        top = np.sort(lt[legal])[::-1]
                        miss.append((float(top[0] - top[1]) if len(top) > 1 else 0.0, bool(per[list(SELF)].any())))
                bad_pkt = pkt_bot != r["want_pkt"] or (r["act"] != 48 and (len(r["sent"]) != 4 or
                                                                           any(v != r["want_pkt"] for v in r["sent"])))
                if bad_pkt:
                    p_bad += 1
                    if pk_bad + p_bad <= 5:
                        print(f"  packet differs at turn {t} (round {r['round']}): bot {pkt_bot:#018x} "
                              f"sim {r['want_pkt']:#018x}")
                cov["own portal report sent"] += bool(pkt_bot >> 13 & 1)
                cov["portal planes lit"] += bool(sim[39:43].any())
                cov["enemy queen in packet"] += bool(pkt_bot >> 1 & 1 and pkt_bot >> 20 & 1)
                cov["queen relayed (age > 0)"] += bool(pkt_bot >> 20 & 1 and (pkt_bot >> 14 & 3) > 0)
                cov["team packets heard (rep_len lit)"] += bool(sim[38].any())
                if FAR0 >= 0 and r["act"] >= FAR0:
                    far_n += 1
                    far_bad += not r["far_ok"]
            n_turns += n
            grid_bad += g_bad
            self_bad += s_bad
            argmax_diff += a_diff
            pk_bad += p_bad
            print(f"{f.stem:22s} seed {seed} {who:5s} (dragon {target}): {n:3d} turns | grid differs {g_bad:3d} "
                  f"(self {s_bad:3d}) | packets differ {p_bad} | argmax differs {a_diff:2d}"
                  f"{'' if n == len(recs) and rc == 0 else f' (bot answered {len(rows)} of {len(recs)}, rc {rc})'}",
                  flush=True)

    print(f"\n{n_turns} turns. grid (excluding self): {grid_bad} turns differ; self channels: {self_bad}")
    bad_ch = [(NAMES[c], int(ch_bad[c])) for c in range(NCH) if ch_bad[c]]
    print("turns with a difference, by channel:", bad_ch or "none")
    if dl_all:
        print(f"logits vs the checkpoint (legal actions): median max|diff| {np.median(dl_all):.4f}, "
              f"p99 {np.percentile(dl_all, 99):.4f}, max {max(dl_all):.4f}; argmax differs on "
              f"{argmax_diff}/{n_turns} turns ({argmax_diff / max(n_turns, 1):.2%})")
    if miss:
        print("argmax misses (checkpoint top-2 margin, self channels differed): "
              + ", ".join(f"{m:.4f}{' self' if sd else ''}" for m, sd in sorted(miss)))
    print(f"int32 accumulator peak {acc_peak:,} ({acc_peak / 2**31:.2%} of overflow)")
    print(f"legal-move mask vs simulator: {mask_bad} of {n_turns} turns differ")
    print(f"mlp0 scalars vs simulator: {scal_bad} of {n_turns} turns differ")
    print(f"sonar packets vs simulator: {pk_bad} of {n_turns} turns differ")
    print(f"far sprints played: {far_n}, path differs from the simulator's on {far_bad}")
    print("coverage (turns):", ", ".join(f"{k} {v}" for k, v in cov.items()))
    fails = []
    if mask_bad:
        fails.append(f"mask differs on {mask_bad} turns")
    if grid_bad:
        fails.append(f"grid differs on {grid_bad} turns")
    if scal_bad:
        fails.append(f"mlp scalars differ on {scal_bad} turns")
    if pk_bad:
        fails.append(f"packet differs on {pk_bad} turns")
    if far_bad:
        fails.append(f"far path differs on {far_bad} turns")
    if self_bad > 0.02 * n_turns:
        fails.append(f"self channels differ on {self_bad} turns")
    if short:
        fails.append(f"{short} games where the bot did not answer every block or exited nonzero")
    if acc_peak >= ACC_LIMIT:
        fails.append("accumulator within 2x of int32 overflow")
    if not dl_all or np.percentile(dl_all, 99) > LOGIT_P99:
        fails.append(f"logits p99 over {LOGIT_P99}")
    if argmax_diff > 0.01 * n_turns:
        fails.append("argmax differs on over 1% of turns")
    print("PARITY FAIL: " + "; ".join(fails) if fails else "PARITY OK")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
