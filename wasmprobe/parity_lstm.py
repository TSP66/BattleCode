"""Checks the LSTM bot (lstmbot/) against the simulator and the checkpoint.

For each map, plays the simulator (sonar on, random legal moves), follows one
dragon from its first turn, and records the protocol block it would be sent and
the simulator's own 38 x 14 x 14 grid. The blocks go to the bot built natively
with -DBC_DUMP, which prints its grid, previous action, mask, logits and action.

  grid parity   the bot's grid (Q12) against the simulator's, per channel. The
                self channels are reported apart: the simulator draws the whole
                body, the bot tracks it and has to guess the part it has never
                seen (a dragon's first turns, a split child).
  net parity    the checkpoint in PyTorch on the bot's own grids and previous
                actions, its LSTM state carried turn to turn, against the bot's
                int16 logits: max difference, and whether the argmax agrees.

    python parity_lstm.py bot_native ckpt.pt [maps_dir] [turns_per_map]

Exits 1 (check_lstm.sh treats it as a failure) unless: the bot's legal-move
mask equals the simulator's on every turn; no grid channel but the
self ones ever differs; the self channels differ on under 2% of turns; the bot
answers every block it is sent; no int32 accumulator comes within 2x of
overflow (the native build tracks the peak in int64 -- wasm would wrap
silently); logits p99 max|diff| is under LOGIT_P99; the argmax differs on
under 1% of turns.
"""

import ctypes
import pathlib
import subprocess
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bcsim"))

import bcsim                              # noqa: E402
from bcsim.env import _lib as VLIB        # noqa: E402
from train import augment                 # noqa: E402
from train.lstm_net import CHANNELS, LSTMPolicy, NO_ACTION  # noqa: E402
from train.net import masked_logits       # noqa: E402

VLIB.bcv_round_block.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
NCH, GC, A = 38, 196, bcsim.N_ACTIONS
SELF = [CHANNELS.index(c) for c in ("self_body", "self_index", "self_tail")]
TOL = 1.5 / 4096                          # Q12 rounding of values the simulator keeps in float
LOGIT_P99 = 0.25                          # bf16 inference, what the evaluations ran, is ~0.08
ACC_LIMIT = 1 << 30                       # half of int32


def block(env) -> str:
    buf = ctypes.create_string_buffer(1 << 16)
    n = VLIB.bcv_round_block(ctypes.c_void_p(env._h), 0, buf, len(buf))
    return buf.raw[:n].decode()


def follow(text: str, seed: int, turns: int):
    m = augment.parse(text)
    env = bcsim.BattlecodeVecEnv([text], num_envs=1, num_threads=1, seed=seed, sonar=True, grid=True)
    obs = env.reset()
    me = int(obs.dragon_id[0])
    team = "AB"[int(obs.team[0])]
    rng = np.random.default_rng(seed)
    blocks, grids, masks = [], [], []
    for _ in range(20000):
        if int(obs.dragon_id[0]) == me:
            blocks.append(block(env))
            grids.append(env.grid[0].copy())
            masks.append(obs.mask[0].astype(bool).copy())
            if len(blocks) >= turns:
                break
        legal = np.nonzero(obs.mask[0])[0]
        steps = legal[legal < 3]
        pick = steps if len(steps) and rng.random() < 0.85 else legal
        a = int(rng.choice(pick)) if len(pick) else 0
        prev_round = int(obs.round[0])
        obs, _, eps = env.step(np.array([a], np.int32))
        if len(eps.rows) or int(obs.round[0]) < prev_round:
            break
    env.close()
    header = f"ID {me}\nTEAM {team}\nMAP {m.w} {m.h}\nUNIT_LIMIT {m.unit_limit or 64}\n"
    # the engine ends every game with ENDGAME; without it the helper throws at EOF
    return header + "".join(blocks) + "ENDGAME\n", grids, masks


def main() -> None:
    bot, ckpt = sys.argv[1], sys.argv[2]
    maps_dir = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "maps-live")
    turns = int(sys.argv[4]) if len(sys.argv) > 4 else 120
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    arch = {k: ck["args"][k] for k in ("c1", "b1", "c2", "b2", "squeeze", "embed", "hidden", "layers")}
    net = LSTMPolicy(**arch)
    net.load_state_dict(ck["net"])
    net.eval()

    n_turns = grid_bad = self_bad = argmax_diff = short_runs = mask_bad = 0
    acc_peak = 0
    ch_bad = np.zeros(NCH, np.int64)
    max_dl, dl_all = 0.0, []
    for f in sorted(pathlib.Path(maps_dir).glob("*.map")):
        for seed in (1, 2):
            transcript, want, want_mask = follow(f.read_text(), seed, turns)
            res = subprocess.run([bot], input=transcript.encode(), capture_output=True, timeout=600)
            lines = res.stderr.decode().splitlines()
            rows = [np.array(l.split()[1:], np.float64) for l in lines if l.startswith("DUMP")]
            peaks = [int(l.split()[1]) for l in lines if l.startswith("ACCPEAK")]
            acc_peak = max([acc_peak] + peaks)
            n = min(len(rows), len(want))
            state = net.initial_state(1)
            g_bad = s_bad = a_diff = 0
            for t in range(n):
                r = rows[t]
                prev = int(r[0])
                g = r[1:1 + NCH * GC].reshape(NCH, 14, 14) / 4096.0
                mask = r[1 + NCH * GC:1 + NCH * GC + A]
                if not np.array_equal(mask > 0, want_mask[t]):
                    mask_bad += 1
                    if mask_bad <= 5:
                        print(f"  mask differs at turn {t}: bot {np.flatnonzero(mask > 0).tolist()} "
                              f"sim {np.flatnonzero(want_mask[t]).tolist()}")
                logits_bot = r[1 + NCH * GC + A:1 + NCH * GC + 2 * A]
                d = np.abs(g - want[t]) > TOL
                per = d.any(axis=(1, 2))
                ch_bad += per
                if per[[c for c in range(NCH) if c not in SELF]].any():
                    g_bad += 1
                if per[SELF].any():
                    s_bad += 1
                with torch.no_grad():
                    lg, _, state = net(torch.from_numpy(g).float()[None],
                                       torch.tensor([prev if prev >= 0 else NO_ACTION]), state)
                m_t = torch.from_numpy(mask > 0)[None]
                lt = masked_logits(lg.float(), m_t)[0].numpy()
                legal = mask > 0
                if legal.any():
                    dl = np.abs(lt[legal] - logits_bot[legal]).max()
                    dl_all.append(dl)
                    max_dl = max(max_dl, dl)
                    if np.argmax(np.where(legal, lt, -1e30)) != np.argmax(np.where(legal, logits_bot, -1e30)):
                        a_diff += 1
            n_turns += n
            grid_bad += g_bad
            self_bad += s_bad
            argmax_diff += a_diff
            short_runs += n != len(want) or res.returncode != 0
            short = "" if n == len(want) else f" (bot answered {len(rows)} of {len(want)})"
            print(f"{f.stem:24s} seed {seed}: {n:3d} turns | grid differs {g_bad:3d} (self {s_bad:3d}) | "
                  f"argmax differs {a_diff:2d}{short}", flush=True)
    print(f"\n{n_turns} turns. grid (excluding self): {grid_bad} turns differ; self channels: {self_bad}")
    bad_ch = [(CHANNELS[c], int(ch_bad[c])) for c in range(NCH) if ch_bad[c]]
    print("turns with a difference, by channel:", bad_ch or "none")
    if dl_all:
        print(f"logits vs PyTorch (legal actions): median max|diff| {np.median(dl_all):.4f}, "
              f"p99 {np.percentile(dl_all, 99):.4f}, max {max_dl:.4f}; argmax differs on "
              f"{argmax_diff}/{n_turns} turns ({argmax_diff / max(n_turns, 1):.2%})")
    print(f"int32 accumulator peak {acc_peak:,} ({acc_peak / 2**31:.2%} of overflow)")
    fails = []
    print(f"legal-move mask vs simulator: {mask_bad} of {n_turns} turns differ")
    if mask_bad:
        fails.append(f"mask differs on {mask_bad} turns")
    if grid_bad:
        fails.append(f"grid differs on {grid_bad} turns")
    if self_bad > 0.02 * n_turns:
        fails.append(f"self channels differ on {self_bad} turns")
    if short_runs:
        fails.append(f"{short_runs} runs where the bot did not answer every block or exited nonzero")
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
