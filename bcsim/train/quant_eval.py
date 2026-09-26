"""What the C++ bot's int16 numerics cost in play, measured in the simulator.

The shipped bot (lstmbot/lstmnet.hpp) runs an LSTM checkpoint with int16
weights at 12 bits per output row, the grid at Q12 and activations between
layers at Q10 (+-32). QuantLSTM emulates exactly those roundings in PyTorch
(validated against the bot's own logits by wasmprobe/parity_lstm.py: p99
max|diff| 0.067 bot vs checkpoint, 0.076 emulation vs checkpoint), so the
two can play each other in the simulator by the thousand.

The fp32 checkpoint is the learner; each opponent is the same checkpoint
through a different numeric path:
  int16_q10   what v15 ships
  int16_q12   the first port's numerics: activations clipped at +-8, which
              cost logits up to 3.4 -- the control, to show the harness sees
              a real degradation when there is one
Score > 0.5 for the fp32 learner means the quantized net plays worse. Each
opponent also counts, over the turns it played, how often its (unmasked)
argmax differs from fp32's on the same input and state.

Runs on the CPU, so it leaves the GPU to training:
    python -m train.quant_eval --ckpt ../runs/ratchet_v8_sab/anchors/gen3.pt --games 48
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from train.distill_lstm import LSTMGreedy  # noqa: E402
from train.lstm_net import LSTMPolicy  # noqa: E402
from train.yardstick import evaluate  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARCH = ("c1", "b1", "c2", "b2", "squeeze", "embed", "hidden", "layers")
WMAX = 2047                                   # export_lstm.py: 12 bits per output row


def load(path: str) -> LSTMPolicy:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = LSTMPolicy(**{k: ck["args"][k] for k in ARCH})
    net.load_state_dict(ck["net"])
    return net.eval()


def quant_rows(w: torch.Tensor) -> torch.Tensor:
    """export_lstm.quant_rows, returned dequantized."""
    w2 = w.reshape(w.shape[0], -1).double()
    s = w2.abs().max(1, keepdim=True).values / WMAX
    s[s == 0] = 1e-12
    return (torch.round(w2 / s).clamp(-WMAX, WMAX) * s).float().reshape(w.shape)


class QuantLSTM(LSTMPolicy):
    """LSTMPolicy computed the way lstmbot/ computes it."""

    def __init__(self, ref: LSTMPolicy, qa: float, arch: dict):
        super().__init__(**arch)
        self.load_state_dict(ref.state_dict())
        self.ref, self.qa = ref, qa
        self.turns = self.flips = 0
        with torch.no_grad():
            for m in [self.stem, self.down, self.squeeze, self.embed, self.pi]:
                m.weight.copy_(quant_rows(m.weight))
            for blk in list(self.res15) + list(self.res8):
                blk.c1.weight.copy_(quant_rows(blk.c1.weight))
                blk.c2.weight.copy_(quant_rows(blk.c2.weight))
            for cell in self.lstm:           # the bot quantizes [W_ih | W_hh] rows together
                w = quant_rows(torch.cat([cell.weight_ih, cell.weight_hh], 1))
                n_in = cell.weight_ih.shape[1]
                cell.weight_ih.copy_(w[:, :n_in])
                cell.weight_hh.copy_(w[:, n_in:])

    def q(self, x, scale):
        return (torch.round(x * scale) / scale).clamp(-32767 / scale, 32767 / scale)

    def forward(self, grid, prev_a, state):
        qa = lambda x: self.q(x, self.qa)                          # noqa: E731
        x = qa(F.silu(self.stem_n(self.stem(self.q(grid.float(), 4096.0)))))
        for b in self.res15:
            h = qa(F.silu(b.n1(b.c1(x))))
            x = qa(F.silu(x + b.n2(b.c2(h))))
        x = qa(F.silu(self.down_n(self.down(x))))
        for b in self.res8:
            h = qa(F.silu(b.n1(b.c1(x))))
            x = qa(F.silu(x + b.n2(b.c2(h))))
        x = qa(F.silu(self.squeeze(x)).flatten(1))
        x = qa(self.in_norm(torch.cat([F.silu(self.embed(x)), self.prev_action(prev_a)], 1)))
        new = []
        for cell, (h, c) in zip(self.lstm, state):
            h, c = cell(x, (h, c))
            h = qa(h)
            new.append((h, c))
            x = h
        logits = self.pi(x)
        with torch.no_grad():                # fp32 on the same input and state
            ref_logits, _, _ = self.ref(grid.float(), prev_a, state)
        self.turns += len(logits)
        self.flips += int((ref_logits.argmax(1) != logits.argmax(1)).sum())
        return logits, self.v(x).squeeze(-1), new


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--games", type=int, default=48, help="per (opponent, map), half each side")
    p.add_argument("--maps", default=str(ROOT / "maps-live"))
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--controls", default="int16_q12", help="comma list of extra arms; '' for none")
    p.add_argument("--out", default="")
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device("cpu")

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    arch = {k: ck["args"][k] for k in ARCH}
    ref = load(a.ckpt)
    arms = {"int16_q10": 1024.0}
    if "int16_q12" in a.controls.split(","):
        arms["int16_q12"] = 4096.0
    maps_p = sorted(pathlib.Path(a.maps).glob("*.map"))
    maps, names = [m.read_text() for m in maps_p], [m.stem for m in maps_p]
    n_envs = len(arms) * len(maps) * max(2, a.games)

    qnets = {k: QuantLSTM(ref, qa, arch).eval() for k, qa in arms.items()}
    opps = [{"name": k, "act": LSTMGreedy(n, dev, n_envs)} for k, n in qnets.items()]
    t0 = time.time()
    with torch.inference_mode():
        res = evaluate(LSTMGreedy(ref, dev, n_envs), opps, maps, names, games=a.games,
                       threads=a.threads, sonar=True, max_seconds=6 * 3600)
    print(f"\n{a.ckpt}: fp32 (learner) vs each numeric path, {a.games} games a map, "
          f"{time.time() - t0:.0f}s")
    out = {"ckpt": a.ckpt, "games_per_map": a.games, "arms": {}}
    for k, n in qnets.items():
        s = res["summary"][k]
        g = s["n"]
        se = 0.5 / np.sqrt(max(g, 1))
        per_map = {m: res["cells"][k][m]["score"] for m in names if m in res["cells"][k]}
        print(f"  {k:10s} fp32 scores {s['score']:.4f} +- {se:.4f} over {g} games | "
              f"argmax differs from fp32 on {n.flips}/{n.turns} turns ({n.flips / max(n.turns, 1):.3%})")
        print("             by map: " + ", ".join(f"{m} {v:.2f}" for m, v in per_map.items()))
        out["arms"][k] = {"score": s["score"], "se": se, "games": g, "flips": n.flips,
                          "turns": n.turns, "by_map": per_map}
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
