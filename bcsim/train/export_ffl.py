"""Exports a feed-forward FFL policy checkpoint (train/ff_net.py FFLPolicy) to lstmbot/weights.hpp.

The same numerics as export_lstm.py (lstmbot/lstmnet.hpp), but int16 weights at 11 bits per output
row (WMAX), Q10 activations between layers, float from the accumulator on. The CNN trunk, previous-action
table and LayerNorm are LSTMPolicy's; the LSTM cells are replaced by `layers` dense layers + SiLU.

The temperature inputs (temp_in) are folded into mlp.0's bias at the temperature the gate plays
greedy at (ratchet.gate player(): args temp_min, else temp), so the bot plays exactly the gate's
policy. mlp.0 then reads [LayerNorm (embed + 48), scalars (scal_in, 5), identity (ident_in, 3)].

Blob (int16, high plane then low plane), in order:
  conv    W [Cout][Cin*k*k]        stem, res15 (c1, c2)..., down, res8 (c1, c2)..., squeeze
  linear  W [K/2][Out][2]          embed, mlp0, mlp1, ..., pi
Floats (FP[]), in the same order, per layer: scales [Out], bias [Out], and for a normed conv
gamma [Cout], beta [Cout]; then the previous-action table [n_actions + 1][48] and the LayerNorm
gamma, beta [embed + 48].

    python -m train.export_ffl --ckpt ../runs/scratch_1002/anchors/gen8.pt
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from train.export_lstm import quant_rows, write_silu  # noqa: E402
from train.ff_net import FFLPolicy, build as build_net  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
# 11-bit weights (export_lstm's are 12): the 15x15 net's last res8 conv (K = 1008) took the int32
# accumulator to 45% of overflow at 12 bits in parity games (wasm wraps silently); 11 bits halve
# every accumulator, and are still far finer than the bf16 the gates play at.
WMAX = 1023


def greedy_temp(args: dict) -> float:
    """The temperature a temperature-conditioned policy is told when played greedy (the gate)."""
    return float(args.get("temp_min", args.get("temp", 0.4)))


def load(path: str) -> tuple[FFLPolicy, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    if a.get("arch") != "ffl":
        raise SystemExit(f"{path} is not an FFL policy checkpoint (arch {a.get('arch')})")
    net = build_net(a)
    net.load_state_dict(ck["net"])
    net.eval()
    if net.temp_in:
        net = net.fold_temp(greedy_temp(a))
    return net.eval(), ck


def mlp0_width(net: FFLPolicy) -> int:
    """mlp0's input width in the bot: the i16 kernel reads pairs, so an odd one gets a zero column."""
    return (net.mlp[0].in_features + 1) // 2 * 2


def build(net: FFLPolicy):
    """(int16 blob, float32 params), in lstmbot's reading order."""
    blob, fp = [], []

    def conv(m, norm=None):
        q, s = quant_rows(m.weight.detach().numpy(), WMAX)
        blob.append(q.reshape(-1))
        fp.extend([s, m.bias.detach().numpy().astype(np.float32)])
        if norm is not None:
            fp.extend([norm.weight.detach().numpy().astype(np.float32),
                       norm.bias.detach().numpy().astype(np.float32)])

    def linear(m):
        w = m.weight.detach().numpy()
        if m is net.mlp[0] and w.shape[1] % 2:
            # ahist_in makes mlp0 387 wide: the bot feeds one zero input more (mlp0_width)
            w = np.concatenate([w, np.zeros((w.shape[0], 1), w.dtype)], 1)
        if w.shape[1] % 2:
            raise SystemExit(f"linear layer with an odd input width {w.shape[1]}: the i16 kernel reads pairs")
        q, s = quant_rows(w, WMAX)
        out_, k = q.shape
        blob.append(q.reshape(out_, k // 2, 2).transpose(1, 0, 2).reshape(-1))
        fp.extend([s, m.bias.detach().numpy().astype(np.float32)])

    conv(net.stem, net.stem_n)
    for blk in net.res15:
        conv(blk.c1, blk.n1)
        conv(blk.c2, blk.n2)
    conv(net.down, net.down_n)
    for blk in net.res8:
        conv(blk.c1, blk.n1)
        conv(blk.c2, blk.n2)
    conv(net.squeeze)
    linear(net.embed)
    for lin in net.mlp:
        linear(lin)
    linear(net.pi)
    fp.append(net.prev_action.weight.detach().numpy().astype(np.float32).reshape(-1))
    fp.extend([net.in_norm.weight.detach().numpy().astype(np.float32),
               net.in_norm.bias.detach().numpy().astype(np.float32)])
    return np.concatenate(blob), np.concatenate([x.reshape(-1) for x in fp])


def write(dest: pathlib.Path, blob: np.ndarray, fp: np.ndarray, net: FFLPolicy, a: dict, source: str) -> None:
    u = blob.view(np.uint16)
    safe = [chr(c) if 32 <= c < 127 and chr(c) not in '"\\?' else "\\%03o" % c for c in range(256)]

    def literal(plane: np.ndarray) -> str:
        data = plane.astype(np.uint8).tobytes()
        return "\n".join('    "' + "".join(safe[c] for c in data[i:i + 2048]) + '"'
                         for i in range(0, len(data), 2048))

    floats = ",\n".join("    " + ", ".join(f"{x:.9e}f" for x in fp[i:i + 8]) for i in range(0, len(fp), 8))
    mlp_in = mlp0_width(net)
    dest.write_text(
        "// GENERATED by bcsim/train/export_ffl.py. Do not edit.\n"
        f"// source: {source}\n// written: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        "#pragma once\n#include <cstdint>\nnamespace wt {\n"
        "constexpr char const ARCH[] = \"ffl\";\n"
        f"constexpr int C1 = {a['c1']}, B1 = {a['b1']}, C2 = {a['c2']}, B2 = {a['b2']};\n"
        f"constexpr int SQ = {a['squeeze']}, EMB = {a['embed']}, HID = {a['hidden']}, LAYERS = {len(net.mlp)};\n"
        f"constexpr int GRID = {a.get('grid', 14)};      // the grid side the net reads\n"
        f"constexpr int IN_CH = {a.get('in_ch', 38)};    // grid channels the CNN reads\n"
        f"constexpr int N_ACT = {a.get('n_actions', 48)};\n"
        f"constexpr int SCAL_IN = {int(net.scal_in)}, IDENT_IN = {int(net.ident_in)}, AHIST_IN = {int(net.ahist_in)};\n"
        f"constexpr int MLP_IN = {mlp_in};      // mlp0's input width (temperature folded out)\n"
        f"constexpr float TEMP = {greedy_temp(a)}f;   // folded into mlp0's bias\n"
        f"constexpr std::uint32_t COUNT = {u.size}u;      // int16 weights\n"
        f"constexpr std::uint32_t N_FP = {fp.size}u;      // float parameters\n"
        f'constexpr char const SOURCE[] = "{source}";\n'
        f"inline constexpr char HI[] =\n{literal(u >> 8)};\n"
        f"inline constexpr char LO[] =\n{literal(u & 0xFF)};\n"
        f"inline constexpr float FP[] = {{\n{floats}\n}};\n"
        "static_assert(sizeof(HI) == COUNT + 1 && sizeof(LO) == COUNT + 1);\n"
        "static_assert(sizeof(FP) == N_FP * sizeof(float));\n"
        "}  // namespace wt\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=str(ROOT / "lstmbot/weights.hpp"))
    a = p.parse_args()
    net, ck = load(a.ckpt)
    args = ck["args"]
    for k, want in (("portal", True), ("s2", True)):
        if bool(args.get(k, False)) != want:
            raise SystemExit(f"lstmbot speaks the portal-report packet (s2 + portal): checkpoint has {k}={args.get(k)}")
    blob, fp = build(net)
    src = f"{pathlib.Path(a.ckpt).parent.name}/{pathlib.Path(a.ckpt).name} iter {ck.get('iter')}"
    write(pathlib.Path(a.out), blob, fp, net, args, src)
    write_silu(pathlib.Path(a.out).with_name("silu_table.hpp"))
    print(f"{blob.size:,} int16 weights + {fp.size:,} floats -> {a.out} ({src}; temperature {greedy_temp(args)} folded)")


if __name__ == "__main__":
    main()
