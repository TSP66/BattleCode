"""Prices a real network against the judge's 100M CPU points per dragon turn.

The judge meters wasm instructions, not time, and the cost of a
multiply-accumulate depends entirely on the loop it sits in. Measured on this
bot (see SUBMITTING.md): a MAC inside the convolution kernels costs about
**3.3 points**, one inside a dot-product loop about **15**, and one through
NumPy about 23. So a MAC total is not a budget -- where the MACs are matters
more than how many there are, and a small recurrent layer can cost more than a
large convolution.

The deployed 64x4 net is the calibration point: 16.46M MAC measured at **69M
points** a turn. Split by kind at the rates above that comes to 68.9M, which is
why this module prices per kind rather than with one blended rate.

Counts come from hooks on the actual module, not from formulas, so an
architecture cannot drift away from its own price.

    python -m train.budget                        # every architecture we have
    python -m train.budget --ckpt runs/x/latest.pt

Reserves: the judge kills a dragon that overruns, and an overrun costs the game,
so the usable figure is well under 100M. RESERVE covers the turn's parsing and
bookkeeping and HEAD_RESERVE the per-turn output; what is left is what the
forward pass may spend.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

ROOT = pathlib.Path(__file__).resolve().parents[2]

MAX_TURN_POINTS = 100_000_000
RESERVE = 12_000_000          # parsing, memory update, bookkeeping
HEAD_RESERVE = 7_000_000      # building and flushing the reply
USABLE = MAX_TURN_POINTS - RESERVE - HEAD_RESERVE

# measured points per MAC, by the loop the MAC sits in
PTS_CONV = 3.3
PTS_DOT = 15.0

# what the deployed net measured, to check the model against reality
DEPLOYED_MAC = 16_462_784
DEPLOYED_POINTS = 69_000_000


class Counter:
    """MACs per layer kind, by forward hook on the module itself."""

    def __init__(self) -> None:
        self.conv = 0
        self.dot = 0
        self.rows: list[tuple[str, str, int]] = []

    def _hook(self, name: str):
        def fn(mod, inp, out):
            if isinstance(mod, nn.Conv2d):
                # one MAC per output element per input channel per kernel cell
                n = (out.numel() // out.shape[0]) * (mod.in_channels
                                                     // mod.groups) * mod.kernel_size[0] \
                    * mod.kernel_size[1]
                self.conv += n
                self.rows.append((name, "conv", n))
            elif isinstance(mod, nn.Linear):
                n = mod.in_features * mod.out_features
                self.dot += n
                self.rows.append((name, "dot", n))
            elif isinstance(mod, (nn.LSTM, nn.LSTMCell, nn.GRU, nn.GRUCell)):
                gates = 4 if isinstance(mod, (nn.LSTM, nn.LSTMCell)) else 3
                h = mod.hidden_size
                n = gates * h * (mod.input_size + h)
                if isinstance(mod, (nn.LSTM, nn.GRU)):
                    n *= mod.num_layers
                    # a sequence costs per step; deployment runs one step a turn
                self.dot += n
                self.rows.append((name, "dot", n))
        return fn

    def attach(self, net: nn.Module) -> list:
        return [m.register_forward_hook(self._hook(n or type(m).__name__))
                for n, m in net.named_modules()
                if isinstance(m, (nn.Conv2d, nn.Linear, nn.LSTM, nn.LSTMCell,
                                  nn.GRU, nn.GRUCell))]

    @property
    def mac(self) -> int:
        return self.conv + self.dot

    @property
    def points(self) -> float:
        return self.conv * PTS_CONV + self.dot * PTS_DOT


def price(net: nn.Module, *args) -> Counter:
    """Runs one forward pass with batch 1 and returns the counted cost."""
    c = Counter()
    handles = c.attach(net)
    try:
        with torch.no_grad():
            net(*args)
    finally:
        for h in handles:
            h.remove()
    return c


def show(name: str, c: Counter, params: int, detail: bool = False) -> None:
    pts = c.points
    frac = pts / MAX_TURN_POINTS
    verdict = "fits" if pts <= USABLE else "OVER"
    print(f"{name:26s} {c.mac:12,} MAC  ({c.conv:11,} conv + {c.dot:9,} dot)")
    print(f"{'':26s} {pts:12,.0f} pts  {frac:6.1%} of the turn cap   "
          f"{USABLE - pts:+12,.0f} spare   {verdict}")
    print(f"{'':26s} {params:12,} params")
    if detail:
        for n, kind, m in sorted(c.rows, key=lambda r: -r[2]):
            rate = PTS_CONV if kind == "conv" else PTS_DOT
            print(f"{'':28s}{n:38s} {kind:4s} {m:11,} MAC  {m * rate:12,.0f} pts")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="", help="price this checkpoint's architecture")
    p.add_argument("--detail", action="store_true", help="per-layer breakdown")
    a = p.parse_args()

    import bcsim
    from train import net as net_mod

    print(f"cap {MAX_TURN_POINTS:,} - reserve {RESERVE:,} - head {HEAD_RESERVE:,} "
          f"= {USABLE:,} points for the forward pass")
    print(f"pricing MACs at {PTS_CONV} points in convolutions, {PTS_DOT} in dot loops")
    # the calibration check: the model has to reproduce what we measured
    print(f"check: the deployed 64x4 net measured {DEPLOYED_POINTS / 1e6:.0f}M points "
          f"at {DEPLOYED_MAC:,} MAC\n")

    nc, na = bcsim.N_CHANNELS, bcsim.N_ACTIONS
    win, ns = bcsim.WINDOW, bcsim.N_SCALARS
    local = torch.zeros(1, nc, win, win)
    scalar = torch.zeros(1, ns)

    if a.ckpt:
        from train.yardstick import load_net
        net, ck = load_net(ROOT / a.ckpt if not pathlib.Path(a.ckpt).is_absolute()
                           else a.ckpt, torch.device("cpu"))
        args = [local, scalar]
        if getattr(net, "wants_wide", False):
            args.append(torch.zeros(1, bcsim.WIDE_CH, bcsim.WIDE_SIDE, bcsim.WIDE_SIDE))
        c = price(net, *args)
        show(f"{pathlib.Path(a.ckpt).stem} ({ck['args'].get('arch', 'flat')})", c,
             sum(q.numel() for q in net.parameters()), a.detail)
        return

    flat = net_mod.ActorCritic(nc, ns, na, width=64, blocks=4, hidden=512)
    show("flat 64x4 (deployed)", price(flat, local, scalar),
         sum(q.numel() for q in flat.parameters()), a.detail)

    if bcsim.WIDE_CH:
        wide = torch.zeros(1, bcsim.WIDE_CH, bcsim.WIDE_SIDE, bcsim.WIDE_SIDE)
        pyr = net_mod.PyramidActorCritic(nc, bcsim.WIDE_CH, na, wide_side=bcsim.WIDE_SIDE)
        show("pyramid 48x3/24x2", price(pyr, local, scalar, wide),
             sum(q.numel() for q in pyr.parameters()), a.detail)


if __name__ == "__main__":
    main()
