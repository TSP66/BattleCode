"""A recurrent memory that keeps its shape: a ConvLSTM in the dragon's frame.

The question this exists to answer is whether recurrence parametrises a
dragon's memory better than the hand-built planes of bc_memory.hpp do. Those
planes are a fixed summary -- age, visits, kelp, pearls -- chosen by us. An
LSTM would learn what is worth remembering instead.

Two things make it affordable, and both come from keeping the state
**structured** rather than flattening it into a vector.

*Cost.* The judge charges about 3.3 points for a MAC inside a convolution and
about 15 for one inside a dot-product loop (see train/budget.py, which
reproduces the real meter to 0.14%). A dense LSTM is all dot products, so its
gates cost the expensive rate; a ConvLSTM's gates are convolutions and cost the
cheap one. The same hidden capacity is roughly 4.5x cheaper as a grid than as a
vector, which is the difference between "fits" and "does not".

*Registration.* A vector state has to re-learn where everything is every time
the dragon moves. A grid state can simply be **moved with it**: each turn the
hidden grid is rolled by the step the dragon took and rotated by any turn it
made, so a cell of the state keeps pointing at the same piece of world. That is
the whole trick, and it is why this is worth trying at all -- without it the
recurrence spends its capacity tracking its own drift.

    python -m train.recurrent --price          # cost against the judge's budget
    python -m train.recurrent --price --hidden 24 --side 15

Nothing here is trained yet. It is defined and priced first because an
architecture that cannot be deployed is not worth distilling into.
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.net import N_PYRAMID_SCALARS, ResBlock, pyramid_scalars


class ConvLSTMCell(nn.Module):
    """One step of a ConvLSTM: the four gates as a single convolution.

    Input and state share a grid, so the gates see a neighbourhood rather than
    a flattened bag, and what the state holds at a cell stays tied to that cell.
    """

    def __init__(self, in_ch: int, hid_ch: int, kernel: int = 3):
        super().__init__()
        self.hid_ch = hid_ch
        self.gates = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel,
                               padding=kernel // 2, bias=True)
        # forget gate biased open, so the state persists before it learns to
        # keep anything; the usual LSTM initialisation, and it matters more here
        # because the useful horizon is hundreds of turns
        with torch.no_grad():
            self.gates.bias.zero_()
            self.gates.bias[hid_ch:2 * hid_ch].fill_(1.0)

    def forward(self, x, state):
        h, c = state
        i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


def roll_state(state, dx, dy, turns):
    """Carries the hidden grid with the dragon: roll by its step, rotate by its turn.

    `dx`, `dy` are the step taken in the *state's* frame and `turns` the
    quarter-turns the dragon made, both per batch row. Anything rolled in from
    the far edge is state about somewhere the dragon has left, so it is zeroed
    rather than wrapped -- the grid is a window on the world, not a torus.
    """
    h, c = state
    out = []
    for t in (h, c):
        # rotate first, then translate, so the step is expressed in the new frame
        k = turns.reshape(-1)
        r = torch.stack([torch.rot90(t[i], int(k[i]) % 4, dims=(1, 2))
                         for i in range(t.shape[0])])
        r = torch.stack([torch.roll(r[i], shifts=(int(dy[i]), int(dx[i])), dims=(1, 2))
                         for i in range(r.shape[0])])
        # blank what wrapped around
        n = r.shape[-1]
        mask = torch.ones_like(r)
        for i in range(r.shape[0]):
            sx, sy = int(dx[i]), int(dy[i])
            if sx > 0:
                mask[i, :, :, :sx] = 0
            elif sx < 0:
                mask[i, :, :, n + sx:] = 0
            if sy > 0:
                mask[i, :, :sy, :] = 0
            elif sy < 0:
                mask[i, :, n + sy:, :] = 0
        out.append(r * mask)
    return out[0], out[1]


class RecurrentActorCritic(nn.Module):
    """The pyramid's near branch, with a ConvLSTM where its wide branch was.

    `wide` still comes in, but only as the *input* to the recurrence for the
    turn -- what the dragon can see and has just seen. What it remembers is the
    cell state, carried between turns by the caller.
    """

    wants_wide = True
    recurrent = True

    def __init__(self, n_channels: int, n_wide_ch: int, n_actions: int,
                 near_width: int = 48, near_blocks: int = 3,
                 hid_ch: int = 24, side: int = 15, in_width: int = 16,
                 near_head: int = 24, mem_head: int = 16, mem_pool: int = 3,
                 hidden: int = 384, n_scalars: int = N_PYRAMID_SCALARS):
        super().__init__()
        self.side = side
        self.hid_ch = hid_ch
        self.n_scalars = n_scalars

        self.near_stem = nn.Sequential(
            nn.Conv2d(n_channels, near_width, 3, padding=1, bias=False),
            nn.GroupNorm(8, near_width), nn.SiLU())
        self.near_blocks = nn.Sequential(*[ResBlock(near_width) for _ in range(near_blocks)])
        self.near_flat = nn.Sequential(nn.Conv2d(near_width, near_head, 1, bias=False),
                                       nn.GroupNorm(8, near_head), nn.SiLU(), nn.Flatten())

        # what the turn contributes to memory, narrowed before the gates
        self.obs_in = nn.Sequential(
            nn.Conv2d(n_wide_ch, in_width, 3, padding=1, bias=False),
            nn.GroupNorm(4, in_width), nn.SiLU())
        self.cell = ConvLSTMCell(in_width, hid_ch)
        self.mem_flat = nn.Sequential(nn.AvgPool2d(mem_pool),
                                      nn.Conv2d(hid_ch, mem_head, 1, bias=False),
                                      nn.GroupNorm(8, mem_head), nn.SiLU(), nn.Flatten())

        self.scalar = nn.Sequential(nn.Linear(n_scalars, 64), nn.SiLU())
        pooled = side // mem_pool
        fuse_in = near_head * 7 * 7 + mem_head * pooled * pooled + 64
        self.fuse = nn.Sequential(nn.Linear(fuse_in, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.zeros_(self.v.bias)

    def initial_state(self, n: int, device=None):
        z = torch.zeros(n, self.hid_ch, self.side, self.side, device=device)
        return z, z.clone()

    def forward(self, local, scalar, wide=None, state=None):
        if wide is None:
            raise ValueError("RecurrentActorCritic needs the env's `wide` tensor")
        if state is None:
            state = self.initial_state(local.shape[0], local.device)
        h, c = self.cell(self.obs_in(wide), state)
        n = self.near_flat(self.near_blocks(self.near_stem(local)))
        m = self.mem_flat(h)
        s = self.scalar(pyramid_scalars(scalar, self.n_scalars))
        z = self.fuse(torch.cat([n, m, s], dim=1))
        return self.pi(z), self.v(z).squeeze(-1), (h, c)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--price", action="store_true", help="cost it against the judge")
    p.add_argument("--hidden-ch", type=int, default=24, help="ConvLSTM state channels")
    p.add_argument("--side", type=int, default=0, help="state grid side (default: wide)")
    p.add_argument("--detail", action="store_true")
    a = p.parse_args()

    import bcsim
    from train.budget import PTS_CONV, PTS_DOT, USABLE, price, show
    from train.net import PyramidActorCritic

    side = a.side or bcsim.WIDE_SIDE
    local = torch.zeros(1, bcsim.N_CHANNELS, bcsim.WINDOW, bcsim.WINDOW)
    scalar = torch.zeros(1, bcsim.N_SCALARS)
    wide = torch.zeros(1, bcsim.WIDE_CH, side, side)

    pyr = PyramidActorCritic(bcsim.N_CHANNELS, bcsim.WIDE_CH, bcsim.N_ACTIONS,
                             wide_side=side)
    show("pyramid (planes)", price(pyr, local, scalar, wide),
         sum(q.numel() for q in pyr.parameters()), a.detail)

    net = RecurrentActorCritic(bcsim.N_CHANNELS, bcsim.WIDE_CH, bcsim.N_ACTIONS,
                               hid_ch=a.hidden_ch, side=side)
    c = price(net, local, scalar, wide)
    show(f"ConvLSTM {a.hidden_ch}ch @{side}", c,
         sum(q.numel() for q in net.parameters()), a.detail)

    # what the same state would cost flattened, which is the comparison that
    # decides whether "structured" is a preference or a requirement
    dense_state = a.hidden_ch * side * side
    dense_in = 16 * side * side
    dense_mac = 4 * dense_state * (dense_in + dense_state)
    print(f"the same state as a dense LSTM ({dense_state} units, {dense_in} in): "
          f"{dense_mac:,} MAC at the dot-loop rate = {dense_mac * PTS_DOT:,.0f} points, "
          f"{'fits' if dense_mac * PTS_DOT <= USABLE else 'OVER the whole budget'}")
    gate_mac = next(m for n, k, m in c.rows if "cell.gates" in n)
    print(f"as a ConvLSTM the gates are {gate_mac:,} MAC at the convolution rate "
          f"= {gate_mac * PTS_CONV:,.0f} points, "
          f"{dense_mac * PTS_DOT / max(gate_mac * PTS_CONV, 1):.0f}x cheaper")


if __name__ == "__main__":
    main()
