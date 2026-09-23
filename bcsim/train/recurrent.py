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

import numpy as np

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


ID_BITS = 12                      # uid = (episode << 12) | dragon_id, bc_vec.hpp
ID_MASK = (1 << ID_BITS) - 1
MAX_IDS = 1 << ID_BITS


class StatePool:
    """Hidden state per *dragon*, not per env slot.

    A step of the env is one dragon's turn, so consecutive steps in the same env
    slot belong to different dragons -- there are thirty-odd alive at once. A
    recurrent policy therefore cannot keep its state in an (env, C, S, S) tensor
    the way a normal vectorised environment lets it. State is kept in a pool and
    a table maps (env, dragon id) to a slot, the same trick rollout.py uses to
    chain a dragon's transitions. Slots are recycled when a dragon dies, so the
    pool only has to be as big as the live population.

    The pool also keeps each dragon's last head position and facing, which is
    what lets the state be carried with it: the shift and rotation are derived
    from the observation, so the env needs no new exports.
    """

    def __init__(self, num_envs: int, per_env: int, ch: int, side: int,
                 device=None):
        self.n = num_envs * per_env
        self.ch, self.side = ch, side
        self.h = torch.zeros(self.n, ch, side, side, device=device)
        self.c = torch.zeros(self.n, ch, side, side, device=device)
        self.slot = np.full((num_envs, MAX_IDS), -1, np.int32)
        self.slot_uid = np.zeros((num_envs, MAX_IDS), np.int64)
        self.owner = np.full(self.n, -1, np.int64)        # slot -> uid, for eviction
        self.owner_env = np.full(self.n, -1, np.int32)
        self.free = list(range(self.n))
        self.prev_x = np.zeros(self.n, np.int32)
        self.prev_y = np.zeros(self.n, np.int32)
        self.prev_face = np.zeros(self.n, np.int8)
        self.fresh = np.zeros(self.n, bool)              # no previous turn yet
        self.evictions = 0
        self.exhausted = 0

    def _take(self, env: int, uid: int) -> int:
        if not self.free:
            # Should not happen with per_env at the unit limit, but a full pool
            # must not corrupt another dragon's memory: drop this one's instead.
            self.exhausted += 1
            return -1
        s = self.free.pop()
        ids = uid & ID_MASK
        self.slot[env, ids] = s
        self.slot_uid[env, ids] = uid
        self.owner[s] = uid
        self.owner_env[s] = env
        self.h[s].zero_()
        self.c[s].zero_()
        self.fresh[s] = True
        return s

    def slots_for(self, envs: np.ndarray, uids: np.ndarray) -> np.ndarray:
        """Slot per row, allocating for a dragon seen for the first time."""
        ids = (uids & ID_MASK).astype(np.int64)
        have = self.slot[envs, ids]
        stale = (have >= 0) & (self.slot_uid[envs, ids] != uids)
        for k in np.flatnonzero(stale):          # id reused by a later episode
            self.release_slot(int(have[k]))
        out = np.where(stale, -1, have)
        for k in np.flatnonzero(out < 0):
            out[k] = self._take(int(envs[k]), int(uids[k]))
        return out

    def release_slot(self, s: int) -> None:
        if s < 0 or self.owner[s] < 0:
            return
        env, uid = int(self.owner_env[s]), int(self.owner[s])
        ids = uid & ID_MASK
        if self.slot[env, ids] == s:
            self.slot[env, ids] = -1
        self.owner[s] = -1
        self.owner_env[s] = -1
        self.free.append(s)
        self.evictions += 1

    def release(self, envs: np.ndarray, uids: np.ndarray) -> None:
        """Called with the closures of dragons whose episode ended."""
        ids = (uids & ID_MASK).astype(np.int64)
        have = self.slot[envs, ids]
        ok = (have >= 0) & (self.slot_uid[envs, ids] == uids)
        for s in have[ok]:
            self.release_slot(int(s))

    def carry(self, slots: np.ndarray, x: np.ndarray, y: np.ndarray,
              face: np.ndarray, w: np.ndarray, hgt: np.ndarray):
        """Roll each row's state to its new frame and return it, gathered.

        The step is taken in world coordinates, wrapped on the torus, then
        rotated into the dragon's new frame -- which is the frame `wide` is
        written in, so the state stays registered with the observation.
        """
        sl = torch.as_tensor(slots, dtype=torch.long, device=self.h.device)
        prev_f = self.prev_face[slots]
        turns = ((prev_f.astype(np.int32) - face.astype(np.int32)) % 4)
        dxw = x.astype(np.int32) - self.prev_x[slots]
        dyw = y.astype(np.int32) - self.prev_y[slots]
        # shortest way round the torus
        dxw = np.where(dxw > w // 2, dxw - w, np.where(dxw < -(w // 2), dxw + w, dxw))
        dyw = np.where(dyw > hgt // 2, dyw - hgt, np.where(dyw < -(hgt // 2), dyw + hgt, dyw))
        # world delta -> the new ego frame (facing north is identity)
        f = face.astype(np.int32)
        ex = np.select([f == 0, f == 1, f == 2], [dxw, dyw, -dxw], default=-dyw)
        ey = np.select([f == 0, f == 1, f == 2], [dyw, -dxw, -dyw], default=dxw)
        # a dragon on its first turn has nothing to carry
        first = self.fresh[slots]
        ex = np.where(first, 0, -ex)
        ey = np.where(first, 0, -ey)
        turns = np.where(first, 0, turns)

        h = self.h[sl]
        c = self.c[sl]
        big = max(self.side, 1)
        ex = np.clip(ex, -big, big)
        ey = np.clip(ey, -big, big)
        # one gather per distinct (turn, dx, dy); there are only a handful a step
        combos = {}
        for i, key in enumerate(zip(turns.tolist(), ex.tolist(), ey.tolist())):
            combos.setdefault(key, []).append(i)
        for (k, sx, sy), rows in combos.items():
            if k == 0 and sx == 0 and sy == 0:
                continue
            r = torch.as_tensor(rows, dtype=torch.long, device=h.device)
            for t in (h, c):
                v = t[r]
                if k:
                    v = torch.rot90(v, int(k), dims=(2, 3))
                if sx or sy:
                    v = torch.roll(v, shifts=(int(sy), int(sx)), dims=(2, 3))
                    n = v.shape[-1]
                    if sx > 0:
                        v[:, :, :, :sx] = 0
                    elif sx < 0:
                        v[:, :, :, n + sx:] = 0
                    if sy > 0:
                        v[:, :, :sy, :] = 0
                    elif sy < 0:
                        v[:, :, n + sy:, :] = 0
                t[r] = v
        return h, c

    def store(self, slots: np.ndarray, h, c, x: np.ndarray, y: np.ndarray,
              face: np.ndarray) -> None:
        sl = torch.as_tensor(slots, dtype=torch.long, device=self.h.device)
        self.h[sl] = h.detach()
        self.c[sl] = c.detach()
        self.prev_x[slots] = x
        self.prev_y[slots] = y
        self.prev_face[slots] = face
        self.fresh[slots] = False


def head_and_face(scalar: np.ndarray) -> tuple:
    """Absolute head cell and facing index, read back out of the base scalars.

    bc_obs.hpp writes head_x / head_y divided by the map size and the facing as a
    one-hot, so the env needs no extra exports for the state to be carried.
    """
    w = np.rint(scalar[:, 10] * 64.0).astype(np.int32)
    h = np.rint(scalar[:, 11] * 64.0).astype(np.int32)
    x = np.rint(scalar[:, 8] * np.maximum(w, 1)).astype(np.int32)
    y = np.rint(scalar[:, 9] * np.maximum(h, 1)).astype(np.int32)
    face = scalar[:, 4:8].argmax(1).astype(np.int8)
    return x, y, face, w, h


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
