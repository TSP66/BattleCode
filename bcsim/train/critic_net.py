"""The privileged critic: a conv trunk over the whole board, not a 7x7 window.

The critic never ships, so it may see what a deployed dragon cannot. Until now
it did not use that licence: it read the same 7x7 window the policy reads plus
eight global numbers, and had to predict the team's result from a keyhole. The
result of a game turns on things that are nowhere in that window -- where the
two swarms are relative to each other, which corridors are held, how much of
the map is still unclaimed.

So the trunk here is the board (bc_obs.hpp writes BOARD_CH planes of 64x64 in
map coordinates, already turned to the acting team's point of view) downsampled
three times. The window is kept as a second branch, because the acting dragon's
immediate situation is what its own shaped return depends on and 8x8 cells of
board have lost it.

Conditioning. A value has to average over whatever the opponent does, and
"whatever the opponent does" is a different function for each opponent, so the
errors of that average land in every advantage. Three things identify the
matchup:

  * a learnable embedding per team, looked up for *both* sides. Slot 0 is
    unknown, which is what self-play and an unrecognised team both get.
  * a sinusoidal encoding of the training iteration. During PPO the opponent
    labelled "self" is a moving target; the iteration index says which version
    of it, so the critic can track the drift instead of averaging over it.
  * a sinusoidal encoding of the round. `priv` carries round/max_rounds, which
    a single linear layer can only read as one ramp; the game's phases are not
    linear in it.

Two heads on the one trunk, as before: `team` gives win/draw/loss logits for the
acting dragon's team, `self_v` its own shaped return with one output per
opponent slot so that no opponent's returns are fitted by another's.

Cost is not a concern the way it is for the policy -- nothing here is metered by
the judge -- but it is still ~55M MAC a row, three times the policy, so the
board is brought down to 16x16 before the width goes up.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from train.net import ResBlock

# Reserved embedding slots. 0 is "unknown" and is what self-play, an
# unrecognised team id and a replay with no metadata all map to, so a critic
# trained on replays can be used on self-play without reshaping anything.
TEAM_UNKNOWN = 0
N_TEAM_SLOTS = 64          # contest has far fewer than this; room to grow


def sinusoidal(x: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Transformer-style encoding of a scalar, (B,) -> (B, dim).

    `x` is passed as a float so a fractional or unknown index is expressible;
    -1 is the caller's convention for unknown and encodes as all zeros.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, device=x.device, dtype=torch.float32) / half)
    ang = x.float().unsqueeze(-1) * freqs.unsqueeze(0)
    out = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    return torch.where((x < 0).unsqueeze(-1), torch.zeros_like(out), out)


def down(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1, bias=False),
                         nn.GroupNorm(8, cout), nn.SiLU())


class BoardCritic(nn.Module):
    def __init__(self, n_channels: int, n_scalars: int, n_context: int,
                 n_board_ch: int, board_side: int = 64, n_priv: int = 8,
                 board_width: tuple[int, int, int] = (24, 48, 64),
                 board_blocks: tuple[int, int] = (2, 2),
                 near_width: int = 64, near_blocks: int = 4,
                 team_dim: int = 16, sin_dim: int = 32, cond_hidden: int = 192,
                 hidden: int = 768, dropout: float = 0.1,
                 n_team_slots: int = N_TEAM_SLOTS):
        super().__init__()
        self.n_context = n_context
        self.n_priv = n_priv
        self.sin_dim = sin_dim
        w0, w1, w2 = board_width
        b1, b2 = board_blocks

        # 64 -> 32 -> 16 -> 8. Most of the board is outside the map on most
        # maps (plane 5 says which), so resolution is spent where the width is
        # cheap and the width is spent once the grid is small.
        self.board = nn.Sequential(
            down(n_board_ch, w0),                       # 32x32
            down(w0, w1),                               # 16x16
            *[ResBlock(w1) for _ in range(b1)],
            down(w1, w2),                               # 8x8
            *[ResBlock(w2) for _ in range(b2)],
            nn.Conv2d(w2, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Flatten())
        side8 = board_side // 8
        board_out = 32 * side8 * side8

        self.near = nn.Sequential(
            nn.Conv2d(n_channels, near_width, 3, padding=1, bias=False),
            nn.GroupNorm(8, near_width), nn.SiLU(),
            *[ResBlock(near_width) for _ in range(near_blocks)],
            nn.Conv2d(near_width, 32, 1, bias=False), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Flatten())

        self.team_emb = nn.Embedding(n_team_slots, team_dim)
        nn.init.normal_(self.team_emb.weight, std=0.02)
        # unknown starts at exactly zero: a matchup we cannot identify should
        # contribute nothing rather than a random direction
        with torch.no_grad():
            self.team_emb.weight[TEAM_UNKNOWN].zero_()

        cond_in = n_priv + n_context + 2 * team_dim + 2 * sin_dim
        self.cond = nn.Sequential(nn.Linear(cond_in, cond_hidden), nn.SiLU(),
                                  nn.Linear(cond_hidden, cond_hidden), nn.SiLU())
        self.scalar = nn.Sequential(nn.Linear(n_scalars, 128), nn.SiLU())

        self.fuse = nn.Sequential(
            nn.Linear(board_out + 32 * 7 * 7 + cond_hidden + 128, hidden), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Dropout(dropout))
        self.team = nn.Linear(hidden, 3)
        self.self_v = nn.Linear(hidden, n_context)
        nn.init.orthogonal_(self.team.weight, 0.1)
        nn.init.zeros_(self.team.bias)
        nn.init.orthogonal_(self.self_v.weight, 1.0)
        nn.init.zeros_(self.self_v.bias)

    def forward(self, local, scalar, context, priv, board,
                team_self=None, team_foe=None, iteration=None, rnd=None):
        """`board` is (B, BOARD_CH, 64, 64) as float.

        `team_self` / `team_foe` are embedding slots, `iteration` the training
        iteration and `rnd` the game round, each (B,). Any of them may be None,
        which is the unknown encoding -- slot 0 or an all-zero encoding -- so a
        caller that does not know the matchup still gets a value.
        """
        b = self.board(board)
        n = self.near(local)
        z = torch.zeros(local.shape[0], device=local.device)
        ts = self.team_emb(team_self if team_self is not None
                           else torch.zeros_like(z, dtype=torch.long))
        tf = self.team_emb(team_foe if team_foe is not None
                           else torch.zeros_like(z, dtype=torch.long))
        si = sinusoidal(iteration if iteration is not None else z - 1, self.sin_dim)
        sr = sinusoidal(rnd if rnd is not None else z - 1, self.sin_dim)
        c = self.cond(torch.cat([priv, context, ts, tf, si, sr], dim=1))
        h = self.fuse(torch.cat([b, n, c, self.scalar(scalar)], dim=1))
        v = (self.self_v(h) * context).sum(1)
        return self.team(h), v


def team_value(logits: torch.Tensor) -> torch.Tensor:
    """p_win - p_loss: the expected terminal reward, win +1 lose -1."""
    p = logits.float().softmax(-1)
    return p[..., 0] - p[..., 2]


class TeamSlots:
    """Contest team id -> embedding slot, stable across runs once saved.

    Slots are handed out in the order teams are first seen and stored in the
    checkpoint, so a critic keeps meaning what it meant. An id past the table's
    size falls back to unknown rather than colliding with another team.
    """

    def __init__(self, mapping: dict | None = None, n_slots: int = N_TEAM_SLOTS):
        self.n_slots = n_slots
        self.map = {int(k): int(v) for k, v in (mapping or {}).items()}

    def slot(self, team_id: int | None, *, add: bool = False) -> int:
        if team_id is None or int(team_id) < 0:
            return TEAM_UNKNOWN
        tid = int(team_id)
        if tid in self.map:
            return self.map[tid]
        if not add:
            return TEAM_UNKNOWN
        nxt = TEAM_UNKNOWN + 1 + len(self.map)
        if nxt >= self.n_slots:
            return TEAM_UNKNOWN
        self.map[tid] = nxt
        return nxt

    def as_dict(self) -> dict:
        return dict(self.map)

    def __len__(self) -> int:
        return len(self.map)
