"""Opponent model on the true board (user, 2026-10-01): not a submittable bot, only an
accurate model of another team, so it may read everything the privileged board shows.

Hybrid (option B): an exact crop of the board around the acting dragon's head, rendered
fresh every turn, plus a coarse whole-board view refreshed once per round per team (far
away, a few moves of staleness move things by a cell or two at 4x pooling).

    crop   (18, W, W)   board planes 0-17 centred on the acting head (W = 27), 0 off the board
    coarse (18, 16, 16) the board at the team's first turn of the round, 4x average-pooled,
                        with planes 8-9 (the acting dragon) replaced by the current ones
    extra  previous action one-hot, facing one-hot, length / 32, units / 32

    crop -> 3x3 c1 -> 3x3 c1 -> s2 c2 (W/2) -> 3x3 c2 -> s2 c2 (W/4) -> 1x1 squeeze
    coarse -> 3x3 48 -> s2 96 (8x8) -> s2 96 (4x4) -> 1x1 32
    concat + extra -> dense 512 -> dense 256 -> pi
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_BINARY = 12
BOARD_CH = 18
COARSE = 16                 # 64 / 4
N_EXTRA = 4 + 2             # facing one-hot, length, units


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, c // 4), c)


def _conv(i, o, s=1):
    return [nn.Conv2d(i, o, 3, stride=s, padding=1), _gn(o), nn.SiLU()]


def board_float(b: torch.Tensor) -> torch.Tensor:
    """uint8 board planes -> float: binary planes as is, byte planes / 255."""
    x = b.float()
    x[:, N_BINARY:] /= 255.0
    return x


def crop_at(board: torch.Tensor, w: int) -> torch.Tensor:
    """(B, C, 64, 64) -> (B, C, w, w) centred on each board's acting head (plane 9)."""
    B, C, H, W = board.shape
    hd = board[:, 9].reshape(B, -1).float().argmax(1)
    hy, hx = hd // W, hd % W
    r = w // 2
    pad = F.pad(board, (r, r, r, r))
    ar = torch.arange(w, device=board.device)
    ys = (hy[:, None] + ar[None])[:, :, None]
    xs = (hx[:, None] + ar[None])[:, None, :]
    bi = torch.arange(B, device=board.device)[:, None, None]
    return pad.permute(0, 2, 3, 1)[bi, ys, xs].permute(0, 3, 1, 2)


def coarse_of(board_f: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(board_f, 4)


class BoardPolicy(nn.Module):
    def __init__(self, crop: int = 27, c1: int = 64, c2: int = 128, squeeze: int = 32, hidden: int = 256,
                 n_actions: int = 49):
        super().__init__()
        self.crop, self.n_actions = crop, n_actions
        self.no_action = n_actions
        self.layers, self.hidden = 0, hidden          # a Pool of these holds only the previous action
        q = (crop + 3) // 4                           # 27 -> 14 -> 7
        self.local = nn.Sequential(*_conv(BOARD_CH, c1), *_conv(c1, c1), *_conv(c1, c2, 2), *_conv(c2, c2),
                                   *_conv(c2, c2, 2), nn.Conv2d(c2, squeeze, 1), nn.SiLU(), nn.Flatten())
        self.wide = nn.Sequential(*_conv(BOARD_CH, 48), *_conv(48, 96, 2), *_conv(96, 96, 2),
                                  nn.Conv2d(96, 32, 1), nn.SiLU(), nn.Flatten())
        d = squeeze * q * q + 32 * 4 * 4 + n_actions + 1 + N_EXTRA
        self.dense = nn.Sequential(nn.Linear(d, 512), nn.SiLU(), nn.Linear(512, hidden), nn.SiLU())
        self.pi = nn.Linear(hidden, n_actions)

    def forward(self, crop: torch.Tensor, coarse: torch.Tensor, prev_a: torch.Tensor, extra: torch.Tensor):
        z = torch.cat([self.local(crop), self.wide(coarse),
                       F.one_hot(prev_a.long(), self.n_actions + 1).to(crop.dtype), extra.to(crop.dtype)], 1)
        return self.pi(self.dense(z))
