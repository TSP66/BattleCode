"""The network small enough to actually run on the judge.

Sized against a measured cost, not an assumed one: the judge's NumPy has no
BLAS, so a multiply-accumulate costs ~23 CPU points whatever shape it is in
(train/budget.py records the measurements). That leaves roughly 2.9M MACs per
turn once parsing and the observation are paid for, so this is about 20x
smaller than the PPO teacher.

Two deliberate differences from the teacher, both about deployment cost:
  * ReLU instead of SiLU -- 38 points per element against 176, and there are
    thousands of elements per turn;
  * no GroupNorm -- it cannot be folded into the weights the way BatchNorm
    can, so it would be ~1.5M points a layer for nothing at this size.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Student(nn.Module):
    def __init__(self, n_channels: int = 23, n_scalars: int = 14, n_actions: int = 48,
                 width: int = 32, blocks: int = 2, hidden: int = 256):
        super().__init__()
        self.width, self.blocks_n, self.hidden = width, blocks, hidden
        self.stem = nn.Conv2d(n_channels, width, 3, padding=1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(width, width, 3, padding=1) for _ in range(blocks)])
        self.fc1 = nn.Linear(width * 49 + n_scalars, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.pi = nn.Linear(hidden, n_actions)

    def forward(self, local, scalar):
        x = F.relu(self.stem(local))
        for c in self.convs:
            x = F.relu(c(x))
        h = torch.cat([x.flatten(1), scalar], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return self.pi(h)

    def macs(self) -> int:
        w, b, h = self.width, self.blocks_n, self.hidden
        total = 49 * (23 * 9) * w
        total += b * 49 * (w * 9) * w
        total += (w * 49 + 14) * h + h * h + h * 48
        return total
