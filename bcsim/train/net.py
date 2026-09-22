"""Policy and value network.

Deliberately wide: the teacher is trained without regard to the judge's CPU
budget and distilled down afterwards. Everything it sees is what a deployed
bot sees, so the actor can be shipped unchanged once it is small enough.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n1 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.n2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        y = F.silu(self.n1(self.c1(x)))
        y = self.n2(self.c2(y))
        return F.silu(x + y)


class ActorCritic(nn.Module):
    """Conv trunk over the 7x7 window, scalars folded in, two heads."""

    def __init__(self, n_channels: int, n_scalars: int, n_actions: int,
                 width: int = 128, blocks: int = 6, hidden: int = 512):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.SiLU())
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        # 7x7 is small enough to keep whole rather than pool away
        self.flat = nn.Sequential(nn.Conv2d(width, 32, 1, bias=False),
                                  nn.GroupNorm(8, 32), nn.SiLU(), nn.Flatten())
        self.scalar = nn.Sequential(nn.Linear(n_scalars, 128), nn.SiLU(),
                                    nn.Linear(128, 128), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(32 * 7 * 7 + 128, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, local, scalar):
        x = self.flat(self.blocks(self.stem(local)))
        h = self.fuse(torch.cat([x, self.scalar(scalar)], dim=1))
        return self.pi(h), self.v(h).squeeze(-1)


NEG = -1e9


def masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Illegal actions are removed. A dragon with no legal action at all dies
    whatever it picks, so the mask is ignored rather than producing NaNs."""
    dead = mask.sum(dim=1, keepdim=True) == 0
    allow = mask.bool() | dead
    return logits.masked_fill(~allow, NEG)


def policy_out(logits, mask, action=None):
    """Returns (action, logprob, entropy) under the masked categorical."""
    logits = masked_logits(logits.float(), mask)
    logp_all = F.log_softmax(logits, dim=1)
    if action is None:
        action = torch.multinomial(logp_all.exp(), 1).squeeze(1)
    logp = logp_all.gather(1, action.unsqueeze(1)).squeeze(1)
    ent = -(logp_all.exp() * logp_all).sum(dim=1)
    return action, logp, ent


class Critic(nn.Module):
    """Value network with no weights shared with the policy.

    Used by finetune.py, where the policy starts from a behaviour clone:
    fitting a value through a shared trunk would move the policy while the
    critic is still learning. The critic never ships, so it may see what the
    deployed bot cannot: `context` is a one-hot of the opponent the episode is
    played against. Without it, one value has to average over opponents of
    very different strength, and its errors land in every advantage. So it
    also has one value output per context: the trunk is shared, but no
    opponent's returns are fitted by another opponent's output.
    """

    def __init__(self, n_channels: int, n_scalars: int, n_context: int,
                 width: int = 64, blocks: int = 4, hidden: int = 512, n_extra: int = 0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(n_channels, width, 3, padding=1, bias=False),
            nn.GroupNorm(8, width), nn.SiLU())
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])
        self.flat = nn.Sequential(nn.Conv2d(width, 32, 1, bias=False),
                                  nn.GroupNorm(8, 32), nn.SiLU(), nn.Flatten())
        self.scalar = nn.Sequential(nn.Linear(n_scalars + n_context + n_extra, 128), nn.SiLU(),
                                    nn.Linear(128, 128), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(32 * 7 * 7 + 128, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.v = nn.Linear(hidden, n_context)
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, local, scalar, context, extra=None):
        """context: one-hot (batch, n_context); extra: privileged features
        (batch, n_extra) when built with them. Returns that context's value."""
        x = self.flat(self.blocks(self.stem(local)))
        parts = [scalar, context] if extra is None else [scalar, context, extra]
        s = self.scalar(torch.cat(parts, dim=1))
        v = self.v(self.fuse(torch.cat([x, s], dim=1)))
        return (v * context).sum(dim=1)
