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

    # whether forward() needs the env's `wide` tensor; lets one call site serve
    # both architectures, and says whether a run must bind it
    wants_wide = False

    def __init__(self, n_channels: int, n_scalars: int, n_actions: int,
                 width: int = 128, blocks: int = 6, hidden: int = 512):
        super().__init__()
        self.n_scalars = n_scalars
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

    def forward(self, local, scalar, wide=None):
        """`wide` is accepted and ignored: callers may pass it unconditionally.

        The scalar row is cut back to the width this net was built for. The env
        appends new features on the end, so a checkpoint trained on 708 keeps
        seeing exactly its 708 and plays identically -- which is what lets the
        frozen league survive the row growing.
        """
        x = self.flat(self.blocks(self.stem(local)))
        h = self.fuse(torch.cat([x, self.scalar(scalar[..., :self.n_scalars])], dim=1))
        return self.pi(h), self.v(h).squeeze(-1)


# The 708-scalar vector is [14 base | 676 mem | 18 memfar]. `mem` is really a
# (4, 13, 13) feature map that Linear(708, 128) flattens, and that linear is
# 0.55% of ActorCritic's forward pass: the geometry is discarded and 92% of the
# compute goes on the inner 49 cells. PyramidActorCritic takes those planes from
# the env's `wide` tensor instead, so the only scalars left to feed are the base
# ones and memfar -- a slice of the same vector, which is why the env, the
# league and every old checkpoint are untouched by this.
N_BASE_SCALARS = 14
N_MEM_SCALARS = 676
N_FAR_SCALARS = 18
N_ECHO_SCALARS = 5                                                   # sonar echoes
N_FLAT_SCALARS = N_BASE_SCALARS + N_MEM_SCALARS + N_FAR_SCALARS      # 708
N_PYRAMID_SCALARS = N_BASE_SCALARS + N_FAR_SCALARS                   # 32
N_PYRAMID_ECHO_SCALARS = N_PYRAMID_SCALARS + N_ECHO_SCALARS          # 37
_FAR_AT = N_BASE_SCALARS + N_MEM_SCALARS                             # 690
_ECHO_AT = N_FLAT_SCALARS                                            # 708


def pyramid_scalars(scalar: torch.Tensor, want: int = N_PYRAMID_SCALARS) -> torch.Tensor:
    """The scalars the pyramid keeps, sliced out of the env's row.

    The env's row grew from 708 to 713 when the sonar echoes were appended, and
    it may grow again. `want` is the width this particular network was built
    for, taken from its own checkpoint, so an older pyramid keeps reading the 32
    it was trained on and a newer one also gets the 5 echoes. Everything is a
    slice of one row, which is why the env can serve every architecture at once.
    """
    if scalar.shape[-1] == want:
        return scalar
    parts = [scalar[..., :N_BASE_SCALARS],
             scalar[..., _FAR_AT:_FAR_AT + N_FAR_SCALARS]]
    if want >= N_PYRAMID_ECHO_SCALARS and scalar.shape[-1] >= _ECHO_AT + N_ECHO_SCALARS:
        parts.append(scalar[..., _ECHO_AT:_ECHO_AT + N_ECHO_SCALARS])
    return torch.cat(parts, dim=-1)


class PyramidActorCritic(nn.Module):
    """Two conv branches in the dragon's own frame, at two scales.

    `local` is the live 7x7 window, ground truth. `wide` is what the dragon
    remembers, as planes rather than a flat bag: six channels at stride 1 out to
    radius 7, then the same six mean-pooled to reach radius 29 (see
    cpp/bc_memory.hpp `wide`). Routing through remembered corridors is a spatial
    problem, and this is the branch that can express it.

    Cheaper than ActorCritic at 64x4, not dearer: 12.7M MAC against 16.5M, and
    ~0.96M parameters against 1.58M, because the flattened 13x13 and the 1696
    -> 512 fuse both go. The judge's budget is the reason -- 69M points a turn
    measured at 16.5M MAC, against a 100M cap with the reserves taken out.
    """

    wants_wide = True

    def __init__(self, n_channels: int, n_wide_ch: int, n_actions: int,
                 near_width: int = 48, near_blocks: int = 3,
                 wide_width: int = 24, wide_blocks: int = 2,
                 near_head: int = 24, wide_head: int = 16,
                 wide_side: int = 15, wide_pool: int = 3, hidden: int = 384,
                 n_scalars: int = N_PYRAMID_SCALARS):
        super().__init__()
        self.n_scalars = n_scalars
        self.near_stem = nn.Sequential(
            nn.Conv2d(n_channels, near_width, 3, padding=1, bias=False),
            nn.GroupNorm(8, near_width), nn.SiLU())
        self.near_blocks = nn.Sequential(*[ResBlock(near_width) for _ in range(near_blocks)])
        self.near_flat = nn.Sequential(nn.Conv2d(near_width, near_head, 1, bias=False),
                                       nn.GroupNorm(8, near_head), nn.SiLU(), nn.Flatten())

        # 225 cells x 9 makes every width-24 conv 1.17M MAC, so this branch gets
        # two blocks and is pooled before the head: full resolution at radius 15
        # (961 cells) is not affordable at any useful width.
        self.wide_stem = nn.Sequential(
            nn.Conv2d(n_wide_ch, wide_width, 3, padding=1, bias=False),
            nn.GroupNorm(8, wide_width), nn.SiLU())
        self.wide_blocks = nn.Sequential(*[ResBlock(wide_width) for _ in range(wide_blocks)])
        self.wide_flat = nn.Sequential(nn.AvgPool2d(wide_pool),
                                       nn.Conv2d(wide_width, wide_head, 1, bias=False),
                                       nn.GroupNorm(8, wide_head), nn.SiLU(), nn.Flatten())

        self.scalar = nn.Sequential(nn.Linear(n_scalars, 64), nn.SiLU())

        pooled = wide_side // wide_pool
        fuse_in = near_head * 7 * 7 + wide_head * pooled * pooled + 64
        self.fuse = nn.Sequential(nn.Linear(fuse_in, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.zeros_(self.v.bias)

    def forward(self, local, scalar, wide=None):
        if wide is None:
            raise ValueError("PyramidActorCritic needs the env's `wide` tensor "
                             "(BattlecodeVecEnv(..., wide=True))")
        n = self.near_flat(self.near_blocks(self.near_stem(local)))
        w = self.wide_flat(self.wide_blocks(self.wide_stem(wide)))
        s = self.scalar(pyramid_scalars(scalar, self.n_scalars))
        h = self.fuse(torch.cat([n, w, s], dim=1))
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
        self.n_scalars = n_scalars
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
