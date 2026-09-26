"""DRAFT: 14x14 remembered grid -> CNN -> flat LSTM stack -> policy.

Nothing imports this yet. It exists to pin the structure down before the
simulator emits the 38-channel grid and before the C++ port is metered.

Shape of one step (batch B):

    grid     (B, 38, 14, 14)   ego frame, rendered each turn from the world-anchored memory
    prev_a   (B,)              the action this dragon took last turn (NO_ACTION on its first)
    state    [(h, c)] * L      each (B, 128)

    -> logits (B, 48), value (B,), new state
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

GRID = 14            # head at row 7, col 7: 7 cells ahead/left, 6 behind/right
N_ACTIONS = 48       # 39 relative paths (1-3 steps of straight/left/right) + 9 splits
NO_ACTION = N_ACTIONS  # the "previous action" of a dragon on its first turn

# ---------------------------------------------------------------- input channels
# Every channel in [0, 1]. Outside the 7x7 live window, "live" channels are 0.
CHANNELS = [
    # static terrain: one channel for seen-now and remembered, exact once seen
    "kelp_n", "kelp_e", "kelp_s", "kelp_w",
    "portal_n", "portal_e", "portal_s", "portal_w",
    "never_spawns",
    # live, 7x7 only
    "pearl", "pearl_timer",                                  # cd / 99
    "ally_head", "ally_body", "enemy_head", "enemy_body",    # one-hot occupancy
    "seg_dir_n", "seg_dir_e", "seg_dir_s", "seg_dir_w",      # relative to our facing
    # self: exact across the whole grid, tracked from our own moves
    "self_body", "self_index", "self_tail",
    # memory with decay (1 / live value inside the window)
    "seen",                  # exp(-age/32); 0 = never seen. The "this is memory" flag
    "pearl_expected",        # (saw a pearl, or its timer has run out) * decay
    "pearl_timer_proj",      # max(cd - age, 0) / 99
    "enemy_mem",             # exp(-age/8) where an enemy was last seen
    "ally_mem",              # exp(-age/8)
    "visited",               # exp(-since/16)
    # global, broadcast as constant planes
    "round",                 # round / max_rounds
    "length",                # min(len, 64) / 64
    "units",                 # alive / unit_limit
    "map_w", "map_h",        # / 64
    "echo_kelp", "echo_ally", "echo_ally_head", "echo_enemy", "echo_enemy_head",  # / 4
]
N_CH = len(CHANNELS)
assert N_CH == 38


class ResBlock(nn.Module):
    """Two 3x3 convs and a skip, GroupNorm + SiLU.

    Not ReLU without a norm, which is what this first shipped with: measured
    2026-09-25, that net never learned (0.563 top-1 against the teacher, i.e.
    "always straight on", against 0.747 for this one on identical data). Its
    ReLUs died during the first Adam steps while the gradient reaching the
    trunk through the LSTM was ~1e-8, and ReLU + GroupNorm did no better. Both
    SiLU (int16 lookup table) and GroupNorm (per-layer stats) port to the int16
    bot for ~1-2M points a turn.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n1 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.n2 = nn.GroupNorm(8, ch)

    def forward(self, x):
        return F.silu(x + self.n2(self.c2(F.silu(self.n1(self.c1(x))))))


class LSTMPolicy(nn.Module):
    # Metered in the judge's cost table (lstmbot/meter.sh), with GroupNorm,
    # SiLU (int16 lookup table) and the LayerNorm: 83.4M points a turn all-in,
    # 86.7M on turn 0, 1.13M params, zip 2.80 of 4 MiB (2026-09-25). The ReLU,
    # norm-free version metered 79.4M but did not learn (see ResBlock). 48/128
    # metered 92.6M and 64/128 108.6M without norms, so the CNN is at its ceiling.
    def __init__(self, c1: int = 48, b1: int = 1, c2: int = 112, b2: int = 2,
                 squeeze: int = 16, embed: int = 256, hidden: int = 128, layers: int = 2):
        super().__init__()
        self.hidden, self.layers = hidden, layers

        # --- spatial: 14x14 at full resolution, then 7x7
        self.stem = nn.Conv2d(N_CH, c1, 3, padding=1)                  # 38x14x14 -> 48x14x14
        self.stem_n = nn.GroupNorm(8, c1)
        self.res15 = nn.Sequential(*[ResBlock(c1) for _ in range(b1)])  # 48x14x14
        self.down = nn.Conv2d(c1, c2, 3, stride=2, padding=1)          # -> 112x7x7
        self.down_n = nn.GroupNorm(8, c2)
        self.res8 = nn.Sequential(*[ResBlock(c2) for _ in range(b2)])  # 112x7x7
        self.squeeze = nn.Conv2d(c2, squeeze, 1)                       # -> 16x7x7 = 784

        # --- flat
        self.embed = nn.Linear(squeeze * 7 * 7, embed)                 # 784 -> 256
        self.prev_action = nn.Embedding(N_ACTIONS + 1, 48)             # one-hot 49 -> 48, as a lookup
        self.in_norm = nn.LayerNorm(embed + 48)                        # what the LSTM reads
        self.lstm = nn.ModuleList(
            nn.LSTMCell(embed + 48 if i == 0 else hidden, hidden) for i in range(layers))
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)       # training only; never exported

        for cell in self.lstm:              # forget gate open, so state persists early on
            with torch.no_grad():
                cell.bias_ih[hidden:2 * hidden].fill_(1.0)
                cell.bias_hh[hidden:2 * hidden].zero_()
        # gain 1, not PPO's usual 0.01: through the LSTM, 0.01 left the trunk a
        # ~1e-8 gradient for the first few hundred distillation steps
        nn.init.orthogonal_(self.pi.weight, 1.0)
        nn.init.zeros_(self.pi.bias)

    def initial_state(self, batch: int, device=None):
        z = lambda: torch.zeros(batch, self.hidden, device=device)   # noqa: E731
        return [(z(), z()) for _ in range(self.layers)]

    def encode(self, grid, prev_a):
        """Everything before the recurrence: no state, so a whole batch of
        turns from many dragons and times can go through at once."""
        x = F.silu(self.stem_n(self.stem(grid)))
        x = self.res15(x)
        x = F.silu(self.down_n(self.down(x)))
        x = self.res8(x)
        x = F.silu(self.squeeze(x)).flatten(1)
        x = F.silu(self.embed(x))
        return self.in_norm(torch.cat([x, self.prev_action(prev_a)], dim=1))

    def step(self, x, state):
        """One turn of the LSTM stack from encode()'s output."""
        new_state = []
        for cell, (h, c) in zip(self.lstm, state):
            h, c = cell(x, (h, c))
            new_state.append((h, c))
            x = h
        return self.pi(x), self.v(x).squeeze(-1), new_state

    def forward(self, grid, prev_a, state):
        return self.step(self.encode(grid, prev_a), state)


if __name__ == "__main__":
    net = LSTMPolicy()
    B = 2
    grid = torch.rand(B, N_CH, GRID, GRID)
    prev = torch.full((B,), NO_ACTION)
    state = net.initial_state(B)

    def tap(name):
        return lambda m, i, o: print(f"  {name:12s} -> {tuple(o.shape) if torch.is_tensor(o) else tuple(o[0].shape)}")
    for name in ["stem", "res15", "down", "res8", "squeeze", "embed", "prev_action", "lstm.0", "lstm.1", "pi"]:
        net.get_submodule(name).register_forward_hook(tap(name))
    print(f"input grid   {tuple(grid.shape)}")
    logits, value, state = net(grid, prev, state)

    print("\nparameters (value head excluded, it never ships):")
    total = 0
    for name, mod in net.named_children():
        if name == "v":
            continue
        n = sum(p.numel() for p in mod.parameters())
        total += n
        print(f"  {name:12s} {n:>10,}")
    print(f"  {'total':12s} {total:>10,}")
