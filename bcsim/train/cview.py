"""The critic view and the critic that reads it (user, 2026-10-01).

The simulator writes one compact byte row per turn (bcsim BattlecodeVecEnv(cview=W),
layout in cpp/bc_vec.hpp cview_stride): the TRUE board -- every body and head, ours and
theirs, not what the dragon remembers -- as

    crop    (16, W, W)   16 view planes about the acting head, wrapped like the policy's
                         window (the world is a torus): board planes 0-11 and the two queens
                         (18-19) bit-packed, then countdown and still-to-move (12-13)
    coarse  (16, 16, 16) the whole 64 x 64 board, 4 x 4 average-pooled, in map coordinates

`decode` turns rows into floats on the GPU. The critic is the critic_lab variant
`crop27+coarse` (train/critic_lab.py `win` + `coarse`, layer for layer, so what r2 measures is
what PPO trains), plus the privileged row (Phi's terms in it), both teams' identities and both
teams' sampling temperatures, and one bit for an OBSERVED game (a server replay, 1) against a
generated one (0; every PPO game): real teams are encoded as greedy (T = 0, user), and the bit
lets the critic tell a team's own games from its clone's played at T = 0. It never ships;
nothing here is metered by the judge.

It NEVER reads the policy's features (user, 2026-10-01, emphatically): they belong to one policy
network, drift as PPO trains it, and say nothing on replays. The critic sees the true board only.

Like PrivValue, it predicts in units of `ret_scale` (`raw`, what the loss regresses) and
`forward` scales back to the return's units.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import bcsim

N_TEMP = 4          # this team's temperature and its sqrt, the other team's and its sqrt (user, 2026-10-01)


class Layout:
    def __init__(self, w: int):
        lay = bcsim.cview_layout(w)
        self.w = w
        self.bits, self.bytes, self.coarse, self.stride = lay["bits"], lay["bytes"], lay["coarse"], lay["stride"]
        self.ch, self.nbits, self.side = lay["ch"], lay["nbits"], lay["side"]


def decode(rows: torch.Tensor, lay: Layout) -> tuple[torch.Tensor, torch.Tensor]:
    """(B, stride) uint8 -> crop (B, ch, W, W), coarse (B, ch, 16, 16), float32 in [0, 1]."""
    if rows.dtype != torch.uint8 or rows.dim() != 2 or rows.shape[1] != lay.stride:
        raise ValueError(f"critic-view rows must be (B, {lay.stride}) uint8, got {tuple(rows.shape)} {rows.dtype}")
    B, W = rows.shape[0], lay.w
    shifts = torch.arange(8, device=rows.device, dtype=torch.uint8)
    bits = ((rows[:, :lay.bits, None] >> shifts) & 1).reshape(B, -1)[:, :lay.nbits * W * W]
    crop = torch.cat([bits.reshape(B, lay.nbits, W, W).float(),
                      rows[:, lay.bytes:lay.coarse].reshape(B, lay.ch - lay.nbits, W, W).float() / 255.0], 1)
    s = lay.side
    coarse = rows[:, lay.coarse:lay.coarse + lay.ch * s * s].reshape(B, lay.ch, s, s).float() / 255.0
    return crop, coarse


IDENT_UNKNOWN = 0     # identity slot 0: a player the table does not know (zero, never trained toward anything)
N_IDENT = 256


class CViewCritic(nn.Module):
    """One value per discount in `alphas` (user, 2026-10-01: horizons 20 / 10 / 5 / 3 rounds); PPO
    trains the head of its own alpha. Both teams' identities come from one table of `n_slots`
    learned embeddings (a clone shares its original team's slot; slot 0 is unknown); the table's
    names travel with the checkpoint (critic_pretrain.py writes them, ratchet_ff_train reads them).

    `wl` (user, 2026-10-03): a win/loss head -- its own MLP on the same encoder (both CNNs, priv,
    identities, temperatures), tanh to [-1, 1]: the expected undiscounted result (+1 win, -1 loss, 0
    draw) for the team to move. Trained by TD(lambda) alongside the alpha heads, for a later switch to
    the sparse reward. Its last layer starts at zero, so a new head says 0 ("no idea")."""

    def __init__(self, w: int, n_priv: int, alphas=(0.95,), n_slots: int = N_IDENT, width: int = 256,
                 wl: bool = False):
        super().__init__()
        self.lay = Layout(w)
        self.w, self.n_priv, self.alphas, self.n_slots = w, n_priv, [float(x) for x in alphas], n_slots
        ch = self.lay.ch
        # critic_lab Critic "win": a true-state window CNN
        self.crop_cnn = nn.Sequential(
            nn.Conv2d(ch, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(2), nn.Flatten(),
            nn.Linear(256, 128), nn.SiLU())
        # critic_lab Critic "coarse", after its AvgPool2d(4) (the simulator pools)
        self.coarse_cnn = nn.Sequential(
            nn.Conv2d(ch, 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(), nn.Linear(256, 128), nn.SiLU())
        self.ident = nn.Embedding(n_slots, 16)
        nn.init.normal_(self.ident.weight, std=0.02)
        with torch.no_grad():
            self.ident.weight[IDENT_UNKNOWN].zero_()
        d = 128 + 128 + n_priv + 2 * 16 + N_TEMP + 1          # + the observed-game bit
        self.mlp = nn.Sequential(nn.Linear(d, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, len(self.alphas)))
        self.d_in, self.width = d, width
        self.wl = None
        if wl:
            self.add_wl_head()
        self.register_buffer("priv_mu", torch.zeros(n_priv))
        self.register_buffer("priv_sd", torch.ones(n_priv))
        self.register_buffer("ret_scale", torch.ones(len(self.alphas)))     # one per head
        self.register_buffer("stats_set", torch.zeros((), dtype=torch.bool))

    def spec(self) -> dict:
        """What a checkpoint needs to rebuild this critic."""
        return {"kind": "cview2", "w": self.w, "n_priv": self.n_priv, "alphas": self.alphas, "n_slots": self.n_slots,
                "temp_feats": "t+sqrt", "observed_bit": True, "wl": self.wl is not None}

    @classmethod
    def from_spec(cls, spec: dict) -> "CViewCritic":
        if spec.get("kind") != "cview2":
            raise ValueError(f"not a cview2 critic: {spec}")
        return cls(spec["w"], spec["n_priv"], spec["alphas"], spec["n_slots"], wl=bool(spec.get("wl", False)))

    def add_wl_head(self) -> None:
        """Give this critic its win/loss head (a no-op if it has one): the alpha heads are untouched."""
        if self.wl is not None:
            return
        dev = self.mlp[0].weight.device
        self.wl = nn.Sequential(nn.Linear(self.d_in, self.width), nn.SiLU(), nn.Linear(self.width, self.width),
                                nn.SiLU(), nn.Linear(self.width, 1)).to(dev)
        with torch.no_grad():
            self.wl[-1].weight.zero_()
            self.wl[-1].bias.zero_()

    def head(self, alpha: float) -> int:
        for k, x in enumerate(self.alphas):
            if abs(x - alpha) < 1e-9:
                return k
        raise ValueError(f"this critic has heads for alphas {self.alphas}, not {alpha}")

    @torch.no_grad()
    def set_priv_stats(self, priv: torch.Tensor):
        self.priv_mu.copy_(priv.float().mean(0))
        self.priv_sd.copy_(priv.float().std(0).clamp(min=1e-3))
        self.stats_set.fill_(True)

    def raw(self, rows, priv, ident, temp, observed=None):
        """(B, len(alphas)) predictions in units of ret_scale (what the loss regresses). `rows` are
        critic-view rows (B, stride) uint8; `ident` (B, 2) identity slots (this team, the other);
        `temp` (B, 2) the two teams' sampling temperatures (0 = greedy), read as T and sqrt(T) each;
        `observed` (B,) 1 for a server replay, None = all generated (every PPO game)."""
        return self.mlp(self.encode(rows, priv, ident, temp, observed))

    def encode(self, rows, priv, ident, temp, observed=None):
        """The shared input of every head: (B, d_in)."""
        crop, coarse = decode(rows, self.lay)
        obs_bit = (torch.zeros(rows.shape[0], 1, device=rows.device) if observed is None
                   else observed.float().reshape(-1, 1))
        z = [self.crop_cnn(crop), self.coarse_cnn(coarse), (priv.float() - self.priv_mu) / self.priv_sd,
             self.ident(ident[:, 0].long()), self.ident(ident[:, 1].long()),
             temp.float(), temp.float().clamp(min=0.0).sqrt(), obs_bit]
        return torch.cat(z, 1)

    def raw_wl(self, rows, priv, ident, temp, observed=None):
        """(raw (B, len(alphas)), win/loss (B,) in [-1, 1]) from one pass of the encoder."""
        z = self.encode(rows, priv, ident, temp, observed)
        return self.mlp(z), torch.tanh(self.wl(z).squeeze(-1))

    def forward(self, rows, priv, ident, temp, observed=None):
        return self.raw(rows, priv, ident, temp, observed) * self.ret_scale
