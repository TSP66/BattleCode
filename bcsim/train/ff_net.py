"""Feed-forward policy on the 14x14 grid: the LSTM net's CNN with no LSTM.

    grid (in_ch x14x14) -> 3x3 conv c1 -> [blocks x 3x3 conv c1] -> 3x3 stride-2 conv c2 (7x7)
    -> 1x1 squeeze -> flatten, + one-hot previous action -> dense hidden -> pi, v

No memory beyond what the grid already carries (remembered cells, sonar reports) and the
previous action. `v` is the policy's own value head, for PPO without the board critic.
Measured 2026-09-30 (the small default, 3.5M MACs): ~150k turns/s rollouts, ~440k/s updates.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from train.lstm_net import LSTMPolicy, blind_portal_planes, fit_grid


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, c // 4), c)


class FFPolicy(nn.Module):
    def __init__(self, c1: int = 32, c2: int = 64, squeeze: int = 16, hidden: int = 128,
                 blocks: int = 0, in_ch: int = 43, n_actions: int = 49, grid: int = 14):
        super().__init__()
        g2 = (grid + 1) // 2
        self.n_actions = n_actions
        self.no_action = n_actions                   # "previous action" at a dragon's first turn
        self.layers, self.hidden = 0, hidden         # no recurrent state: a Pool of these holds only prev
        body = [nn.Conv2d(in_ch, c1, 3, padding=1), _gn(c1), nn.SiLU()]
        for _ in range(blocks):
            body += [nn.Conv2d(c1, c1, 3, padding=1), _gn(c1), nn.SiLU()]
        body += [nn.Conv2d(c1, c2, 3, stride=2, padding=1), _gn(c2), nn.SiLU(),
                 nn.Conv2d(c2, squeeze, 1), nn.SiLU(), nn.Flatten()]
        self.body = nn.Sequential(*body)
        self.dense = nn.Linear(squeeze * g2 * g2 + n_actions + 1, hidden)
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def features(self, grid: torch.Tensor, prev_a: torch.Tensor):
        g = fit_grid(grid, self.body[0].in_channels)
        if getattr(self, "blind_portal", False):
            g = blind_portal_planes(g)
        z = self.body(g)
        z = torch.cat([z, F.one_hot(prev_a.long(), self.n_actions + 1).to(z.dtype)], 1)
        return F.silu(self.dense(z))

    def forward(self, grid: torch.Tensor, prev_a: torch.Tensor, state=None):
        """(logits, v); with a state (LSTMPolicy's calling convention) also the empty new state."""
        z = self.features(grid, prev_a)
        out = self.pi(z), self.v(z).squeeze(-1)
        return out if state is None else (*out, [])


def macs(c1=32, c2=64, squeeze=16, hidden=128, blocks=0, in_ch=43, n_actions=49) -> int:
    return (196 * 9 * in_ch * c1 + blocks * 196 * 9 * c1 * c1 + 49 * 9 * c1 * c2 + 49 * c2 * squeeze
            + (49 * squeeze + n_actions + 1) * hidden + hidden * n_actions)


N_TEMP_FEATS = 2


def temp_feats(t: torch.Tensor) -> torch.Tensor:
    """(n,) temperatures -> (n, 2) [T, sqrt(T)] (user, 2026-10-01). 0 (greedy) is a valid input."""
    t = t.float().clamp(min=0.0)
    return torch.stack([t, t.sqrt()], 1)


# scal_in (user, 2026-10-03): five scalars straight into the first dense layer, read off the
# grid's constant planes (cpp/bc_obs.hpp glob[], lstm_net QUEEN_CHANNELS) at the head's cell:
# round / 500, (round / 500)^2, min(len, 64) / 64, (alive / unit_limit)^2, is_queen.
N_SCAL_FEATS = 5
SCAL_ROUND, SCAL_LENGTH, SCAL_UNITS, SCAL_IS_QUEEN = 28, 29, 30, 43


# ident_in (user, 2026-10-03): three more that fingerprint the dragon, so dragons can take on
# different roles: birth round / 500, sin(id / 7), sin(id / 43). The simulator paints them on
# grid planes 54-56 (cpp/bc_memory.hpp BIRTH..ID43), past every channel a CNN reads.
N_IDENT_FEATS = 3
IDENT_BIRTH = 54


def ident_feats(grid: torch.Tensor) -> torch.Tensor:
    """(n, C >= 57, G, G) grid -> (n, 3); constant planes, so the centre cell is the value."""
    if grid.shape[1] < IDENT_BIRTH + N_IDENT_FEATS:
        raise ValueError(f"ident_in reads grid planes {IDENT_BIRTH}-{IDENT_BIRTH + N_IDENT_FEATS - 1}: "
                         f"this simulator's grid has {grid.shape[1]} channels (rebuild bcsim)")
    c = grid.shape[-1] // 2
    return grid[:, IDENT_BIRTH:IDENT_BIRTH + N_IDENT_FEATS, c, c]


# ahist_in (user, 2026-10-05): the dragon's own action history EXCLUDING its last action (that one
# is the one-hot prev input): a(t-2) 0.5, a(t-3) 0.25, ... summed per action id, so an action
# repeated forever reads 1. The simulator keeps it (bc_vec.hpp note_action) and writes it as a
# vector into the first n_actions cells of grid plane 57 (cpp/bc_memory.hpp AHIST; BC_AHIST builds).
AHIST_PLANE = 57


def ahist_feats(grid: torch.Tensor, n_actions: int) -> torch.Tensor:
    """(n, C >= 58, G, G) grid -> (n, n_actions)."""
    if grid.shape[1] <= AHIST_PLANE:
        raise ValueError(f"ahist_in reads grid plane {AHIST_PLANE}: this simulator's grid has {grid.shape[1]} "
                         "channels (a BC_AHIST build: make -C bcsim s2g15p)")
    return grid[:, AHIST_PLANE].flatten(1)[:, :n_actions]


def scal_feats(grid: torch.Tensor) -> torch.Tensor:
    """(n, C, G, G) grid -> (n, 5); the planes are constant, so the centre cell is the value."""
    c = grid.shape[-1] // 2
    r, ln, u, q = (grid[:, k, c, c] for k in (SCAL_ROUND, SCAL_LENGTH, SCAL_UNITS, SCAL_IS_QUEEN))
    return torch.stack([r, r * r, ln, u * u, q], 1)


class FFLPolicy(LSTMPolicy):
    """The LSTM net with its recurrence replaced by dense layers of the same width.

    Same CNN trunk, previous-action embedding and LayerNorm as LSTMPolicy (so the
    same `encode`), then `layers` x Linear(hidden) + SiLU in place of the LSTM
    cells. Isolates what memory is worth: everything else matches the LSTM clone.
    """

    def __init__(self, c1: int = 48, b1: int = 1, c2: int = 112, b2: int = 2,
                 squeeze: int = 16, embed: int = 256, hidden: int = 128, layers: int = 2,
                 in_ch: int = 43, n_actions: int = 49, grid: int = 14, temp_in: bool = False,
                 temp_ref: float = 0.4, scal_in: bool = False, ident_in: bool = False, ahist_in: bool = False):
        super().__init__(c1, b1, c2, b2, squeeze, embed, hidden, layers, in_ch, n_actions, grid)
        del self.lstm
        self.layers = 0                              # no recurrent state (see FFPolicy)
        self.n_mlp = layers
        # temp_in (user, 2026-10-01): the policy is told the temperature it samples at, as T and
        # sqrt(T) (user: the non-linear one may say more) on two extra inputs of the first dense
        # layer. `temp_default` is what a caller that passes no temperature gets (gates, panels: set it).
        self.temp_in, self.temp_ref = bool(temp_in), float(temp_ref)
        self.register_buffer("temp_default", torch.tensor(float(temp_ref)), persistent=False)
        # mlp[0] reads [encode (embed + 48), scalars (scal_in), identity (ident_in), temperature (temp_in)],
        # in that order: the temperature columns stay last, which fold_temp relies on
        self.scal_in, self.ident_in, self.ahist_in = bool(scal_in), bool(ident_in), bool(ahist_in)
        if self.scal_in and in_ch <= SCAL_IS_QUEEN:
            raise ValueError(f"scal_in reads is_queen (channel {SCAL_IS_QUEEN}): needs in_ch > {SCAL_IS_QUEEN}, got {in_ch}")
        first = (embed + 48 + (N_SCAL_FEATS if scal_in else 0) + (N_IDENT_FEATS if ident_in else 0)
                 + (n_actions if ahist_in else 0) + (N_TEMP_FEATS if temp_in else 0))
        self.mlp = nn.ModuleList(nn.Linear(first if i == 0 else hidden, hidden) for i in range(layers))

    def temp_feature(self, temp, n: int, like: torch.Tensor) -> torch.Tensor:
        """(n, 2) [T, sqrt(T)]; T is a scalar, an (n,) tensor or None (temp_default)."""
        t = self.temp_default if temp is None else temp
        t = torch.as_tensor(t, device=like.device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1:
            t = t.expand(n)
        return temp_feats(t).to(like.dtype)

    def features(self, grid, prev_a, temp=None):
        x = self.encode(grid, prev_a)
        if self.scal_in:
            x = torch.cat([x, scal_feats(fit_grid(grid, self.in_ch)).to(x.dtype)], 1)
        if self.ident_in:
            x = torch.cat([x, ident_feats(grid).to(x.dtype)], 1)     # the full grid: past in_ch
        if self.ahist_in:
            x = torch.cat([x, ahist_feats(grid, self.n_actions).to(x.dtype)], 1)
        if self.temp_in:
            x = torch.cat([x, self.temp_feature(temp, x.shape[0], x)], 1)
        for lin in self.mlp:
            x = F.silu(lin(x))
        return x

    def add_temp_input(self) -> None:
        """Give a net built without temp_in its temperature inputs, as zero columns: its
        outputs are unchanged at every temperature until training moves them."""
        if self.temp_in:
            return
        old = self.mlp[0]
        new = nn.Linear(old.in_features + N_TEMP_FEATS, old.out_features).to(old.weight.device, old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :old.in_features] = old.weight
            new.bias.copy_(old.bias)
        self.mlp[0] = new
        self.temp_in = True

    def add_scal_input(self) -> None:
        """Give a net built without scal_in its five scalar inputs, as zero columns inserted before
        the temperature ones: its outputs are unchanged until training moves them."""
        if self.scal_in:
            return
        if self.in_ch <= SCAL_IS_QUEEN:
            raise ValueError(f"scal_in reads is_queen (channel {SCAL_IS_QUEEN}): needs in_ch > {SCAL_IS_QUEEN}")
        old = self.mlp[0]
        n0 = old.in_features - (N_TEMP_FEATS if self.temp_in else 0)
        new = nn.Linear(old.in_features + N_SCAL_FEATS, old.out_features).to(old.weight.device, old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :n0] = old.weight[:, :n0]
            new.weight[:, n0 + N_SCAL_FEATS:] = old.weight[:, n0:]
            new.bias.copy_(old.bias)
        self.mlp[0] = new
        self.scal_in = True

    def add_ident_input(self) -> None:
        """Give a net built without ident_in its three identity inputs, as zero columns inserted
        before the temperature ones: its outputs are unchanged until training moves them."""
        if self.ident_in:
            return
        old = self.mlp[0]
        n0 = old.in_features - (N_TEMP_FEATS if self.temp_in else 0)
        new = nn.Linear(old.in_features + N_IDENT_FEATS, old.out_features).to(old.weight.device, old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :n0] = old.weight[:, :n0]
            new.weight[:, n0 + N_IDENT_FEATS:] = old.weight[:, n0:]
            new.bias.copy_(old.bias)
        self.mlp[0] = new
        self.ident_in = True

    def add_ahist_input(self) -> None:
        """Give a net built without ahist_in its n_actions history inputs, as zero columns inserted
        before the temperature ones: its outputs are unchanged until training moves them."""
        if self.ahist_in:
            return
        old = self.mlp[0]
        k = self.n_actions
        n0 = old.in_features - (N_TEMP_FEATS if self.temp_in else 0)
        new = nn.Linear(old.in_features + k, old.out_features).to(old.weight.device, old.weight.dtype)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :n0] = old.weight[:, :n0]
            new.weight[:, n0 + k:] = old.weight[:, n0:]
            new.bias.copy_(old.bias)
        self.mlp[0] = new
        self.ahist_in = True

    def fold_temp(self, temp: float) -> "FFLPolicy":
        """A copy without the temperature inputs, its first bias absorbing them at `temp`:
        what an exported bot plays at a fixed temperature (identical outputs at `temp`)."""
        import copy
        net = copy.deepcopy(self)
        if not net.temp_in:
            return net
        lin = net.mlp[0]
        k = N_TEMP_FEATS
        new = nn.Linear(lin.in_features - k, lin.out_features).to(lin.weight.device, lin.weight.dtype)
        with torch.no_grad():
            new.weight.copy_(lin.weight[:, :-k])
            f = temp_feats(torch.tensor([float(temp)], device=lin.weight.device))[0].to(lin.weight.dtype)
            new.bias.copy_(lin.bias + lin.weight[:, -k:] @ f)
        net.mlp[0] = new
        net.temp_in = False
        return net

    def forward(self, grid, prev_a, state=None, temp=None):
        x = self.features(grid, prev_a, temp)
        out = self.pi(x), self.v(x).squeeze(-1)
        return out if state is None else (*out, [])


# every per-dragon policy on the 14x14 grid; all play through the LSTM's Pool/LSTMGreedy path
GRID_ARCHS = ("lstm", "ff", "ffl")
FF_KEYS = ("c1", "c2", "squeeze", "hidden", "blocks")
LSTM_KEYS = ("c1", "b1", "c2", "b2", "squeeze", "embed", "hidden", "layers")


def build(args: dict):
    """The net a checkpoint's args describe (arch lstm, ff or ffl). args["uniform"] marks the
    uniform-random player (train/make_random.py): every player samples it uniformly over the
    legal moves whatever its temperature, greedy included."""
    arch = args.get("arch")
    io = {"in_ch": args.get("in_ch", 38), "n_actions": args.get("n_actions", 48), "grid": args.get("grid", 14)}
    if arch == "ff":
        net = FFPolicy(**{k: args[k] for k in FF_KEYS}, **io)
    elif arch == "ffl":
        net = FFLPolicy(**{k: args[k] for k in LSTM_KEYS}, **io, temp_in=bool(args.get("temp_in", False)),
                        temp_ref=float(args.get("temp_ref", 0.4)), scal_in=bool(args.get("scal_in", False)),
                        ident_in=bool(args.get("ident_in", False)), ahist_in=bool(args.get("ahist_in", False)))
    elif arch == "lstm":
        net = LSTMPolicy(**{k: args[k] for k in LSTM_KEYS}, **io)
    else:
        raise ValueError(f"not a grid policy: {arch}")
    net.uniform = bool(args.get("uniform", False))
    # BC_PORTALREP (2026-10-02): in a portal simulator, a net not trained on the portal planes
    # (args["portal"]) reads zeros there instead of planes it has never seen
    net.blind_portal = portal_sim() and not bool(args.get("portal", False))
    return net


def portal_sim() -> bool:
    """Whether this process's simulator library is a BC_PORTALREP build (bcsim already loaded)."""
    import sys
    bc = sys.modules.get("bcsim")
    return bool(getattr(bc, "PORTAL_BUILD", False))


class PrivValue(nn.Module):
    """The critic as another output of the policy (user, 2026-10-01).

    Reads the policy's own features (its last hidden layer) plus inputs the policy
    never sees: the engine's privileged row (global summary + v8 Phi terms) and which
    opponent this game is against. Nothing flows the other way at play time: the
    policy's logits are a function of the grid and previous action only, so the bot
    never needs any of this. Training shares the trunk (the value loss reaches it
    through `feat`), which is what makes it cheap.

    The privileged row is standardised by stats fixed from the first rollout
    (`set_priv_stats`); the output is in units of `ret_scale`, an EMA of the
    returns' spread, so the regression is well-conditioned whatever the reward scale.
    """

    def __init__(self, feat: int, n_priv: int, n_opp: int = 64, width: int = 256):
        super().__init__()
        self.opp = nn.Embedding(n_opp, 16)
        self.mlp = nn.Sequential(nn.Linear(feat + n_priv + 16, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        self.register_buffer("priv_mu", torch.zeros(n_priv))
        self.register_buffer("priv_sd", torch.ones(n_priv))
        self.register_buffer("ret_scale", torch.ones(()))
        self.register_buffer("stats_set", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def set_priv_stats(self, priv: torch.Tensor):
        self.priv_mu.copy_(priv.float().mean(0))
        self.priv_sd.copy_(priv.float().std(0).clamp(min=1e-3))
        self.stats_set.fill_(True)

    def raw(self, feat, priv, opp):
        """Prediction in units of ret_scale (what the loss regresses)."""
        x = torch.cat([feat.float(), (priv.float() - self.priv_mu) / self.priv_sd, self.opp(opp)], 1)
        return self.mlp(x).squeeze(-1)

    def forward(self, feat, priv, opp):
        return self.raw(feat, priv, opp) * self.ret_scale
