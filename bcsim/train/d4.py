"""The dihedral group of the map (8 flips/rotations) on everything the critic reads.

A transform is (fx, fy, tr): mirror x, then mirror y, then transpose. The map is a
torus in the top-left W x H corner of the 64 x 64 board. Edges are stored on the
tile they bound -- the h planes on a tile's NORTH side, the v planes on its WEST
side (bc_core edge_on_side) -- so a mirror moves an edge to the far side of the
tile, i.e. onto the next tile: v'[y, x] = v[y, (W - x) mod W] under a mirror in x,
h'[y, x] = h[(H - y) mod H, x] under a mirror in y; a transpose swaps h and v.

    board    (B, 18, 64, 64) critic board (bc_vec.hpp BOARD_CH 18):
             tiles 0-5, 8, 9, 12, 13; edges h = 6 (kelp), 10 (portal), 14/15 (portal
             partner x/y); v = 7, 11, 16/17. Partner values are the partner edge's
             tile, 1 + round(254 v / (n - 1)), and are moved like the edge itself.
    local    (B, 23, 7, 7) the acting dragon's window, ALREADY rotated to its facing:
             a rotation of the map leaves it unchanged, a reflection flips it left-right
             and swaps east/west in the direction channels (face, kelp, portal).
    scalars  (B, 14) bcsim.SCALARS: facing one-hot, head x/y, map w/h move; the rest stay.
    priv     invariant.

Not exact in one place, and the engine's fault: a length-1 dragon faces N at spawn
whatever the map's orientation, until its first move (bc_core spawn).

map_d4 applies the same transform to a map (augment.Map), for testing against the
engine: obs(T(map)) must equal T(obs(map)).
"""

from __future__ import annotations

import numpy as np
import torch

S = 64
TILE = [0, 1, 2, 3, 4, 5, 8, 9, 12, 13]
LC_E_W = [(10, 12), (14, 16), (18, 20)]         # FACE, KELP, PORTAL: east <-> west
SC_FACE, SC_HX, SC_HY, SC_MW, SC_MH = 4, 8, 9, 10, 11
DIR_FX = [0, 3, 2, 1]                           # N E S W under a mirror in x
DIR_FY = [2, 1, 0, 3]
DIR_TR = [3, 2, 1, 0]                           # transpose: N <-> W, E <-> S


def all8():
    return [(fx, fy, tr) for fx in (0, 1) for fy in (0, 1) for tr in (0, 1)]


def _dec(v, n):
    return torch.round((v - 1) / 254 * (n - 1).clamp(min=1))


def _enc(c, n):
    return 1 + torch.round(254 * c / (n - 1).clamp(min=1))


def board(x: torch.Tensor, W: torch.Tensor, H: torch.Tensor, fx, fy, tr) -> torch.Tensor:
    """x (B, 18, 64, 64) float (any scaling of the value planes except 14-17, which must
    be raw 0..255). W, H (B,) long; fx, fy, tr (B,) bool."""
    B, dev = x.shape[0], x.device
    ar = torch.arange(S, device=dev)
    yo, xo = ar.view(1, S, 1).expand(B, S, S), ar.view(1, 1, S).expand(B, S, S)
    W_, H_ = W.view(B, 1, 1).long(), H.view(B, 1, 1).long()
    fxv, fyv, trv = fx.view(B, 1, 1), fy.view(B, 1, 1), tr.view(B, 1, 1)
    Wo, Ho = torch.where(trv, H_, W_), torch.where(trv, W_, H_)
    inside = (xo < Wo) & (yo < Ho)
    y1, x1 = torch.where(trv, xo, yo), torch.where(trv, yo, xo)      # the flipped frame

    def src(kind):
        sx = torch.where(fxv, ((W_ - x1) % W_) if kind == "v" else (W_ - 1 - x1), x1)
        sy = torch.where(fyv, ((H_ - y1) % H_) if kind == "h" else (H_ - 1 - y1), y1)
        return (sy.clamp(0, S - 1) * S + sx.clamp(0, S - 1)).reshape(B, 1, S * S)

    flat = x.reshape(B, x.shape[1], S * S)
    ti, hi, vi = src("tile"), src("h"), src("v")
    g = lambda p, idx: flat[:, p:p + 1].gather(2, idx).reshape(B, 1, S, S)     # noqa: E731
    out = torch.zeros_like(x)
    for p in TILE:
        out[:, p:p + 1] = g(p, ti)
    t4 = tr.view(B, 1, 1, 1)
    # kelp and portal edges: a transpose turns west edges into north edges and back
    for hp, vp in ((6, 7), (10, 11)):
        hh, vv, hv, vh = g(hp, hi), g(vp, vi), g(vp, vi), g(hp, hi)
        out[:, hp:hp + 1] = torch.where(t4, hv, hh)
        out[:, vp:vp + 1] = torch.where(t4, vh, vv)
    # portal partners: gathered like their edge, then the partner's own position moved
    W4, H4 = W.view(B, 1, 1, 1).float(), H.view(B, 1, 1, 1).float()
    fx4, fy4 = fx.view(B, 1, 1, 1), fy.view(B, 1, 1, 1)

    def partner(px_p, py_p, idx, kind):
        vx, vy = g(px_p, idx), g(py_p, idx)
        has = vx > 0
        px, py = _dec(vx, W4), _dec(vy, H4)
        if kind == "h":
            px = torch.where(fx4, W4 - 1 - px, px)
            py = torch.where(fy4, (H4 - py) % H4, py)
        else:
            px = torch.where(fx4, (W4 - px) % W4, px)
            py = torch.where(fy4, H4 - 1 - py, py)
        return has, px, py

    hh_has, hh_x, hh_y = partner(14, 15, hi, "h")          # h edges, not transposed
    vv_has, vv_x, vv_y = partner(16, 17, vi, "v")
    vh_has, vh_x, vh_y = partner(16, 17, vi, "v")          # v edges that become h under a transpose
    hv_has, hv_x, hv_y = partner(14, 15, hi, "h")
    Wo4, Ho4 = torch.where(t4, H4, W4), torch.where(t4, W4, H4)
    # output h planes: from h edges, or (transposed) from v edges with x/y swapped
    h_has = torch.where(t4, vh_has, hh_has)
    h_x = torch.where(t4, vh_y, hh_x); h_y = torch.where(t4, vh_x, hh_y)
    v_has = torch.where(t4, hv_has, vv_has)
    v_x = torch.where(t4, hv_y, vv_x); v_y = torch.where(t4, hv_x, vv_y)
    out[:, 14:15] = torch.where(h_has, _enc(h_x, Wo4), torch.zeros_like(h_x))
    out[:, 15:16] = torch.where(h_has, _enc(h_y, Ho4), torch.zeros_like(h_y))
    out[:, 16:17] = torch.where(v_has, _enc(v_x, Wo4), torch.zeros_like(v_x))
    out[:, 17:18] = torch.where(v_has, _enc(v_y, Ho4), torch.zeros_like(v_y))
    return out * inside.unsqueeze(1)


def local(l: torch.Tensor, fx, fy, tr) -> torch.Tensor:
    """(B, 23, 7, 7) egocentric window: a reflection (odd fx+fy+tr) flips it left-right."""
    refl = (fx.long() + fy.long() + tr.long()) % 2 == 1
    if not refl.any():
        return l
    f = l.flip(-1).clone()
    for e, w in LC_E_W:
        f[:, [e, w]] = f[:, [w, e]]
    return torch.where(refl.view(-1, 1, 1, 1), f, l)


def scalars(s: torch.Tensor, fx, fy, tr) -> torch.Tensor:
    """(B, 14) bcsim.SCALARS rows."""
    s = s.clone()
    W = torch.round(s[:, SC_MW] * 64); H = torch.round(s[:, SC_MH] * 64)
    hx = torch.round(s[:, SC_HX] * W); hy = torch.round(s[:, SC_HY] * H)
    face = s[:, SC_FACE:SC_FACE + 4]
    perm = lambda f, p: f[:, p]                 # noqa: E731
    face = torch.where(fx.view(-1, 1), perm(face, DIR_FX), face)
    face = torch.where(fy.view(-1, 1), perm(face, DIR_FY), face)
    face = torch.where(tr.view(-1, 1), perm(face, DIR_TR), face)
    hx = torch.where(fx, W - 1 - hx, hx)
    hy = torch.where(fy, H - 1 - hy, hy)
    hx, hy = torch.where(tr, hy, hx), torch.where(tr, hx, hy)
    W, H = torch.where(tr, H, W), torch.where(tr, W, H)
    s[:, SC_FACE:SC_FACE + 4] = face
    s[:, SC_HX], s[:, SC_HY] = hx / W, hy / H
    s[:, SC_MW], s[:, SC_MH] = W / 64, H / 64
    return s


def random_syms(n: int, dev) -> tuple:
    r = torch.randint(0, 2, (3, n), device=dev).bool()
    return r[0], r[1], r[2]


# ------------------------------------------------------------------ maps (for tests)
def map_d4(m, fx: bool, fy: bool, tr: bool):
    """The same transform on an augment.Map: tiles, edges (with portal ids), dragons."""
    from train import augment
    out = m.copy()
    W, H = m.w, m.h
    Wo, Ho = (H, W) if tr else (W, H)
    tiles = {k: np.zeros((Ho, Wo), getattr(m, k).dtype) for k in ("min_gap", "max_gap")}
    hk = np.zeros((Ho, Wo), m.hk.dtype); vk = np.zeros((Ho, Wo), m.vk.dtype)
    hp = np.full((Ho, Wo), -1, m.hp.dtype); vp = np.full((Ho, Wo), -1, m.vp.dtype)

    def tile(x, y):
        x = W - 1 - x if fx else x
        y = H - 1 - y if fy else y
        return (y, x) if tr else (x, y)

    for y in range(H):
        for x in range(W):
            X, Y = tile(x, y)
            for k in tiles:
                tiles[k][Y, X] = getattr(m, k)[y, x]
            # north edge of (x, y)
            ex, ey = (W - 1 - x if fx else x), ((H - y) % H if fy else y)
            if tr:
                vk[ex, ey], vp[ex, ey] = m.hk[y, x], m.hp[y, x]
            else:
                hk[ey, ex], hp[ey, ex] = m.hk[y, x], m.hp[y, x]
            # west edge of (x, y)
            ex, ey = ((W - x) % W if fx else x), (H - 1 - y if fy else y)
            if tr:
                hk[ex, ey], hp[ex, ey] = m.vk[y, x], m.vp[y, x]
            else:
                vk[ey, ex], vp[ey, ex] = m.vk[y, x], m.vp[y, x]
    out.w, out.h = Wo, Ho
    out.min_gap, out.max_gap = tiles["min_gap"], tiles["max_gap"]
    out.hk, out.vk, out.hp, out.vp = hk, vk, hp, vp
    out.dragons = [(t, [tile(x, y) for x, y in body]) for t, body in m.dragons]
    if tr:
        swap = {augment.SYM_FLIP_X: augment.SYM_FLIP_Y, augment.SYM_FLIP_Y: augment.SYM_FLIP_X}
        out.sym = swap.get(out.sym, out.sym)
        out.aug_sym = swap.get(out.aug_sym, out.aug_sym)
    return out
