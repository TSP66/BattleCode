"""The rollout's GAE pass (ratchet_ff_train), compiled with numba (user, 2026-10-05: throughput).

`gae_py` is the trainer's original Python loop, kept as the reference; `gae` is the same loop compiled,
with every operation in the reference's precision (numpy 2 scalar promotion: float32 where both sides are
float32 or a Python float meets a float32, float64 where a power of alpha/lam enters), so the two agree
bit for bit (tests/test_fast_gae.py). The powers come from tables built with the reference's own
expression (`x ** np.int32(d)`), so no pow() implementation difference can creep in.
"""

from __future__ import annotations

import numba
import numpy as np

MAX_GAP = 4096          # rounds between a team's consecutive turns (a game is 500 rounds)


def pow_table(x: float) -> np.ndarray:
    return np.array([x ** np.int32(d) for d in range(MAX_GAP)], np.float64)


def gae_py(rows_l, nx, er, is_l, ph, tmf, rf, v_all, alpha, lam, wl, vw=None, ew=None, wl_lam=0.0):
    """The trainer's loop as it was (ratchet_ff_train before 2026-10-05)."""
    n = len(ph)
    adv = np.zeros(n, np.float32)
    usable = np.zeros(n, bool)
    rw = np.zeros(n, np.float32)
    adv_w = np.zeros(n, np.float32) if wl else None
    for row in rows_l[::-1]:
        n_ = nx[row]
        if n_ >= 0 and is_l[n_]:
            r = ph[n_] - ph[row]
            disc = alpha ** (rf[n_] - rf[row])
            delta = r + disc * v_all[n_] - v_all[row]
            lam_r = lam ** (rf[n_] - rf[row])
            adv[row] = delta + (disc * lam_r * adv[n_] if usable[n_] else 0.0)
            if wl:
                adv_w[row] = vw[n_] - vw[row] + (wl_lam ** (rf[n_] - rf[row]) * adv_w[n_] if usable[n_] else 0.0)
            usable[row] = True
            rw[row] = r
        elif er[row] >= 0:
            e_r = er[row]
            q = ph[e_r] if tmf[e_r] == tmf[row] else -ph[e_r]
            r = 0.0 if e_r == row else q - ph[row]
            adv[row] = r - v_all[row]
            if wl:
                w_ = ew[e_r]
                adv_w[row] = (0.0 if w_ < 0 else (1.0 if w_ == tmf[row] else -1.0)) - vw[row]
            usable[row] = True
            rw[row] = r
    return adv, usable, rw, adv_w


@numba.njit(cache=True, fastmath=False, nogil=True)
def _gae_nb(rows_l, nx, er, is_l, ph, tmf, rf, v_all, pa, pl, wl, vw, ew, pw, adv, usable, rw, adv_w):
    z32 = np.float32(0.0)
    for k in range(len(rows_l) - 1, -1, -1):
        row = rows_l[k]
        n_ = nx[row]
        if n_ >= 0 and is_l[n_]:
            r = np.float32(ph[n_] - ph[row])                           # float32 - float32
            d = rf[n_] - rf[row]
            disc = pa[d]                                                # float64
            delta = (np.float64(r) + disc * np.float64(v_all[n_])) - np.float64(v_all[row])
            if usable[n_]:
                adv[row] = np.float32(delta + (disc * pl[d]) * np.float64(adv[n_]))
            else:
                adv[row] = np.float32(delta + 0.0)
            if wl:
                dw = np.float32(vw[n_] - vw[row])                       # float32 - float32
                if usable[n_]:
                    adv_w[row] = np.float32(np.float64(dw) + pw[d] * np.float64(adv_w[n_]))
                else:
                    adv_w[row] = np.float32(dw + z32)                   # float32 + a Python 0.0: float32
            usable[row] = True
            rw[row] = r
        elif er[row] >= 0:
            e_r = er[row]
            q = ph[e_r] if tmf[e_r] == tmf[row] else np.float32(-ph[e_r])
            r = z32 if e_r == row else np.float32(q - ph[row])
            adv[row] = np.float32(r - v_all[row])                      # float32 (0.0 - float32: float32)
            if wl:
                w_ = ew[e_r]
                zw = z32 if w_ < 0 else (np.float32(1.0) if w_ == tmf[row] else np.float32(-1.0))
                adv_w[row] = np.float32(zw - vw[row])
            usable[row] = True
            rw[row] = r


class GAE:
    """Holds the power tables (one set per run: alpha, lam, wl_lam are fixed)."""

    def __init__(self, alpha: float, lam: float, wl_lam: float = 0.0):
        self.pa, self.pl, self.pw = pow_table(alpha), pow_table(lam), pow_table(wl_lam)

    def __call__(self, rows_l, nx, er, is_l, ph, tmf, rf, v_all, wl, vw=None, ew=None):
        n = len(ph)
        adv = np.zeros(n, np.float32)
        usable = np.zeros(n, bool)
        rw = np.zeros(n, np.float32)
        adv_w = np.zeros(n, np.float32)
        if len(rows_l):
            gap = rf[nx[rows_l]] - rf[rows_l]
            if int(gap[nx[rows_l] >= 0].max(initial=0)) >= MAX_GAP:
                raise ValueError("a team's turns more than MAX_GAP rounds apart")
        _gae_nb(np.ascontiguousarray(rows_l, np.int64), nx, er, is_l, ph, tmf, rf, v_all,
                self.pa, self.pl, bool(wl),
                vw if wl else np.zeros(1, np.float32), ew if wl else np.zeros(1, np.int8), self.pw,
                adv, usable, rw, adv_w)
        return adv, usable, rw, (adv_w if wl else None)
