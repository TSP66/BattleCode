"""The trained actor, run forward in NumPy inside the judge's point budget.

Convolutions are done as im2col plus one matmul each, which is the only shape
NumPy is fast at: the 3x3 kernels become (Cout, Cin*9) matrices and the 7x7
window becomes a (Cin*9, 49) column matrix, so a layer is a single BLAS call.
The gather indices are built once at import.

GroupNorm cannot be folded into the kernels the way BatchNorm can -- it
normalises against the current activation, not a stored running mean -- so it
stays as a cheap elementwise pass.
"""

import os

import numpy as np

CELLS = 49
WINDOW = 7
PAD = 9
EPS = 1e-5


def _indices():
    """Where each output cell reads from, in the zero-padded 9x9 board."""
    pad_idx = np.array([(r + 1) * PAD + (c + 1)
                        for r in range(WINDOW) for c in range(WINDOW)], dtype=np.intp)
    k_idx = np.zeros((9, CELLS), dtype=np.intp)
    for kh in range(3):
        for kw in range(3):
            for r in range(WINDOW):
                for c in range(WINDOW):
                    k_idx[kh * 3 + kw, r * WINDOW + c] = (r + kh) * PAD + (c + kw)
    return pad_idx, k_idx


PAD_IDX, K_IDX = _indices()


def silu(x):
    # exp overflows to inf for very negative x, and x/inf is the 0 we want
    with np.errstate(over="ignore"):
        return x / (1.0 + np.exp(-x))


def group_norm(x, w, b, groups=8):
    """x is (C, 49); normalise over each group's channels and cells together."""
    y = x.reshape(groups, -1)
    m = y.mean(axis=1, keepdims=True)
    v = y.var(axis=1, keepdims=True)
    y = (y - m) / np.sqrt(v + EPS)
    return y.reshape(x.shape) * w[:, None] + b[:, None]


class Policy:
    def __init__(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights.npz")
        with np.load(path) as z:
            # widened once here rather than per turn; float16 only ever exists
            # so the upload fits under the 4MB cap
            self.w = {k: (z[k].astype(np.float32) if z[k].dtype == np.float16 else z[k])
                      for k in z.files}
        self.blocks = int(self.w["meta"][1])
        self.width = int(self.w["meta"][0])
        self._pad = np.zeros((self.width, PAD * PAD), dtype=np.float32)

    def _conv3(self, x, kernel, buf=None):
        """3x3 same-padding convolution as one matmul."""
        if buf is None or buf.shape[0] != x.shape[0]:
            buf = np.zeros((x.shape[0], PAD * PAD), dtype=np.float32)
        else:
            buf[:, :] = 0.0
        buf[:, PAD_IDX] = x
        cols = buf[:, K_IDX].reshape(-1, CELLS)
        return kernel @ cols, buf

    def act(self, local, scalars, mask):
        """Returns the highest-scoring legal action id."""
        w = self.w
        x = local.reshape(-1, CELLS)

        buf = None
        h, _ = self._conv3(x, w["stem_c"])
        h = silu(group_norm(h, w["stem_n_w"], w["stem_n_b"]))
        buf = self._pad
        for i in range(self.blocks):
            y, buf = self._conv3(h, w[f"b{i}_c1"], buf)
            y = silu(group_norm(y, w[f"b{i}_n1_w"], w[f"b{i}_n1_b"]))
            y, buf = self._conv3(y, w[f"b{i}_c2"], buf)
            y = group_norm(y, w[f"b{i}_n2_w"], w[f"b{i}_n2_b"])
            h = silu(h + y)

        f = silu(group_norm(w["flat_c"] @ h, w["flat_n_w"], w["flat_n_b"], groups=8))

        s = silu(w["sc0_w"] @ scalars + w["sc0_b"])
        s = silu(w["sc2_w"] @ s + w["sc2_b"])

        z = np.concatenate([f.reshape(-1), s])
        z = silu(w["fu0_w"] @ z + w["fu0_b"])
        z = silu(w["fu2_w"] @ z + w["fu2_b"])
        logits = w["pi_w"] @ z + w["pi_b"]

        if mask.any():
            logits = np.where(mask > 0, logits, -np.inf)
        return int(np.argmax(logits))
