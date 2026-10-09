"""The critic view (cpp/bc_vec.hpp cview_stride, train/cview.py) against the full board.

Plays random legal games on real maps with the board AND the critic view bound, and checks
every row of every step:
  1. the bytes equal a numpy reference cut from the board (wrapped crop, packbits little,
     (sum + 8) // 16 pooling with binary planes as 255) -- byte for byte, padding included;
  2. a second env with ONLY the critic view bound (the thread-local scratch board path),
     same seed and actions, writes identical bytes;
  3. train/cview.decode of the rows equals the float reference (crop planes 0-11 as 0/1,
     12-13 / 255, coarse / 255) exactly.
Run for several crop sizes, including 1 and 63, and maps of several sizes.

    BCSIM_LIB=bcsim/libbcvec_priv_s2.so python3 tests/test_cview.py
"""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("BCSIM_LIB", str(HERE.parent / "bcsim" / "libbcvec_priv_s2.so"))

import bcsim  # noqa: E402
import torch  # noqa: E402

from train.cview import Layout, decode  # noqa: E402

ROOT = HERE.parent.parent


def reference(board: np.ndarray, w: int, lay: dict) -> np.ndarray:
    """(N, BOARD_CH, 64, 64) board -> (N, stride) critic-view bytes, the view planes taken
    from the board by lay["src"]."""
    N = len(board)
    out = np.zeros((N, lay["stride"]), np.uint8)
    r = w // 2
    for i in range(N):
        b = board[i, lay["src"]]
        inmap = b[5]
        mw = int(inmap.any(0).sum())
        mh = int(inmap.any(1).sum())
        hy, hx = np.argwhere(b[9])[0]
        ys = (hy - r + np.arange(w)) % mh
        xs = (hx - r + np.arange(w)) % mw
        crop = b[:, ys][:, :, xs]                                    # (ch, w, w)
        nb, ch = lay["nbits"], lay["ch"]
        bits = np.packbits((crop[:nb] != 0).reshape(-1), bitorder="little")
        out[i, :lay["bits"]] = bits
        out[i, lay["bytes"]:lay["coarse"]] = crop[nb:ch].reshape(-1)
        v = b.astype(np.int64)
        v[:nb] = (v[:nb] != 0) * 255
        s = v.reshape(ch, 16, 4, 16, 4).sum((2, 4))
        out[i, lay["coarse"]:lay["coarse"] + ch * 256] = ((s + 8) // 16).reshape(-1).astype(np.uint8)
    return out


def float_reference(board: np.ndarray, w: int, lay: dict) -> tuple[np.ndarray, np.ndarray]:
    N = len(board)
    r = w // 2
    nb, ch = lay["nbits"], lay["ch"]
    crops = np.zeros((N, ch, w, w), np.float32)
    for i in range(N):
        b = board[i, lay["src"]]
        mw, mh = int(b[5].any(0).sum()), int(b[5].any(1).sum())
        hy, hx = np.argwhere(b[9])[0]
        c = b[:, (hy - r + np.arange(w)) % mh][:, :, (hx - r + np.arange(w)) % mw].astype(np.float32)
        c[:nb] = c[:nb] != 0
        c[nb:] /= 255.0
        crops[i] = c
    v = board[:, lay["src"]].astype(np.int64)
    v[:, :nb] = (v[:, :nb] != 0) * 255
    coarse = ((v.reshape(N, ch, 16, 4, 16, 4).sum((3, 5)) + 8) // 16).astype(np.float32) / 255.0
    return crops, coarse


def check_queens(board: np.ndarray, priv: np.ndarray) -> int:
    """Board planes 18-19 (our queen, theirs) lie on the dragon planes of their own side and
    are exactly as long as the queen lengths in the privileged row (log1p(len) / 5)."""
    own = board[:, 0] | board[:, 1]
    foe = board[:, 2] | board[:, 3]
    assert not (board[:, 18] & ~own).any() and not (board[:, 19] & ~foe).any(), "a queen plane off its side"
    for side, col in ((18, 8), (19, 9)):
        n = board[:, side].reshape(len(board), -1).sum(1)
        want = np.rint(np.expm1(priv[:, col] * 5.0))
        assert (n == want).all(), f"queen plane {side} length {n[n != want][:3]} vs priv {want[n != want][:3]}"
    return int((board[:, 19].reshape(len(board), -1).sum(1) > 0).sum())


def play(maps: list[str], w: int, steps: int, seed: int, n_envs: int = 24) -> int:
    lay = bcsim.cview_layout(w)
    A = bcsim.BattlecodeVecEnv(maps, num_envs=n_envs, num_threads=4, seed=seed, privileged=True,
                               board=True, cview=w)
    B = bcsim.BattlecodeVecEnv(maps, num_envs=n_envs, num_threads=4, seed=seed, privileged=True,
                               board=False, cview=w)
    oa, ob = A.reset(), B.reset()
    rng = np.random.default_rng(seed)
    tl = Layout(w)
    checked = 0
    queens_seen = 0
    for step in range(steps):
        assert (oa.uid == ob.uid).all() and (oa.round == ob.round).all(), "the two envs drifted apart"
        # every row must carry a head on plane 9 (the crop's centre)
        assert A.board[:, 9].reshape(n_envs, -1).any(1).all(), "a row without an acting head"
        ref = reference(A.board, w, lay)
        bad = np.flatnonzero((ref != A.cview).any(1))
        assert not len(bad), f"w {w} step {step}: env {bad[0]} differs from the board reference " \
                             f"at bytes {np.flatnonzero(ref[bad[0]] != A.cview[bad[0]])[:10]}"
        assert (A.cview == B.cview).all(), f"w {w} step {step}: the scratch-board path differs"
        queens_seen += check_queens(A.board, oa.priv)
        if step % 25 == 0:
            crop, coarse = decode(torch.from_numpy(A.cview.copy()), tl)
            fc, fco = float_reference(A.board, w, lay)
            assert torch.equal(crop, torch.from_numpy(fc)), f"w {w} step {step}: decoded crop differs"
            assert torch.equal(coarse, torch.from_numpy(fco)), f"w {w} step {step}: decoded coarse differs"
        checked += n_envs
        legal = oa.mask.astype(bool)
        act = np.array([rng.choice(np.flatnonzero(m)) if m.any() else 0 for m in legal], np.int32)
        oa, _, _ = A.step(act)
        ob, _, _ = B.step(act)
    assert queens_seen > 0, "no row ever showed the enemy queen: the queen planes are untested"
    return checked


def main() -> None:
    files = (sorted((ROOT / "maps-train-official").glob("*.map"))[:6] + sorted((ROOT / "maps-loong").glob("*.map"))[:4]
             + [ROOT / "maps-live/portals.map"])
    maps = bcsim.load_maps([str(f) for f in files if f.exists()])
    sizes = sorted({(m.count("\n")) for m in maps})
    print(f"{len(maps)} maps; crop sizes 1, 15, 27, 63")
    # layout sanity
    for w in (1, 15, 27, 63):
        lay = bcsim.cview_layout(w)
        assert lay["bits"] == (lay["nbits"] * w * w + 7) // 8
        assert lay["coarse"] == lay["bytes"] + (lay["ch"] - lay["nbits"]) * w * w
        assert lay["stride"] % 16 == 0 and lay["stride"] >= lay["coarse"] + lay["ch"] * 256
        assert lay["src"] == list(range(12)) + [18, 19, 12, 13], lay["src"]
    for bad_w in (0, 2, 64, -1):
        try:
            bcsim.cview_layout(bad_w)
        except ValueError:
            pass
        else:
            raise AssertionError(f"crop {bad_w} accepted")
    total = 0
    for w, steps, seed in ((27, 1500, 1), (15, 600, 2), (1, 200, 3), (63, 200, 4)):
        n = play(maps, w, steps, seed)
        total += n
        print(f"  crop {w}: {n:,} rows exact (bytes, scratch path, decode)", flush=True)
    print(f"PASS: {total:,} rows")


if __name__ == "__main__":
    main()
