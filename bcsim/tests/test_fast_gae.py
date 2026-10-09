"""fast_gae.gae (numba) == the trainer's Python loop, bit for bit, on rollout-shaped random data."""
import sys, time, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from train.fast_gae import GAE, gae_py


def rollout(rng, T, N, end_p):
    """Trainer-shaped links: per env, teams act in some order, rounds rise, games end."""
    nx = np.full((T, N), -1, np.int64); er = np.full((T, N), -1, np.int64)
    team = rng.integers(0, 2, (T, N)).astype(np.int8)
    rnd = np.zeros((T, N), np.int32); ew = np.full((T, N), -2, np.int8)
    is_l = rng.random((T, N)) < 0.8
    for e in range(N):
        last = [-1, -1]; r = int(rng.integers(0, 500))
        for t in range(T):
            r += int(rng.integers(0, 3)) if rng.random() < 0.97 else int(rng.integers(3, 60))
            rnd[t, e] = r; row = t * N + e; tm = team[t, e]
            if last[tm] >= 0:
                nx.flat[last[tm]] = row
            last[tm] = row
            if rng.random() < end_p:                     # the game ends at this row
                ew.flat[row] = rng.choice([-1, 0, 1])
                for x in (0, 1):
                    if last[x] >= 0:
                        er.flat[last[x]] = row
                last = [-1, -1]; r = 0
    ph = (rng.standard_normal((T, N)) * rng.choice([0.01, 1, 30], (T, N))).astype(np.float32)
    v = (rng.standard_normal((T, N)) * 3).astype(np.float32)
    vw = np.tanh(rng.standard_normal((T, N))).astype(np.float32)
    return nx, er, team, rnd, ew, is_l, ph, v, vw


def same(a, b):
    return a.dtype == b.dtype and np.array_equal(a.view(np.uint32) if a.dtype == np.float32 else a,
                                                 b.view(np.uint32) if b.dtype == np.float32 else b)


rng = np.random.default_rng(0)
cases = [(16, 64, 0.05, 0.8, 0.95, 0.98)] * 20 + [(64, 256, 0.01, 0.8, 0.95, 0.98)] * 5 + [(32, 128, 0.2, 0.9, 0.9, 0.97)] * 5
bad = 0
for k, (T, N, p, al, la, wla) in enumerate(cases):
    nx, er, tm, rf, ew, il, ph, v, vw = rollout(rng, T, N, p)
    args = [x.reshape(-1) for x in (nx, er, il, ph, tm, rf, v)]
    rows_l = np.flatnonzero(args[2])
    for wl in (False, True):
        ref = gae_py(rows_l, *args, al, la, wl, vw.reshape(-1), ew.reshape(-1), wla)
        got = GAE(al, la, wla)(rows_l, *args, wl, vw.reshape(-1), ew.reshape(-1))
        ok = all(same(x, y) for x, y in zip(ref[:3], got[:3])) and (not wl or same(ref[3], got[3]))
        bad += not ok
        if not ok:
            print("MISMATCH case", k, "wl", wl, [np.abs(x.astype(np.float64) - y).max() for x, y in zip(ref[:3], got[:3])])
print(f"{len(cases) * 2} cases, {bad} mismatches")
# a full-size rollout (128 x 2048), timed
nx, er, tm, rf, ew, il, ph, v, vw = rollout(rng, 128, 2048, 0.003)
args = [x.reshape(-1) for x in (nx, er, il, ph, tm, rf, v)]
rows_l = np.flatnonzero(args[2]); g = GAE(0.8, 0.95, 0.98)
t0 = time.perf_counter(); ref = gae_py(rows_l, *args, 0.8, 0.95, True, vw.reshape(-1), ew.reshape(-1), 0.98); t1 = time.perf_counter()
got = g(rows_l, *args, True, vw.reshape(-1), ew.reshape(-1)); t2 = time.perf_counter()
print(f"full size: {len(rows_l):,} rows, python {t1 - t0:.3f}s, numba {(t2 - t1) * 1e3:.1f}ms, identical "
      f"{all(same(x, y) for x, y in zip(ref, got))}, usable {ref[1].mean():.3f}")
sys.exit(1 if bad else 0)
