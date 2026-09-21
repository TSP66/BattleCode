"""Throughput of the simulator alone: dragon turns per second, no network."""
import pathlib, sys, time
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import bcsim

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root


def bench(num_envs, threads, steps=400, maps=None):
    env = bcsim.BattlecodeVecEnv(maps, num_envs=num_envs, num_threads=threads, seed=0)
    obs = env.reset()
    rng = np.random.default_rng(0)
    acts = [rng.integers(0, bcsim.N_ACTIONS, size=num_envs).astype(np.int32) for _ in range(16)]
    for i in range(20):
        env.step(acts[i % 16])
    t0 = time.perf_counter()
    for i in range(steps):
        env.step(acts[i % 16])
    dt = time.perf_counter() - t0
    env.close()
    return num_envs * steps / dt


if __name__ == "__main__":
    maps = bcsim.load_maps(str(ROOT / "maps"))
    big = bcsim.load_maps(str(ROOT / "maps/big_empty.map"))
    print(f"{'envs':>6} {'threads':>8} {'turns/sec':>14}")
    for envs, threads in [(64, 1), (256, 1), (256, 4), (1024, 8), (1024, 16), (4096, 16)]:
        rate = bench(envs, threads, maps=maps)
        print(f"{envs:6d} {threads:8d} {rate:14,.0f}")
    print("\n64x64 map only (worst case for pearl ticks):")
    for envs, threads in [(256, 1), (1024, 16)]:
        print(f"{envs:6d} {threads:8d} {bench(envs, threads, maps=big):14,.0f}")
