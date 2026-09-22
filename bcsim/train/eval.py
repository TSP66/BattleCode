"""Behavioural read-out of a checkpoint, for finding failure modes.

Answers the questions the training curves cannot: what is the policy actually
doing with its turns, does it grow, and is it better than picking a legal move
at random.

    python -m train.eval --ckpt runs/v0/latest.pt --games 400
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                    # noqa: E402
from train.net import ActorCritic, policy_out   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root

N_MOVES = 39            # 3 + 9 + 27 relative paths
SPLIT_K = [2, 3, 4, 5, 6, 8, 12, 16, -1]


def action_kind(a: int) -> str:
    if a >= N_MOVES:
        return "split"
    if a < 3:
        return "step"
    if a < 12:
        return "sprint2"
    return "sprint3"


def run(policy, env, steps: int, rng: np.random.Generator | None = None):
    """Plays `steps` dragon turns and gathers behaviour statistics."""
    obs = env.reset()
    n = env.num_envs
    acc = {
        "turns": 0, "kinds": collections.Counter(), "deaths": 0.0, "pearls": 0.0,
        "kills": 0.0, "closed": 0, "len_sum": 0.0, "len_max": 0,
        "legal_sum": 0, "revisit": 0, "visited_checks": 0,
    }
    ep_rows = []
    si = {k: bcsim.SCALARS.index(k) for k in ("length_raw", "head_x", "head_y", "round")}
    # a short memory of where each dragon's head has been, to spot circling
    recent: list[dict[int, collections.deque]] = [collections.defaultdict(
        lambda: collections.deque(maxlen=12)) for _ in range(n)]

    for _ in range(steps):
        acts = policy(obs)
        lengths = obs.scalar[:, si["length_raw"]]
        acc["len_sum"] += float(lengths.sum())
        acc["len_max"] = max(acc["len_max"], int(lengths.max()))
        acc["legal_sum"] += int(obs.mask.sum())
        acc["turns"] += n
        for a in acts:
            acc["kinds"][action_kind(int(a))] += 1
        hx = obs.scalar[:, si["head_x"]]
        hy = obs.scalar[:, si["head_y"]]
        for e in range(n):
            did = int(obs.dragon_id[e])
            cell = (round(float(hx[e]), 4), round(float(hy[e]), 4))
            hist = recent[e][did]
            if len(hist) == hist.maxlen:
                acc["visited_checks"] += 1
                if cell in hist:
                    acc["revisit"] += 1
            hist.append(cell)

        obs, closures, eps = env.step(acts)
        if len(closures.uid):
            c = closures.comps
            acc["closed"] += len(c)
            acc["deaths"] += float(c[:, bcsim.REWARD_COMPS.index("died")].sum())
            acc["pearls"] += float(c[:, bcsim.REWARD_COMPS.index("pearls")].sum())
            acc["kills"] += float(c[:, bcsim.REWARD_COMPS.index("kills")].sum())
        if len(eps.rows):
            ep_rows.append(eps.rows.copy())
    acc["episodes"] = np.concatenate(ep_rows) if ep_rows else np.zeros((0, bcsim.EP_COLS), np.int32)
    return acc


def report(name: str, acc: dict, maps: list[str]) -> None:
    t = acc["turns"]
    rows = acc["episodes"]
    cols = bcsim.EpisodeStats.COLUMNS
    print(f"\n=== {name} ===")
    print(f"  turns                {t:,}")
    print(f"  mean length          {acc['len_sum']/t:6.2f}   (max seen {acc['len_max']})")
    print(f"  pearls / turn        {acc['pearls']/t:6.4f}")
    print(f"  deaths / turn        {acc['deaths']/t:6.4f}")
    print(f"  kills / turn         {acc['kills']/t:6.4f}")
    print(f"  legal actions / turn {acc['legal_sum']/t:6.2f} of {bcsim.N_ACTIONS}")
    if acc["visited_checks"]:
        print(f"  head revisit rate    {acc['revisit']/acc['visited_checks']:6.3f}   "
              f"(share of turns whose cell was seen in the last 12)")
    total = sum(acc["kinds"].values())
    mix = "  ".join(f"{k} {acc['kinds'][k]/total:5.1%}"
                    for k in ("step", "sprint2", "sprint3", "split"))
    print(f"  action mix           {mix}")
    if len(rows):
        rounds = rows[:, cols.index("rounds")]
        winner = rows[:, cols.index("winner")]
        longest = np.maximum(rows[:, cols.index("a_longest")], rows[:, cols.index("b_longest")])
        units = rows[:, cols.index("a_units")] + rows[:, cols.index("b_units")]
        print(f"  games                {len(rows)}")
        print(f"  mean rounds          {rounds.mean():6.1f}   "
              f"({(rounds >= 500).mean():.0%} hit the 500 cap)")
        print(f"  draws                {(winner < 0).mean():6.1%}")
        print(f"  longest at end       {longest.mean():6.2f}")
        print(f"  dragons alive at end {units.mean():6.2f}")
        by_map = collections.defaultdict(list)
        for r in rows:
            by_map[int(r[cols.index("map")])].append(r)
        print("  per map:")
        for mi in sorted(by_map):
            g = np.array(by_map[mi])
            lg = np.maximum(g[:, cols.index("a_longest")], g[:, cols.index("b_longest")])
            print(f"    map {mi:2d}  games {len(g):4d}  rounds {g[:, cols.index('rounds')].mean():6.1f}"
                  f"  longest {lg.mean():6.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "runs/v0/latest.pt"))
    p.add_argument("--maps", default=str(ROOT / "maps"))
    p.add_argument("--envs", type=int, default=256)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--baseline", action="store_true", help="also run a random legal policy")
    args = p.parse_args()

    dev = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    a = ck["args"]
    net = ActorCritic(bcsim.N_CHANNELS, bcsim.N_SCALARS, bcsim.N_ACTIONS,
                      width=a["width"], blocks=a["blocks"],
                      hidden=next(v for k, v in ck["net"].items() if k.endswith("fuse.0.weight")).shape[0]).to(dev)
    state = {k.replace("_orig_mod.", ""): v for k, v in ck["net"].items()}
    net.load_state_dict(state)
    net.eval()
    print(f"checkpoint iter {ck['iter']}, width {a['width']} blocks {a['blocks']}, "
          f"{'greedy' if args.greedy else 'sampled'}")

    maps = bcsim.load_maps(args.maps)
    env = bcsim.BattlecodeVecEnv(maps, num_envs=args.envs, num_threads=16, seed=7,
                                 closure_capacity=max(8192, args.envs * 160))

    def learned(obs):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            local = torch.from_numpy(obs.local).to(dev)
            scalar = torch.from_numpy(obs.scalar).to(dev)
            mask = torch.from_numpy(obs.mask).to(dev).bool()
            logits, _ = net(local, scalar)
            if args.greedy:
                from train.net import masked_logits
                act = masked_logits(logits.float(), mask).argmax(dim=1)
            else:
                act, _, _ = policy_out(logits, mask)
        return act.to(torch.int32).cpu().numpy()

    report(f"learned ({'greedy' if args.greedy else 'sampled'})",
           run(learned, env, args.steps), maps)

    if args.baseline:
        rng = np.random.default_rng(0)

        def random_legal(obs):
            m = obs.mask.astype(np.float32)
            m[m.sum(1) == 0] = 1.0
            c = m.cumsum(1)
            r = rng.random(len(m)) * c[:, -1]
            return (c < r[:, None]).sum(1).astype(np.int32)

        env2 = bcsim.BattlecodeVecEnv(maps, num_envs=args.envs, num_threads=16, seed=7,
                                      closure_capacity=max(8192, args.envs * 160))
        report("random legal", run(random_legal, env2, args.steps), maps)


if __name__ == "__main__":
    main()
