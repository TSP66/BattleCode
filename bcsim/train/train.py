"""PPO self-play for Battlecode.

One policy controls every dragon on both teams. Rewards are already written
from each dragon's own point of view, so self-play needs no sign flipping: the
same network supplies A's moves and B's, and learns from both.

    python -m train.train --envs 1024 --steps 128 --width 128 --blocks 6
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                    # noqa: E402
from train import augment                       # noqa: E402
from train.net import ActorCritic, policy_out   # noqa: E402
from train.rollout import Rollout               # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root

# Reward spec v1, after v0 trained a policy that shredded itself into ~34 tiny
# dragons per team. v0 paid split_cost +0.05 to cancel the parent's length loss,
# which made splitting free, and paid win +2 to every survivor, which made extra
# survivors worth having for their own sake. The game is won by the *longest*
# dragon, and v0's reward never mentioned length concentration at all.
#
# v1 is written over team state instead of the dragon's own body. team_len,
# team_max, foe_len and foe_max are differences of a team potential taken
# between one agent's consecutive turns, so they telescope and leave the optimal
# policy alone while paying every dragon for what the team did while it was
# away. team_max is the win condition priced directly: growing the leader pays
# the whole team, and splitting the leader bills the whole team.
REWARD_V1 = {
    # this dragon's own turn: dense and low variance
    "length_delta": 0.03,      # prices growth, sprint steps and death-by-length
    # team potentials, the reason a sacrifice can now pay
    "team_len": 0.01,
    "team_max": 0.06,          # concentration
    "foe_len": -0.005,
    "foe_max": -0.04,          # also makes kills scale with the victim's size
    # splitting now costs something: the parent keeps its length loss
    # (split_cost stays 0) and every split pays a flat fee on top
    "splits": -0.25,
    "died": -1.0,
    "kills": 0.5,
    # halved from v0: the terminal result reaches survivors only, so a large
    # weight quietly rewards having many survivors, which is what went wrong
    "win": 1.0,
    "lose": -1.0,
}

# v2: v1 with kills worth half as much again, to see whether a policy that is
# paid more to kill plays more aggressively. Everything else is unchanged so
# the two runs can be compared.
REWARD_V2 = {**REWARD_V1, "kills": 0.75}

# v3, after v2 (runs/anchors/v2_64x4_511M.pt):
#  * zero-sum team terms: foe weights mirror team weights, so the shaping is
#    d(us - them). v1/v2 paid a gain in our length about twice what they
#    charged for the same gain by the enemy, so an even trade scored as a loss
#    and a sacrifice had to take out twice its size to pay at all. Now a
#    sacrifice is paid exactly (victim's length - own length).
#  * the potentials are discounted, gamma * phi(s') - phi(s) (the env's
#    potential_gamma is set to the PPO gamma), and phi is kept, not zeroed,
#    at a death: a dead dragon keeps the team position it left behind.
#  * died -1 -> -0.25: a death was charged four times (died, length_delta,
#    team_len, team_max); the other three already price it.
#  * draw is written out as 0 so the three results are all stated.
#  * kills are paid only to a killer that survives (env rule). The first v3
#    launch paid both dragons in a head-on: +0.75 kill - 0.25 died = +0.5
#    each, and self-play converged on mutual kamikaze within 30 iterations
#    (90% draws, ~2 deaths and ~2 kills per team per game). A head-on is now
#    judged by the trade alone, through the zero-sum team terms.
REWARD_V3 = {
    "length_delta": 0.03,
    "team_len": 0.01,
    "foe_len": -0.01,
    "team_max": 0.06,
    "foe_max": -0.06,
    "splits": -0.25,
    "died": -0.25,
    "kills": 0.75,
    "win": 1.0,
    "lose": -1.0,
    "draw": 0.0,
}

# v4: v3 with the split fee cut from -0.25 to -0.05, because the policies
# were not splitting enough. A split still carries the parent's length_delta
# (-0.03 per segment handed over) and, when it splits the leader, the drop in
# team_max; it is not free, which is what exploded in v0.
REWARD_V4 = {**REWARD_V3, "splits": -0.05}

# v5: v4 with no split fee at all. A split already pays the parent's
# length_delta (-0.03 per segment handed over) and, when it splits the leader,
# the drop in team_max, so the flat fee was a third charge for one event.
# died -0.25 -> -0.2; an even head-on still nets -0.2 - 0.03 * length per
# dragon, so mutual kamikaze stays a loss.
REWARD_V5 = {**REWARD_V4, "splits": 0.0, "died": -0.2}

# v6: v5 against mobbing. Opponents that split early and swarm our 1-5
# dragons with small heads were winning by elimination, because a head-on
# kills both dragons whatever their size.
#  * eliminated: the result was paid only to survivors, so a wiped-out team
#    was never charged `lose` and elimination cost less than losing on length
#    at round 500. Charged to the dragons that die on the wiping turn (not on
#    a mutual wipe-out, which is a draw).
#  * team_units / foe_units: zero-sum potentials over log(1 + living units),
#    pricing how exposed a small team is to elimination. Concave so it cannot
#    rebuild v0's shredding: 1 -> 2 dragons pays +0.12, 20 -> 21 pays +0.014,
#    while halving a length-20 leader costs ~-0.6 through team_max.
REWARD_V6 = {**REWARD_V5, "eliminated": -1.0, "team_units": 0.3, "foe_units": -0.3}

# v7 (planned, not implemented): v6 plus a zero-sum exposure potential,
# phi = -0.02 * sum over our dragons of max(0, our_len - their_len) for enemy
# heads within Manhattan 2 of our head (mirrored for theirs). Teaches the
# leader to keep away from small heads before the trade rather than after,
# and pays the shared policy to mob, so self-play produces the opponent it
# has to learn to defend against. Kept separate from v6 to attribute effects.

# v8: ONE bounded zero-sum team potential plus the true result, and nothing else.
# See REWARDS.md for the derivation and for every measurement behind it.
#
# There are no per-dragon terms at all -- no died, no pearls, no kills, no
# length. Every dragon on the team receives the identical reward stream, which is
# what makes a sacrifice learnable, and the constraint is that no dragon of ours
# may ever be rewarded at the expense of another of ours. Dragons compete for
# ground (a tile a teammate covered is used up) and never for reward.
#
# The five v8_* components arrive from the engine ALREADY scaled by
# kappa * lambda_i(t) / sum(lambda(t)), so their weights here are 1.0 and are not
# knobs -- the mix lives in the lambda schedules in cpp/bc_reward8.hpp and the
# strength lives in kappa (--v8-kappa). `outcome` is the only non-telescoping
# term and the only one that decides what the optimal policy is.
#
# Expect early learning to be SLOWER than v6's: with no per-dragon term a pearl
# that paid +0.03 through length_delta now pays about 2/T of a tanh diluted across
# the whole team, call it 100x weaker. That is the credit-assignment cost of a
# team reward and it is meant to come out of the critic (a counterfactual baseline
# conditioned on the acting dragon), not out of the reward.
REWARD_V8 = {
    "v8_win": 1.0, "v8_len": 1.0, "v8_top3": 1.0, "v8_kill": 1.0, "v8_exp": 1.0,
    "outcome": 1.0,
}

REWARDS = {"v1": REWARD_V1, "v2": REWARD_V2, "v3": REWARD_V3, "v4": REWARD_V4,
           "v5": REWARD_V5, "v6": REWARD_V6, "v8": REWARD_V8}
# v1/v2 used plain differences in the team potentials
POTENTIAL_DISCOUNT = {"v1": False, "v2": False, "v3": True, "v4": True, "v5": True,
                      "v6": True, "v8": True}


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--maps", default=str(ROOT / "maps"))
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--blocks", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--reward", default="v4", choices=sorted(REWARDS))
    p.add_argument("--v8-kappa", type=float, default=1.0,
                   help="reward v8 only: shaping strength against the terminal result. "
                        "The one knob -- it rescales every term at once and cannot "
                        "change the mix, so it is the first thing to sweep")
    p.add_argument("--sonar", action="store_true",
                   help="broadcast a sonar in all four directions every turn and feed "
                        "the five echo counts to the policy (scalars 708-712, zero "
                        "without this). Sonar is sensing, not an action: it costs no "
                        "turn and 0.51M judge points, and the simulator is "
                        "byte-identical to the engine (SONAR.md). It changes what the "
                        "OPPONENT sees too, through num_msgs, so a league built "
                        "without it is not strictly comparable")
    p.add_argument("--ent", type=float, default=0.01, help="entropy bonus at the start")
    p.add_argument("--ent-end", type=float, default=0.003, help="floor it decays toward")
    # v2 halved every 500M and its entropy had collapsed (~0.32) by 500M turns
    p.add_argument("--ent-half-life", type=float, default=1.5e9,
                   help="turns for the bonus to halve its distance to the floor")
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=16384)
    p.add_argument("--iters", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "runs/selfplay"))
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", default="", help="checkpoint to continue from (weights, "
                   "optimiser, turn count); the run's other flags come from the command line")
    # map mix: every base map plus augmented variants, smaller maps more often
    p.add_argument("--aug-per-map", type=int, default=48, help="0 = original maps only")
    p.add_argument("--aug-original-share", type=float, default=0.25)
    p.add_argument("--size-alpha", type=float, default=0.75,
                   help="map weight = (256 / area) ** alpha; 0 = uniform")
    # small maps are where a young policy learns fastest, but staying biased
    # toward them trains for the wrong mix, so alpha falls linearly to 0
    # (every base map equally likely) over this many turns
    p.add_argument("--size-alpha-decay", type=float, default=2e9,
                   help="turns for alpha to reach 0; 0 = keep --size-alpha fixed")
    p.add_argument("--aug-refresh", type=int, default=0,
                   help="rebuild the variants every this many iterations (0 = never); "
                        "restarts every game in progress")
    p.add_argument("--snapshot-every", type=float, default=12.5e6,
                   help="turns between frozen snapshots, which train.yardstick evaluates")
    return p.parse_args()


def main() -> None:
    a = parse()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_file = (out / "log.jsonl").open("a")

    ck = torch.load(a.resume, map_location=dev, weights_only=False) if a.resume else None
    start_iter = ck["iter"] + 1 if ck else 0
    turns_per_iter = a.envs * a.steps
    # older checkpoints predate total_turns, but their runs had a fixed batch
    base_turns = ck.get("total_turns", (ck["iter"] + 1) * ck["args"]["envs"] * ck["args"]["steps"]) \
        if ck else 0
    # the entropy schedule is anchored where the run (or its first resume) began
    ent_t0 = ck["args"].get("ent_t0", base_turns) if ck else 0
    a.ent_t0 = ent_t0

    def make_env(generation: int):
        seed = a.seed + 7919 * start_iter + 104729 * generation
        texts, map_w, names, areas = augment.build_pool(a.maps, a.aug_per_map, seed,
                                                        a.size_alpha, a.aug_original_share)
        e = bcsim.BattlecodeVecEnv(texts, num_envs=a.envs, num_threads=a.threads,
                                   seed=seed, closure_capacity=max(8192, a.envs * 160),
                                   sonar=a.sonar)
        e.set_map_weights(map_w)
        if POTENTIAL_DISCOUNT[a.reward]:
            e.set_potential_gamma(a.gamma)
        if a.reward == "v8":
            e.set_reward_v8(True, a.v8_kappa)
        return e, names, map_w, areas

    def size_alpha(total: int) -> float:
        if a.size_alpha_decay <= 0:
            return a.size_alpha
        return a.size_alpha * max(0.0, 1.0 - total / a.size_alpha_decay)

    env, map_base, map_w, map_area = make_env(0)
    base_names = sorted(set(map_base))
    share = {b: float(map_w[[n == b for n in map_base]].sum() / map_w.sum()) for b in base_names}
    print(f"{len(map_base)} maps in the pool; sampling share by base map: " +
          ", ".join(f"{b} {v:.0%}" for b, v in share.items()), flush=True)
    weights = bcsim.reward_vector(REWARDS[a.reward])

    net = ActorCritic(bcsim.N_CHANNELS, bcsim.N_SCALARS, bcsim.N_ACTIONS,
                      width=a.width, blocks=a.blocks).to(dev).to(memory_format=torch.channels_last)
    if ck:
        net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["net"].items()})
    raw_net = net
    if a.compile:
        net = torch.compile(net)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.0, eps=1e-5, fused=True)
    if ck:
        opt.load_state_dict(ck["opt"])
        print(f"resumed {a.resume} at iter {ck['iter']}, {base_turns / 1e6:.1f}M turns", flush=True)
        del ck
    snap_dir = out / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    next_snap = (base_turns // int(a.snapshot_every) + 1) * int(a.snapshot_every)
    params = sum(p.numel() for p in net.parameters())
    print(f"{params/1e6:.2f}M params | {a.envs} envs x {a.steps} steps "
          f"= {a.envs*a.steps:,} dragon turns per iteration", flush=True)

    roll = Rollout(a.steps, a.envs, bcsim.N_CHANNELS, bcsim.WINDOW, bcsim.N_SCALARS,
                   bcsim.N_ACTIONS, dev)
    obs = env.reset()
    hist: list[dict] = []
    t_start = time.perf_counter()

    def ent_coef(total: int) -> float:
        k = 0.5 ** (max(total - ent_t0, 0) / a.ent_half_life)
        return a.ent_end + (a.ent - a.ent_end) * k

    for it in range(start_iter, start_iter + a.iters):
        done_turns = base_turns + (it - start_iter) * turns_per_iter
        ent_w = ent_coef(done_turns)
        if a.aug_refresh and it > start_iter and (it - start_iter) % a.aug_refresh == 0:
            env.close()
            env, map_base, map_w, map_area = make_env((it - start_iter) // a.aug_refresh)
            obs = env.reset()
        alpha = size_alpha(done_turns)
        # takes effect from each env's next episode
        env.set_map_weights(augment.reweight(map_w, map_area, a.size_alpha, alpha))
        net.eval()
        roll.begin()
        ep_rows, t_env, t_fwd = [], 0.0, 0.0
        t0 = time.perf_counter()
        for t in range(a.steps):
            ta = time.perf_counter()
            staged = roll.stage(obs)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, value = net(staged[0], staged[1])
                action, logp, _ = policy_out(logits, staged[2])
            acts = action.to(torch.int32).cpu().numpy()
            t_fwd += time.perf_counter() - ta

            ta = time.perf_counter()
            roll.record(t, staged, obs, action, logp, value.float())
            obs, closures, eps = env.step(acts)
            roll.close(closures, weights)
            if len(eps.rows):
                ep_rows.append(eps.rows.copy())
            t_env += time.perf_counter() - ta
        t_roll = time.perf_counter() - t0

        adv, ret, valid = roll.finish(a.gamma, a.lam)
        batch = roll.flat_batch(adv, ret, valid)
        n = batch["action"].numel()
        if n == 0:
            continue
        adv_b = batch["adv"]
        batch["adv"] = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
        # how much of the return the value head actually explains: 1 is perfect,
        # 0 is no better than predicting the mean, negative is worse than that
        ret_var = batch["ret"].var()
        ev = 1.0 - (batch["ret"] - batch["value"]).var() / (ret_var + 1e-8)
        ev = float(ev.item())
        split_frac = float((batch["action"] >= bcsim.N_ACTIONS - 9).float().mean().item())
        sprint_frac = float(((batch["action"] >= 3) & (batch["action"] < 39)).float().mean().item())

        net.train()
        t0 = time.perf_counter()
        stats = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0}
        nb = 0
        for _ in range(a.epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, a.minibatch):
                idx = perm[s:s + a.minibatch]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, value = net(batch["local"][idx], batch["scalar"][idx])
                _, logp, ent = policy_out(logits, batch["mask"][idx], batch["action"][idx])
                ratio = (logp - batch["logp"][idx]).exp()
                mb_adv = batch["adv"][idx]
                pg = -torch.min(ratio * mb_adv,
                                ratio.clamp(1 - a.clip, 1 + a.clip) * mb_adv).mean()
                v_pred = value.float()
                v_old = batch["value"][idx]
                v_clip = v_old + (v_pred - v_old).clamp(-a.clip, a.clip)
                v_loss = 0.5 * torch.max((v_pred - batch["ret"][idx]) ** 2,
                                         (v_clip - batch["ret"][idx]) ** 2).mean()
                ent_m = ent.mean()
                loss = pg + a.vf * v_loss - ent_w * ent_m
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                with torch.no_grad():
                    stats["pg"] += pg.item()
                    stats["v"] += v_loss.item()
                    stats["ent"] += ent_m.item()
                    stats["kl"] += (batch["logp"][idx] - logp).mean().item()
                    stats["clipfrac"] += ((ratio - 1).abs() > a.clip).float().mean().item()
                nb += 1
        t_opt = time.perf_counter() - t0
        for k in stats:
            stats[k] /= max(nb, 1)

        turns = a.envs * a.steps
        row = {
            "iter": it,
            "turns": turns,
            "total_turns": done_turns + turns,
            "ent_coef": round(ent_w, 6),
            "size_alpha": round(alpha, 4),
            "sps": turns / (t_roll + t_opt),
            "roll_sps": turns / t_roll,
            "t_env": round(t_env, 2), "t_fwd": round(t_fwd, 2), "t_opt": round(t_opt, 2),
            "usable": round(n / turns, 3),
            "orphans": roll.orphans, "overwrites": roll.overwrites,
            "reward_mean": float(roll.reward[valid].mean().item()),
            "return_mean": float(batch["ret"].mean().item()),
            "value_mean": float(batch["value"].mean().item()),
            "explained_var": float(ev),
            "elapsed": round(time.perf_counter() - t_start, 1),
            "split_frac": round(split_frac, 4),
            "sprint_frac": round(sprint_frac, 4),
            **{k: round(v, 4) for k, v in stats.items()},
        }
        if roll.comp_total is not None and roll.n_closed:
            per = roll.comp_total / roll.n_closed
            row.update({f"r_{name}": round(float(per[i]), 5)
                        for i, name in enumerate(bcsim.REWARD_COMPS)})
        if ep_rows:
            rows = np.concatenate(ep_rows)
            cols = bcsim.EpisodeStats.COLUMNS
            winner = rows[:, cols.index("winner")]
            row["episodes"] = len(rows)
            row["rounds_mean"] = float(rows[:, cols.index("rounds")].mean())
            row["draw_rate"] = float((winner < 0).mean())
            row["longest"] = float(np.maximum(rows[:, cols.index("a_longest")],
                                              rows[:, cols.index("b_longest")]).mean())
            row["units_mean"] = float((rows[:, cols.index("a_units")] +
                                       rows[:, cols.index("b_units")]).mean() / 2)
            # segments in the leader per dragon on the board: the number v0 got
            # backwards, since the game is won by the longest dragon, not by
            # having the most of them
            row["conc"] = row["longest"] / max(row["units_mean"], 1e-6)
            # per team per game, so it reads the same whatever the map mix
            row["kills_per_game"] = float((rows[:, cols.index("a_kills")] +
                                           rows[:, cols.index("b_kills")]).mean() / 2)
            row["deaths_per_game"] = float((rows[:, cols.index("a_deaths")] +
                                            rows[:, cols.index("b_deaths")]).mean() / 2)
            # trades: segments a team's kills removed vs segments it lost, per
            # team per game, and whether the side that traded better won
            killed = rows[:, [cols.index("a_len_killed"), cols.index("b_len_killed")]]
            lost = rows[:, [cols.index("a_len_lost"), cols.index("b_len_lost")]]
            row["len_killed_per_game"] = float(killed.mean())
            row["len_lost_per_game"] = float(lost.mean())
            margin = (killed[:, 0] - lost[:, 0]) - (killed[:, 1] - lost[:, 1])
            decided = (margin != 0) & (winner >= 0)
            if decided.any():
                better = np.where(margin[decided] > 0, 0, 1)
                row["trade_win_rate"] = float((better == winner[decided]).mean())
            row["headon_per_game"] = float((rows[:, cols.index("a_headon")] +
                                            rows[:, cols.index("b_headon")]).mean() / 2)
            small = [map_base[int(m)] in ("small", "arena", "Colloseum", "default_small")
                     for m in rows[:, cols.index("map")]]
            row["small_map_frac"] = float(np.mean(small))
        hist.append(row)
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()

        if it % 5 == 0 or it < 3:
            print(f"it {it:5d} | {row['sps']:>9,.0f} turns/s | "
                  f"ep {row.get('episodes', 0):4d} rounds {row.get('rounds_mean', 0):6.1f} "
                  f"longest {row.get('longest', 0):5.1f} | "
                  f"ent {stats['ent']:.3f} kl {stats['kl']:+.4f} v {stats['v']:.3f} | "
                  f"usable {row['usable']:.2f}", flush=True)
        if roll.overwrites:
            raise RuntimeError(f"slot reused before closing ({roll.overwrites}): "
                               "the transition bookkeeping is wrong, stopping")
        total = done_turns + turns
        if it % 50 == 0:
            tmp = out / "latest.pt.tmp"
            torch.save({"net": raw_net.state_dict(), "opt": opt.state_dict(), "iter": it,
                        "total_turns": total, "args": vars(a)}, tmp)
            tmp.replace(out / "latest.pt")      # never leave a half-written file
        if total >= next_snap:
            torch.save({"net": raw_net.state_dict(), "iter": it, "total_turns": total,
                        "args": vars(a)}, snap_dir / f"turns_{total}.pt")
            next_snap = (total // int(a.snapshot_every) + 1) * int(a.snapshot_every)


if __name__ == "__main__":
    main()
