"""One training segment of the ratchet (see ratchet.py).

finetune_team.py cut down to what the ft1-ft7 post-mortem left standing:

  * the result is the only reward: win +1 / draw 0 / loss -1 for the TEAM,
    reaching every dragon through the team's chain of turns (no per-dragon
    shaping, no self head, no alpha mix);
  * the team value comes from a FROZEN copy of the replay-pretrained critic
    (runs/team_critic/pretrained.pt), so it cannot drift (ft5, ft6). Its
    calibration against real results is still logged, as the staleness check;
  * the KL teacher is the current anchor (the policy this segment started
    from), not the original clone, so the leash moves with each promotion;
  * frozen opponents are weighted (--opp-weights, set by the supervisor from
    the anchor's gate scores) and sample their moves instead of playing argmax;
  * no warm-up: the critic is pretrained and never trains;
  * it stops after --turns and always writes final.pt (with optimiser state,
    so a segment can be extended); latest.pt every 10 iterations for crash
    resumes;
  * it aborts with exit code 3 on a non-finite loss or parameter, KL to the
    teacher above --max-kl, or entropy below --min-ent: the supervisor then
    discards the candidate rather than gate a broken one.

The logged total_turns is the experiment-wide counter (--turn-base), so the
dashboard's x axis runs on across segments.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# the critic's global features need the privileged build, chosen before bcsim loads
os.environ.setdefault("BCSIM_LIB", str(pathlib.Path(__file__).resolve().parents[1]
                                       / "bcsim" / "libbcvec_priv.so"))

import bcsim                                    # noqa: E402
from train import augment, team_critic          # noqa: E402
from train.finetune import load_policy          # noqa: E402
from train.net import masked_logits, policy_out  # noqa: E402
from train.rollout import Rollout               # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
N_PRIV = team_critic.N_PRIV
ABORT = 3                                       # exit code: the candidate is broken


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="policy to start from (the anchor, or a "
                                                 "candidate's final.pt when extending)")
    p.add_argument("--teacher", required=True, help="KL teacher: the current anchor")
    p.add_argument("--continue", dest="cont", action="store_true",
                   help="--init is this candidate's own checkpoint: keep its optimiser "
                        "state and turn count (an extension or a crash resume)")
    p.add_argument("--critic", default=str(ROOT / "runs/team_critic/pretrained.pt"))
    p.add_argument("--opponents", default="", help="frozen networks, comma separated")
    p.add_argument("--opp-names", default="", help="their display names, comma separated")
    p.add_argument("--opp-weights", default="", help="relative shares, comma separated")
    p.add_argument("--self-frac", type=float, default=0.2)
    p.add_argument("--out", required=True, help="the candidate's directory")
    p.add_argument("--log", required=True, help="experiment-wide log.jsonl to append to")
    p.add_argument("--turns", type=float, default=50e6, help="this segment's budget")
    p.add_argument("--turn-base", type=int, default=0, help="experiment turns before it")
    p.add_argument("--gen", type=int, default=0)
    p.add_argument("--segment", type=int, default=0)
    p.add_argument("--maps", default=str(ROOT / "runs/ft3/maps"))
    p.add_argument("--live-maps", default="",
                   help="directory of the maps the ladder is played on; with --live-share, "
                        "they take that share of sampling and every other map in --maps shares "
                        "the rest. Empty (the default) weights every base map equally.")
    p.add_argument("--live-share", type=float, default=0.0,
                   help="0 = off. 0.6 keeps the live maps at 60%% of training when --maps holds "
                        "invented maps as well (see MAPS_PROPOSAL.md)")
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--team-lam", type=float, default=0.9, help="TD lambda per ROUND")
    p.add_argument("--kl-coef", type=float, default=0.5)
    p.add_argument("--ent", type=float, default=0.001)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--max-kl", type=float, default=0.1, help="abort above this KL to teacher")
    p.add_argument("--min-ent", type=float, default=0.1, help="abort below this entropy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aug-per-map", type=int, default=48)
    p.add_argument("--aug-original-share", type=float, default=0.25)
    p.add_argument("--calib-share", type=float, default=1 / 32)
    p.add_argument("--explore", type=float, default=0.0,
                   help="share of the learner's sampling spread uniformly over legal "
                        "actions (see the note at the sampling site)")
    return p.parse_args()


def frozen(path: str, dev):
    net, _ = load_policy(path, dev)
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    return net


def main() -> None:
    a = parse()
    if not hasattr(bcsim.env._lib, "bcv_bind_priv"):
        raise SystemExit("needs libbcvec_priv.so (make priv)")
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.backends.cudnn.benchmark = True
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_file = open(a.log, "a")

    # ---- networks
    ck = torch.load(a.init, map_location=dev, weights_only=False)
    policy, _ = load_policy(a.init, dev)
    width, blocks = ck["args"]["width"], ck["args"]["blocks"]
    popt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0, eps=1e-5)
    done0 = 0                                    # turns this candidate already trained
    if a.cont and "opt" in ck and ck.get("ratchet"):
        popt.load_state_dict(ck["opt"])
        for g in popt.param_groups:
            g["lr"] = a.lr
        done0 = int(ck.get("cand_turns", 0))
    del ck
    teacher = frozen(a.teacher, dev)
    opp_paths = [x for x in a.opponents.split(",") if x]
    opp_names = [x for x in a.opp_names.split(",") if x] or [pathlib.Path(x).stem for x in opp_paths]
    opps = [frozen(x, dev) for x in opp_paths]
    # pretrained on replays with no opponent slots: its context input is unused
    tcrit, _ = team_critic.load(a.critic, dev, n_context=1)
    tcrit.eval()
    for q in tcrit.parameters():
        q.requires_grad_(False)

    opp_p = None
    if opps:
        wts = [float(x) for x in a.opp_weights.split(",")] if a.opp_weights else [1.0] * len(opps)
        if len(wts) != len(opps) or len(opp_names) != len(opps):
            raise SystemExit("--opp-weights/--opp-names need one entry per opponent")
        opp_p = np.array(wts) / sum(wts)
    print(f"gen {a.gen} segment {a.segment}: policy {width}x{blocks} from {a.init}; "
          f"teacher {a.teacher} (KL {a.kl_coef}); lr {a.lr}; self-play {a.self_frac}; "
          "opponents " + ", ".join(f"{n} {w:.2f}" for n, w in zip(opp_names, opp_p if opps else [])),
          flush=True)

    # ---- env
    texts, map_w, map_names, _ = augment.build_pool(a.maps, a.aug_per_map, a.seed, 0.0,
                                                    a.aug_original_share)
    if a.live_maps and a.live_share > 0:
        live = {f.stem for f in pathlib.Path(a.live_maps).glob("*.map")}
        map_w = augment.set_group_share(map_w, map_names, live, a.live_share)
        held = sorted(set(map_names) - live)
        print(f"map sampling: {len(live & set(map_names))} live maps at {a.live_share:.0%}, "
              f"{len(held)} others at {1 - a.live_share:.0%} ({', '.join(held)})", flush=True)
    env = bcsim.BattlecodeVecEnv(texts, num_envs=a.envs, num_threads=a.threads, seed=a.seed,
                                 closure_capacity=max(8192, a.envs * 160), privileged=True)
    env.set_map_weights(map_w)
    no_reward = bcsim.reward_vector({})         # closures only keep the chains honest

    N, T = a.envs, a.steps
    slot = np.zeros(N, np.int64)                 # 0 self-play, k + 1 = opponent k
    learner = np.full(N, -1, np.int8)            # side the learner plays, -1 both

    def assign(envs: np.ndarray) -> None:
        if not len(envs):
            return
        fz = rng.random(len(envs)) >= a.self_frac if opps else np.zeros(len(envs), bool)
        pick = rng.choice(len(opps), len(envs), p=opp_p) if opps else np.zeros(len(envs), np.int64)
        slot[envs] = np.where(fz, 1 + pick, 0)
        learner[envs] = np.where(fz, rng.integers(0, 2, len(envs)), -1)

    assign(np.arange(N))
    zero_ctx = torch.zeros(1, 1, device=dev)
    outcome_u = torch.tensor(team_critic.OUTCOME_VALUE, device=dev)
    roll = Rollout(T, N, bcsim.N_CHANNELS, bcsim.WINDOW, bcsim.N_SCALARS, bcsim.N_ACTIONS, dev)
    priv_buf = torch.zeros(T, N, N_PRIV, device=dev)
    probs = torch.zeros(T, N, 3, device=dev)
    zero_v = torch.zeros(N, device=dev)
    nxt_team = np.full((T, N), -1, np.int64)
    term_cls = np.full((T, N), -1, np.int64)
    rnd_cpu = np.zeros((T, N), np.int64)
    team_cpu = np.zeros((T, N), np.int8)
    epid_cpu = np.zeros((T, N), np.int64)
    learn_cpu = np.zeros((T, N), bool)
    last = np.full((2, N), -1, np.int64)
    envs_ix = np.arange(N)
    pend = {"key": np.zeros(0, np.int64), "team": np.zeros(0, np.int8),
            "round": np.zeros(0, np.int64), "it": np.zeros(0, np.int64),
            "p": np.zeros((0, 3), np.float32)}
    resolved = collections.deque(maxlen=40)
    games = collections.deque(maxlen=4000)       # (slot, score) of finished learner games

    obs = env.reset()
    turns_per_iter = N * T
    n_iters = max(1, math.ceil(a.turns / turns_per_iter))
    t_start = time.perf_counter()

    def save(path: pathlib.Path, it: int, cand: int) -> None:
        ckpt = {"net": policy.state_dict(), "opt": popt.state_dict(), "ratchet": True,
                "iter": it, "total_turns": cand, "cand_turns": cand,
                "args": {**vars(a), "width": width, "blocks": blocks}}
        tmp = path.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(path)

    def abort(why: str) -> None:
        print(f"ABORT: {why}", flush=True)
        log_file.write(json.dumps({"gen": a.gen, "segment": a.segment, "abort": why,
                                   "total_turns": a.turn_base}) + "\n")
        log_file.flush()
        sys.exit(ABORT)

    for it in range(n_iters):
        policy.eval()
        roll.begin()
        nxt_team.fill(-1)
        term_cls.fill(-1)
        last.fill(-1)
        ended = {}
        t0 = time.perf_counter()
        for t in range(T):
            staged = roll.stage(obs)
            learn = (learner < 0) | (obs.team == learner)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = policy(staged[0], staged[1])
                if a.explore > 0:
                    # Behaviour = (1 - eps) * policy + eps * uniform over legal moves.
                    # The recorded logp is the POLICY's, not the mixture's: an exact
                    # importance weight (pi / mixture ~ 1e-6 / 5e-4 for a dead action,
                    # e.g. gen0's 3-step sprints) would scale its gradient to nothing,
                    # so it could never come back. This biases updates toward the
                    # tried actions, bounded by the PPO clip on each step.
                    lm = masked_logits(logits.float(), staged[2])
                    logp_all = torch.log_softmax(lm, dim=1)
                    legal = (lm > -1e8).float()
                    q = (1 - a.explore) * logp_all.exp() + a.explore * legal / legal.sum(1, keepdim=True)
                    action = torch.multinomial(q, 1).squeeze(1)
                    logp = logp_all.gather(1, action.unsqueeze(1)).squeeze(1)
                else:
                    action, logp, _ = policy_out(logits, staged[2])
                for k, onet in enumerate(opps):
                    rows = torch.from_numpy(~learn & (slot == k + 1)).to(dev)
                    if rows.any():
                        ol, _ = onet(staged[0][rows], staged[1][rows])
                        ol = masked_logits(ol.float(), staged[2][rows])
                        action[rows] = torch.multinomial(ol.softmax(1), 1).squeeze(1)
            roll.record(t, staged, obs, action, logp, zero_v, learn=learn)
            priv_buf[t] = torch.from_numpy(obs.priv).to(dev)
            tm = obs.team.astype(np.int64)
            rnd_cpu[t] = obs.round
            team_cpu[t] = obs.team
            epid_cpu[t] = obs.uid >> 12
            learn_cpu[t] = learn
            prev = last[tm, envs_ix]
            has = prev >= 0
            nxt_team[prev[has], envs_ix[has]] = t
            last[tm, envs_ix] = t
            obs, closures, eps = env.step(action.to(torch.int32).cpu().numpy())
            roll.close(closures, no_reward)
            if len(eps.rows):
                e = eps.rows[:, 0].astype(np.int64)
                w = eps.rows[:, bcsim.EpisodeStats.COLUMNS.index("winner")].astype(np.int64)
                for k in (0, 1):
                    lt = last[k, e]
                    ok = lt >= 0
                    term_cls[lt[ok], e[ok]] = np.where(w[ok] < 0, 1, np.where(w[ok] == k, 0, 2))
                last[:, e] = -1
                for ee, ww in zip(e, w):
                    ended[(int(ee) << 40) | int(epid_cpu[t, ee])] = int(ww)
                    if slot[ee] > 0:
                        games.append((int(slot[ee]), 0.5 if ww < 0 else float(ww == learner[ee])))
                assign(e)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s0 in range(0, T, 16):
                sl = slice(s0, min(s0 + 16, T))
                k = sl.stop - sl.start
                tl, _ = tcrit(roll.local[sl].reshape(k * N, *roll.local.shape[2:]).float(),
                              roll.scalar[sl].reshape(k * N, -1).float(),
                              zero_ctx.expand(k * N, 1), priv_buf[sl].reshape(k * N, -1))
                probs[sl] = tl.float().softmax(-1).reshape(k, N, 3)
        t_roll = time.perf_counter() - t0

        # ---- team TD(lambda) along each team's turns, the result at the end
        nxt = torch.from_numpy(nxt_team).to(dev)
        term = torch.from_numpy(term_cls).to(dev)
        rnd = torch.from_numpy(rnd_cpu).to(dev)
        target = probs.clone()
        idx = torch.arange(N, device=dev)
        onehot = torch.eye(3, device=dev)
        for t in range(T - 1, -1, -1):
            nx = nxt[t]
            safe = nx.clamp(min=0)
            lam_e = a.team_lam ** (rnd[safe, idx] - rnd[t]).clamp(min=0).float()
            boot = (1 - lam_e)[:, None] * probs[safe, idx] + lam_e[:, None] * target[safe, idx]
            y = torch.where((nx >= 0)[:, None], boot, probs[t])
            target[t] = torch.where((term[t] >= 0)[:, None], onehot[term[t].clamp(min=0)], y)
        valid = (term >= 0) | (nxt >= 0)
        v_team = probs @ outcome_u
        ret = target @ outcome_u
        adv_raw = ret - v_team

        learn_t = torch.from_numpy(learn_cpu).to(dev)
        sel = (learn_t & valid).reshape(-1).nonzero(as_tuple=True)[0]
        n = sel.numel()
        if n < a.minibatch:
            print(f"it {it}: only {n} usable rows, skipped", flush=True)
            continue
        flat = lambda x: x.reshape(T * N, *x.shape[2:])[sel]
        b_local, b_scalar, b_mask = flat(roll.local), flat(roll.scalar), flat(roll.mask)
        b_action, b_logp = flat(roll.action), flat(roll.logp)
        b_adv = flat(adv_raw)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        ev_team = float(1.0 - (flat(ret) - flat(v_team)).var() / (flat(ret).var() + 1e-8))

        # ---- calibration of the frozen critic against how games really ended
        cand_m = (learn_t & (torch.rand(T, N, device=dev) < a.calib_share)).cpu().numpy()
        ct, ce = np.nonzero(cand_m)
        if len(ct):
            pend["key"] = np.concatenate([pend["key"], (ce << 40) | epid_cpu[ct, ce]])
            pend["team"] = np.concatenate([pend["team"], team_cpu[ct, ce]])
            pend["round"] = np.concatenate([pend["round"], rnd_cpu[ct, ce]])
            pend["it"] = np.concatenate([pend["it"], np.full(len(ct), it)])
            pend["p"] = np.concatenate([pend["p"], probs.cpu().numpy()[ct, ce]])
        if ended and len(pend["key"]):
            hit = np.isin(pend["key"], np.fromiter(ended.keys(), np.int64))
            if hit.any():
                w = np.array([ended[int(x)] for x in pend["key"][hit]])
                tmh = pend["team"][hit].astype(np.int64)
                resolved.append((pend["p"][hit], np.where(w < 0, 1, np.where(w == tmh, 0, 2)),
                                 pend["round"][hit]))
            keep = ~hit & (pend["it"] > it - 200)
            pend = {k: v[keep] for k, v in pend.items()}

        # ---- one policy epoch
        policy.train()
        t0 = time.perf_counter()
        st = {"pg": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0, "kl_teacher": 0.0}
        nb = 0
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, a.minibatch):
            ix = perm[s:s + a.minibatch]
            lb, sb, mask_b = b_local[ix].float(), b_scalar[ix].float(), b_mask[ix]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                t_logits, _ = teacher(lb, sb)
            t_logp = torch.log_softmax(masked_logits(t_logits.float(), mask_b), dim=1)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = policy(lb, sb)
            _, logp, ent = policy_out(logits, mask_b, b_action[ix])
            ratio = (logp - b_logp[ix]).exp()
            mb = b_adv[ix]
            pg = -torch.min(ratio * mb, ratio.clamp(1 - a.clip, 1 + a.clip) * mb).mean()
            s_logp = torch.log_softmax(masked_logits(logits.float(), mask_b), dim=1)
            kl_t = (t_logp.exp() * (t_logp - s_logp)).nan_to_num(0.0).sum(1).mean()
            loss = pg - a.ent * ent.mean() + a.kl_coef * kl_t
            if not torch.isfinite(loss):
                abort(f"non-finite loss at iteration {it}")
            popt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            popt.step()
            with torch.no_grad():
                st["pg"] += pg.item()
                st["ent"] += ent.mean().item()
                st["kl"] += (b_logp[ix] - logp).mean().item()
                st["clipfrac"] += ((ratio - 1).abs() > a.clip).float().mean().item()
                st["kl_teacher"] += kl_t.item()
            nb += 1
        t_opt = time.perf_counter() - t0
        for k in st:
            st[k] /= max(nb, 1)
        if not all(torch.isfinite(q).all() for q in policy.parameters()):
            abort(f"non-finite parameters at iteration {it}")

        cand = done0 + (it + 1) * turns_per_iter
        total = a.turn_base + (it + 1) * turns_per_iter
        row = {"gen": a.gen, "segment": a.segment, "iter": it, "total_turns": total,
               "cand_turns": cand, "lr": a.lr, "explore": a.explore,
               "sps": turns_per_iter / (t_roll + t_opt), "t_roll": round(t_roll, 2),
               "t_opt": round(t_opt, 2), "usable": round(n / turns_per_iter, 3),
               "orphans": roll.orphans, "overwrites": roll.overwrites,
               "return_mean": float(flat(ret).mean()), "value_mean": float(flat(v_team).mean()),
               "explained_var": round(ev_team, 4),
               "elapsed": round(time.perf_counter() - t_start, 1),
               **{k: round(v, 5) for k, v in st.items()}}
        if resolved and it % 5 == 0:
            p_ = np.concatenate([r[0] for r in resolved])
            y_ = np.concatenate([r[1] for r in resolved])
            r_ = np.concatenate([r[2] for r in resolved])
            row.update({f"calib_{k}": v for k, v in team_critic.calibration(p_, y_, r_).items()})
            row["calib_n"] = int(len(y_))
        # win rates over the last few thousand games, not this iteration's handful
        if games:
            g = np.array(games)
            for k, name in enumerate(opp_names):
                m = g[:, 0] == k + 1
                if m.sum() >= 30:
                    row[f"vs_{name}"] = round(float(g[m, 1].mean()), 4)
                    row[f"n_{name}"] = int(m.sum())
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()
        if it % 5 == 0 or it == n_iters - 1:
            vs = " ".join(f"{k[3:]} {v:.2f}" for k, v in row.items() if k.startswith("vs_"))
            cal = f" | calib ev {row.get('calib_ev')}" if "calib_ev" in row else ""
            print(f"g{a.gen}.{a.segment} it {it:4d}/{n_iters} | {cand / 1e6:6.1f}M | "
                  f"{row['sps']:>7,.0f} t/s | ev {ev_team:.3f} | ent {st['ent']:.3f} "
                  f"klT {st['kl_teacher']:.4f} clip {st['clipfrac']:.3f}{cal} | {vs}", flush=True)
        if roll.overwrites:
            abort(f"slot reused before closing ({roll.overwrites})")
        if st["kl_teacher"] > a.max_kl:
            abort(f"KL to teacher {st['kl_teacher']:.4f} > {a.max_kl}")
        if st["ent"] < a.min_ent:
            abort(f"entropy {st['ent']:.4f} < {a.min_ent}")
        if it % 10 == 9:
            save(out / "latest.pt", it, cand)
    save(out / "final.pt", n_iters - 1, done0 + n_iters * turns_per_iter)
    (out / "latest.pt").unlink(missing_ok=True)
    print(f"segment done in {time.perf_counter() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
