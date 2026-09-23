"""PPO fine-tuning from a clone with a team-level critic (see team_critic.py).

finetune.py with the critic and the advantage replaced. Everything else is
ft3's: a frozen-policy warm-up, the opponent pool, KL toward the clone, a low
learning rate and one policy epoch.

Why: ft1-ft3 fitted each dragon's own return, which carries the result only
for survivors, and its critic explained 7-12% of it. Here the advantage is

    A = alpha * A_team / sd + (1 - alpha) * A_self / sd

  * A_team: TD(lambda) along the TEAM's chain of turns (every turn the team
    takes, whichever dragon takes it) of the team value p_win - p_loss, with
    the result (+1 / 0 / -1) at the end of the game. lambda decays per round,
    not per turn, since a round is one turn for each living dragon. A dragon
    that dies still gets the team's result through the turns that follow.
  * A_self: finetune.py's per-dragon GAE, on the dense terms only (win, lose,
    draw and eliminated removed: the team term carries the result).

The critic's team head is trained on the distributional TD(lambda) target
(soft win/draw/loss classes), its self head on the per-dragon returns.
Calibration is tracked by round against how games actually ended.

    python -m train.finetune_team --init ../runs/anchors/sss_r2_bc_64x4.pt \\
        --critic-init ../runs/team_critic/pretrained.pt \\
        --opponents a.pt,b.pt --out ../runs/ft4
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# the global features need the privileged build, chosen before bcsim loads
os.environ.setdefault("BCSIM_LIB", str(pathlib.Path(__file__).resolve().parents[1]
                                       / "bcsim" / "libbcvec_priv.so"))

import bcsim                                    # noqa: E402
from train import augment, team_critic          # noqa: E402
from train.finetune import load_policy, reward_weights  # noqa: E402
from train.net import masked_logits, policy_out  # noqa: E402
from train.rollout import Rollout               # noqa: E402
from train.train import POTENTIAL_DISCOUNT, REWARDS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULT = ("win", "lose", "draw", "eliminated")  # left to the team term
N_PRIV = team_critic.N_PRIV


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="", help="policy to start from (a clone)")
    p.add_argument("--resume", default="", help="a finetune_team checkpoint to continue")
    p.add_argument("--critic-init", default="",
                   help="team_critic pretrain output; default a fresh critic")
    p.add_argument("--teacher", default="", help="frozen policy for the KL term; default --init")
    p.add_argument("--kl-coef", type=float, default=0.5)
    p.add_argument("--opponents", default="", help="frozen networks, comma separated")
    p.add_argument("--self-frac", type=float, default=0.5)
    p.add_argument("--opp-sample", action="store_true")
    p.add_argument("--maps", default=str(ROOT / "runs/ft3/maps"))
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--critic-width", type=int, default=128)
    p.add_argument("--critic-blocks", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--critic-lr", type=float, default=1e-4)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.997, help="per-dragon (self) discount")
    p.add_argument("--lam", type=float, default=0.95, help="per-dragon GAE lambda")
    p.add_argument("--team-lam", type=float, default=0.9,
                   help="team TD lambda, per ROUND between the team's turns")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="weight of the team advantage; 1 - alpha on the dragon's own")
    p.add_argument("--self-coef", type=float, default=1.0, help="self-head loss weight")
    p.add_argument("--ent", type=float, default=0.001)
    p.add_argument("--reward", default="v6", choices=sorted(REWARDS))
    p.add_argument("--shaping-scale", type=float, default=0.25)
    p.add_argument("--critic-warmup", type=float, default=25e6,
                   help="turns with the policy frozen while the critic learns")
    p.add_argument("--self-ev-gate", type=float, default=0.3,
                   help="after --critic-warmup, keep the policy frozen until the self "
                        "head's explained variance (EMA) reaches this")
    p.add_argument("--mc-coef", type=float, default=1.0,
                   help="weight of the team head's cross-entropy on REAL game results "
                        "(sampled positions kept until their game ends); 0 = off")
    p.add_argument("--td-coef", type=float, default=0.25,
                   help="weight of the team head's TD(lambda) target loss")
    p.add_argument("--mc-buffer", type=int, default=200_000,
                   help="labelled positions kept for the real-result loss")
    p.add_argument("--mc-batch", type=int, default=4096,
                   help="real-result positions per critic step")
    p.add_argument("--mc-min", type=int, default=20_000,
                   help="labelled positions needed before the real-result loss starts")
    p.add_argument("--freeze-team", default="",
                   help="take the team value from a FROZEN copy of this critic (trunk and "
                        "head); the trained critic then fits the self head only")
    p.add_argument("--opp-weights", default="",
                   help="relative share of each --opponents entry among the frozen-opponent "
                        "games, comma separated (default: equal)")
    p.add_argument("--gate-max", type=float, default=80e6,
                   help="turns after which the policy trains even if the gate is unmet")
    p.add_argument("--epochs", type=int, default=2, help="critic passes per iteration")
    p.add_argument("--policy-epochs", type=int, default=1)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--iters", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "runs/ft4"))
    p.add_argument("--aug-per-map", type=int, default=48)
    p.add_argument("--aug-original-share", type=float, default=0.25)
    p.add_argument("--size-alpha", type=float, default=0.0)
    p.add_argument("--snapshot-every", type=float, default=12.5e6)
    p.add_argument("--calib-share", type=float, default=1 / 32,
                   help="share of learner turns kept to check against the game's end")
    p.add_argument("--gpu-frac", type=float, default=0.0,
                   help="cap this process's GPU memory share (0 = no cap)")
    return p.parse_args()


def main() -> None:
    a = parse()
    if not (a.init or a.resume):
        raise SystemExit("give --init (a clone) or --resume")
    if not hasattr(bcsim.env._lib, "bcv_bind_priv"):
        raise SystemExit("needs libbcvec_priv.so (make priv)")
    dev = torch.device("cuda")
    if a.gpu_frac:
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.backends.cudnn.benchmark = True
    out = pathlib.Path(a.out)
    (out / "snapshots").mkdir(parents=True, exist_ok=True)
    log_file = (out / "log.jsonl").open("a")

    # ---- networks
    ck = torch.load(a.resume, map_location=dev, weights_only=False) if a.resume else None
    policy, init_ck = load_policy(a.resume or a.init, dev)
    width, blocks = init_ck["args"]["width"], init_ck["args"]["blocks"]
    opp_paths = [x for x in a.opponents.split(",") if x]
    opps = []
    for path in opp_paths:
        net, _ = load_policy(path, dev)
        net.eval()
        for q in net.parameters():
            q.requires_grad_(False)
        opps.append(net)
    n_ctx = 1 + len(opps)                        # slot 0 = self-play
    teacher_path = a.teacher or (ck["args"].get("teacher") if ck else "") or a.init
    a.teacher = teacher_path
    teacher, _ = load_policy(teacher_path, dev)
    teacher.eval()
    for q in teacher.parameters():
        q.requires_grad_(False)

    if ck and ck["args"].get("opponents", "") != a.opponents:
        raise SystemExit("--opponents differs from the checkpoint's: the critic's "
                         "opponent slots would change meaning")
    # --critic-init with --resume: the policy continues, the critic restarts from
    # that file (ft6: the drifted team head replaced by the pretrained one)
    critic_reset = bool(ck and a.critic_init)
    if ck and not critic_reset:
        critic = team_critic.build(bcsim.N_CHANNELS,
                                   ck["critic"]["scalar.0.weight"].shape[1], n_ctx,
                                   ck["critic_args"]["width"], ck["critic_args"]["blocks"]).to(dev)
        critic.load_state_dict(ck["critic"])
        c_args = ck["critic_args"]
    elif a.critic_init:
        critic, c_args = team_critic.load(a.critic_init, dev, n_context=n_ctx)
    else:
        critic = team_critic.build(bcsim.N_CHANNELS, bcsim.N_SCALARS, n_ctx,
                                   a.critic_width, a.critic_blocks).to(dev)
        c_args = {"width": a.critic_width, "blocks": a.critic_blocks, "n_context": n_ctx,
                  "n_priv": N_PRIV}
    tcrit = None
    if a.freeze_team:
        # ft5/ft6: a team head trained online drifted away from real results
        # (bootstrapped targets, and the self head reshaping the shared trunk);
        # a frozen pretrained copy cannot
        tcrit, _ = team_critic.load(a.freeze_team, dev, n_context=n_ctx)
        tcrit.eval()
        for q in tcrit.parameters():
            q.requires_grad_(False)
        print(f"team value from a frozen critic: {a.freeze_team}", flush=True)
    # a critic saved before the memory features is narrower than the env is now
    # wide; the base scalars keep indices 0..13, so the leading slice is exactly
    # what it was trained on (see migrate_scalars.py)
    N_CRITIC_SCALARS = critic.scalar[0].weight.shape[1]
    N_TCRIT_SCALARS = tcrit.scalar[0].weight.shape[1] if tcrit is not None else 0
    popt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0, eps=1e-5)
    copt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr, weight_decay=0.0, eps=1e-5)
    start_iter, base_turns = 0, 0
    # The self head predicts the dragon's shaped return divided by self_scale
    # (an EMA of that return's spread). Unscaled, the return spreads ~0.03, its
    # MSE (~3e-4) is ~1000x the team head's cross-entropy in the shared trunk,
    # and in ft4's first launch the head never got past predicting the mean.
    self_scale, ev_self_ema = None, None
    if ck:
        if "opt" in ck:
            popt.load_state_dict(ck["opt"])
            if not critic_reset:
                copt.load_state_dict(ck["copt"])
        else:
            print("no optimiser state in the checkpoint (a snapshot): fresh optimisers",
                  flush=True)
        start_iter, base_turns = ck["iter"] + 1, ck["total_turns"]
        self_scale = ck.get("self_scale")
        # a fresh self head has to earn the gate again
        ev_self_ema = None if critic_reset else ck.get("ev_self_ema")
        if critic_reset:
            print(f"critic reset from {a.critic_init}; policy and its optimiser continue",
                  flush=True)
        del ck
    print(f"teacher {teacher_path} (KL coef {a.kl_coef})", flush=True)
    print(f"policy {width}x{blocks} from {a.resume or a.init}; team critic "
          f"{c_args['width']}x{c_args['blocks']} from "
          f"{a.resume or a.critic_init or 'scratch'}; alpha {a.alpha}, team lambda "
          f"{a.team_lam}/round; opponents: self"
          + "".join(f", {pathlib.Path(p).stem}" for p in opp_paths), flush=True)

    # ---- rewards: the dragon's own dense terms; the result goes to the team term
    rw = {k: v for k, v in reward_weights(a.reward, a.shaping_scale).items()
          if k not in RESULT}
    weights = bcsim.reward_vector(rw)
    print("self reward weights: " + ", ".join(f"{k} {v:g}" for k, v in rw.items())
          + " | team: win +1, draw 0, loss -1", flush=True)

    # ---- env and opponent assignment
    texts, map_w, _, _ = augment.build_pool(a.maps, a.aug_per_map, a.seed,
                                            a.size_alpha, a.aug_original_share)
    env = bcsim.BattlecodeVecEnv(texts, num_envs=a.envs, num_threads=a.threads,
                                 seed=a.seed + 7919 * start_iter,
                                 closure_capacity=max(8192, a.envs * 160), privileged=True)
    env.set_map_weights(map_w)
    if POTENTIAL_DISCOUNT[a.reward]:
        env.set_potential_gamma(a.gamma)

    N, T = a.envs, a.steps
    slot = np.zeros(N, np.int64)
    learner = np.full(N, -1, np.int8)

    opp_p = None
    if opps:
        wts = ([float(x) for x in a.opp_weights.split(",")] if a.opp_weights
               else [1.0] * len(opps))
        if len(wts) != len(opps):
            raise SystemExit("--opp-weights needs one weight per --opponents entry")
        opp_p = np.array(wts) / sum(wts)
        print("frozen-opponent shares: " + ", ".join(
            f"{pathlib.Path(q).stem} {w:.2f}" for q, w in zip(opp_paths, opp_p)), flush=True)

    def assign(envs: np.ndarray) -> None:
        if not len(envs):
            return
        frozen = rng.random(len(envs)) >= a.self_frac if opps else np.zeros(len(envs), bool)
        pick = (rng.choice(len(opps), len(envs), p=opp_p) if opps
                else np.zeros(len(envs), np.int64))
        slot[envs] = np.where(frozen, 1 + pick, 0)
        learner[envs] = np.where(frozen, rng.integers(0, 2, len(envs)), -1)

    assign(np.arange(N))
    ctx_eye = torch.eye(n_ctx, device=dev)
    outcome_u = torch.tensor(team_critic.OUTCOME_VALUE, device=dev)

    roll = Rollout(T, N, bcsim.N_CHANNELS, bcsim.WINDOW, bcsim.N_SCALARS,
                   bcsim.N_ACTIONS, dev)
    ctx_buf = torch.zeros(T, N, dtype=torch.int64, device=dev)
    priv_buf = torch.zeros(T, N, N_PRIV, device=dev)
    probs = torch.zeros(T, N, 3, device=dev)     # team head, per stored turn
    zero_v = torch.zeros(N, device=dev)
    # team chains (CPU while stepping): the next turn of the same team in the
    # same game, and the result class where a team's last turn of a game is
    nxt_team = np.full((T, N), -1, np.int64)
    term_cls = np.full((T, N), -1, np.int64)
    rnd_cpu = np.zeros((T, N), np.int64)
    team_cpu = np.zeros((T, N), np.int8)
    epid_cpu = np.zeros((T, N), np.int64)
    learn_cpu = np.zeros((T, N), bool)
    last = np.full((2, N), -1, np.int64)
    envs_ix = np.arange(N)
    # calibration: sampled predictions wait here until their game ends
    pend = {k: np.zeros(0, dt) for k, dt in (("key", np.int64), ("team", np.int8),
                                               ("round", np.int64), ("it", np.int64))}
    pend["p"] = np.zeros((0, 3), np.float32)
    resolved = collections.deque(maxlen=40)      # (p, y, round) arrays per iteration
    # real-result training: the same sampled positions keep their critic inputs
    # (CPU) until the game ends, then enter a ring buffer on the GPU with the
    # team's actual result
    for k, shp, dt in (("local", (bcsim.N_CHANNELS, bcsim.WINDOW, bcsim.WINDOW), np.float16),
                       ("scalar", (bcsim.N_SCALARS,), np.float16), ("priv", (N_PRIV,), np.float32),
                       ("ctx", (), np.int64)):
        pend[k] = np.zeros((0,) + shp, dt)
    MB = a.mc_buffer
    mc = {"local": torch.zeros(MB, bcsim.N_CHANNELS, bcsim.WINDOW, bcsim.WINDOW,
                               dtype=torch.float16, device=dev),
          "scalar": torch.zeros(MB, bcsim.N_SCALARS, dtype=torch.float16, device=dev),
          "priv": torch.zeros(MB, N_PRIV, device=dev),
          "ctx": torch.zeros(MB, dtype=torch.int64, device=dev),
          "y": torch.zeros(MB, dtype=torch.int64, device=dev)}
    mc_n, mc_pos = 0, 0

    obs = env.reset()
    turns_per_iter = N * T
    next_snap = (base_turns // int(a.snapshot_every) + 1) * int(a.snapshot_every)
    t_start = time.perf_counter()

    for it in range(start_iter, start_iter + a.iters):
        done_turns = base_turns + (it - start_iter) * turns_per_iter
        # frozen during the warm-up, then until the self head is usable
        # (explained variance EMA >= --self-ev-gate), up to --gate-max
        warm = done_turns < a.critic_warmup or (
            done_turns < a.gate_max and (ev_self_ema is None or ev_self_ema < a.self_ev_gate))
        policy.eval()
        critic.eval()
        roll.begin()
        nxt_team.fill(-1)
        term_cls.fill(-1)
        last.fill(-1)
        ended = {}                               # env << 40 | episode -> winner
        ep_rows, ep_slot, ep_side = [], [], []
        t0 = time.perf_counter()
        for t in range(T):
            staged = roll.stage(obs)
            learn = (learner < 0) | (obs.team == learner)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = policy(staged[0], staged[1])
                action, logp, _ = policy_out(logits, staged[2])
                for k, onet in enumerate(opps):
                    rows = torch.from_numpy(~learn & (slot == k + 1)).to(dev)
                    if rows.any():
                        ol, _ = onet(staged[0][rows], staged[1][rows])
                        ol = masked_logits(ol.float(), staged[2][rows])
                        action[rows] = (torch.multinomial(ol.softmax(1), 1).squeeze(1)
                                        if a.opp_sample else ol.argmax(1))
            roll.record(t, staged, obs, action, logp, zero_v, learn=learn)
            ctx_buf[t] = torch.from_numpy(slot).to(dev)
            priv_buf[t] = torch.from_numpy(obs.priv).to(dev)   # copied: env reuses it
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
            roll.close(closures, weights)
            if len(eps.rows):
                e = eps.rows[:, 0].astype(np.int64)
                w = eps.rows[:, bcsim.EpisodeStats.COLUMNS.index("winner")].astype(np.int64)
                for k in (0, 1):
                    lt = last[k, e]
                    ok = lt >= 0
                    # this team's result: 0 win, 1 draw, 2 loss
                    term_cls[lt[ok], e[ok]] = np.where(w[ok] < 0, 1, np.where(w[ok] == k, 0, 2))
                last[:, e] = -1
                for ee, ww in zip(e, w):
                    ended[(int(ee) << 40) | int(epid_cpu[t, ee])] = int(ww)
                ep_rows.append(eps.rows.copy())
                ep_slot.append(slot[e].copy())
                ep_side.append(learner[e].copy())
                assign(e)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s0 in range(0, T, 16):
                sl = slice(s0, min(s0 + 16, T))
                k = sl.stop - sl.start
                tl, sv = critic(roll.local[sl].reshape(k * N, *roll.local.shape[2:]).float(),
                                roll.scalar[sl].reshape(k * N, -1)[:, :N_CRITIC_SCALARS].float(),
                                ctx_eye[ctx_buf[sl].reshape(-1)],
                                priv_buf[sl].reshape(k * N, -1))
                if tcrit is not None:
                    tl, _ = tcrit(roll.local[sl].reshape(k * N, *roll.local.shape[2:]).float(),
                                  roll.scalar[sl].reshape(k * N, -1)[:, :N_TCRIT_SCALARS].float(),
                                  ctx_eye[ctx_buf[sl].reshape(-1)],
                                  priv_buf[sl].reshape(k * N, -1))
                probs[sl] = tl.float().softmax(-1).reshape(k, N, 3)
                roll.value[sl] = sv.float().reshape(k, N) * (self_scale or 1.0)
        t_roll = time.perf_counter() - t0

        # ---- self: per-dragon GAE on the dense terms
        adv_s, ret_s, valid_s = roll.finish(a.gamma, a.lam)
        sd_now = float(ret_s[valid_s & torch.from_numpy(learn_cpu).to(dev)].std())
        scale_used = self_scale or sd_now      # the scale this iteration's targets use

        # ---- team: distributional TD(lambda) along each team's turns
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
            is_term = term[t] >= 0
            y = torch.where((nx >= 0)[:, None], boot, probs[t])
            target[t] = torch.where(is_term[:, None], onehot[term[t].clamp(min=0)], y)
        valid_t = (term >= 0) | (nxt >= 0)
        v_team = probs @ outcome_u
        adv_t = target @ outcome_u - v_team

        learn_t = torch.from_numpy(learn_cpu).to(dev)
        sel = (learn_t & (valid_s | valid_t)).reshape(-1).nonzero(as_tuple=True)[0]
        if sel.numel() == 0:
            continue
        flat = lambda x: x.reshape(T * N, *x.shape[2:])[sel]
        batch = {"local": flat(roll.local), "scalar": flat(roll.scalar), "mask": flat(roll.mask),
                 "action": flat(roll.action), "logp": flat(roll.logp),
                 "ctx": flat(ctx_buf), "priv": flat(priv_buf),
                 "v_self": flat(roll.value), "ret_s": flat(ret_s), "adv_s": flat(adv_s),
                 "ok_s": flat(valid_s), "target": flat(target), "v_team": flat(v_team),
                 "adv_t": flat(adv_t), "ok_t": flat(valid_t)}
        n = sel.numel()
        ok_p = batch["ok_s"] & batch["ok_t"]     # rows the policy trains on
        sd_t = batch["adv_t"][ok_p].std() + 1e-8
        sd_s = batch["adv_s"][ok_p].std() + 1e-8
        adv = a.alpha * batch["adv_t"] / sd_t + (1 - a.alpha) * batch["adv_s"] / sd_s
        m_, s_ = adv[ok_p].mean(), adv[ok_p].std() + 1e-8
        batch["adv"] = torch.where(ok_p, (adv - m_) / s_, torch.zeros_like(adv))

        def ev(r, v, m):
            if m.sum() < 100:
                return None
            return round(float(1.0 - (r[m] - v[m]).var() / (r[m].var() + 1e-8)), 4)

        ret_t = batch["target"] @ outcome_u
        ev_team = ev(ret_t, batch["v_team"], batch["ok_t"])
        ev_self = ev(batch["ret_s"], batch["v_self"], batch["ok_s"])
        if ev_self is not None:
            ev_self_ema = ev_self if ev_self_ema is None else 0.9 * ev_self_ema + 0.1 * ev_self

        # ---- calibration: keep a few predictions until their game ends
        cand = (learn_t & (torch.rand(T, N, device=dev) < a.calib_share)).cpu().numpy()
        ct, ce = np.nonzero(cand)
        if len(ct):
            pend["key"] = np.concatenate([pend["key"], (ce << 40) | epid_cpu[ct, ce]])
            pend["team"] = np.concatenate([pend["team"], team_cpu[ct, ce]])
            pend["round"] = np.concatenate([pend["round"], rnd_cpu[ct, ce]])
            pend["it"] = np.concatenate([pend["it"], np.full(len(ct), it)])
            pend["p"] = np.concatenate([pend["p"], probs.cpu().numpy()[ct, ce]])
            ct_t, ce_t = torch.from_numpy(ct).to(dev), torch.from_numpy(ce).to(dev)
            pend["local"] = np.concatenate([pend["local"], roll.local[ct_t, ce_t].cpu().numpy()])
            pend["scalar"] = np.concatenate([pend["scalar"],
                                             roll.scalar[ct_t, ce_t].cpu().numpy()])
            pend["priv"] = np.concatenate([pend["priv"], priv_buf[ct_t, ce_t].cpu().numpy()])
            pend["ctx"] = np.concatenate([pend["ctx"], ctx_buf[ct_t, ce_t].cpu().numpy()])
        if ended and len(pend["key"]):
            hit = np.isin(pend["key"], np.fromiter(ended.keys(), np.int64))
            if hit.any():
                w = np.array([ended[int(x)] for x in pend["key"][hit]])
                tmh = pend["team"][hit].astype(np.int64)
                y = np.where(w < 0, 1, np.where(w == tmh, 0, 2))
                resolved.append((pend["p"][hit], y, pend["round"][hit]))
                # scored above before being trained on, so calibration stays honest
                m = len(y)
                at = (mc_pos + np.arange(m)) % MB
                at_t = torch.from_numpy(at).to(dev)
                mc["local"][at_t] = torch.from_numpy(pend["local"][hit]).to(dev)
                mc["scalar"][at_t] = torch.from_numpy(pend["scalar"][hit]).to(dev)
                mc["priv"][at_t] = torch.from_numpy(pend["priv"][hit]).to(dev)
                mc["ctx"][at_t] = torch.from_numpy(pend["ctx"][hit]).to(dev)
                mc["y"][at_t] = torch.from_numpy(y.astype(np.int64)).to(dev)
                mc_pos, mc_n = (mc_pos + m) % MB, min(MB, mc_n + m)
            keep = ~hit & (pend["it"] > it - 200)   # a game outlives 200 iterations: drop
            pend = {k: v[keep] for k, v in pend.items()}

        # ---- optimise
        policy.train(not warm)
        critic.train()
        t0 = time.perf_counter()
        st = {"pg": 0.0, "v_team": 0.0, "v_mc": 0.0, "v_self": 0.0, "ent": 0.0, "kl": 0.0,
              "clipfrac": 0.0, "kl_teacher": 0.0}
        nb, npb = 0, 0
        for ep_i in range(a.epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, a.minibatch):
                ix = perm[s:s + a.minibatch]
                lb = batch["local"][ix].float()
                sb = batch["scalar"][ix][:, :N_CRITIC_SCALARS].float()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    tlog, sv = critic(lb, sb, ctx_eye[batch["ctx"][ix]], batch["priv"][ix])
                okt, oks = batch["ok_t"][ix].float(), batch["ok_s"][ix].float()
                ce_t = -(batch["target"][ix] * torch.log_softmax(tlog.float(), 1)).sum(1)
                l_t = (ce_t * okt).sum() / okt.sum().clamp(min=1)
                # the anchor: cross-entropy on real results, so the team head
                # cannot drift away from them on its own bootstrapped targets
                l_mc = torch.zeros((), device=dev)
                if a.mc_coef and mc_n >= a.mc_min and tcrit is None:
                    j = torch.randint(0, mc_n, (a.mc_batch,), device=dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        mlog, _ = critic(mc["local"][j].float(),
                                         mc["scalar"][j][:, :N_CRITIC_SCALARS].float(),
                                         ctx_eye[mc["ctx"][j]], mc["priv"][j])
                    l_mc = torch.nn.functional.cross_entropy(mlog.float(), mc["y"][j])
                l_s = (0.5 * (sv.float() - batch["ret_s"][ix] / scale_used) ** 2 * oks).sum() \
                    / oks.sum().clamp(min=1)
                copt.zero_grad(set_to_none=True)
                # with a frozen team critic the trained one fits the self head only
                team_w = 0.0 if tcrit is not None else 1.0
                (team_w * (a.td_coef * l_t + a.mc_coef * l_mc) + a.self_coef * l_s).backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                copt.step()
                st["v_team"] += l_t.detach().item()
                st["v_mc"] += l_mc.detach().item()
                st["v_self"] += l_s.detach().item()
                nb += 1
                mask_b = batch["mask"][ix]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    t_logits, _ = teacher(lb, sb)
                t_logp = torch.log_softmax(masked_logits(t_logits.float(), mask_b), dim=1)
                if not warm and ep_i < a.policy_epochs:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = policy(lb, sb)
                    _, logp, ent = policy_out(logits, mask_b, batch["action"][ix])
                    okp = ok_p[ix].float()
                    ratio = (logp - batch["logp"][ix]).exp()
                    mb = batch["adv"][ix]
                    pg_i = -torch.min(ratio * mb, ratio.clamp(1 - a.clip, 1 + a.clip) * mb)
                    pg = (pg_i * okp).sum() / okp.sum().clamp(min=1)
                    s_logp = torch.log_softmax(masked_logits(logits.float(), mask_b), dim=1)
                    kl_t = (t_logp.exp() * (t_logp - s_logp)).nan_to_num(0.0).sum(1).mean()
                    loss = pg - a.ent * ent.mean() + a.kl_coef * kl_t
                    popt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    popt.step()
                    with torch.no_grad():
                        st["pg"] += pg.item()
                        st["ent"] += ent.mean().item()
                        st["kl"] += (batch["logp"][ix] - logp).mean().item()
                        st["clipfrac"] += ((ratio - 1).abs() > a.clip).float().mean().item()
                        st["kl_teacher"] += kl_t.item()
                    npb += 1
                elif warm:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = policy(lb, sb)
                    s_logp = torch.log_softmax(masked_logits(logits.float(), mask_b), dim=1)
                    st["kl_teacher"] += (t_logp.exp() * (t_logp - s_logp)).nan_to_num(0.0) \
                        .sum(1).mean().item()
                    # measured while frozen too, so the chart shows the real entropy
                    st["ent"] += -(s_logp.exp() * s_logp).nan_to_num(0.0).sum(1).mean().item()
                    npb += 1
        t_opt = time.perf_counter() - t0
        for k in st:
            st[k] /= max(nb if k.startswith("v_") else npb, 1)

        total = done_turns + turns_per_iter
        row = {"iter": it, "total_turns": total, "warmup": warm,
               "sps": turns_per_iter / (t_roll + t_opt), "t_roll": round(t_roll, 2),
               "t_opt": round(t_opt, 2), "usable": round(n / turns_per_iter, 3),
               "policy_rows": round(float(ok_p.float().mean()), 3),
               "orphans": roll.orphans, "overwrites": roll.overwrites,
               "return_mean": float(ret_t[batch["ok_t"]].mean()),
               "value_mean": float(batch["v_team"][batch["ok_t"]].mean()),
               # explained_var is the team head's, so the dashboard shows it
               "explained_var": ev_team, "ev_self": ev_self,
               "ev_self_ema": None if ev_self_ema is None else round(ev_self_ema, 4),
               "self_scale": round(scale_used, 5),
               "mc_n": mc_n,
               "adv_corr": round(float(torch.corrcoef(torch.stack(
                   [batch["adv_t"][ok_p], batch["adv_s"][ok_p]]))[0, 1]), 4),
               "elapsed": round(time.perf_counter() - t_start, 1),
               **{k: round(v, 5) for k, v in st.items()}}
        for k in range(n_ctx):
            m = batch["ok_t"] & (batch["ctx"] == k)
            e_k = ev(ret_t, batch["v_team"], m)
            if e_k is not None:
                row[f"ev_{k}"] = e_k
        if resolved and it % 5 == 0:
            p_ = np.concatenate([r[0] for r in resolved])
            y_ = np.concatenate([r[1] for r in resolved])
            r_ = np.concatenate([r[2] for r in resolved])
            row.update({f"calib_{k}": v for k, v in
                        team_critic.calibration(p_, y_, r_).items()})
            row["calib_n"] = int(len(y_))
        if ep_rows:
            rows = np.concatenate(ep_rows)
            sl_, side = np.concatenate(ep_slot), np.concatenate(ep_side)
            cols = bcsim.EpisodeStats.COLUMNS
            winner = rows[:, cols.index("winner")]
            row["episodes"] = len(rows)
            row["rounds_mean"] = float(rows[:, cols.index("rounds")].mean())
            row["draw_rate"] = float((winner < 0).mean())
            for k, path in enumerate(opp_paths):
                m = sl_ == k + 1
                if m.any():
                    score = np.where(winner[m] < 0, 0.5, (winner[m] == side[m]).astype(float))
                    row[f"vs_{pathlib.Path(path).stem}"] = round(float(score.mean()), 4)
                    row[f"n_{pathlib.Path(path).stem}"] = int(m.sum())
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()
        if it % 5 == 0 or it < 3:
            vs = " ".join(f"{k[3:]} {v:.2f}" for k, v in row.items() if k.startswith("vs_"))
            cal = (f" | calib ev {row.get('calib_ev')} ece {row['calib_ece_win']:.3f}"
                   if "calib_ece_win" in row else "")
            print(f"it {it:5d} | {total / 1e6:7.1f}M | {'WARMUP ' if warm else ''}"
                  f"{row['sps']:>8,.0f} turns/s | ev team {ev_team} self {ev_self} | "
                  f"ent {st['ent']:.3f} klT {st['kl_teacher']:.4f} "
                  f"clip {st['clipfrac']:.3f}{cal} | {vs}", flush=True)
        if roll.overwrites:
            raise RuntimeError(f"slot reused before closing ({roll.overwrites})")

        # the next iteration's values use the updated scale; a slow EMA, so a
        # head trained against one scale is never read with a very different one
        self_scale = sd_now if self_scale is None else 0.97 * self_scale + 0.03 * sd_now
        ckpt = {"net": policy.state_dict(), "critic": critic.state_dict(),
                "self_scale": self_scale, "ev_self_ema": ev_self_ema,
                "critic_args": c_args, "opt": popt.state_dict(), "copt": copt.state_dict(),
                "iter": it, "total_turns": total,
                "args": {**vars(a), "width": width, "blocks": blocks, "reward_weights": rw,
                         "team_critic": True}}
        if it % 25 == 0:
            tmp = out / "latest.pt.tmp"
            torch.save(ckpt, tmp)
            tmp.replace(out / "latest.pt")
        if total >= next_snap:
            torch.save({k: v for k, v in ckpt.items() if k not in ("opt", "copt")},
                       out / "snapshots" / f"turns_{total}.pt")
            next_snap = (total // int(a.snapshot_every) + 1) * int(a.snapshot_every)


if __name__ == "__main__":
    main()