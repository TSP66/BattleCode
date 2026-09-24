"""PPO fine-tuning from a behaviour-cloned policy (see imitate.py).

Differs from train.py where starting from a strong policy changes the rules:

  * a separate critic (net.Critic), no weights shared with the policy, so
    fitting values cannot move the policy;
  * a critic warm-up: for the first --critic-warmup turns the policy is frozen
    and plays as cloned while only the critic trains. PPO's advantages come
    from the critic, and a cold one would feed the clone noise at full
    learning rate;
  * an opponent pool: each episode is self-play or against a frozen network
    (the clones, fixed anchors). Against a frozen opponent only the learner's
    turns train. The critic is told which opponent the episode is against
    (a one-hot input, and a value output of its own per opponent), so one
    value does not have to average over them;
  * rewards: a train.py reward with every non-terminal term scaled by
    --shaping-scale. Terminal results (win, lose, draw, eliminated) keep
    their weight; the team potentials, which pay sacrifices, survive scaled;
  * a small fixed entropy bonus, a lower learning rate and a tighter clip.

  * a KL penalty toward the clone: loss += --kl-coef * KL(teacher || policy)
    over the legal actions of each minibatch's states. This direction charges
    the policy for dropping moves the teacher makes, which is how a fine-tune
    forgets what it was cloned from. The teacher is frozen (--teacher,
    default the --init policy).

    python -m train.finetune --init ../runs/imitate_vibing_r3/best.pt \\
        --opponents ../runs/anchors/vibing_bc_64x4.pt,../runs/anchors/sss_bc_64x4.pt \\
        --out ../runs/ft1

Checkpoints keep train.py's layout ("net" is the policy, an ActorCritic whose
value head is unused), so yardstick, export_cpp and check_bot work unchanged;
the critic rides along under "critic".
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import os

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# --privileged needs a build with the global-feature buffer; chosen before
# bcsim is imported, since the library is loaded at import
if "--privileged" in sys.argv:
    os.environ.setdefault("BCSIM_LIB", str(pathlib.Path(__file__).resolve().parents[1]
                                           / "bcsim" / "libbcvec_priv.so"))

import bcsim                                    # noqa: E402
from train import augment                       # noqa: E402
from train import net as net_mod                 # noqa: E402
from train.net import ActorCritic, Critic, masked_logits, policy_out  # noqa: E402
from train.rollout import Rollout               # noqa: E402
from train.train import POTENTIAL_DISCOUNT, REWARDS  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
TERMINAL = ("win", "lose", "draw", "eliminated")


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="", help="policy to start from (a clone)")
    p.add_argument("--resume", default="", help="a finetune checkpoint to continue")
    p.add_argument("--teacher", default="", help="frozen policy for the KL term; "
                   "default --init (on --resume, the checkpoint's recorded teacher)")
    p.add_argument("--kl-coef", type=float, default=0.1,
                   help="weight of KL(teacher || policy); 0 turns the term off")
    p.add_argument("--opponents", default="",
                   help="frozen networks to play against, comma separated")
    p.add_argument("--self-frac", type=float, default=0.5,
                   help="share of episodes played as self-play; the rest are split "
                        "evenly over --opponents")
    p.add_argument("--opp-sample", action="store_true",
                   help="frozen opponents sample their policy; default is argmax, "
                        "as they are deployed")
    p.add_argument("--maps", default=str(ROOT / "maps"))
    p.add_argument("--envs", type=int, default=1024)
    p.add_argument("--steps", type=int, default=256)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--critic-width", type=int, default=128)
    p.add_argument("--critic-blocks", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ent", type=float, default=0.001)
    p.add_argument("--reward", default="v6", choices=sorted(REWARDS))
    p.add_argument("--shaping-scale", type=float, default=0.25,
                   help="multiplies every non-terminal reward weight")
    p.add_argument("--critic-warmup", type=float, default=50e6,
                   help="turns with the policy frozen while the critic learns")
    p.add_argument("--epochs", type=int, default=2, help="critic passes per iteration")
    p.add_argument("--policy-epochs", type=int, default=0,
                   help="policy passes per iteration (0 = same as --epochs)")
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--iters", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "runs/ft"))
    p.add_argument("--aug-per-map", type=int, default=48)
    p.add_argument("--aug-original-share", type=float, default=0.25)
    p.add_argument("--size-alpha", type=float, default=0.0,
                   help="0 = every base map equally likely; a clone needs no small-map "
                        "curriculum")
    p.add_argument("--snapshot-every", type=float, default=12.5e6)
    p.add_argument("--privileged", action="store_true",
                   help="give the critic both teams' lengths, longest dragon, unit "
                        "counts and the round (it never ships, so it may see them)")
    return p.parse_args()


def reward_weights(name: str, scale: float) -> dict[str, float]:
    return {k: (v if k in TERMINAL else v * scale) for k, v in REWARDS[name].items()}


def load_policy(path: str, dev) -> tuple[ActorCritic, dict]:
    """The one place a policy checkpoint becomes a network.

    `args["arch"]` picks the architecture. Checkpoints written before the
    pyramid existed have no such key, which is the flat 708-scalar
    ActorCritic -- so every clone and league member still loads unchanged.

    Read onto the host and moved to the device once full, for the reason given
    in yardstick.load_net: parameters filled by a device-to-device copy break a
    later CUDA graph capture.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck["args"]
    hidden = next(v for k, v in ck["net"].items() if k.endswith("fuse.0.weight")).shape[0]
    # The scalar width comes from the checkpoint, never from the env. The env's
    # row grows as features are appended (708 -> 713 with the sonar echoes) and
    # every older net has to go on reading exactly the columns it was trained
    # on, or the frozen league stops being frozen.
    n_scalars = next(v for k, v in ck["net"].items()
                     if k.endswith("scalar.0.weight")).shape[1]
    if a.get("arch") == "pyramid":
        net = net_mod.PyramidActorCritic(
            bcsim.N_CHANNELS, bcsim.WIDE_CH, bcsim.N_ACTIONS,
            near_width=a["near_width"], near_blocks=a["near_blocks"],
            wide_width=a["wide_width"], wide_blocks=a["wide_blocks"],
            wide_side=bcsim.WIDE_SIDE, hidden=hidden, n_scalars=n_scalars)
    else:
        net = ActorCritic(bcsim.N_CHANNELS, n_scalars, bcsim.N_ACTIONS,
                          width=a["width"], blocks=a["blocks"], hidden=hidden)
    net.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["net"].items()})
    net.to(dev)
    return net, ck


def main() -> None:
    a = parse()
    if not (a.init or a.resume):
        raise SystemExit("give --init (a clone) or --resume")
    dev = torch.device("cuda")
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
    # base features only; the v8 Phi components are sliced off by the net
    n_priv = bcsim.PRIV_BASE if a.privileged else 0
    critic = Critic(bcsim.N_CHANNELS, bcsim.N_SCALARS, n_ctx,
                    width=a.critic_width, blocks=a.critic_blocks, n_extra=n_priv).to(dev)
    popt = torch.optim.AdamW(policy.parameters(), lr=a.lr, weight_decay=0.0, eps=1e-5)
    copt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr, weight_decay=0.0, eps=1e-5)
    start_iter, base_turns = 0, 0
    if ck:
        critic.load_state_dict(ck["critic"])
        # snapshots carry no optimiser state: resuming from one (a rollback)
        # starts both optimisers fresh
        if "opt" in ck:
            popt.load_state_dict(ck["opt"])
            copt.load_state_dict(ck["copt"])
        else:
            print("no optimiser state in the checkpoint (a snapshot): fresh optimisers",
                  flush=True)
        start_iter, base_turns = ck["iter"] + 1, ck["total_turns"]
        if bool(ck["args"].get("privileged", False)) != a.privileged:
            raise SystemExit("--privileged differs from the checkpoint's critic")
        if ck["args"].get("opponents", "") != a.opponents:
            raise SystemExit("--opponents differs from the checkpoint's: the critic's "
                             "opponent slots would change meaning")
        del ck
    print(f"teacher {teacher_path} (KL coef {a.kl_coef})", flush=True)
    print(f"policy {width}x{blocks} from {a.resume or a.init}; critic "
          f"{a.critic_width}x{a.critic_blocks}; opponents: self"
          + "".join(f", {pathlib.Path(p).stem}" for p in opp_paths), flush=True)

    # ---- rewards
    rw = reward_weights(a.reward, a.shaping_scale)
    weights = bcsim.reward_vector(rw)
    print("reward weights: " + ", ".join(f"{k} {v:g}" for k, v in rw.items()), flush=True)

    # ---- env and opponent assignment
    texts, map_w, map_base, _ = augment.build_pool(a.maps, a.aug_per_map, a.seed,
                                                   a.size_alpha, a.aug_original_share)
    env = bcsim.BattlecodeVecEnv(texts, num_envs=a.envs, num_threads=a.threads,
                                 seed=a.seed + 7919 * start_iter,
                                 closure_capacity=max(8192, a.envs * 160),
                                 privileged=a.privileged)
    env.set_map_weights(map_w)
    if POTENTIAL_DISCOUNT[a.reward]:
        env.set_potential_gamma(a.gamma)

    # per env: which opponent slot (0 self) and which team the learner is
    # (-1: both, in self-play)
    slot = np.zeros(a.envs, np.int64)
    learner = np.full(a.envs, -1, np.int8)

    def assign(envs: np.ndarray) -> None:
        if not len(envs):
            return
        frozen = rng.random(len(envs)) >= a.self_frac if opps else np.zeros(len(envs), bool)
        slot[envs] = np.where(frozen, 1 + rng.integers(0, max(len(opps), 1), len(envs)), 0)
        learner[envs] = np.where(frozen, rng.integers(0, 2, len(envs)), -1)

    assign(np.arange(a.envs))
    ctx_eye = torch.eye(n_ctx, device=dev)

    roll = Rollout(a.steps, a.envs, bcsim.N_CHANNELS, bcsim.WINDOW, bcsim.N_SCALARS,
                   bcsim.N_ACTIONS, dev)
    # the critic's context per stored slot, alongside the rollout's own arrays
    ctx_buf = torch.zeros(a.steps, a.envs, dtype=torch.int64, device=dev)
    zero_v = torch.zeros(a.envs, device=dev)
    priv_buf = torch.zeros(a.steps, a.envs, max(n_priv, 1), device=dev)
    obs = env.reset()
    turns_per_iter = a.envs * a.steps
    next_snap = (base_turns // int(a.snapshot_every) + 1) * int(a.snapshot_every)
    t_start = time.perf_counter()

    for it in range(start_iter, start_iter + a.iters):
        done_turns = base_turns + (it - start_iter) * turns_per_iter
        warm = done_turns < a.critic_warmup
        policy.eval()
        critic.eval()
        roll.begin()
        ep_rows, ep_slot, ep_side = [], [], []
        t0 = time.perf_counter()
        for t in range(a.steps):
            staged = roll.stage(obs)
            learn = (learner < 0) | (obs.team == learner)
            s_t = torch.from_numpy(slot).to(dev)
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
            # values are filled in after the rollout, in a few large batches
            roll.record(t, staged, obs, action, logp, zero_v, learn=learn)
            ctx_buf[t] = s_t
            if n_priv:
                priv_buf[t] = torch.from_numpy(obs.priv).to(dev)   # copied: env reuses it
            obs, closures, eps = env.step(action.to(torch.int32).cpu().numpy())
            roll.close(closures, weights)
            if len(eps.rows):
                e = eps.rows[:, 0].astype(np.int64)
                ep_rows.append(eps.rows.copy())
                ep_slot.append(slot[e].copy())
                ep_side.append(learner[e].copy())
                assign(e)                         # each env's next episode
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for s0 in range(0, a.steps, 16):
                sl = slice(s0, min(s0 + 16, a.steps))
                k = sl.stop - sl.start
                roll.value[sl] = critic(
                    roll.local[sl].reshape(k * a.envs, *roll.local.shape[2:]).float(),
                    roll.scalar[sl].reshape(k * a.envs, -1).float(),
                    ctx_eye[ctx_buf[sl].reshape(-1)],
                    priv_buf[sl].reshape(k * a.envs, -1) if n_priv else None
                ).float().reshape(k, a.envs)
        t_roll = time.perf_counter() - t0

        adv, ret, valid = roll.finish(a.gamma, a.lam)
        batch = roll.flat_batch(adv, ret, valid)
        sel_v = valid.reshape(-1).nonzero(as_tuple=True)[0]
        batch["ctx"] = ctx_buf.reshape(-1)[sel_v]
        if n_priv:
            batch["priv"] = priv_buf.reshape(-1, n_priv)[sel_v]
        n = batch["action"].numel()
        if n == 0:
            continue
        adv_b = batch["adv"]
        batch["adv"] = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)
        ev = float(1.0 - (batch["ret"] - batch["value"]).var() / (batch["ret"].var() + 1e-8))

        policy.train(not warm)
        critic.train()
        t0 = time.perf_counter()
        st = {"pg": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0, "kl_teacher": 0.0}
        nb = 0
        p_epochs = a.policy_epochs or a.epochs
        npb = 0                                   # minibatches that touched the policy
        for ep_i in range(a.epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, a.minibatch):
                idx = perm[s:s + a.minibatch]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    v_pred = critic(batch["local"][idx].float(), batch["scalar"][idx].float(),
                                    ctx_eye[batch["ctx"][idx]],
                                    batch["priv"][idx] if n_priv else None).float()
                # a separate critic needs no value clipping: nothing it does
                # can move the policy
                v_loss = 0.5 * ((v_pred - batch["ret"][idx]) ** 2).mean()
                copt.zero_grad(set_to_none=True)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                copt.step()
                st["v"] += v_loss.item()
                lb, sb = batch["local"][idx].float(), batch["scalar"][idx].float()
                mask_b = batch["mask"][idx]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    t_logits, _ = teacher(lb, sb)
                t_logp = torch.log_softmax(masked_logits(t_logits.float(), mask_b), dim=1)
                if not warm and ep_i < p_epochs:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = policy(lb, sb)
                    _, logp, ent = policy_out(logits, mask_b, batch["action"][idx])
                    ratio = (logp - batch["logp"][idx]).exp()
                    mb = batch["adv"][idx]
                    pg = -torch.min(ratio * mb, ratio.clamp(1 - a.clip, 1 + a.clip) * mb).mean()
                    s_logp = torch.log_softmax(masked_logits(logits.float(), mask_b), dim=1)
                    # illegal actions carry ~0 teacher mass; nan_to_num drops them
                    kl_t = (t_logp.exp() * (t_logp - s_logp)).nan_to_num(0.0).sum(1).mean()
                    loss = pg - a.ent * ent.mean() + a.kl_coef * kl_t
                    popt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                    popt.step()
                    with torch.no_grad():
                        st["pg"] += pg.item()
                        st["ent"] += ent.mean().item()
                        st["kl"] += (batch["logp"][idx] - logp).mean().item()
                        st["clipfrac"] += ((ratio - 1).abs() > a.clip).float().mean().item()
                        st["kl_teacher"] += kl_t.item()
                    npb += 1
                elif warm:
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, _ = policy(lb, sb)
                    s_logp = torch.log_softmax(masked_logits(logits.float(), mask_b), dim=1)
                    st["kl_teacher"] += (t_logp.exp() * (t_logp - s_logp)).nan_to_num(0.0) \
                        .sum(1).mean().item()
                    npb += 1
                nb += 1
        t_opt = time.perf_counter() - t0
        for k in st:
            st[k] /= max(nb if k == "v" else npb, 1)

        total = done_turns + turns_per_iter
        row = {"iter": it, "total_turns": total, "warmup": warm,
               "sps": turns_per_iter / (t_roll + t_opt), "t_roll": round(t_roll, 2),
               "t_opt": round(t_opt, 2), "usable": round(n / turns_per_iter, 3),
               "orphans": roll.orphans, "overwrites": roll.overwrites,
               "return_mean": float(batch["ret"].mean()), "value_mean": float(batch["value"].mean()),
               "explained_var": ev, "elapsed": round(time.perf_counter() - t_start, 1),
               **{k: round(v, 5) for k, v in st.items()}}
        # explained variance per opponent slot: the failure mode to watch is
        # one slot's values being dragged by another's
        for k in range(n_ctx):
            m = batch["ctx"] == k
            if m.sum() > 100:
                r, v = batch["ret"][m], batch["value"][m]
                row[f"ev_{k}"] = round(float(1.0 - (r - v).var() / (r.var() + 1e-8)), 4)
        if ep_rows:
            rows = np.concatenate(ep_rows)
            sl, side = np.concatenate(ep_slot), np.concatenate(ep_side)
            cols = bcsim.EpisodeStats.COLUMNS
            winner = rows[:, cols.index("winner")]
            row["episodes"] = len(rows)
            row["rounds_mean"] = float(rows[:, cols.index("rounds")].mean())
            row["draw_rate"] = float((winner < 0).mean())
            for k, path in enumerate(opp_paths):
                m = sl == k + 1
                if m.any():
                    score = np.where(winner[m] < 0, 0.5, (winner[m] == side[m]).astype(float))
                    row[f"vs_{pathlib.Path(path).stem}"] = round(float(score.mean()), 4)
                    row[f"n_{pathlib.Path(path).stem}"] = int(m.sum())
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()
        if it % 5 == 0 or it < 3:
            vs = " ".join(f"{k[3:]} {v:.2f}" for k, v in row.items() if k.startswith("vs_"))
            print(f"it {it:5d} | {total / 1e6:7.1f}M | {'WARMUP ' if warm else ''}"
                  f"{row['sps']:>8,.0f} turns/s | ev {ev:+.3f} v {st['v']:.4f} | "
                  f"ent {st['ent']:.3f} kl {st['kl']:+.4f} klT {st['kl_teacher']:.4f} "
                  f"clip {st['clipfrac']:.3f} | {vs}",
                  flush=True)
        if roll.overwrites:
            raise RuntimeError(f"slot reused before closing ({roll.overwrites})")

        ckpt = {"net": policy.state_dict(), "critic": critic.state_dict(),
                "opt": popt.state_dict(), "copt": copt.state_dict(), "iter": it,
                "total_turns": total,
                "args": {**vars(a), "width": width, "blocks": blocks, "reward_weights": rw}}
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
