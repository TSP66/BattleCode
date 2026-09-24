"""Distils the PPO teacher into the network that fits the judge's budget.

On-policy rather than from a buffer: the teacher drives the simulator and the
student is trained on exactly the states the teacher visits, so the student
never has to generalise to a distribution it will not see. Both networks see
the same observation, so the target is just the teacher's masked action
distribution.

    python -m train.distill --teacher ../runs/ft6/snapshots/turns_112721920.pt \\
        --width 48 --blocks 3 --out ../runs/distill_48x3

--arch actorcritic (the default) trains the same ActorCritic the C++ bot runs
(mybot/net.hpp), at any width/blocks/hidden, and saves it in train.py's layout
("net", "args" with width and blocks), so yardstick, export_cpp, check_bot and
submit.sh take it as it is. --arch student trains train/student.py's ReLU,
norm-free net, which is cheaper but has NO C++ implementation yet.
See DISTILL.md at the repo root.
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
from train import memfeat                        # noqa: E402
from train import net as net_mod                  # noqa: E402
from train.net import ActorCritic, PyramidActorCritic, masked_logits  # noqa: E402
from train.student import Student                 # noqa: E402
from train.yardstick import load_net              # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--maps", default=str(ROOT / "runs/ft3/maps"), help="the 8 live maps")
    p.add_argument("--teacher", default=str(ROOT / "runs/v1/latest.pt"))
    p.add_argument("--out", default=str(ROOT / "runs/distill"))
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--wide-width", type=int, default=24, help="--arch pyramid: memory branch width")
    p.add_argument("--wide-blocks", type=int, default=2, help="--arch pyramid: memory branch blocks")
    p.add_argument("--arch", choices=["actorcritic", "student", "pyramid", "convlstm"],
                   default="actorcritic",
                   help="actorcritic ships through export_cpp; student has no C++ path; "
                        "convlstm is recurrent and trains through a different loop")
    p.add_argument("--hid-ch", type=int, default=24,
                   help="--arch convlstm: channels of the recurrent state grid")
    p.add_argument("--width", type=int, default=None, help="default 64 (actorcritic) / 32")
    p.add_argument("--blocks", type=int, default=None, help="default 4 (actorcritic) / 2")
    p.add_argument("--hidden", type=int, default=None, help="default 512 (actorcritic) / 256")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--memchan", action="store_true",
                   help="distil with the team's shared map on (train/memfeat.py). Match "
                        "this to the PPO run that follows: the channel rewrites memfar, "
                        "and a student cloned on the unmerged row starts PPO reading a "
                        "feature whose distribution has shifted under it")
    return p.parse_args()


def distill_recurrent(a, teacher, ck, dev, out, log) -> None:
    """Distil into a ConvLSTM, which cannot use the shuffled buffer above.

    The flat path records a batch of turns, shuffles them and replays them
    through the student. A recurrent student cannot be trained that way: its
    output depends on a state built from that dragon's own earlier turns, and
    shuffling destroys the order. So this trains online, stepping the env and
    carrying each dragon's state through a StatePool keyed by uid.

    **The state is detached between turns.** Gradients therefore say how the
    state it already has affects this turn's logits, but not how a turn's input
    should change the state for later use -- a one-step objective. Real
    truncated BPTT means keeping the graph across several of a dragon's turns,
    which interleave with thirty other dragons in the same env, and that is a
    bigger change than tonight allows. So this is a floor on what recurrence can
    do here, not a measurement of it, and the number it produces should be read
    that way.
    """
    from train.recurrent import RecurrentActorCritic, StatePool, head_and_face

    net = RecurrentActorCritic(bcsim.N_CHANNELS, bcsim.WIDE_CH, bcsim.N_ACTIONS,
                               near_width=a.width, near_blocks=a.blocks,
                               hid_ch=a.hid_ch, side=bcsim.WIDE_SIDE,
                               hidden=a.hidden).to(dev)
    params = list(net.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)
    print(f"teacher iter {ck['iter']}")
    print(f"student convlstm {a.width}x{a.blocks} state {a.hid_ch}x"
          f"{bcsim.WIDE_SIDE}x{bcsim.WIDE_SIDE} hidden {a.hidden}: "
          f"{sum(p.numel() for p in params) / 1e6:.3f}M params", flush=True)

    env = bcsim.BattlecodeVecEnv(bcsim.load_maps(a.maps), num_envs=a.envs,
                                 num_threads=a.threads, seed=11,
                                 closure_capacity=max(8192, a.envs * 160), wide=True)
    obs = env.reset()
    n = a.envs
    # a slot per live dragon; the unit limit caps how many there can be
    pool = StatePool(n, 96, a.hid_ch, bcsim.WIDE_SIDE, device=dev)
    envs_idx = np.arange(n)
    prev_round = np.asarray(obs.round, np.int64).copy()
    t_start = time.perf_counter()

    for it in range(a.iters):
        tot_kl, tot_agree, nb = 0.0, 0.0, 0
        opt.zero_grad(set_to_none=True)
        for t in range(a.steps):
            local = torch.from_numpy(obs.local).to(dev, non_blocking=True)
            scalar = torch.from_numpy(obs.scalar).to(dev, non_blocking=True)
            mask = torch.from_numpy(obs.mask).to(dev, non_blocking=True).bool()
            wide = torch.from_numpy(env.wide).to(dev, non_blocking=True)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                t_logits, _ = teacher(local, scalar)
            t_logp = F.log_softmax(masked_logits(t_logits.float(), mask), dim=1)

            slots = pool.slots_for(envs_idx, obs.uid)
            live = slots >= 0
            x, y, face, w, hgt = head_and_face(obs.scalar)
            sl = slots[live]
            h, c = pool.carry(sl, x[live], y[live], face[live], w[live], hgt[live])
            s_logits, _, (h2, c2) = net(local[live], scalar[live], wide[live], (h, c))
            s_logp = F.log_softmax(masked_logits(s_logits, mask[live]), dim=1)
            tp = t_logp[live].exp()
            kl = (tp * (t_logp[live] - s_logp)).nan_to_num(0.0).sum(dim=1).mean()
            (kl / a.steps).backward()
            pool.store(sl, h2, c2, x[live], y[live], face[live])
            with torch.no_grad():
                tot_kl += kl.item()
                tot_agree += (s_logp.argmax(1) == t_logp[live].argmax(1)).float().mean().item()
            nb += 1

            action = torch.multinomial(t_logp.exp(), 1).squeeze(1)
            obs, closures, _ = env.step(action.to(torch.int32).cpu().numpy())
            if len(closures.env):
                done = closures.done.astype(bool)
                if done.any():
                    pool.release(closures.env[done].astype(np.int64), closures.uid[done])
            # A finished game restarts its env, and every dragon in it is gone;
            # closures alone leak slots (see StatePool.release_envs).
            cur = np.asarray(obs.round, np.int64)
            restarted = np.flatnonzero(cur < prev_round)
            if len(restarted):
                pool.release_envs(restarted)
            prev_round = cur.copy()

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        row = {"iter": it, "kl": tot_kl / nb, "agree": tot_agree / nb,
               "turns": (it + 1) * a.steps * n, "live": int(pool.n - len(pool.free)),
               "evictions": pool.evictions, "pool_exhausted": pool.exhausted,
               "elapsed": round(time.perf_counter() - t_start, 1)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if it % 10 == 0:
            print(f"it {it:4d} | kl {row['kl']:.4f} | top-1 agree {row['agree']:.3f} "
                  f"| {row['turns']:,} turns | live {row['live']} "
                  f"| {row['elapsed']:.0f}s", flush=True)
        if it % 25 == 0 or it == a.iters - 1:
            torch.save({"net": net.state_dict(), "iter": it,
                        "total_turns": (it + 1) * a.steps * n,
                        "args": {**vars(a), "arch": "convlstm",
                                 "near_width": a.width, "near_blocks": a.blocks,
                                 "hid_ch": a.hid_ch, "side": bcsim.WIDE_SIDE}},
                       out / "latest.pt")
    env.close()


def main() -> None:
    a = parse()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "log.jsonl").open("a")

    # load_net reads width, blocks and the fuse width from the checkpoint, so
    # any policy checkpoint works (train, finetune, imitate, ratchet)
    teacher, ck = load_net(a.teacher, dev)
    ta = ck["args"]
    ck.setdefault("iter", -1)
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --memchan is only correct when the TEACHER was trained with the channel on,
    # and the reason is a base scalar rather than anything to do with memfar.
    #
    # num_msgs is scalar 12, inside the 14 every architecture reads. Broadcasting
    # four ways from every dragon takes it from 100% zeros to 78% non-zero
    # (0:22% 1:14% 2:28% 3:15% 4:20%, measured on maps-all), while a clone of a
    # real team was fitted where it is 0 in 89% of rows -- Sabotage-d submission
    # 3952, 813,413 rows. So the channel drives the teacher onto an input it
    # effectively never saw, its weight on that input is barely trained, and the
    # distribution the student is asked to copy is partly noise.
    #
    # Nor is there anything to gain. A teacher that cannot see memfar cannot act
    # on the shared map, so the merged columns are uninformative for predicting
    # its action and the student learns to ignore them either way. Distillation
    # cannot teach the use of a channel the teacher is blind to; only PPO can.
    if a.memchan and not bool(ta.get("memchan", False)):
        raise SystemExit(
            f"--memchan with a teacher that was not trained with it ({a.teacher}).\n"
            "Broadcasting moves num_msgs (scalar 12, which every net reads) from ~0 to\n"
            "1-4 on most turns, and a behaviour clone never saw that, so the target you\n"
            "would be copying is partly untrained behaviour. It buys nothing either: a\n"
            "teacher blind to memfar cannot act on the shared map, so the student learns\n"
            "to ignore it regardless. Distil without --memchan and let PPO learn the\n"
            "channel, or pass a teacher from a --memchan run.")

    ac = a.arch == "actorcritic"
    pyr = a.arch == "pyramid"
    if a.arch == "convlstm":
        # the pyramid's near branch, so the two are comparable at the same width
        a.width = a.width or 48
        a.blocks = a.blocks or 3
        a.hidden = a.hidden or 384
        return distill_recurrent(a, teacher, ck, dev, out, log)
    a.width = a.width or (64 if ac else 48 if pyr else 32)
    a.blocks = a.blocks or (4 if ac else 3 if pyr else 2)
    a.hidden = a.hidden or (512 if ac else 384 if pyr else 256)
    if pyr:
        # the warm start for a PPO run on the new architecture: no weight of a
        # 708-scalar net transfers, but the pyramid sees a superset of what the
        # teacher sees (radius 29 against radius 6, off the same memory), so
        # there is no information gap and it can match the teacher outright.
        net = PyramidActorCritic(bcsim.N_CHANNELS, bcsim.WIDE_CH, bcsim.N_ACTIONS,
                                 near_width=a.width, near_blocks=a.blocks,
                                 wide_width=a.wide_width, wide_blocks=a.wide_blocks,
                                 wide_side=bcsim.WIDE_SIDE, hidden=a.hidden).to(dev)
        student = lambda local, scalar, wide: net(local, scalar, wide)[0]   # noqa: E731
        params = list(net.parameters())
    elif ac:
        # pinned to the flat architecture's own width, not the env's current
        # row: this is the one that ships through export_cpp and mybot/net.hpp,
        # and the env's row grows as features are appended
        net = ActorCritic(bcsim.N_CHANNELS, net_mod.N_FLAT_SCALARS, bcsim.N_ACTIONS,
                          width=a.width, blocks=a.blocks, hidden=a.hidden).to(dev)
        student = lambda local, scalar, wide=None: net(local, scalar)[0]    # noqa: E731
        params = list(net.parameters())
    else:
        net = Student(bcsim.N_CHANNELS, bcsim.N_SCALARS, bcsim.N_ACTIONS,
                      width=a.width, blocks=a.blocks, hidden=a.hidden).to(dev)
        params = list(net.parameters())
        student = lambda local, scalar, wide=None: net(local, scalar)       # noqa: E731
    n_params = sum(p.numel() for p in params)
    print(f"teacher iter {ck['iter']} (width {ta['width']} blocks {ta['blocks']})")
    print(f"student {a.arch} {a.width}x{a.blocks} hidden {a.hidden}: "
          f"{n_params/1e6:.3f}M params", flush=True)

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)

    env = bcsim.BattlecodeVecEnv(bcsim.load_maps(a.maps), num_envs=a.envs,
                                 num_threads=a.threads, seed=11,
                                 closure_capacity=max(8192, a.envs * 160), wide=pyr)
    chan = memfeat.MemChannel(a.envs) if a.memchan else None
    if chan is not None:
        print("shared-map channel on: the student is cloned on the merged memfar row",
              flush=True)
    obs = env.reset()
    n = a.envs
    buf_local = torch.zeros(a.steps, n, bcsim.N_CHANNELS, bcsim.WINDOW, bcsim.WINDOW,
                            dtype=torch.float16, device=dev)
    buf_scalar = torch.zeros(a.steps, n, bcsim.N_SCALARS, dtype=torch.float16, device=dev)
    buf_mask = torch.zeros(a.steps, n, bcsim.N_ACTIONS, dtype=torch.bool, device=dev)
    buf_logp = torch.zeros(a.steps, n, bcsim.N_ACTIONS, dtype=torch.float16, device=dev)
    buf_wide = (torch.zeros(a.steps, n, bcsim.WIDE_CH, bcsim.WIDE_SIDE, bcsim.WIDE_SIDE,
                            dtype=torch.float16, device=dev) if pyr else None)
    t_start = time.perf_counter()

    for it in range(a.iters):
        net.eval()
        for t in range(a.steps):
            if chan is not None:
                chan.receive(obs)       # before anything reads the row
            local = torch.from_numpy(obs.local).to(dev, non_blocking=True)
            scalar = torch.from_numpy(obs.scalar).to(dev, non_blocking=True)
            mask = torch.from_numpy(obs.mask).to(dev, non_blocking=True).bool()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = teacher(local, scalar)
            logp = F.log_softmax(masked_logits(logits.float(), mask), dim=1)
            buf_local[t] = local.half()
            buf_scalar[t] = scalar.half()
            buf_mask[t] = mask
            buf_logp[t] = logp.half()
            if pyr:
                buf_wide[t] = torch.from_numpy(env.wide).to(dev, non_blocking=True).half()
            action = torch.multinomial(logp.exp(), 1).squeeze(1)
            acts = action.to(torch.int32).cpu().numpy()
            obs, _, _ = (env.step(acts, *chan.send(obs)) if chan is not None
                         else env.step(acts))

        flat = a.steps * n
        fl = buf_local.reshape(flat, bcsim.N_CHANNELS, bcsim.WINDOW, bcsim.WINDOW)
        fs = buf_scalar.reshape(flat, bcsim.N_SCALARS)
        fm = buf_mask.reshape(flat, bcsim.N_ACTIONS)
        fp = buf_logp.reshape(flat, bcsim.N_ACTIONS)
        fw = (buf_wide.reshape(flat, bcsim.WIDE_CH, bcsim.WIDE_SIDE, bcsim.WIDE_SIDE)
              if pyr else None)

        net.train()
        tot_kl, tot_agree, nb = 0.0, 0.0, 0
        for _ in range(a.epochs):
            perm = torch.randperm(flat, device=dev)
            for s in range(0, flat, a.minibatch):
                idx = perm[s:s + a.minibatch]
                m = fm[idx]
                t_logp = fp[idx].float()
                s_logits = student(fl[idx].float(), fs[idx].float(),
                                   fw[idx].float() if pyr else None)
                s_logp = F.log_softmax(masked_logits(s_logits, m), dim=1)
                # the teacher's -inf entries contribute nothing, so the KL is
                # taken over legal actions only and stays finite
                p = t_logp.exp()
                kl = (p * (t_logp - s_logp)).nan_to_num(0.0).sum(dim=1).mean()
                opt.zero_grad(set_to_none=True)
                kl.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                with torch.no_grad():
                    tot_kl += kl.item()
                    tot_agree += (s_logp.argmax(1) == t_logp.argmax(1)).float().mean().item()
                nb += 1
        sched.step()

        row = {"iter": it, "kl": tot_kl / nb, "agree": tot_agree / nb,
               "turns": (it + 1) * flat, "elapsed": round(time.perf_counter() - t_start, 1)}
        log.write(json.dumps(row) + "\n")
        log.flush()
        if it % 10 == 0:
            print(f"it {it:4d} | kl {row['kl']:.4f} | top-1 agree {row['agree']:.3f} "
                  f"| {row['turns']:,} turns | {row['elapsed']:.0f}s", flush=True)
        if it % 25 == 0 or it == a.iters - 1:
            # train.py's layout: args carries width/blocks for load_net / export_cpp
            torch.save({"net": net.state_dict(), "iter": it, "total_turns": (it + 1) * flat,
                        "args": {**vars(a), "width": a.width, "blocks": a.blocks,
                                 # load_policy dispatches on this; a pyramid also
                                 # needs its two branch shapes named the way it
                                 # rebuilds them
                                 **({"arch": "pyramid", "near_width": a.width,
                                     "near_blocks": a.blocks,
                                     "wide_width": a.wide_width,
                                     "wide_blocks": a.wide_blocks} if pyr else {})},
                        "teacher_iter": ck["iter"]}, out / "latest.pt")


if __name__ == "__main__":
    main()
