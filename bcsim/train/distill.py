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
from train.net import ActorCritic, masked_logits  # noqa: E402
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
    p.add_argument("--arch", choices=["actorcritic", "student"], default="actorcritic",
                   help="actorcritic ships through export_cpp; student has no C++ path")
    p.add_argument("--width", type=int, default=None, help="default 64 (actorcritic) / 32")
    p.add_argument("--blocks", type=int, default=None, help="default 4 (actorcritic) / 2")
    p.add_argument("--hidden", type=int, default=None, help="default 512 (actorcritic) / 256")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--temp", type=float, default=1.0)
    return p.parse_args()


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

    ac = a.arch == "actorcritic"
    a.width = a.width or (64 if ac else 32)
    a.blocks = a.blocks or (4 if ac else 2)
    a.hidden = a.hidden or (512 if ac else 256)
    if ac:
        net = ActorCritic(bcsim.N_CHANNELS, bcsim.N_SCALARS, bcsim.N_ACTIONS,
                          width=a.width, blocks=a.blocks, hidden=a.hidden).to(dev)
        student = lambda local, scalar: net(local, scalar)[0]      # noqa: E731
        params = list(net.parameters())
    else:
        net = Student(bcsim.N_CHANNELS, bcsim.N_SCALARS, bcsim.N_ACTIONS,
                      width=a.width, blocks=a.blocks, hidden=a.hidden).to(dev)
        student, params = net, list(net.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"teacher iter {ck['iter']} (width {ta['width']} blocks {ta['blocks']})")
    print(f"student {a.arch} {a.width}x{a.blocks} hidden {a.hidden}: "
          f"{n_params/1e6:.3f}M params", flush=True)

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)

    env = bcsim.BattlecodeVecEnv(bcsim.load_maps(a.maps), num_envs=a.envs,
                                 num_threads=a.threads, seed=11,
                                 closure_capacity=max(8192, a.envs * 160))
    obs = env.reset()
    n = a.envs
    buf_local = torch.zeros(a.steps, n, bcsim.N_CHANNELS, bcsim.WINDOW, bcsim.WINDOW,
                            dtype=torch.float16, device=dev)
    buf_scalar = torch.zeros(a.steps, n, bcsim.N_SCALARS, dtype=torch.float16, device=dev)
    buf_mask = torch.zeros(a.steps, n, bcsim.N_ACTIONS, dtype=torch.bool, device=dev)
    buf_logp = torch.zeros(a.steps, n, bcsim.N_ACTIONS, dtype=torch.float16, device=dev)
    t_start = time.perf_counter()

    for it in range(a.iters):
        net.eval()
        for t in range(a.steps):
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
            action = torch.multinomial(logp.exp(), 1).squeeze(1)
            obs, _, _ = env.step(action.to(torch.int32).cpu().numpy())

        flat = a.steps * n
        fl = buf_local.reshape(flat, bcsim.N_CHANNELS, bcsim.WINDOW, bcsim.WINDOW)
        fs = buf_scalar.reshape(flat, bcsim.N_SCALARS)
        fm = buf_mask.reshape(flat, bcsim.N_ACTIONS)
        fp = buf_logp.reshape(flat, bcsim.N_ACTIONS)

        net.train()
        tot_kl, tot_agree, nb = 0.0, 0.0, 0
        for _ in range(a.epochs):
            perm = torch.randperm(flat, device=dev)
            for s in range(0, flat, a.minibatch):
                idx = perm[s:s + a.minibatch]
                m = fm[idx]
                t_logp = fp[idx].float()
                s_logits = student(fl[idx].float(), fs[idx].float())
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
                        "args": {**vars(a), "width": a.width, "blocks": a.blocks},
                        "teacher_iter": ck["iter"]}, out / "latest.pt")


if __name__ == "__main__":
    main()
