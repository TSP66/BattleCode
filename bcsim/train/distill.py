"""Distils the PPO teacher into the network that fits the judge's budget.

On-policy rather than from a buffer: the teacher drives the simulator and the
student is trained on exactly the states the teacher visits, so the student
never has to generalise to a distribution it will not see. Both networks see
the same observation, so the target is just the teacher's masked action
distribution.

    python -m train.distill --teacher runs/v1/latest.pt --iters 400
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

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--maps", default=str(ROOT / "maps"))
    p.add_argument("--teacher", default=str(ROOT / "runs/v1/latest.pt"))
    p.add_argument("--out", default=str(ROOT / "runs/distill"))
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--hidden", type=int, default=256)
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

    ck = torch.load(a.teacher, map_location=dev, weights_only=False)
    ta = ck["args"]
    teacher = ActorCritic(bcsim.N_CHANNELS, bcsim.N_SCALARS, bcsim.N_ACTIONS,
                          width=ta["width"], blocks=ta["blocks"]).to(dev)
    teacher.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["net"].items()})
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = Student(bcsim.N_CHANNELS, bcsim.N_SCALARS, bcsim.N_ACTIONS,
                      width=a.width, blocks=a.blocks, hidden=a.hidden).to(dev)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"teacher iter {ck['iter']} (width {ta['width']} blocks {ta['blocks']})")
    print(f"student {n_params/1e6:.3f}M params, {student.macs():,} MACs "
          f"= {student.macs()*23.2/1e6:.1f}M judge points", flush=True)

    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.0)
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
        student.eval()
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

        student.train()
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
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
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
            torch.save({"net": student.state_dict(), "iter": it,
                        "args": vars(a), "teacher_iter": ck["iter"]}, out / "latest.pt")


if __name__ == "__main__":
    main()
