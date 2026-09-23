"""Squeezing a dragon's remembered map into the 56 bits a sonar can carry.

A split makes a child with a blank memory standing where its parent stood. The
parent knows the corridors behind it, where pearls were, which way it came --
and 1.0.0 lets it send 64 bits in a chosen direction. Eight of those are a
fixed tag so a child that hears several sonars on its first turn can tell which
one was its parent (see SONAR_TAG). That leaves 56 bits to say something
useful.

What is learned here is a codec, not a policy:

    encoder   the parent's wide planes (bc_memory.hpp `wide`) -> 56 bits
    decoder   56 bits + the child's own live window -> those planes back

and the loss is reconstruction. Doing it this way rather than "emit four
latents and let PPO work it out" is deliberate: **no gradient crosses between
dragons**. The message leaves the parent, goes through the simulator as data,
and arrives at the child, so the child's loss can never reach the parent's
encoder. A latent code would have to be discovered by credit assignment
through the reward, which is the slowest signal available. Reconstruction is
supervised, needs no reward at all, and is measurable before any PPO is run.

The decoder is conditioned on the child's own 7x7 window, which is the trick
that makes 56 bits go further than it sounds: at the moment of a split the
child can see most of what is near it, so the code does not have to spend bits
on any of that. It only has to carry what the child cannot see -- the far
structure, the parts of the map behind them both.

What comes out is planes, not advice, and those planes go where the child's own
memory planes would be. A newborn's memory is empty; this gives it a prior.

    python -m train.memcodec collect --ckpt runs/pyr_warm/latest.pt --turns 400000
    python -m train.memcodec train --bits 56
    python -m train.memcodec train --bits 56 --no-window   # ablation

Both halves are metered: the encoder runs on the parent and the decoder on the
child, both inside the judge's per-turn budget, so `--report-cost` prices them
with train.budget before anything is trained. Encoder 3.3M points, decoder 14.8M,
18.1M together against the 81M a turn available -- with the pyramid's 51.2M that
is 69.3M, so it fits with 11.7M spare.

Measured on 120,000 planes from real play, 10% held out, as the share of variance
recovered that predicting the training mean leaves:

    56 bits, decoder sees the child's window     62.2%
    56 bits, decoder blind to it                 55.8%
     8 bits, decoder sees the child's window     50.9%

**Eight bits already get 50.9%, and the other forty-eight buy 11 points.** So the
message channel is not the bottleneck and "use all 64 bits" is the wrong target:
a newborn's memory is largely predictable from what it can see plus a little side
information, and what is scarce is information the child cannot already infer --
of which there is not 56 bits. Conditioning on the child's window is worth 6.4
points, which is the reason it is built that way.

Per channel at 56 bits, the code carries terrain and history well and enemies not
at all: remembered-age 0.93 near / 0.87 far, pearl countdown 0.86 far, kelp 0.74
near, visit recency 0.64 near -- against **0.20 for near enemy sightings and 0.09
for far**. Enemy positions are sparse and stale within a few turns, so they do
not survive compression. A message should carry terrain and where the parent has
been, not where it last saw an enemy.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                    # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/memcodec"

# The eight bits that say "your parent sent this". A child hears every sonar
# that reaches it, and on its first turn it may hear several, so the tag has to
# be recognisable rather than merely present: this value is one byte with four
# transitions, which no run of zeros or ones can be mistaken for.
SONAR_TAG = 0b10110100
TAG_BITS = 8
PAYLOAD_BITS = 64 - TAG_BITS                    # 56


class Encoder(nn.Module):
    """Wide planes -> `bits` logits, binarised by sign with a straight-through
    gradient. Kept small: this runs on the parent, every turn it broadcasts."""

    def __init__(self, n_wide_ch: int, side: int, bits: int = PAYLOAD_BITS,
                 width: int = 16, width2: int = 32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_wide_ch, width, 3, padding=1, bias=False),
            nn.GroupNorm(4, width), nn.SiLU(),
            nn.AvgPool2d(2),                                    # 15 -> 7
            nn.Conv2d(width, width2, 3, padding=1, bias=False),
            nn.GroupNorm(4, width2), nn.SiLU(),
            nn.Flatten())
        self.head = nn.Linear(width2 * (side // 2) * (side // 2), bits)

    def forward(self, wide):
        return self.head(self.body(wide))


def binarise(logits: torch.Tensor) -> torch.Tensor:
    """-1/+1 by sign, with the gradient of a tanh passed straight through.

    The forward value is what actually travels in the message, so it has to be
    exactly one bit per channel; the backward path has to be something other
    than zero, hence the surrogate.
    """
    soft = torch.tanh(logits)
    hard = torch.where(logits >= 0, 1.0, -1.0)
    return soft + (hard - soft).detach()


class Decoder(nn.Module):
    """`bits` + optionally the child's own window -> the parent's wide planes.

    The window branch is what lets the code skip everything the child can
    already see. Set use_window False for the ablation that says how much that
    is worth.
    """

    def __init__(self, n_wide_ch: int, side: int, n_local_ch: int,
                 bits: int = PAYLOAD_BITS, use_window: bool = True,
                 hidden: int = 256, width: int = 32):
        super().__init__()
        self.side = side
        self.width = width
        self.use_window = use_window
        self.n_wide_ch = n_wide_ch
        in_dim = bits
        if use_window:
            self.win = nn.Sequential(
                nn.Conv2d(n_local_ch, 16, 3, padding=1, bias=False),
                nn.GroupNorm(4, 16), nn.SiLU(),
                nn.Conv2d(16, 8, 1, bias=False), nn.GroupNorm(4, 8), nn.SiLU(),
                nn.Flatten())
            in_dim += 8 * 7 * 7
        self.seed = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                  nn.Linear(hidden, width * 5 * 5), nn.SiLU())
        # 5 -> 15 in one nearest-neighbour step, then convolutions to clean it
        self.up = nn.Sequential(
            nn.Upsample(size=(side, side), mode="nearest"),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(4, width), nn.SiLU(),
            nn.Conv2d(width, n_wide_ch, 3, padding=1))

    def forward(self, code, local=None):
        parts = [code]
        if self.use_window:
            if local is None:
                raise ValueError("this decoder was built to use the child's window")
            parts.append(self.win(local))
        h = self.seed(torch.cat(parts, dim=1))
        h = h.view(-1, self.width, 5, 5)
        return self.up(h)


class Codec(nn.Module):
    def __init__(self, n_wide_ch: int, side: int, n_local_ch: int,
                 bits: int = PAYLOAD_BITS, use_window: bool = True):
        super().__init__()
        self.enc = Encoder(n_wide_ch, side, bits)
        self.dec = Decoder(n_wide_ch, side, n_local_ch, bits, use_window)
        self.bits = bits

    def forward(self, wide, local=None):
        code = binarise(self.enc(wide))
        return self.dec(code, local), code


# ------------------------------------------------------------------ collect
def collect(a) -> None:
    """Records wide planes and the matching live window from real play.

    Sampled from a policy's own games rather than from random walks: the
    memories that need compressing are the ones a trained dragon actually
    builds, and those are nothing like the memories of a random one.
    """
    from train.finetune import load_policy
    from train.net import masked_logits

    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    net = load_policy(str(ROOT / a.ckpt) if not pathlib.Path(a.ckpt).is_absolute()
                      else a.ckpt, dev)[0].eval()
    wants = getattr(net, "wants_wide", False)
    maps_dir = pathlib.Path(a.maps if pathlib.Path(a.maps).is_absolute()
                            else ROOT / a.maps)
    maps = [p.read_text() for p in sorted(maps_dir.glob("*.map"))]
    env = bcsim.BattlecodeVecEnv(maps, num_envs=a.envs, num_threads=a.threads,
                                 seed=a.seed, wide=True)
    obs = env.reset()
    rng = np.random.default_rng(a.seed)
    keep_w, keep_l = [], []
    got = 0
    t0 = time.time()
    while got < a.turns:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            lg, _ = net(torch.from_numpy(obs.local).to(dev),
                        torch.from_numpy(obs.scalar).to(dev),
                        torch.from_numpy(env.wide).to(dev) if wants else None)
            lg = masked_logits(lg.float(), torch.from_numpy(obs.mask).to(dev).bool())
            act = torch.distributions.Categorical(logits=lg).sample()
        # A dragon whose memory is still blank has nothing to compress, and
        # early turns are almost all blank, so rows are taken by how much the
        # memory actually holds rather than uniformly.
        filled = (env.wide[:, :6] > 0).mean(axis=(1, 2, 3))
        pick = np.flatnonzero((filled > a.min_filled) & (rng.random(a.envs) < a.keep))
        if len(pick):
            keep_w.append(env.wide[pick].astype(np.float16))
            keep_l.append(obs.local[pick].astype(np.float16))
            got += len(pick)
        obs, _, _ = env.step(act.to(torch.int32).cpu().numpy())
        if got and len(keep_w) % 200 == 0:
            print(f"  {got:,}/{a.turns:,} rows, {time.time() - t0:.0f}s", flush=True)
    env.close()
    w = np.concatenate(keep_w)[:a.turns]
    l = np.concatenate(keep_l)[:a.turns]
    dest = OUT / "planes.npz"
    np.savez(dest, wide=w, local=l)
    print(f"{len(w):,} rows -> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
    print(f"wide {w.shape} local {l.shape}")


# ------------------------------------------------------------------- train
def channel_report(pred: np.ndarray, true: np.ndarray, base: np.ndarray) -> dict:
    """Per-channel explained variance against predicting the training mean.

    A raw MSE says nothing on its own here: most of these planes are mostly
    zero, so a decoder that emits zeros everywhere scores well. What matters is
    how much of the variance the 56 bits recover that the mean does not.
    """
    out = {}
    for c in range(true.shape[1]):
        t, p, b = true[:, c], pred[:, c], base[c]
        var = float(((t - b) ** 2).mean())
        mse = float(((t - p) ** 2).mean())
        out[f"ev_ch{c}"] = round(1.0 - mse / var, 4) if var > 1e-12 else None
    return out


def train(a) -> None:
    src = OUT / "planes.npz"
    if not src.exists():
        raise SystemExit(f"no data at {src}; run `collect` first")
    d = np.load(src)
    wide, local = d["wide"], d["local"]
    n = len(wide)
    n_va = max(1, int(n * 0.1))
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)

    W = torch.from_numpy(wide)
    L = torch.from_numpy(local)
    tr = slice(0, n - n_va)
    va = slice(n - n_va, n)
    print(f"{n:,} rows: {n - n_va:,} train, {n_va:,} held out", flush=True)

    side = wide.shape[-1]
    net = Codec(wide.shape[1], side, local.shape[1], bits=a.bits,
                use_window=not a.no_window).to(dev)
    n_enc = sum(q.numel() for q in net.enc.parameters())
    n_dec = sum(q.numel() for q in net.dec.parameters())
    print(f"encoder {n_enc:,} params, decoder {n_dec:,} params, {a.bits} bits, "
          f"window {'off' if a.no_window else 'on'}", flush=True)

    from train.budget import PTS_CONV, PTS_DOT, USABLE, price
    ce = price(net.enc, W[:1].float().to(dev))
    cd = price(net.dec, torch.zeros(1, a.bits, device=dev),
               *([] if a.no_window else [L[:1].float().to(dev)]))
    print(f"encoder {ce.mac:,} MAC = {ce.points:,.0f} pts (on the parent)", flush=True)
    print(f"decoder {cd.mac:,} MAC = {cd.points:,.0f} pts (on the child)", flush=True)
    print(f"together {ce.points + cd.points:,.0f} of {USABLE:,} available", flush=True)
    if a.report_cost:
        return

    # the baseline every number is measured against: emit the training mean
    base = W[tr].float().mean(dim=0).to(dev)
    base_np = base.mean(dim=(1, 2)).cpu().numpy()

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * ((n - n_va + a.batch - 1) // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)

    def run_val():
        net.eval()
        preds, trues = [], []
        with torch.no_grad():
            for s in range(va.start, n, a.batch):
                e = min(s + a.batch, n)
                w = W[s:e].float().to(dev)
                l = None if a.no_window else L[s:e].float().to(dev)
                p, _ = net(w, l)
                preds.append(p.cpu())
                trues.append(w.cpu())
        net.train()
        return torch.cat(preds).numpy(), torch.cat(trues).numpy()

    OUT.mkdir(parents=True, exist_ok=True)
    log = []
    best = None
    t0 = time.time()
    nt = n - n_va
    for ep in range(a.epochs):
        perm = torch.randperm(nt)
        tot = 0.0
        for s in range(0, nt, a.batch):
            idx = perm[s:s + a.batch]
            w = W[idx].float().to(dev)
            l = None if a.no_window else L[idx].float().to(dev)
            pred, code = net(w, l)
            loss = F.mse_loss(pred, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.detach().item() * len(idx)
        pred, true = run_val()
        mse = float(((pred - true) ** 2).mean())
        bmse = float(((true - base.cpu().numpy()[None]) ** 2).mean())
        # how much of the code is actually used: a bit that is always the same
        # carries nothing
        with torch.no_grad():
            _, code = net(W[va][:4096].float().to(dev),
                          None if a.no_window else L[va][:4096].float().to(dev))
            used = int(((code > 0).float().mean(0) - 0.5).abs().lt(0.49).sum())
        row = {"epoch": ep, "train_mse": round(tot / nt, 6), "val_mse": round(mse, 6),
               "mean_baseline_mse": round(bmse, 6),
               "ev_overall": round(1.0 - mse / bmse, 4),
               "live_bits": used, "elapsed": round(time.time() - t0),
               **channel_report(pred, true, base_np)}
        log.append(row)
        print(json.dumps(row), flush=True)
        if best is None or mse < best:
            best = mse
            torch.save({"codec": net.state_dict(), "eval": row,
                        "args": {"bits": a.bits, "use_window": not a.no_window,
                                 "n_wide_ch": int(wide.shape[1]), "side": int(side),
                                 "n_local_ch": int(local.shape[1]),
                                 "tag": SONAR_TAG, "tag_bits": TAG_BITS}},
                       OUT / ("codec_nowin.pt" if a.no_window else "codec.pt"))
    (OUT / ("train_nowin_log.json" if a.no_window else "train_log.json")).write_text(
        json.dumps(log, indent=1))
    print(f"best held-out mse {best:.6f}, {log[-1]['ev_overall']:.1%} of the variance "
          f"the mean leaves")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--ckpt", required=True)
    c.add_argument("--maps", default="maps-all")
    c.add_argument("--turns", type=int, default=400_000)
    c.add_argument("--envs", type=int, default=256)
    c.add_argument("--threads", type=int, default=8)
    c.add_argument("--keep", type=float, default=0.02)
    c.add_argument("--min-filled", type=float, default=0.02,
                   help="skip rows whose memory is still nearly blank")
    c.add_argument("--seed", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--bits", type=int, default=PAYLOAD_BITS)
    t.add_argument("--no-window", action="store_true",
                   help="ablation: decode without the child's own view")
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--batch", type=int, default=512)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--wd", type=float, default=0.01)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--report-cost", action="store_true",
                   help="price the two halves and stop")
    a = p.parse_args()
    {"collect": collect, "train": train}[a.cmd](a)


if __name__ == "__main__":
    main()
