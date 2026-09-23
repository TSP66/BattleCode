"""Checks the C++ bot's forward pass is the checkpoint's, turn by turn.

Reads the bot's -DBC_DUMP output (observation, mask, logits and chosen action
per turn), runs the checkpoint in PyTorch on those same observations with its
weights rounded to bf16 as the bot stores them, and requires the logits to
agree and the argmax to be the action the bot played. Together with
parity_obs.py (bot observation == simulator observation) this covers the
whole path from protocol text to move.

    python parity_net.py ckpt.pt dump.txt [dump2.txt ...]
"""

import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]  # repo root
sys.path.insert(0, str(ROOT / "bcsim"))

import bcsim                                           # noqa: E402
from train.net import ActorCritic, masked_logits      # noqa: E402

C, S, A = bcsim.N_CHANNELS, bcsim.N_SCALARS, bcsim.N_ACTIONS
TOL = 0.02          # bf16 weights, fast_exp and summation order all differ a hair


def main() -> None:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    state = {}
    for k, v in ck["net"].items():
        u = v.float().contiguous().view(torch.int32)
        state[k.replace("_orig_mod.", "")] = (((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16) \
            .view(torch.float32)
    # a clone trained with remembered features wants more scalars than the
    # observation has; the bot appends them in the order the checkpoint names
    n_sc = int(state["scalar.0.weight"].shape[1])
    net = ActorCritic(C, n_sc, A, width=ck["args"]["width"], blocks=ck["args"]["blocks"])
    net.load_state_dict(state)
    net.eval()

    rows = []
    for path in sys.argv[2:]:
        rows += [np.array(l.split()[1:], np.float64) for l in open(path)
                 if l.startswith("DUMP")]
    ran = [r for r in rows if r[0] == 1]
    if not ran:
        print("no turn ran the network")
        sys.exit(1)
    n_dump = len(ran[0]) - (2 + C * 49 + 2 * A)      # every scalar the bot fed
    if n_dump < n_sc:
        print(f"the bot dumped {n_dump} scalars, the checkpoint wants {n_sc}")
        sys.exit(1)
    loc = np.stack([r[2:2 + C * 49] for r in ran]).reshape(-1, C, 7, 7)
    sc = np.stack([r[2 + C * 49:2 + C * 49 + n_sc] for r in ran])
    base = 2 + C * 49 + n_dump
    mask = np.stack([r[base:base + A] for r in ran]) > 0
    lg = np.stack([r[base + A:base + 2 * A] for r in ran])
    act = np.array([int(r[1]) for r in ran])
    with torch.no_grad():
        tl, _ = net(torch.tensor(loc, dtype=torch.float32), torch.tensor(sc, dtype=torch.float32))
        want = masked_logits(tl, torch.from_numpy(mask)).argmax(1).numpy()
    err = np.abs(tl.numpy() - lg).max(axis=1)
    # a near tie may legitimately flip under bf16; only count clear ones
    top2 = np.sort(np.where(mask, tl.numpy(), -1e9), axis=1)[:, -2:]
    clear = (top2[:, 1] - top2[:, 0]) > 2 * TOL
    wrong = (want != act) & clear
    print(f"{len(ran)} network turns: max logit error {err.max():.4f} (tolerance {TOL}), "
          f"{int((want == act).sum())} argmax agree, {int(wrong.sum())} clear disagreements")
    sys.exit(1 if err.max() > TOL or wrong.any() else 0)


if __name__ == "__main__":
    main()
