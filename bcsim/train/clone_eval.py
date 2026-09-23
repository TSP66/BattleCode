"""Head-to-head win rates for clones, including ones with memory features.

yardstick.py's evaluate() plays the games; this supplies the policies. A
checkpoint from imitate2.py names its extra features, and its policy keeps a
clone_features.MemoryTracker per (env, dragon), fed every turn the dragon
takes, exactly as the features were built from replays. Runs on CUDA, MPS or
CPU (no CUDA graphs), greedy, both sides of every map.

    python -m train.clone_eval --ckpt ../runs/i2/full_mem/best.pt \
        --opponents ../runs/submitted/v10.pt,../runs/anchors/sss_r3_bc_64x4.pt \
        --maps ../runs/eval_maps --games 12 --dump ../runs/i2/full_mem/eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                                               # noqa: E402
from train.clone_features import MemoryTracker, msg_rows   # noqa: E402
from train.imitate2 import LC_PEARL_TIME                   # noqa: E402,F401
from train.net import ActorCritic, masked_logits          # noqa: E402

TRACKED = ("mem", "memfar")


def load(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["net"].items()}
    n_ch = sd["stem.0.weight"].shape[1]
    n_sc = sd["scalar.0.weight"].shape[1]
    a = ck["args"]
    net = ActorCritic(n_ch, n_sc, bcsim.N_ACTIONS, width=a["width"], blocks=a["blocks"],
                      hidden=sd["fuse.0.weight"].shape[0])
    net.load_state_dict(sd)
    net.to(dev).eval()
    return net, ck.get("features", [])


class Policy:
    """Greedy policy; stateful when it uses remembered features."""

    SMALL = 64      # batches below this run on a CPU copy: a GPU round trip costs more

    def __init__(self, net, features, dev):
        self.net, self.features, self.dev = net, list(features), dev
        self.cpu_net = None
        if dev.type != "cpu":
            import copy
            self.cpu_net = copy.deepcopy(net).to("cpu").eval()
        unknown = set(self.features) - set(TRACKED) - {"msg"}
        if unknown:
            raise ValueError(f"no online version of features {sorted(unknown)}")
        self.stateful = any(f in TRACKED for f in self.features)
        self.tracker = MemoryTracker() if self.stateful else None

    def _forward(self, local, scalar, mask):
        small = self.cpu_net is not None and len(local) < self.SMALL
        net, dev = (self.cpu_net, torch.device("cpu")) if small else (self.net, self.dev)
        with torch.inference_mode():
            logits, _ = net(torch.from_numpy(np.ascontiguousarray(local, np.float32)).to(dev),
                            torch.from_numpy(np.ascontiguousarray(scalar, np.float32)).to(dev))
            m = torch.from_numpy(mask).to(dev).bool()
            return masked_logits(logits.float(), m).argmax(1).to(torch.int32).cpu().numpy()

    def __call__(self, local, scalar, mask):
        return self._forward(local, scalar, mask)

    def rows(self, obs, rows):
        idx = np.flatnonzero(rows)
        local, scalar = obs.local[idx], obs.scalar[idx]
        keys = [(int(e), int(d)) for e, d in zip(idx, obs.dragon_id[idx])]
        got = self.tracker.step(keys, local, scalar, obs.round[idx],
                                want=[f for f in self.features if f in TRACKED])
        parts = [scalar]
        for f in self.features:             # the order imitate2.py concatenated them in
            parts.append(msg_rows(scalar, obs.msgs[idx]) if f == "msg" else got[f])
        return self._forward(local, np.concatenate(parts, 1).astype(np.float32), obs.mask[idx])

    def forget(self, env):
        self.tracker.forget(lambda k: k[0] == env)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--opponents", required=True, help="comma list of checkpoints")
    p.add_argument("--maps", default="../runs/eval_maps")
    p.add_argument("--games", type=int, default=12, help="per (opponent, map), half each side")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--max-seconds", type=float, default=10_800)
    p.add_argument("--dump", default="")
    a = p.parse_args()
    from train.yardstick import evaluate
    torch.set_num_threads(4)
    dev = torch.device(a.device)
    maps = bcsim.load_maps(a.maps)
    names = [f.stem for f in sorted(pathlib.Path(a.maps).glob("*.map"))]
    net, feats = load(a.ckpt, dev)
    learner = Policy(net, feats, dev)
    opps = []
    for o in filter(None, a.opponents.split(",")):
        onet, ofeats = load(o, dev)
        opps.append({"name": pathlib.Path(o).parent.name + "/" + pathlib.Path(o).stem
                     if pathlib.Path(o).stem in ("best", "latest") else pathlib.Path(o).stem,
                     "act": Policy(onet, ofeats, dev), "path": o})
    t0 = time.perf_counter()
    res = evaluate(learner, opps, maps, names, games=a.games, threads=a.threads,
                   seed=a.seed, max_seconds=a.max_seconds, progress=300)
    for name, s in res["summary"].items():
        print(f"{name:32s} n {s['n']:4d}  score {s['score']:.3f}  win {s['win']:.3f}  "
              f"draw {s['draw']:.3f}", flush=True)
    print(f"{time.perf_counter() - t0:.0f}s", flush=True)
    if a.dump:
        with open(a.dump, "a") as fh:
            fh.write(json.dumps({"ckpt": a.ckpt, "features": feats, "games": a.games,
                                 "seed": a.seed, "time": time.time(), **res}) + "\n")


if __name__ == "__main__":
    main()
