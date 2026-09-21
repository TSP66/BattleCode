"""Packs a checkpoint into the flat float16 blob the submitted bot loads.

The judge caps the upload at 4MB, so the weights ship as float16 and are
widened once at start-up. Everything that can be folded ahead of time is:
conv kernels arrive already reshaped for the im2col matmul the bot uses, and
linear weights arrive transposed, so the bot does no setup arithmetic beyond
the widening itself.

    python -m train.export --ckpt runs/v0/latest.pt --out mybot/weights.npz
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=str(ROOT / "ckpt/v0_iter_snapshot.pt"))
    p.add_argument("--out", default=str(ROOT / "mybot/weights.npz"))
    args = p.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    st = {k.replace("_orig_mod.", ""): v.float().numpy() for k, v in ck["net"].items()}
    cfg = ck["args"]
    width, blocks = cfg["width"], cfg["blocks"]

    out: dict[str, np.ndarray] = {}

    def conv(name: str, w: np.ndarray) -> None:
        """(Cout, Cin, 3, 3) -> (Cout, Cin*9), matching the im2col row order."""
        out[name] = w.reshape(w.shape[0], -1).astype(np.float16)

    def lin(name: str, w: np.ndarray, b: np.ndarray) -> None:
        out[name + "_w"] = w.astype(np.float16)
        out[name + "_b"] = b.astype(np.float32)

    def norm(name: str, w: np.ndarray, b: np.ndarray) -> None:
        out[name + "_w"] = w.astype(np.float32)
        out[name + "_b"] = b.astype(np.float32)

    conv("stem_c", st["stem.0.weight"])
    norm("stem_n", st["stem.1.weight"], st["stem.1.bias"])
    for i in range(blocks):
        conv(f"b{i}_c1", st[f"blocks.{i}.c1.weight"])
        norm(f"b{i}_n1", st[f"blocks.{i}.n1.weight"], st[f"blocks.{i}.n1.bias"])
        conv(f"b{i}_c2", st[f"blocks.{i}.c2.weight"])
        norm(f"b{i}_n2", st[f"blocks.{i}.n2.weight"], st[f"blocks.{i}.n2.bias"])
    # the 1x1 projection is a plain matmul over channels, no im2col needed
    out["flat_c"] = st["flat.0.weight"].reshape(32, -1).astype(np.float16)
    norm("flat_n", st["flat.1.weight"], st["flat.1.bias"])
    lin("sc0", st["scalar.0.weight"], st["scalar.0.bias"])
    lin("sc2", st["scalar.2.weight"], st["scalar.2.bias"])
    lin("fu0", st["fuse.0.weight"], st["fuse.0.bias"])
    lin("fu2", st["fuse.2.weight"], st["fuse.2.bias"])
    lin("pi", st["pi.weight"], st["pi.bias"])
    out["meta"] = np.array([width, blocks], dtype=np.int32)

    dest = pathlib.Path(args.out)
    np.savez(dest, **out)
    nbytes = sum(v.nbytes for v in out.values())
    print(f"iter {ck['iter']}, width {width} blocks {blocks}")
    print(f"{len(out)} arrays, {nbytes/1e6:.2f} MB raw, {dest.stat().st_size/1e6:.2f} MB on disk")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
