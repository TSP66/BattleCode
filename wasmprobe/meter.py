"""Prices wasm functions in the judge's own CPU points, locally.

The toolkit ships the judge's metering pass (unswbc.metering) because
`unswbc run --sandbox` needs it to reproduce the judge's numbers for Python
bots. That pass is language-agnostic -- it rewrites any wasm module to
decrement a points global -- so pointing it at our own compiled kernels gives
the same numbers the judge would charge, without a submission round trip.

    python meter.py kernel.wasm
"""

from __future__ import annotations

import pathlib
import sys

sys.path[:0] = [str(p) for p in pathlib.Path.home().glob(
    ".local/share/uv/tools/unswbc/lib/python3*/site-packages")]

from unswbc import metering                                   # noqa: E402
from wasmtime import Engine, Instance, Module, Store          # noqa: E402

REMAINING = "wasmer_metering_remaining_points"

# name, args, the multiply-accumulates it performs (0 when not a gemm)
CASES = [
    ("gemm",       (96, 864),   96 * 864 * 52),     # one width-96 conv
    ("gemm",       (96, 207),   96 * 207 * 52),     # the stem
    ("gemm",       (32, 96),    32 * 96 * 52),      # the 1x1 projection
    ("gemm",       (48, 512),   48 * 512 * 52),     # sized like the policy head
    ("gemm_simd",  (96, 864),   96 * 864 * 52),
    ("gemm_simd4", (96, 864),   96 * 864 * 52),
    ("gemm_t4",    (96, 864),   96 * 864 * 52),
    ("gemm_t8",    (96, 864),   96 * 864 * 52),
    ("gemm_t44",   (96, 864),   96 * 864 * 52),
    ("gemm_t2",    (96, 864),   96 * 864 * 52),
    ("im2col",     (96,),       0),
    ("group_norm", (96, 8),     0),
    ("silu",       (96 * 52,),  0),
    ("widen",      (82_944,),   0),
]


def main() -> None:
    path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "kernel.wasm")
    blob = metering.instrument(path.read_bytes())

    engine = Engine()
    module = Module(engine, blob)

    print(f"{'function':12s} {'args':>14s} {'points':>14s} {'MACs':>12s} {'pts/MAC':>9s}")
    for name, args, macs in CASES:
        store = Store(engine)
        inst = Instance(store, module, [])
        meter = inst.exports(store)[REMAINING]
        fn = inst.exports(store)[name]

        before = meter.value(store)
        fn(store, *args)
        spent = before - meter.value(store)

        rate = f"{spent / macs:9.2f}" if macs else " " * 9
        mac_s = f"{macs:,}" if macs else "-"
        print(f"{name:12s} {str(args):>14s} {spent:14,} {mac_s:>12s} {rate}")


if __name__ == "__main__":
    main()
