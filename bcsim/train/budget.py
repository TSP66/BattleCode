"""Prices a network against the judge's 100M CPU points per dragon per turn.

The judge meters wasm instructions, not time. A multiply-accumulate in a NumPy
float32 matmul costs roughly 1 point with SIMD and up to 2 once loads, stores
and loop overhead are counted, so this prices MACs at a configurable rate and
keeps a margin. The real figure comes from `unswbc run --sandbox -v` once a
NumPy inference path exists; until then this is the conservative estimate.
"""

from __future__ import annotations

POINTS_BUDGET = 100_000_000
PARSE_COST = 10_000_000      # docs: parsing one round in the Python helper
OUTPUT_COST = 2_600_000      # one flushed write per turn
POINTS_PER_MAC = 2.0         # conservative; 1.0 is the optimistic SIMD case


def macs(n_channels: int, window: int, n_scalars: int, n_actions: int,
         width: int, blocks: int, hidden: int = 512, head: int = 32) -> int:
    hw = window * window
    total = hw * n_channels * width * 9                 # stem conv
    total += blocks * 2 * hw * width * width * 9        # residual 3x3 convs
    total += hw * width * head                          # 1x1 projection
    total += n_scalars * 128 + 128 * 128                # scalar MLP
    total += (hw * head + 128) * hidden + hidden * hidden
    total += hidden * (n_actions + 1)                   # heads
    return int(total)


def report(n_channels=23, window=7, n_scalars=14, n_actions=48,
           points_per_mac: float = POINTS_PER_MAC) -> None:
    avail = POINTS_BUDGET - PARSE_COST - OUTPUT_COST
    print(f"budget {POINTS_BUDGET:,} - parse {PARSE_COST:,} - output {OUTPUT_COST:,} "
          f"= {avail:,} points for the network")
    print(f"pricing MACs at {points_per_mac} points each\n")
    print(f"{'width':>6} {'blocks':>7} {'MACs':>14} {'points':>14} {'params':>11}  fits")
    best = None
    for width in (48, 64, 96, 128):
        for blocks in (2, 3, 4, 6, 8):
            m = macs(n_channels, window, n_scalars, n_actions, width, blocks)
            pts = m * points_per_mac
            params = (n_channels * width * 9 + blocks * 2 * width * width * 9
                      + width * 32 + (window * window * 32 + 128) * 512 + 512 * 512
                      + 512 * (n_actions + 1))
            fits = pts <= avail
            if fits and (best is None or m > best[0]):
                best = (m, width, blocks)
            print(f"{width:6d} {blocks:7d} {m:14,} {pts:14,.0f} {params:11,}  "
                  f"{'yes' if fits else 'NO'}")
    if best:
        print(f"\nlargest that fits: width {best[1]}, blocks {best[2]} "
              f"({best[0]:,} MACs, {best[0]*points_per_mac/POINTS_BUDGET:.0%} of the turn budget)")


if __name__ == "__main__":
    report()
