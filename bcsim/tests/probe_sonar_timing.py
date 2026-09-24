"""WHEN does the engine deliver a sonar message? Asked of the engine, decisively.

The delivery rule is what parity_sonar.py's first difference is about: at round 0
our simulator hands a dragon messages the engine does not. Three models of inbox
clearing have already been guessed at and rejected, so this stops guessing and
runs a controlled experiment instead.

Exactly ONE dragon sends exactly ONE sonar, in one direction, on one round, with
a payload that encodes who sent it and when. Nobody else ever sends. So every
received message is unambiguous, and the gap between the send round and the
receive round is the rule.

    python tests/probe_sonar_timing.py [map]
"""

from __future__ import annotations

import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from blockparse import Block                    # noqa: E402
from oracle import OracleGame                   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOP = 1 << 63


def tag(sender: int, rnd: int, d: int) -> int:
    """sender, round and direction, all recoverable from the payload alone."""
    return TOP | (sender & 0xFF) << 32 | (rnd & 0xFFFF) << 8 | (d & 0x3)


def untag(v: int) -> tuple[int, int, int] | None:
    if not (v & TOP):
        return None
    return ((v >> 32) & 0xFF, (v >> 8) & 0xFFFF, v & 0x3)


def run(map_text: str, send_dir: str = "N", every: int = 25, all_dirs: bool = False) -> None:
    order: list[tuple[int, int]] = []       # (round, dragon) as the engine asks
    sent: dict[int, tuple[int, int]] = {}   # payload -> (round, dragon)
    got: list[tuple[int, int, int, int]] = []   # recv_round, recv_dragon, send_round, sender
    seat = {}                               # dragon -> its index within a round

    def policy(did: int, text: str) -> str:
        b = Block(text)
        r = b.round
        order.append((r, did))
        seat.setdefault((r, did), sum(1 for (rr, _) in order if rr == r) - 1)

        for v in b.msgs:
            u = untag(v)
            if u is not None:
                sender, sr, _ = u
                got.append((r, did, sr, sender))

        reply = []
        # ONE sender, ONE direction, on selected rounds only
        if did == 0 and r % every == 0:
            for d in ("NESW" if all_dirs else send_dir):
                p = tag(did, r, "NESW".index(d))
                sent[p] = (r, did)
                reply.append(f"SONAR {d} {p}")
        # stay alive without moving into anything: hold a direction that is clear
        safe = b.safe_dirs()
        reply.append(f"MOVE {safe[0] if safe else b.dir}")
        reply.append("PROTOCOL 3")
        reply.append("ENDTURN")
        return "\n".join(reply) + "\n"

    OracleGame(map_text, policy).run()

    print(f"turns {len(order)}, sonars sent {len(sent)}, messages received {len(got)}")
    if not got:
        print("  NOTHING was received -- widen `every` or pick a map where a ray "
              "can reach another dragon")
        return
    lag = collections.Counter(r - sr for (r, _, sr, _) in got)
    print("  receive_round - send_round:")
    for d, c in sorted(lag.items()):
        print(f"    lag {d:+d} rounds : {c:5d} messages")
    # The decisive case: a receiver whose SEAT in the round is after the
    # sender's. "next round" and "next turn after the ray" differ only there.
    print("  seat of sender vs receiver, for cross-dragon deliveries:")
    shown = 0
    for (r, did, sr, sender) in got:
        if did == sender:
            continue
        ss, rs = seat.get((sr, sender)), seat.get((r, did))
        print(f"    {sender} (round {sr}, seat {ss}) -> {did} (round {r}, seat {rs})"
              f"   lag {r - sr:+d}"
              + ("   <- receiver sat LATER in the send round"
                 if r == sr and rs is not None and ss is not None and rs > ss else ""))
        shown += 1
        if shown >= 14:
            break
    if not shown:
        print("    none: every delivery was the sender hearing its own wrapped ray")


if __name__ == "__main__":
    args = sys.argv[1:]
    mp = pathlib.Path(args[0]) if args else ROOT / "maps-official" / "big_empty.map"
    print(f"=== {mp.stem}")
    run(mp.read_text(), every=1, all_dirs=True)
