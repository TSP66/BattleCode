"""Reads a server replay: who acted when, what they did, and how they died.

The server strips LOG and engine messages from replays, so the only way to
see why a submission's dragons died is the event stream itself. Replays are
gzipped Cap'n Proto; this is just enough of the wire format to walk it, with
field offsets taken from the replay viewer's generated schema code.

    python replay.py battle.replay
"""

from __future__ import annotations

import collections
import gzip
import struct
import sys

EVENTS = ["roundStart", "turnStart", "pearlCountdown", "tileChange", "dragonAction",
          "engineLog", "dragonLog", "dragonIndicator", "debugDraw", "dragonUpdate",
          "dragonSplit", "dragonDeath", "sonarPing"]
DEATH = ["HIT_WALL", "HIT_SELF", "HIT_OTHER_BODY", "HIT_HEAD_TO_HEAD", "NO_VALID_ACTION"]
END = {0: "?", 1: "elimination", 2: "round limit"}


def unpack(raw: bytes) -> bytes:
    """Cap'n Proto packed encoding: a tag byte per word says which of its eight
    bytes are non-zero; tag 0x00 is followed by a count of further zero words,
    tag 0xFF by a count of words copied verbatim."""
    out = bytearray()
    i = 0
    n = len(raw)
    while i < n:
        tag = raw[i]
        i += 1
        word = bytearray(8)
        for b in range(8):
            if tag & (1 << b):
                word[b] = raw[i]
                i += 1
        out += word
        if tag == 0x00:
            out += bytes(8 * raw[i])
            i += 1
        elif tag == 0xFF:
            count = raw[i]
            i += 1
            out += raw[i:i + 8 * count]
            i += 8 * count
    return bytes(out)


class Msg:
    def __init__(self, raw: bytes):
        n = struct.unpack_from("<I", raw, 0)[0] + 1
        sizes = struct.unpack_from(f"<{n}I", raw, 4)
        off = 4 + 4 * n
        off += (8 - off % 8) % 8
        self.segs = []
        for s in sizes:
            self.segs.append(raw[off:off + 8 * s])
            off += 8 * s

    def word(self, seg: int, w: int) -> int:
        return struct.unpack_from("<Q", self.segs[seg], 8 * w)[0]

    def follow(self, seg: int, w: int):
        """Resolves the pointer at (seg, w) to (seg, target word, pointer word)."""
        p = self.word(seg, w)
        if p & 3 == 2:                                   # far pointer
            landing = (p >> 3) & ((1 << 29) - 1)
            tseg = p >> 32
            if (p >> 2) & 1:
                raise NotImplementedError("double-far pointer")
            return self.follow(tseg, landing)
        off = (p >> 2) & ((1 << 30) - 1)
        if off & (1 << 29):
            off -= 1 << 30
        return seg, w + 1 + off, p

    def struct(self, seg: int, w: int):
        if self.word(seg, w) == 0:
            return None
        s, t, p = self.follow(seg, w)
        data = (p >> 32) & 0xFFFF
        return (s, t, data)

    def list_structs(self, seg: int, w: int):
        s, t, p = self.follow(seg, w)
        tag = self.word(s, t)
        count = (tag >> 2) & ((1 << 30) - 1)
        data = (tag >> 32) & 0xFFFF
        ptrs = (tag >> 48) & 0xFFFF
        for i in range(count):
            yield (s, t + 1 + i * (data + ptrs), data)

    def u16(self, st, byte: int) -> int:
        s, t, _ = st
        return struct.unpack_from("<H", self.segs[s], 8 * t + byte)[0]

    def i32(self, st, byte: int) -> int:
        s, t, _ = st
        return struct.unpack_from("<i", self.segs[s], 8 * t + byte)[0]

    def ptr(self, st, index: int):
        s, t, data = st
        return self.struct(s, t + data + index)


def main() -> None:
    raw = open(sys.argv[1], "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    m = Msg(unpack(raw))
    root = m.struct(0, 0)
    s, t, data = root

    counts = collections.Counter()
    rnd = -1
    acted = collections.defaultdict(list)
    deaths = []
    for ev in m.list_structs(s, t + data + 3):
        kind = m.u16(ev, 0)
        name = EVENTS[kind] if kind < len(EVENTS) else f"#{kind}"
        counts[name] += 1
        body = m.ptr(ev, 0)
        if body is None:
            continue
        if name == "roundStart":
            rnd = m.i32(body, 0)
        elif name == "dragonAction":
            acted[m.i32(body, 0)].append(rnd)
        elif name == "dragonDeath":
            reason = m.u16(body, 4)
            deaths.append((rnd, m.i32(body, 0), DEATH[reason] if reason < len(DEATH) else reason))

    by_reason = collections.Counter(d[2] for d in deaths)
    # a CPU overrun is charged as NO_VALID_ACTION, and it bites on a dragon's
    # first turn, where start-up and weight loading are paid
    overruns = [d for d in deaths
                if d[2] == "NO_VALID_ACTION" and len(acted.get(d[1], [])) <= 1]
    print(f"rounds played: {rnd + 1}   dragons: {len(acted)}   "
          f"dragon-turns: {sum(len(v) for v in acted.values()):,}")
    print(f"deaths by reason: {dict(by_reason) or 'none'}")
    print(f"first-turn NO_VALID_ACTION (CPU overrun signature): {len(overruns)}"
          + ("   <-- BUDGET PROBLEM" if overruns else "   (ok)"))
    for d in deaths[:10]:
        print(f"  round {d[0]:3d}  dragon {d[1]:3d}  {d[2]}")
    res = m.struct(s, t + data + 4)
    if res is not None:
        print(f"\nresult: endReason {END.get(m.u16(res, 2), m.u16(res, 2))}, "
              f"winner {'AB'[m.u16(res, 6)] if m.u16(res, 4) == 1 else 'none'}")


if __name__ == "__main__":
    main()
