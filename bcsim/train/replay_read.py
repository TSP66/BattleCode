"""Reads a server replay into plain Python.

Replays are gzipped, packed Cap'n Proto. The schema (field offsets below) was
read off the generated code inside the replay viewer that ships with unswbc
(replay-viewer.vsix, dist/webview/webview.js):

    Replay        ptrs: map Text, botA Text, botB Text, events List(Event), result GameResult
    Event         u16 which @0, ptr 0 -> body
    PlayerAction  u16 which @0 (0 move, 1 split, 2 suicide);
                  move = List(UInt16) directions (0 N 1 E 2 S 3 W); split = i32 @4
    ...see EVENTS for the per-event fields.
"""

from __future__ import annotations

import gzip
import struct
from dataclasses import dataclass, field


def unpack(raw: bytes) -> bytes:
    """Cap'n Proto packed encoding."""
    out = bytearray()
    i, n = 0, len(raw)
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

    def word(self, seg, w):
        return struct.unpack_from("<Q", self.segs[seg], 8 * w)[0]

    def follow(self, seg, w):
        """(seg, target word, pointer word) for the pointer at (seg, w)."""
        p = self.word(seg, w)
        if p & 3 == 2:                                    # far pointer
            if (p >> 2) & 1:
                raise NotImplementedError("double-far pointer")
            return self.follow(p >> 32, (p >> 3) & ((1 << 29) - 1))
        off = (p >> 2) & ((1 << 30) - 1)
        if off & (1 << 29):
            off -= 1 << 30
        return seg, w + 1 + off, p

    # a struct reference is (seg, word, data words)
    def struct(self, seg, w):
        if self.word(seg, w) == 0:
            return None
        s, t, p = self.follow(seg, w)
        return (s, t, (p >> 32) & 0xFFFF)

    def ptr(self, st, i):
        s, t, data = st
        return self.struct(s, t + data + i)

    def list_structs(self, st, i):
        s, t, data = st
        s, t, _ = self.follow(s, t + data + i)
        tag = self.word(s, t)
        count = (tag >> 2) & ((1 << 30) - 1)
        dw, pw = (tag >> 32) & 0xFFFF, (tag >> 48) & 0xFFFF
        return [(s, t + 1 + k * (dw + pw), dw) for k in range(count)]

    def list_u16(self, st, i):
        s, t, data = st
        if self.word(s, t + data + i) == 0:
            return []
        s, t, p = self.follow(s, t + data + i)
        count = p >> 35
        return list(struct.unpack_from(f"<{count}H", self.segs[s], 8 * t))

    def text(self, st, i):
        s, t, data = st
        if self.word(s, t + data + i) == 0:
            return ""
        s, t, p = self.follow(s, t + data + i)
        n = p >> 35
        return self.segs[s][8 * t:8 * t + n - 1].decode()

    def u16(self, st, byte):
        s, t, _ = st
        return struct.unpack_from("<H", self.segs[s], 8 * t + byte)[0]

    def i32(self, st, byte):
        s, t, _ = st
        return struct.unpack_from("<i", self.segs[s], 8 * t + byte)[0]

    def u32(self, st, byte):
        s, t, _ = st
        return struct.unpack_from("<I", self.segs[s], 8 * t + byte)[0]

    def u64(self, st, byte):
        s, t, _ = st
        return struct.unpack_from("<Q", self.segs[s], 8 * t + byte)[0]

    def bit(self, st, byte):
        s, t, _ = st
        return self.segs[s][8 * t + byte] & 1

    def point(self, st, i):
        p = self.ptr(st, i)
        return None if p is None else (self.i32(p, 0), self.i32(p, 4))


EVENTS = ["roundStart", "turnStart", "pearlCountdown", "tileChange", "dragonAction",
          "engineLog", "dragonLog", "dragonIndicator", "debugDraw", "dragonUpdate",
          "dragonSplit", "dragonDeath", "sonarPing"]
DEATH = ["HIT_WALL", "HIT_SELF", "HIT_OTHER_BODY", "HIT_HEAD_TO_HEAD", "NO_VALID_ACTION"]


@dataclass
class Turn:
    """One dragon's turn: what it asked for and what came of it."""
    round: int
    dragon: int
    kind: int = -1              # 0 move, 1 split, 2 suicide, -1 no action given
    dirs: list[int] = field(default_factory=list)
    split_k: int = 0
    # Every ray this dragon cast, as (direction index 0..3 = N,E,S,W, payload).
    # A turn may hold up to one per direction, and the payload is 64 bits wide.
    # This replaced a single `sonar: int | None` read with u32 off the 32-bit
    # legacy field, which both truncated the value and let a second ping in the
    # same turn overwrite the first.
    sonars: list[tuple[int, int]] = field(default_factory=list)
    head_after: tuple[int, int] | None = None
    died: str | None = None


@dataclass
class Replay:
    map_text: str
    turns: list[Turn]
    rounds: int
    end_reason: int             # 0 team eliminated, 1 round limit
    winner: int                 # 0 A, 1 B, -1 none
    events: list[tuple]         # (round, name, fields) for anything else wanted


def read(path) -> Replay:
    raw = open(path, "rb").read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    m = Msg(unpack(raw))
    root = m.struct(0, 0)
    map_text = m.text(root, 0)

    turns: list[Turn] = []
    events: list[tuple] = []
    rnd = -1
    cur: Turn | None = None
    for ev in m.list_structs(root, 3):
        kind = m.u16(ev, 0)
        name = EVENTS[kind] if kind < len(EVENTS) else f"#{kind}"
        body = m.ptr(ev, 0)
        if name == "roundStart":
            rnd = m.i32(body, 0) if body else rnd + 1
            cur = None
        elif name == "turnStart":
            cur = Turn(rnd, m.i32(body, 0) if body else 0)
            turns.append(cur)
        elif name == "dragonAction" and body is not None:
            did = m.i32(body, 0)
            act = m.ptr(body, 0)
            assert cur is not None and cur.dragon == did, "action outside its turn"
            which = m.u16(act, 0) if act else 2
            cur.kind = which
            if which == 0:
                cur.dirs = m.list_u16(act, 0)
            elif which == 1:
                cur.split_k = m.i32(act, 4)
        elif name == "sonarPing" and body is not None:
            sender = m.i32(body, 0)
            # value64 at offset 16 superseded the 32-bit value at 8; older
            # formats only filled the narrow one. Layout verified against the
            # engine's own accessors, see tests/replay.py.
            value = m.u64(body, 16) or m.u32(body, 8)
            direction = m.u16(body, 4)
            if cur is not None and cur.dragon == sender and direction < 4:
                cur.sonars.append((direction, value))
            events.append((rnd, name, {"sender": sender, "direction": direction,
                                       "value": value}))
        elif name == "dragonUpdate" and body is not None:
            if cur is not None and cur.dragon == m.i32(body, 0):
                cur.head_after = m.point(body, 0)
        elif name == "dragonDeath" and body is not None:
            did, reason = m.i32(body, 0), m.u16(body, 4)
            why = DEATH[reason] if reason < len(DEATH) else str(reason)
            if cur is not None and cur.dragon == did:
                cur.died = why
            events.append((rnd, name, {"id": did, "reason": why}))
        elif name == "dragonSplit" and body is not None:
            events.append((rnd, name, {"parent": m.i32(body, 0), "child": m.i32(body, 4)}))
        elif name == "tileChange" and body is not None:
            events.append((rnd, name, {"tile": m.point(body, 0), "pearl": m.bit(body, 0)}))

    res = m.ptr(root, 4)
    end_reason, winner = -1, -1
    if res is not None:
        end_reason = m.u16(res, 2)
        winner = m.u16(res, 6) if m.u16(res, 4) == 1 else -1
    return Replay(map_text, turns, rnd + 1, end_reason, winner, events)


if __name__ == "__main__":
    import collections
    import sys
    r = read(sys.argv[1])
    print(f"{r.rounds} rounds, {len(r.turns)} turns, end {r.end_reason}, winner {r.winner}")
    print("action kinds:", collections.Counter(t.kind for t in r.turns))
    print("move lengths:", collections.Counter(len(t.dirs) for t in r.turns if t.kind == 0))
    print("split k:", collections.Counter(t.split_k for t in r.turns if t.kind == 1))
    print("sonar rays:", sum(len(t.sonars) for t in r.turns),
          "turns that cast:", sum(1 for t in r.turns if t.sonars),
          "wide payloads:", sum(1 for t in r.turns for _, v in t.sonars if v > 0xFFFFFFFF))
    for t in r.turns[:6]:
        print(t)
