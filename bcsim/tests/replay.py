"""Reads the reference engine's Cap'n Proto replay, which is the source of truth.

Why this exists: the observation block a bot receives is a *lossy* view of what
the engine did. For sonar it gives an aggregate `ECHOES` line with no bearing and
a bag of messages with no sender, so a targeting bug and a classification bug look
identical from inside a bot. Every earlier sonar probe worked around that by
encoding tags into payloads and inferring what must have happened.

The replay does not need inferring. `EventSonarPing` records, for every single ray:

    senderId  direction  value64  origin  end  hitId  hitKind

so the tile the ray started on, **the tile it stopped on**, the dragon it reached
and how that dragon was classified are all stated outright. That turns sonar
parity from an inference problem into a diff.

The schema is not shipped as a .capnp file; it was read off the accessors in the
replay viewer that unswbc ships (`replay-viewer.vsix`,
`extension/dist/webview/webview.js`), which is generated from the real schema.
Struct ids are recorded below so a format change is detectable rather than
silently misread: `formatVersion` is checked on every parse.

    python bcsim/tests/replay.py <replay.bcr>        # summarise a replay file
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field

# ---------------------------------------------------------------- wire format

WORD = 8


class CapnpError(RuntimeError):
    pass


def unpack(blob: bytes) -> bytes:
    """Cap'n Proto packed encoding -> flat words. The engine writes packed.

    Each word is preceded by a tag byte whose bit i says whether byte i of the
    word is present; absent bytes are zero. Tag 0x00 is followed by a count of
    extra all-zero words, and tag 0xFF by a count of extra words copied verbatim.
    """
    out = bytearray()
    i, n = 0, len(blob)
    while i < n:
        tag = blob[i]
        i += 1
        word = bytearray(8)
        for b in range(8):
            if tag >> b & 1:
                if i >= n:
                    raise CapnpError("packed replay ends inside a word")
                word[b] = blob[i]
                i += 1
        out += word
        if tag == 0x00:
            if i >= n:
                raise CapnpError("packed replay ends after a zero tag")
            out += b"\0" * (8 * blob[i])
            i += 1
        elif tag == 0xFF:
            if i >= n:
                raise CapnpError("packed replay ends after a literal tag")
            count = blob[i] * 8
            i += 1
            out += blob[i : i + count]
            i += count
    return bytes(out)


def is_packed(blob: bytes) -> bool:
    """A flat message starts with a plausible small segment count; packed does not."""
    if len(blob) < 8:
        return False
    nsegs = struct.unpack_from("<I", blob, 0)[0] + 1
    return not 1 <= nsegs <= 512


@dataclass
class _Ptr:
    """A resolved struct: where its data and pointer sections live."""

    msg: "Message"
    seg: int
    data: int          # byte offset of the data section
    dwords: int
    ptrs: int          # byte offset of the pointer section
    pwords: int

    # -- data section accessors. Reads beyond the struct's declared data
    # section return 0, which is what Cap'n Proto requires for forward
    # compatibility (an older writer simply had a shorter struct).
    def _raw(self, off: int, n: int) -> bytes:
        if off + n > self.dwords * WORD:
            return b"\0" * n
        return self.msg.segs[self.seg][self.data + off : self.data + off + n]

    def u8(self, off: int) -> int:
        return self._raw(off, 1)[0]

    def u16(self, off: int) -> int:
        return struct.unpack_from("<H", self._raw(off, 2))[0]

    def i32(self, off: int) -> int:
        return struct.unpack_from("<i", self._raw(off, 4))[0]

    def u32(self, off: int) -> int:
        return struct.unpack_from("<I", self._raw(off, 4))[0]

    def u64(self, off: int) -> int:
        return struct.unpack_from("<Q", self._raw(off, 8))[0]

    def bit(self, off: int) -> bool:
        return bool(self.u8(off // 8) >> (off % 8) & 1)

    # -- pointer section
    def _word(self, i: int) -> int:
        if i >= self.pwords:
            return 0
        return struct.unpack_from("<Q", self.msg.segs[self.seg],
                                  self.ptrs + i * WORD)[0]

    def struct(self, i: int) -> "_Ptr | None":
        return self.msg._deref(self.seg, self.ptrs + i * WORD, self._word(i))

    def text(self, i: int) -> str:
        lst = self.msg._list(self.seg, self.ptrs + i * WORD, self._word(i))
        if lst is None:
            return ""
        seg, off, count, _esz, _ = lst
        return self.msg.segs[seg][off : off + max(0, count - 1)].decode(
            "utf-8", "replace")

    def u16_list(self, i: int) -> list[int]:
        """A list of 16-bit values, which is how enums travel (e.g. move steps)."""
        lst = self.msg._list(self.seg, self.ptrs + i * WORD, self._word(i))
        if lst is None:
            return []
        seg, off, count, esz, _ = lst
        if esz != 3:
            raise CapnpError(f"expected a two-byte list, got element size {esz}")
        return list(struct.unpack_from(f"<{count}H", self.msg.segs[seg], off))

    def structs(self, i: int) -> "list[_Ptr]":
        """A composite list, which is the only list of structs the replay uses."""
        lst = self.msg._list(self.seg, self.ptrs + i * WORD, self._word(i))
        if lst is None:
            return []
        seg, off, count, esz, tag = lst
        if esz != 7:
            raise CapnpError(f"expected a composite list, got element size {esz}")
        n, dwords, pwords = tag
        out = []
        base = off + WORD                      # past the tag word
        stride = (dwords + pwords) * WORD
        for k in range(n):
            out.append(_Ptr(self.msg, seg, base + k * stride, dwords,
                            base + k * stride + dwords * WORD, pwords))
        return out


class Message:
    def __init__(self, blob: bytes):
        if len(blob) < 4:
            raise CapnpError("replay is too short to be a Cap'n Proto message")
        nsegs = struct.unpack_from("<I", blob, 0)[0] + 1
        if not 1 <= nsegs <= 512:
            raise CapnpError(f"implausible segment count {nsegs}")
        sizes = [struct.unpack_from("<I", blob, 4 + 4 * i)[0] for i in range(nsegs)]
        off = 4 + 4 * nsegs
        off += -off % WORD                     # the table is padded to a word
        self.segs: list[bytes] = []
        for s in sizes:
            self.segs.append(blob[off : off + s * WORD])
            off += s * WORD

    def _deref(self, seg: int, at: int, word: int) -> _Ptr | None:
        if word == 0:
            return None
        kind = word & 3
        if kind == 2:                          # far pointer
            tseg, target, pad = _far(word)
            inner = struct.unpack_from("<Q", self.segs[tseg], target)[0]
            if not pad:
                return self._deref(tseg, target, inner)
            # Two-word landing pad: the first word is a struct pointer whose
            # offset is relative to the start of the *content*, which the second
            # word (a far pointer) locates.
            tag = struct.unpack_from("<Q", self.segs[tseg], target + WORD)[0]
            cseg, content, _ = _far(inner)
            dwords = (tag >> 32) & 0xFFFF
            pwords = (tag >> 48) & 0xFFFF
            return _Ptr(self, cseg, content, dwords,
                        content + dwords * WORD, pwords)
        if kind != 0:
            raise CapnpError(f"expected a struct pointer, got kind {kind}")
        offw = _signed30(word >> 2)
        data = at + WORD + offw * WORD
        dwords = (word >> 32) & 0xFFFF
        pwords = (word >> 48) & 0xFFFF
        return _Ptr(self, seg, data, dwords, data + dwords * WORD, pwords)

    def _list(self, seg: int, at: int, word: int):
        if word == 0:
            return None
        kind = word & 3
        if kind == 2:
            tseg, target, pad = _far(word)
            inner = struct.unpack_from("<Q", self.segs[tseg], target)[0]
            if not pad:
                return self._list(tseg, target, inner)
            tag = struct.unpack_from("<Q", self.segs[tseg], target + WORD)[0]
            cseg, content, _ = _far(inner)
            esz = (tag >> 32) & 7
            count = tag >> 35
            ctag = None
            if esz == 7:
                tw = struct.unpack_from("<Q", self.segs[cseg], content)[0]
                ctag = (_signed30(tw >> 2), (tw >> 32) & 0xFFFF,
                        (tw >> 48) & 0xFFFF)
            return cseg, content, count, esz, ctag
        if kind != 1:
            raise CapnpError(f"expected a list pointer, got kind {kind}")
        offw = _signed30(word >> 2)
        esz = (word >> 32) & 7
        count = word >> 35
        off = at + WORD + offw * WORD
        tag = None
        if esz == 7:
            tw = struct.unpack_from("<Q", self.segs[seg], off)[0]
            tag = (_signed30(tw >> 2), (tw >> 32) & 0xFFFF, (tw >> 48) & 0xFFFF)
        return seg, off, count, esz, tag

    def root(self) -> _Ptr:
        word = struct.unpack_from("<Q", self.segs[0], 0)[0]
        r = self._deref(0, 0, word)
        if r is None:
            raise CapnpError("replay has a null root pointer")
        return r


def _far(word: int) -> tuple[int, int, int]:
    """(segment, byte offset, landing-pad-is-two-words) from a far pointer."""
    return word >> 32, ((word >> 3) & 0x1FFFFFFF) * WORD, (word >> 2) & 1


def _signed30(v: int) -> int:
    v &= 0x3FFFFFFF
    return v - (1 << 30) if v & (1 << 29) else v


# ------------------------------------------------------------------- schema
# Read off the shipped replay viewer; ids let a schema change be caught.

STRUCT_IDS = {
    "Replay": "bee8ca02914a028b",
    "Event": "ac3551e2b979a5ac",
    "EventSonarPing": "b307bb24c5a60cac",
    "Point": "f46b7269eb2bbb4c",
}

FORMAT_VERSIONS_KNOWN = (1, 2, 3, 4, 5)

EVENT_ROUND_START, EVENT_TURN_START = 0, 1
EVENT_PEARL_COUNTDOWN, EVENT_TILE_CHANGE = 2, 3
EVENT_DRAGON_ACTION, EVENT_ENGINE_LOG, EVENT_DRAGON_LOG = 4, 5, 6
EVENT_DRAGON_INDICATOR, EVENT_DEBUG_DRAW = 7, 8
EVENT_DRAGON_UPDATE, EVENT_DRAGON_SPLIT, EVENT_DRAGON_DEATH = 9, 10, 11
EVENT_SONAR_PING = 12

DIRECTIONS = ("N", "E", "S", "W")          # NORTH EAST SOUTH WEST
HIT_KINDS = ("unknown", "empty", "kelp", "ally", "ally_head", "enemy", "enemy_head")

SONAR_NO_HIT, SONAR_HIT_ID = 0, 1


@dataclass
class SonarPing:
    """One ray, exactly as the engine resolved it."""

    round: int
    sender: int
    direction: str
    value: int
    origin: tuple[int, int]
    end: tuple[int, int]
    hit: int | None               # dragon id, or None for no hit
    kind: str                     # one of HIT_KINDS

    def __str__(self) -> str:
        where = f"dragon {self.hit}" if self.hit is not None else "nobody"
        return (f"r{self.round} d{self.sender} {self.direction} "
                f"{self.origin}->{self.end} {where} ({self.kind})")


@dataclass
class Replay:
    map_name: str
    bot_a: str
    bot_b: str
    format_version: int
    pings: list[SonarPing] = field(default_factory=list)
    deaths: list[tuple[int, int, str]] = field(default_factory=list)
    rounds: int = 0


DEATH_REASONS = ("W", "S", "O", "H", "A")


def _point(p: _Ptr | None) -> tuple[int, int]:
    return (-1, -1) if p is None else (p.i32(0), p.i32(4))


def parse(blob: bytes) -> Replay:
    if is_packed(blob):
        blob = unpack(blob)
    root = Message(blob).root()
    ver = root.u32(0)
    if ver not in FORMAT_VERSIONS_KNOWN:
        raise CapnpError(
            f"replay formatVersion {ver} is not one this parser was written "
            f"against ({FORMAT_VERSIONS_KNOWN}); re-read the schema from the "
            f"shipped replay viewer before trusting a diff")
    out = Replay(map_name=root.text(0), bot_a=root.text(1), bot_b=root.text(2),
                 format_version=ver)
    rnd = 0
    for ev in root.structs(3):
        which = ev.u16(0)
        body = ev.struct(0)
        if body is None:
            continue
        if which == EVENT_ROUND_START:
            rnd = body.i32(0)
            out.rounds = max(out.rounds, rnd)
        elif which == EVENT_SONAR_PING:
            hit_union = body.u16(6)
            # value64 superseded value; older formats only filled the 32-bit one.
            val = body.u64(16) or body.u32(8)
            out.pings.append(SonarPing(
                round=rnd,
                sender=body.i32(0),
                direction=DIRECTIONS[body.u16(4)] if body.u16(4) < 4 else "?",
                value=val,
                origin=_point(body.struct(0)),
                end=_point(body.struct(1)),
                hit=body.i32(12) if hit_union == SONAR_HIT_ID else None,
                kind=HIT_KINDS[body.u16(24)] if body.u16(24) < len(HIT_KINDS)
                else "?",
            ))
        elif which == EVENT_DRAGON_DEATH:
            r = body.u16(4)
            out.deaths.append((body.i32(0), rnd,
                               DEATH_REASONS[r] if r < len(DEATH_REASONS) else "?"))
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[-1])
        return 2
    rep = parse(open(argv[1], "rb").read())
    print(f"map {rep.map_name!r}  {rep.bot_a} vs {rep.bot_b}  "
          f"format {rep.format_version}  rounds {rep.rounds}")
    print(f"{len(rep.pings)} sonar pings, {len(rep.deaths)} deaths")
    kinds: dict[str, int] = {}
    for p in rep.pings:
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
    print("by echo kind:", dict(sorted(kinds.items(), key=lambda kv: -kv[1])))
    for p in rep.pings[:10]:
        print("  ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
