"""Our sonar against the engine's, turn by turn, block for block.

This checks that we do what the engine does: the reference engine and our text simulator are driven in lockstep
with identical replies, and every round block is compared byte for byte. That
covers NUM_MSGS and the 64-bit payloads, the ECHOES line and its five counts,
and the fact that ECHOES only appears once a team has declared protocol 3.

The policy deliberately broadcasts in all four directions, splits often enough
to make allies, and picks moves that keep dragons alive, because a sonar rule
that is only exercised on an empty board is not exercised at all.

    python tests/parity_sonar.py [map ...]
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from diffsim import Sim                          # noqa: E402
from oracle import OracleGame                   # noqa: E402
from blockparse import Block                     # noqa: E402


DIRS = "NESW"


def payload(did: int, d: int) -> int:
    """A payload that uses the whole word: the top bit, the bottom bits, and the
    sender and direction in the middle."""
    return 1 << 63 | (did & 0xFFF) << 16 | (d & 0x3) << 8 | 0x5A


def narrow(did: int, d: int) -> int:
    """A payload inside 32 bits, which any receiver can be handed."""
    return ((did & 0xFFF) << 8 | (d & 0x3) << 4 | 0xA) & 0xFFFFFFFF

ROOT = pathlib.Path(__file__).resolve().parents[2]


def run(map_text: str, protocol) -> dict:
    """3 declares it every turn; 2 never does, so ECHOES must stay away.

    "mixed" declares protocol 3 on even dragon ids only. That pins two rules
    that were measured off the engine and are easy to get wrong:

      * the protocol is per DRAGON, not per team -- when only some dragons
        declare it, the others get no ECHOES line at all, teammates included;
      * a payload wider than 32 bits is DROPPED for a receiver still on the
        legacy protocol, rather than truncated, while the sender's echo still
        counts the hit.

    A split child inherits its parent's protocol, so an odd-id child of an
    even-id parent speaks protocol 3 without ever declaring it. That is the
    engine's behaviour and it is what makes this case worth checking.
    """
    sim = Sim(map_text)
    st = {"turns": 0, "diff": 0, "first": None, "echo_turns": 0, "msg_turns": 0,
          "echo_nonzero": 0, "msgs": 0, "desync": 0}

    def policy(did: int, text: str) -> str:
        b = Block(text)
        st["turns"] += 1
        if b.echoes is not None:
            st["echo_turns"] += 1
            if any(b.echoes):
                st["echo_nonzero"] += 1
        if b.msgs:
            st["msg_turns"] += 1
            st["msgs"] += len(b.msgs)

        di = sim.next_turn()
        if di < 0 or sim.dragon_id(di) != did:
            st["desync"] += 1
        else:
            ours = sim.round_block(di)
            if ours != text and st["diff"] == 0:
                st["first"] = (did, b.round, ours, text)
            st["diff"] += int(ours != text)

        # Half the rays carry a payload that fits 32 bits and half do not, so
        # both sides of the "a legacy receiver cannot be handed a wide payload"
        # rule are exercised on every map. Using only wide payloads made the
        # protocol-2 pass vacuous: every message was dropped, so 0 arrived and
        # the ray itself was never compared at all.
        reply = [f"SONAR {d} {payload(did, k) if k % 2 else narrow(did, k)}"
                 for k, d in enumerate(DIRS)]
        ok = b.safe_dirs()
        if b.length >= 6 and (b.round % 7) == 0:
            reply.append(f"SPLIT {b.length // 2}")
        else:
            reply.append(f"MOVE {ok[0] if ok else b.dir}")
        declare = (did % 2 == 0) if protocol == "mixed" else protocol >= 3
        if declare:
            reply.append("PROTOCOL 3")
        reply.append("ENDTURN")
        out = "\n".join(reply) + "\n"
        if di >= 0:
            sim.reply(di, out)
        return out

    g = OracleGame(map_text, policy)
    try:
        g.run()
    except Exception as ex:
        st["error"] = f"{type(ex).__name__}: {ex}"
    return st


def main() -> int:
    args = sys.argv[1:]
    maps = ([pathlib.Path(a) for a in args] if args else
            sorted((ROOT / "maps-official").glob("*.map")))
    bad = 0
    for protocol in (3, 2, "mixed"):
        label = {3: "### bots declaring PROTOCOL 3",
                 2: "### bots declaring PROTOCOL 2 (never declared: ECHOES must not appear)",
                 "mixed": "### only even dragon ids declare PROTOCOL 3 "
                          "(per-dragon protocol, and 64-bit messages dropped for legacy receivers)"}[protocol]
        print(f"\n{label}")
        for mp in maps:
            st = run(mp.read_text(), protocol)
            ok = st["diff"] == 0 and st["desync"] == 0 and "error" not in st
            if protocol == "mixed":
                # some dragons must have echoes and some must not, or the case
                # is not actually mixed and proves nothing
                ok = ok and 0 < st["echo_turns"] < st["turns"]
            elif protocol >= 3:
                ok = ok and st["echo_turns"] > 0
            else:
                ok = ok and st["echo_turns"] == 0
            bad += not ok
            note = ""
            if "error" in st:
                note = "  " + st["error"]
            print(f"  {mp.stem:22s} {st['turns']:6d} turns  blocks differing "
                  f"{st['diff']:4d}  echo turns {st['echo_turns']:6d} "
                  f"(non-zero {st['echo_nonzero']:5d})  messages {st['msgs']:5d}"
                  f"  {'ok' if ok else 'FAIL'}{note}")
            if st["first"]:
                did, rnd, ours, theirs = st["first"]
                print(f"    first difference, dragon {did} round {rnd}:")
                for a, b in zip(ours.split("\n"), theirs.split("\n")):
                    if a != b:
                        print(f"      ours   {a!r}")
                        print(f"      engine {b!r}")
                        break
    print("\n" + ("sonar matches the engine" if not bad else f"{bad} map(s) FAILED"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
