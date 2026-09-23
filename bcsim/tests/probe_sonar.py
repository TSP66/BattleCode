"""What the 1.0.0 engine actually does with sonar, asked of the engine itself.

The helper headers give the protocol -- `SONAR <dir> <uint64>` out, `NUM_MSGS`
and an optional `ECHOES` line with five counts in -- but not the rules, and
guessing rules costs hours. So this drives the reference engine through
OracleGame with a policy that broadcasts in all four directions every turn, and
reports what comes back:

  * whether ECHOES appears at all, and whether it is one aggregate for all four
    sonars or one per sonar;
  * whether the counts ever exceed one, which says whether a ray passes through
    what it hits or stops at the first thing;
  * who receives a message -- allies only, or enemies too (if enemies do, then
    anything we encode is readable by them);
  * whether a dragon hears its own sonar, and whether its own body blocks it;
  * whether a full 64-bit payload survives the round trip.

    python tests/probe_sonar.py [map ...]
"""

from __future__ import annotations

import collections
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from oracle import OracleGame                   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
DIRS = "NESW"
STEP = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}

# A payload that exercises the whole word: top bit, bottom bit, and a field in
# the middle carrying the sender and the direction, so a receiver can say where
# each message came from.
TOP = 1 << 63


def payload(did: int, d: int) -> int:
    return TOP | (did & 0xFFF) << 16 | (d & 0x3) << 8 | 0x5A


def parse(text: str) -> dict:
    """Round block -> the parts this probe needs, tolerating the ECHOES line."""
    lines = text.split("\n")
    out = {"round": int(lines[0].split()[1]), "dir": lines[1].split()[1],
           "length": int(lines[2].split()[1]), "units": int(lines[3].split()[1])}
    n = int(lines[4].split()[1])
    out["msgs"] = [int(lines[5 + k]) for k in range(n)]
    i = 5 + n
    out["echoes"] = None
    if lines[i].split() and lines[i].split()[0] == "ECHOES":
        vals = lines[i].split()[1:]
        out["echoes"] = tuple(int(v) for v in vals)
        i += 1
    tiles, order = {}, []
    for k in range(49):
        x, y, pearl, cd = (int(v) for v in lines[i + k].split())
        tiles[(x, y)] = (pearl, cd)
        order.append((x, y))
    i += 49
    out["head"] = order[24]
    out["window"] = order
    nb = int(lines[i].split()[1])
    i += 1
    bodies = {}
    for k in range(nb):
        team, did, x, y, facing, is_head = lines[i + k].split()
        bodies[(int(x), int(y))] = (team, int(did), facing, is_head == "1")
    out["bodies"] = bodies
    i += nb
    out["hedges"] = [lines[i + r].split() for r in range(8)]
    i += 8
    out["vedges"] = [lines[i + r].split() for r in range(7)]
    return out


def edge(b: dict, d: str) -> str:
    r = c = 3
    if d == "N":
        return b["hedges"][r][c]
    if d == "S":
        return b["hedges"][r + 1][c]
    if d == "W":
        return b["vedges"][r][c]
    return b["vedges"][r][c + 1]


def safe_dirs(b: dict) -> list[str]:
    """Directions that are not certain death. Kelp is lowercase 'w' in a block
    (bc_text.hpp append_edge_symbol); anything that is neither 'w' nor '.' is a
    portal, whose far side cannot be seen from here."""
    out = []
    for d, (dx, dy) in STEP.items():
        sym = edge(b, d)
        if sym == "w":
            continue
        if sym != ".":
            out.append(d)
            continue
        if b["window"][(3 + dy) * 7 + (3 + dx)] in b["bodies"]:
            continue
        out.append(d)
    return out


def run(map_text: str, name: str, rounds_cap: int = 120) -> dict:
    stat = {
        "turns": 0, "with_echoes": 0, "echo_lens": collections.Counter(),
        "echo_max": [0] * 5, "echo_any": collections.Counter(),
        "msgs_per_turn": collections.Counter(),
        "own_msg": 0, "ally_msg": 0, "enemy_msg": 0, "unknown_msg": 0,
        "payload_ok": 0, "payload_bad": 0, "bad_example": None,
        "teams": {}, "sent": 0,
    }

    def policy(did: int, text: str) -> str:
        b = parse(text)
        stat["turns"] += 1
        # who is this dragon? the head in the middle of its own window
        me = b["bodies"].get(b["head"])
        my_team = me[0] if me else "?"
        stat["teams"][did] = my_team

        if b["echoes"] is not None:
            stat["with_echoes"] += 1
            e = b["echoes"]
            stat["echo_lens"][len(e)] += 1
            for k in range(min(5, len(e))):
                stat["echo_max"][k] = max(stat["echo_max"][k], e[k])
            if any(e):
                stat["echo_any"][e] += 1

        stat["msgs_per_turn"][len(b["msgs"])] += 1
        for m in b["msgs"]:
            if not (m & TOP) or (m & 0xFF) != 0x5A:
                stat["payload_bad"] += 1
                if stat["bad_example"] is None:
                    stat["bad_example"] = m
                continue
            stat["payload_ok"] += 1
            sender = (m >> 16) & 0xFFF
            if sender == did:
                stat["own_msg"] += 1
            elif stat["teams"].get(sender) == my_team:
                stat["ally_msg"] += 1
            elif sender in stat["teams"]:
                stat["enemy_msg"] += 1
            else:
                stat["unknown_msg"] += 1

        reply = []
        for k, d in enumerate(DIRS):
            reply.append(f"SONAR {d} {payload(did, k)}")
            stat["sent"] += 1
        ok = safe_dirs(b)
        # split now and then, so that there are allies to hear the broadcasts
        if b["length"] >= 6 and b["units"] < b.get("unit_limit", 64) and \
                (b["round"] % 7) == 0:
            reply.append(f"SPLIT {b['length'] // 2}")
        else:
            reply.append(f"MOVE {ok[0] if ok else b['dir']}")
        # The engine only speaks protocol 3 -- 64-bit directed sonar and the
        # ECHOES line -- to a bot that says so, and the shipped helper says so
        # on *every* turn, immediately before ENDTURN, not once at startup.
        reply.append("PROTOCOL 3")
        reply.append("ENDTURN")
        return "\n".join(reply) + "\n"

    g = OracleGame(map_text, policy)
    try:
        g.run()
    except Exception as ex:                       # a probe is not a test
        stat["error"] = f"{type(ex).__name__}: {ex}"
    return stat


def main() -> int:
    args = sys.argv[1:]
    maps = ([pathlib.Path(a) for a in args] if args else
            sorted((ROOT / "maps-official").glob("*.map"))[:4])
    for mp in maps:
        st = run(mp.read_text(), mp.stem)
        print(f"\n=== {mp.stem} ===")
        if "error" in st:
            print(f"  engine stopped: {st['error']}")
        print(f"  {st['turns']} dragon turns, {st['sent']} sonars sent "
              f"(4 per turn, one per direction)")
        print(f"  ECHOES line present on {st['with_echoes']}/{st['turns']} turns; "
              f"value counts per line {dict(st['echo_lens'])}")
        print(f"  max echo per field  kelp={st['echo_max'][0]} ally={st['echo_max'][1]} "
              f"ally_head={st['echo_max'][2]} enemy={st['echo_max'][3]} "
              f"enemy_head={st['echo_max'][4]}")
        top = st["echo_any"].most_common(5)
        if top:
            print("  most common non-zero echo tuples "
                  "(kelp, ally, ally_head, enemy, enemy_head):")
            for tup, c in top:
                print(f"    {tup}  x{c}")
        print(f"  messages received per turn {dict(sorted(st['msgs_per_turn'].items()))}")
        print(f"  by sender: own {st['own_msg']}, ally {st['ally_msg']}, "
              f"enemy {st['enemy_msg']}, unseen sender {st['unknown_msg']}")
        print(f"  payload intact {st['payload_ok']}, corrupted {st['payload_bad']}"
              + (f" (e.g. {st['bad_example']})" if st["bad_example"] is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
