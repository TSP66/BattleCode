"""Sends one odd reply into a real game and checks our sim reacts identically.

Used to pin down the reply parser, which is full of C library behaviour
(sscanf's %u accepting a negative, trailing junk rules, and so on) that is far
easier to measure than to read out of the disassembly.
"""

from __future__ import annotations

import random
import sys

import diffsim
import policies
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root

PROBES = [
    # movement
    "MOVE N", "MOVE NN", "MOVE NNE", "MOVE  N", "MOVE N ", "MOVE N\t", "MOVE\tN",
    "MOVE NN N", "MOVE N extra", "MOVE N 2", "MOVEN", " MOVE N", "\tMOVE N",
    "move n", "Move N", "MOVE n", "MOVE X", "MOVE NX", "MOVE", "MOVE ",
    "MOVE NNNNNNNN", "MOVE NESW", "MOVE N#c", "MOVE N # c",
    # splits
    "SPLIT 2", "SPLIT 3", "SPLIT  2", "SPLIT 2 3", "SPLIT 2x", "SPLIT x2",
    "SPLIT +2", "SPLIT -1", "SPLIT 2.0", "SPLIT 0", "SPLIT 1", "SPLIT 99",
    "SPLIT", "SPLIT 2 ",
    # sonar
    "MOVE N\nSONAR 5", "MOVE N\nSONAR -1", "MOVE N\nSONAR 4294967295",
    "MOVE N\nSONAR 4294967296", "MOVE N\nSONAR 0x10", "MOVE N\nSONAR 12 34",
    "MOVE N\nSONAR +5", "MOVE N\nSONAR 007", "MOVE N\nSONAR", "MOVE N\nSONAR ",
    "SONAR 7",
    # ordering and junk
    "MOVE N\nMOVE E", "MOVE E\nMOVE N", "SPLIT 2\nMOVE N", "MOVE N\nSPLIT 2",
    "SONAR 5\nSONAR 6\nMOVE N", "JUNK\nMOVE N", "MOVE N\nJUNK",
    "LOG hi\nMOVE N", "INDICATOR x\nMOVE N", "DOT 1 2 3 4 5\nMOVE N",
    "LINE 1 2 3 4 5 6 7\nMOVE N", "#MOVE N", "MOVE N\r", "\nMOVE N\n\n",
    "", "MOVE N\nENDTURN\nMOVE E", "ENDTURN\nMOVE E", "MOVE N\nendturn",
    "MOVE N\nENDTURN extra",
]


def probe_policy(text: str, rng: random.Random, target_turn: int = 6):
    """Plays safely, then delivers the probe on one chosen turn."""
    base = policies.survivor_policy(rng, split_rate=0.0, sonar_rate=0.0)
    state = {"turn": 0}

    def policy(dragon_id: int, block: str) -> str:
        state["turn"] += 1
        if state["turn"] == target_turn:
            return text if text.endswith("\n") else text + "\nENDTURN\n"
        return base(dragon_id, block)

    return policy


def main(map_path: str):
    map_text = open(map_path).read()
    bad = []
    for text in PROBES:
        try:
            diffsim.compare(map_text, probe_policy(text, random.Random(5)),
                            label=f"probe {text!r}")
            print(f"  match   {text!r}")
        except diffsim.Mismatch as e:
            bad.append((text, str(e)))
            print(f"  DIFFERS {text!r}")
    print(f"\n{len(PROBES) - len(bad)}/{len(PROBES)} probes match")
    for text, err in bad:
        print(f"\n--- {text!r}\n{err[:600]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else str(ROOT / "maps/big_empty.map")))
