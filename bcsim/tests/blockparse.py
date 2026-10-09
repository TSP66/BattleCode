"""Minimal parser for a round block, used only by tests to write policies that
keep dragons alive long enough to exercise the interesting rules."""

from __future__ import annotations

VISION = 3
STEP = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}


class Block:
    def __init__(self, text: str):
        lines = text.split("\n")
        i = 0
        self.round = int(lines[0].split()[1])
        self.dir = lines[1].split()[1]
        self.length = int(lines[2].split()[1])
        self.units = int(lines[3].split()[1])
        nmsgs = int(lines[4].split()[1])
        self.msgs = [int(lines[5 + k]) for k in range(nmsgs)]
        i = 5 + nmsgs

        # Protocol 3 puts ECHOES between the messages and the tiles, so it shifts
        # every offset after it. Without this, a protocol-3 block parses the echo
        # line as a tile and then reads everything one line out -- silently, with
        # plausible-looking numbers.
        self.echoes = None
        head = lines[i].split()
        if head and head[0] == "ECHOES":
            self.echoes = tuple(int(v) for v in head[1:])
            i += 1

        self.tiles = {}
        order = []
        for k in range(49):
            x, y, pearl, cd = (int(v) for v in lines[i + k].split())
            self.tiles[(x, y)] = (pearl, cd)
            order.append((x, y))
        i += 49
        self.head = order[24]  # centre of the 7x7 window

        nbodies = int(lines[i].split()[1])
        i += 1
        self.bodies = {}
        for k in range(nbodies):
            team, did, x, y, facing, is_head = lines[i + k].split()
            self.bodies[(int(x), int(y))] = (team, int(did), facing, is_head == "1")
        i += nbodies

        self.hedges = [lines[i + r].split() for r in range(8)]
        i += 8
        self.vedges = [lines[i + r].split() for r in range(7)]

    def edge(self, direction: str) -> str:
        """The edge symbol on the given side of the head, by window position."""
        r, c = VISION, VISION
        if direction == "N":
            return self.hedges[r][c]
        if direction == "S":
            return self.hedges[r + 1][c]
        if direction == "W":
            return self.vedges[r][c]
        return self.vedges[r][c + 1]

    def safe_dirs(self) -> list[str]:
        """Directions that are not certain death, judged only from what the
        dragon can see: no kelp, and no dragon segment on the far side."""
        out = []
        for d, (dx, dy) in STEP.items():
            symbol = self.edge(d)
            if symbol == "w":
                continue
            if symbol != ".":       # a portal: cannot see the far side
                out.append(d)
                continue
            # neighbour by window offset, wrapping handled by the block itself
            target = list(self.tiles.keys())[(VISION + dy) * 7 + (VISION + dx)]
            if target in self.bodies:
                continue
            out.append(d)
        return out
