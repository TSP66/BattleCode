"""Test policies: survival-minded, so games last and the rules get exercised."""
import random
from blockparse import Block


def smart_policy(rng: random.Random, chaos: float = 0.05, split_rate: float = 0.02,
                 sonar_rate: float = 0.2, sprint_rate: float = 0.08):
    def policy(dragon_id: int, text: str) -> str:
        b = Block(text)
        lines = []
        roll = rng.random()
        if roll < chaos:
            lines.append(rng.choice(["MOVE", "MOVE X", "SPLIT", "SPLIT abc", "JUNK", "",
                                     "MOVE NN N", "SONAR -1", "SPLIT 0", "SPLIT 1",
                                     "SPLIT 99", "MOVE nne", "ENDTURN extra"]))
        elif roll < chaos + split_rate and b.length >= 4:
            lines.append(f"SPLIT {rng.randint(2, b.length - 2)}")
        else:
            safe = b.safe_dirs()
            if not safe:
                safe = list("NESW")
            # prefer carrying on, and head for a pearl when one is adjacent
            weights = []
            for d in safe:
                w = 3.0 if d == b.dir else 1.0
                dx, dy = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}[d]
                keys = list(b.tiles.keys())
                nb = keys[(3 + dy) * 7 + (3 + dx)]
                if b.tiles[nb][0] == 1:
                    w += 6.0
                weights.append(w)
            step = rng.choices(safe, weights)[0]
            if rng.random() < sprint_rate and b.length > 3:
                extra = "".join(rng.choice("NESW") for _ in range(rng.randint(1, 3)))
                lines.append(f"MOVE {step}{extra}")
            else:
                lines.append(f"MOVE {step}")
        if rng.random() < sonar_rate:
            lines.append(f"SONAR {rng.randrange(0, 2**32)}")
        if rng.random() < 0.03:
            lines.append("LOG note")
        if rng.random() < 0.03:
            lines.append(f"INDICATOR d{dragon_id}")
        return "\n".join(lines) + "\nENDTURN\n"
    return policy


def survivor_policy(rng: random.Random, split_rate: float = 0.01, sonar_rate: float = 0.25):
    """Never deliberately suicides: picks a visible-safe direction, sprints only
    straight into a tile it can see is clear. Produces long games."""
    def policy(dragon_id: int, text: str) -> str:
        b = Block(text)
        keys = list(b.tiles.keys())
        lines = []
        if rng.random() < split_rate and b.length >= 6:
            lines.append(f"SPLIT {rng.randint(2, b.length - 2)}")
        else:
            safe = b.safe_dirs() or ["N"]
            weights = []
            for d in safe:
                dx, dy = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}[d]
                nb = keys[(3 + dy) * 7 + (3 + dx)]
                weights.append((6.0 if b.tiles[nb][0] else 1.0) + (2.0 if d == b.dir else 0.0))
            step = rng.choices(safe, weights)[0]
            move = step
            if b.length > 4 and rng.random() < 0.15:
                dx, dy = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}[step]
                nb2 = keys[(3 + 2 * dy) * 7 + (3 + 2 * dx)]
                if nb2 not in b.bodies and b.edge(step) == ".":
                    move = step + step
            lines.append(f"MOVE {move}")
        if rng.random() < sonar_rate:
            lines.append(f"SONAR {rng.randrange(0, 2**32)}")
        return "\n".join(lines) + "\nENDTURN\n"
    return policy


def portal_policy(rng: random.Random, sonar_rate: float = 0.3):
    """Actively seeks portals, so the teleport rules get hammered."""
    def policy(dragon_id: int, text: str) -> str:
        b = Block(text)
        keys = list(b.tiles.keys())
        portal_dirs, plain = [], []
        for d in "NESW":
            sym = b.edge(d)
            if sym == "w":
                continue
            if sym != ".":
                portal_dirs.append(d)
                continue
            dx, dy = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}[d]
            if keys[(3 + dy) * 7 + (3 + dx)] not in b.bodies:
                plain.append(d)
        if portal_dirs and rng.random() < 0.7:
            step = rng.choice(portal_dirs)
        elif plain:
            step = rng.choices(plain, [3.0 if d == b.dir else 1.0 for d in plain])[0]
        else:
            step = rng.choice("NESW")
        move = step
        if b.length > 3 and rng.random() < 0.25:
            move = step + "".join(rng.choice("NESW") for _ in range(rng.randint(1, 2)))
        lines = [f"MOVE {move}"]
        if rng.random() < 0.02 and b.length >= 5:
            lines = [f"SPLIT {rng.randint(2, b.length - 2)}"]
        if rng.random() < sonar_rate:
            lines.append(f"SONAR {rng.randrange(0, 2**32)}")
        return "\n".join(lines) + "\nENDTURN\n"
    return policy


def splitter_policy(rng: random.Random):
    """Splits at every opportunity, to drive a team into the 64 unit limit."""
    def policy(dragon_id: int, text: str) -> str:
        b = Block(text)
        if b.length >= 4:
            return f"SPLIT {b.length // 2}\nENDTURN\n"
        safe = b.safe_dirs() or ["N"]
        keys = list(b.tiles.keys())
        best = max(safe, key=lambda d: b.tiles[keys[(3 + {"N": -1, "S": 1}.get(d, 0)) * 7
                                                   + (3 + {"E": 1, "W": -1}.get(d, 0))]][0])
        return f"MOVE {best}\nENDTURN\n"
    return policy


def queen_sprint_policy(rng: random.Random, sprint_rate: float = 0.35, sonar_rate: float = 0.2):
    """The unswbc 1.2.3 rules: survives to grow long, sprints past its free steps (ceil(L/4))
    and into paid ones, and in most games one queen (dragon 0 or 1) gives up late, so the
    round-500 verdict is often decided by the queen and not the longest dragon."""
    doomed = rng.choice([0, 1]) if rng.random() < 0.7 else -1
    doom_round = rng.randint(250, 499)

    def policy(dragon_id: int, text: str) -> str:
        b = Block(text)
        if dragon_id == doomed and b.round >= doom_round:
            return "JUNK\nENDTURN\n"
        keys = list(b.tiles.keys())
        lines = []
        if rng.random() < 0.01 and b.length >= 8:
            lines.append(f"SPLIT {rng.randint(2, b.length - 2)}")
        else:
            safe = b.safe_dirs() or ["N"]
            weights = []
            for d in safe:
                dx, dy = {"N": (0, -1), "E": (1, 0), "S": (0, 1), "W": (-1, 0)}[d]
                nb = keys[(3 + dy) * 7 + (3 + dx)]
                weights.append((6.0 if b.tiles[nb][0] else 1.0) + (2.0 if d == b.dir else 0.0))
            step = rng.choices(safe, weights)[0]
            move = step
            if rng.random() < sprint_rate:
                free = (b.length + 3) // 4
                n = rng.randint(2, free + 2)          # into the paid steps, sometimes past paying
                if rng.random() < 0.7:
                    move = step * n                   # straight on
                else:
                    move = step + "".join(rng.choice("NESW") for _ in range(n - 1))
            lines.append(f"MOVE {move}")
        if rng.random() < sonar_rate:
            lines.append(f"SONAR {rng.randrange(0, 2**32)}")
        return "\n".join(lines) + "\nENDTURN\n"
    return policy
