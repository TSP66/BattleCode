"""Runs our simulator in lockstep with the official engine and compares
everything observable: the init block and round block for every dragon turn,
every death with its reason and round, and the final result."""

from __future__ import annotations

import ctypes
import pathlib
import random

from oracle import OracleGame

LIB = ctypes.CDLL(str(pathlib.Path(__file__).resolve().parents[1] / "bcsim" / "libbctext.so"))
LIB.bct_create.restype = ctypes.c_void_p
LIB.bct_create.argtypes = [ctypes.c_char_p, ctypes.c_uint, ctypes.c_char_p, ctypes.c_int]
LIB.bct_destroy.argtypes = [ctypes.c_void_p]
LIB.bct_next.argtypes = [ctypes.c_void_p]
LIB.bct_round.argtypes = [ctypes.c_void_p]
LIB.bct_dragon_id.argtypes = [ctypes.c_void_p, ctypes.c_int]
LIB.bct_num_dragons.argtypes = [ctypes.c_void_p]
LIB.bct_init_block.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
LIB.bct_round_block.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
LIB.bct_reply.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
LIB.bct_deaths.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
LIB.bct_result.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
LIB.bct_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong), ctypes.c_int]

STAT_NAMES = ["step", "portal_step", "sprint_step", "pearl_eaten", "pearl_spawn",
              "death_wall", "death_self", "death_other", "death_head", "death_action",
              "split_ok", "split_illegal", "split_limit", "sonar_cast", "sonar_hit",
              "sonar_self", "sonar_lost", "blocked_spawn"]

ENGINE_SEED = 1592614637


class Mismatch(Exception):
    pass


class Sim:
    """Thin ctypes wrapper over the text-mode driver."""

    def __init__(self, map_text: str, seed: int = ENGINE_SEED):
        err = ctypes.create_string_buffer(512)
        self.h = LIB.bct_create(map_text.encode(), seed, err, 512)
        if not self.h:
            raise ValueError(f"map rejected: {err.value.decode()}")
        self._buf = ctypes.create_string_buffer(1 << 16)

    def next_turn(self) -> int:
        return LIB.bct_next(ctypes.c_void_p(self.h))

    def dragon_id(self, di: int) -> int:
        return LIB.bct_dragon_id(ctypes.c_void_p(self.h), di)

    def init_block(self, di: int) -> str:
        n = LIB.bct_init_block(ctypes.c_void_p(self.h), di, self._buf, len(self._buf))
        return self._buf.raw[:n].decode()

    def round_block(self, di: int) -> str:
        n = LIB.bct_round_block(ctypes.c_void_p(self.h), di, self._buf, len(self._buf))
        return self._buf.raw[:n].decode()

    def reply(self, di: int, text: str) -> None:
        LIB.bct_reply(ctypes.c_void_p(self.h), di, text.encode())

    def deaths(self) -> list[tuple[int, int, str]]:
        out = (ctypes.c_int * (3 * 256))()
        got = []
        while True:
            n = LIB.bct_deaths(ctypes.c_void_p(self.h), out, 256)
            got += [(out[i * 3], out[i * 3 + 1], chr(out[i * 3 + 2])) for i in range(n)]
            if n < 256:
                return got

    def stats(self) -> dict:
        out = (ctypes.c_longlong * len(STAT_NAMES))()
        n = LIB.bct_stats(ctypes.c_void_p(self.h), out, len(STAT_NAMES))
        return {STAT_NAMES[i]: out[i] for i in range(n)}

    def result(self) -> dict:
        out = (ctypes.c_int * 9)()
        LIB.bct_result(ctypes.c_void_p(self.h), out)
        return {"rounds": out[0], "winner": out[1], "end_reason": out[2],
                "a_dragons": out[3], "b_dragons": out[4],
                "a_length": out[5], "b_length": out[6],
                "a_longest": out[7], "b_longest": out[8]}

    def __del__(self):
        if getattr(self, "h", None):
            LIB.bct_destroy(ctypes.c_void_p(self.h))


def random_policy(rng: random.Random, chaos: float = 0.25):
    """Mostly plausible moves, with a helping of malformed and illegal ones."""

    def policy(dragon_id: int, block: str) -> str:
        lines = []
        roll = rng.random()
        if roll < chaos * 0.20:
            lines.append(rng.choice(["MOVE", "MOVE X", "SPLIT", "SPLIT abc", "JUNK",
                                     "MOVE NN N", "SONAR -1", "SONAR 99999999999",
                                     "", "MOVE nne", "SPLIT 0", "SPLIT 1", "SPLIT 99"]))
        elif roll < chaos * 0.55:
            lines.append("SPLIT " + str(rng.randint(1, 6)))
        elif roll < chaos:
            steps = "".join(rng.choice("NESW") for _ in range(rng.randint(1, 5)))
            lines.append("MOVE " + steps)
        else:
            lines.append("MOVE " + rng.choice("NESW"))
        if rng.random() < 0.3:
            lines.append(f"SONAR {rng.randrange(0, 2**32)}")
        if rng.random() < 0.1:  # a second action: the last one read wins
            lines.append("MOVE " + rng.choice("NESW"))
        if rng.random() < 0.1:
            lines.append("LOG hello")
        return "\n".join(lines) + "\nENDTURN\n"

    return policy


def compare(map_text: str, policy, seed: int = ENGINE_SEED, label: str = "") -> dict:
    """Runs both engines on the same map and replies. Raises Mismatch on any
    difference. Returns a summary of what was checked."""
    sim = Sim(map_text, seed)
    turns = 0
    seen_ids: set[int] = set()
    problems: list[str] = []

    def bridge(dragon_id: int, block: str) -> str:
        nonlocal turns
        di = sim.next_turn()
        if di < 0:
            problems.append(f"our sim ended early, engine still asking dragon {dragon_id}")
            raise Mismatch(problems[-1])
        ours_id = sim.dragon_id(di)
        if ours_id != dragon_id:
            problems.append(f"turn order: engine says dragon {dragon_id}, we say {ours_id}")
            raise Mismatch(problems[-1])
        if dragon_id not in seen_ids:
            seen_ids.add(dragon_id)
            ours_init = sim.init_block(di)
            theirs_init = game.inits.get(dragon_id)
            if theirs_init is not None and ours_init != theirs_init:
                problems.append(f"init block for dragon {dragon_id}:\n"
                                f"--- engine ---\n{theirs_init}\n--- ours ---\n{ours_init}")
                raise Mismatch(problems[-1])
        ours = sim.round_block(di)
        if ours != block:
            problems.append(_diff(block, ours, f"{label} round block, dragon {dragon_id}, turn {turns}"))
            raise Mismatch(problems[-1])
        turns += 1
        text = policy(dragon_id, block)
        sim.reply(di, text)
        return text

    game = OracleGame(map_text, bridge)
    game.run()

    if game.notices:
        raise ValueError(f"engine corrected this map, so it is not a fair test: {game.notices[:3]}")

    ours_deaths = sim.deaths()
    theirs_deaths = game.deaths
    if ours_deaths != theirs_deaths:
        raise Mismatch(f"{label} deaths differ\nengine: {theirs_deaths[:20]}\nours:   {ours_deaths[:20]}")

    if sim.next_turn() >= 0:
        raise Mismatch(f"{label} our sim wants more turns after the engine finished")

    ours = sim.result()
    theirs = game.result
    mismatched = {}
    for key, value in (("rounds", theirs.rounds), ("end_reason", theirs.end_reason),
                       ("a_dragons", theirs.a_dragons), ("b_dragons", theirs.b_dragons),
                       ("a_length", theirs.a_length), ("b_length", theirs.b_length)):
        if ours[key] != value:
            mismatched[key] = (value, ours[key])
    winner = {None: -1, "A": 0, "B": 1}[theirs.winner]
    if ours["winner"] != winner:
        mismatched["winner"] = (winner, ours["winner"])
    if mismatched:
        raise Mismatch(f"{label} result differs (engine, ours): {mismatched}")

    summary = {"turns": turns, "rounds": theirs.rounds + 1, "deaths": len(theirs_deaths),
               "dragons": len(seen_ids), "winner": theirs.winner}
    summary["stats"] = sim.stats()
    return summary


def _diff(theirs: str, ours: str, what: str) -> str:
    a, b = theirs.split("\n"), ours.split("\n")
    out = [f"{what}: blocks differ"]
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else "<missing>"
        y = b[i] if i < len(b) else "<missing>"
        if x != y:
            out.append(f"  line {i}: engine {x!r} != ours {y!r}")
            if len(out) > 8:
                break
    return "\n".join(out)
