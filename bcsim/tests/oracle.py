"""Drives the official unswbc WASM engine in-process as a ground-truth oracle.

The engine calls back once per dragon turn with the exact observation block it
would send to a bot, and takes the reply text back. That makes it a perfect
reference: we can feed our own simulator the same replies and compare the
blocks it produces byte for byte.
"""

from __future__ import annotations

import os
import pathlib
import sys

# UNSWBC_PKG: a directory holding another release's `unswbc` package (an unpacked wheel),
# to compare engine versions; default the installed toolkit.
_PKG = pathlib.Path(os.environ["UNSWBC_PKG"]) if os.environ.get("UNSWBC_PKG") else None
for p in pathlib.Path.home().glob(".local/share/uv/tools/unswbc/lib/*/site-packages"):
    _PKG = _PKG or p
if _PKG is None:
    raise SystemExit("unswbc toolkit not found")
sys.path.insert(0, str(_PKG))

from unswbc.engine import EngineModule, MatchResult  # noqa: E402

DEATH_REASONS = {"W": "hit wall", "S": "hit self", "O": "hit other body",
                 "H": "head to head", "A": "no valid action"}


class OracleGame:
    """One match against the reference engine, driven by a python callback.

    `policy(dragon_id, block) -> reply_text` is called once per dragon turn.
    Every (dragon_id, init_block, round_block, reply) is recorded in `log`.
    """

    def __init__(self, map_text: str, policy, debug: int = 15, seed: int = 0):
        self.map_text = map_text
        self.seed = seed
        self.policy = policy
        self.log: list[dict] = []
        self.deaths: list[tuple[int, int, str]] = []
        self.notices: list[str] = []
        self.inits: dict[int, str] = {}
        self._engine = EngineModule()
        self.result: MatchResult | None = None
        self._debug = debug

    def run(self) -> MatchResult:
        def spawn(dragon_id: int, init: bytes) -> None:
            self.inits[dragon_id] = init.decode()

        def reply(dragon_id: int, block: bytes) -> bytes:
            text = block.decode()
            out = self.policy(dragon_id, text)
            self.log.append({"id": dragon_id, "block": text, "reply": out})
            return out.encode()

        def death(dragon_id: int, round_num: int, reason: str) -> None:
            self.deaths.append((dragon_id, round_num, reason))

        self.result = self._engine.run(
            self.map_text.encode(), reply, death, spawn,
            self.notices.append, self._debug, seed=self.seed)
        return self.result


def run_oracle(map_text: str, policy, debug: int = 15, seed: int = 0) -> OracleGame:
    game = OracleGame(map_text, policy, debug, seed)
    game.run()
    return game
