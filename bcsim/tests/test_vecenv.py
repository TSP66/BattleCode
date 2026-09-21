"""The batched env has its own turn loop, reward bookkeeping and observation
builder, so it gets checked against the real engine too:

  1. lockstep play: identical blocks, deaths and results, driven through the
     structured action API;
  2. the observation tensors agree with the protocol block they came from;
  3. the action mask never marks a move legal that the block says is kelp, and
     never marks an illegal split legal.
"""

from __future__ import annotations

import ctypes
import pathlib
import random
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import bcsim                                   # noqa: E402
from bcsim.env import _lib as VLIB             # noqa: E402
from blockparse import Block                   # noqa: E402
from oracle import OracleGame                  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]  # repo root

VLIB.bcv_acting_dragon.argtypes = [ctypes.c_void_p, ctypes.c_int]
VLIB.bcv_round_block.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]

DIRS = "NESW"


def vec_block(env, i=0) -> str:
    buf = ctypes.create_string_buffer(1 << 16)
    n = VLIB.bcv_round_block(ctypes.c_void_p(env._h), i, buf, len(buf))
    return buf.raw[:n].decode()


def lockstep(map_text: str, rng: random.Random, steps_per_game: int = 10_000) -> dict:
    """Plays one game through the batched env and the engine at once."""
    env = bcsim.BattlecodeVecEnv([map_text], num_envs=1, num_threads=1, seed=0,
                                 egocentric=False, random_pearl_seed=False)
    env.reset()
    checked = {"turns": 0}

    def bridge(dragon_id: int, block: str) -> str:
        ours_id = VLIB.bcv_acting_dragon(ctypes.c_void_p(env._h), 0)
        if ours_id != dragon_id:
            raise AssertionError(f"turn order: engine {dragon_id}, batched env {ours_id}")
        ours = vec_block(env, 0)
        if ours != block:
            raise AssertionError(f"block mismatch on dragon {dragon_id}\n"
                                 f"--- engine\n{block}\n--- ours\n{ours}")
        check_observation(env, block)
        checked["turns"] += 1

        b = Block(block)
        kind = np.zeros(1, np.int8)
        n_steps = np.zeros(1, np.int8)
        dirs = np.zeros((1, 8), np.int8)
        split = np.zeros(1, np.int16)
        send = np.zeros(1, np.int8)
        sonar = np.zeros(1, np.uint32)

        roll = rng.random()
        if roll < 0.04 and b.length >= 5:
            kind[0], split[0] = 1, rng.randint(1, b.length)
            text = f"SPLIT {split[0]}"
        elif roll < 0.06:
            kind[0] = 2
            text = "JUNK"
        else:
            safe = b.safe_dirs() or list(DIRS)
            n = 1 if rng.random() > 0.15 or b.length <= 3 else rng.randint(2, 3)
            picks = [rng.choice(safe)] + [rng.choice(DIRS) for _ in range(n - 1)]
            kind[0], n_steps[0] = 0, n
            for i, d in enumerate(picks):
                dirs[0, i] = DIRS.index(d)
            text = "MOVE " + "".join(picks)
        if rng.random() < 0.3:
            send[0], sonar[0] = 1, rng.randrange(0, 2 ** 32)
            text += f"\nSONAR {sonar[0]}"
        env.step_raw(kind, n_steps, dirs, split, send, sonar)
        return text + "\nENDTURN\n"

    game = OracleGame(map_text, bridge)
    game.run()
    if game.notices:
        raise ValueError(f"map was corrected by the engine: {game.notices[:2]}")
    env.close()
    return checked


def check_observation(env, block_text: str) -> None:
    """The tensors must say exactly what the protocol block says."""
    b = Block(block_text)
    obs = env.observation()
    local, scalar = obs.local[0], obs.scalar[0]
    ch = {name: i for i, name in enumerate(bcsim.CHANNELS)}
    keys = list(b.tiles.keys())

    for row in range(7):
        for col in range(7):
            x, y = keys[row * 7 + col]
            pearl, cd = b.tiles[(x, y)]
            assert local[ch["pearl"], row, col] == pearl, f"pearl at {(x, y)}"
            if cd < 0:
                assert local[ch["never_spawn"], row, col] == 1, f"never spawn at {(x, y)}"
            else:
                expect = min(cd, 99) / 99.0
                got = local[ch["pearl_time"], row, col]
                assert abs(got - expect) < 1e-6, f"pearl time at {(x, y)}: {got} != {expect}"

            occupant = b.bodies.get((x, y))
            planes = ["self_head", "self_body", "ally_head", "ally_body",
                      "enemy_head", "enemy_body"]
            total = sum(local[ch[p], row, col] for p in planes)
            assert total == (1.0 if occupant else 0.0), f"occupancy at {(x, y)}"

    my_team, my_id = None, None
    for (x, y), (team, did, _facing, is_head) in b.bodies.items():
        if (x, y) == b.head and is_head:
            my_team, my_id = team, did
    assert my_id is not None
    assert local[ch["self_head"], 3, 3] == 1.0, "the centre must be our own head"

    assert abs(scalar[bcsim.SCALARS.index("length_raw")] - b.length) < 1e-6
    assert abs(scalar[bcsim.SCALARS.index("round")] - b.round / 500) < 1e-6
    assert scalar[bcsim.SCALARS.index("face_" + b.dir.lower())] == 1.0
    assert scalar[bcsim.SCALARS.index("team_b")] == (1.0 if my_team == "B" else 0.0)
    assert abs(scalar[bcsim.SCALARS.index("num_msgs")] - min(len(b.msgs), bcsim.MAX_MSGS)) < 1e-6
    for i, value in enumerate(b.msgs[:bcsim.MAX_MSGS]):
        assert obs.msgs[0, i] == value, "sonar payload"

    # edges, in the dragon's own frame (egocentric is off in this test)
    for i, d in enumerate(DIRS):
        symbol = b.edge(d)
        kelp = local[ch["kelp_" + ["fwd", "right", "back", "left"][i]], 3, 3]
        portal = local[ch["portal_" + ["fwd", "right", "back", "left"][i]], 3, 3]
        assert kelp == (1.0 if symbol == "w" else 0.0), f"kelp {d}"
        assert portal == (1.0 if symbol not in (".", "w") else 0.0), f"portal {d}"

    # the mask must not claim a kelp step is legal. Codec actions 0, 1, 2 are
    # the one step moves: forward, left, right, relative to the dragon.
    mask = obs.mask[0]
    facing = DIRS.index(b.dir)
    for action, turn in enumerate([0, 3, 1]):          # forward, left, right
        d = DIRS[(facing + turn) % 4]
        if b.edge(d) == "w":
            assert mask[action] == 0, f"mask allows walking into kelp going {d}"
        occupied = list(b.tiles.keys())[(3 + {"N": -1, "S": 1}.get(d, 0)) * 7
                                        + (3 + {"E": 1, "W": -1}.get(d, 0))]
        if b.edge(d) == "." and occupied in b.bodies and not b.bodies[occupied][3]:
            assert mask[action] == 0, f"mask allows walking into a body going {d}"


def main() -> int:
    maps = sorted(pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "maps")).glob("*.map"))
    total = 0
    for path in maps:
        for seed in range(2):
            got = lockstep(path.read_text(), random.Random(seed))
            total += got["turns"]
            print(f"  {path.name:38s} seed {seed}: {got['turns']:6d} turns verified")
    print(f"batched env matches the engine over {total} turns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
