"""Can a parent actually hand its state to the child it just created?

This is the property the whole memory-transfer idea rests on, and it is a fact
about the game rather than about any codec: a dragon splits, and on the child's
very first turn the child either has the parent's 64 bits or it does not. No
amount of clever encoding helps if the bits never arrive, and a codec trained
against a simulator that silently truncated them would learn to send 32.

Two things are measured, both end to end through the public env API:

  1. delivery -- of every split, how often the child's FIRST observation carries
     the exact payload the parent cast, at full 64-bit width;
  2. the inbox high-water mark -- the protocol puts no cap on NUM_MSGS, so
     MAX_MSGS is the simulator's own choice and has to be shown to be enough.

The parent-to-child pairing comes from the engine's own split event
(`env.last_splits`), not from guessing which new dragon id turned up: dragons
also spawn mid-game, and an earlier version of this test that assumed "the first
previously-unseen dragon" scored those spawns as children that heard nothing.

    python3 tests/test_msg_transfer.py [maps_dir]
"""

from __future__ import annotations

import collections
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import bcsim                       # noqa: E402
from bcsim.env import N_MOVES      # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]

SPLIT_K = [2, 3, 4, 5, 6, 8, 12, 16, -1]
SPLIT_HALF = N_MOVES + SPLIT_K.index(-1)      # split into halves

# A payload with the top bit set, so anything that narrows it to 32 bits (as the
# observation buffer used to) cannot possibly compare equal.
TAG = 0xA5
LEN_RAW = bcsim.SCALARS.index("length_raw")


def payload(seq: int) -> int:
    return (1 << 63) | (seq << 8) | TAG


def seq_of(value: int) -> int | None:
    if not (value >> 63) & 1 or (value & 0xFF) != TAG:
        return None
    return (value & ~(1 << 63)) >> 8


def transfer(maps: list[str], num_envs: int = 64, steps: int = 4000,
             seed: int = 11) -> dict:
    """Splits as often as it legally can, and checks what the child heard."""
    env = bcsim.BattlecodeVecEnv(maps, num_envs=num_envs, num_threads=8, seed=seed,
                                 closure_capacity=max(8192, num_envs * 160))
    obs = env.reset()
    rng = np.random.default_rng(seed)
    n = env.num_envs

    # (env, child dragon id) -> seq the parent cast on the turn it split
    waiting: dict[tuple[int, int], int] = {}
    seen: list[set[int]] = [set() for _ in range(n)]
    seq = 1
    acc = collections.Counter()
    inbox_max = 0
    # every payload we cast, so a delivery can be checked for exactness
    cast: dict[int, int] = {}
    plen: dict[int, int] = {}
    by_len = collections.defaultdict(lambda: [0, 0])   # parent length -> [children, heard]
    by_k = collections.defaultdict(lambda: [0, 0])     # child segments -> [children, heard]
    kof: dict[int, int] = {}

    # Every dragon declares protocol 3 on every turn, as a real bot does. The
    # child inherits it at birth because the declaration is applied before the
    # action, which is what lets a newborn receive a payload wider than 32 bits.
    proto = np.full(n, 3, np.int8)

    for _ in range(steps):
        acts = np.zeros(n, np.int32)
        send = np.zeros(n, np.uint8)
        value = np.zeros((n, bcsim.SONAR_DIRS), np.uint64)
        splitting: dict[int, int] = {}

        for e in range(n):
            did = int(obs.dragon_id[e])
            first_turn = did not in seen[e]
            seen[e].add(did)

            # A child we are waiting on, taking its very first turn.
            if first_turn and (e, did) in waiting:
                want = waiting.pop((e, did))
                acc["children"] += 1
                got = [int(v) for v in obs.msgs[e, :int(obs.num_msgs[e])]]
                hit = [v for v in got if seq_of(v) == want]
                if hit:
                    acc["children_heard_parent"] += 1
                    # exactness, not just arrival: the whole point is that all 64
                    # bits survive the trip
                    if hit[0] == cast[want]:
                        acc["exact"] += 1
                        if cast[want] > 0xFFFFFFFF:
                            acc["exact_wide"] += 1
                    else:
                        acc["mangled"] += 1
                        print(f"  MANGLED: cast {cast[want]:#018x} "
                              f"arrived {hit[0]:#018x}")
                else:
                    acc["children_heard_nothing"] += 1
                by_len[plen.get(want, 0)][0] += 1
                by_k[kof.get(want, 0)][0] += 1
                if hit:
                    by_len[plen.get(want, 0)][1] += 1
                    by_k[kof.get(want, 0)][1] += 1

            legal = np.nonzero(obs.mask[e])[0]
            if len(legal) == 0:
                continue
            if obs.mask[e][SPLIT_HALF]:
                acts[e] = SPLIT_HALF
                cast[seq] = payload(seq)
                # Cast the same payload in all four directions: we are asking
                # whether the child is reachable at all, not from where.
                send[e] = 0b1111
                value[e, :] = np.uint64(cast[seq])
                splitting[e] = seq
                plen[seq] = int(obs.scalar[e, LEN_RAW])
                seq += 1
            else:
                steps_only = legal[legal < 3]
                pick = steps_only if len(steps_only) else legal
                acts[e] = int(rng.choice(pick))

        inbox_max = max(inbox_max, int(obs.num_msgs.max()))
        obs, _, eps = env.step(acts, send_dirs=send, sonar=value, protocol=proto)

        # Ask the engine which splits actually happened -- an illegal split kills
        # the dragon instead, so asking for one is not the same as getting one.
        for e, want in splitting.items():
            for parent, child, k in env.last_splits(e):
                acc["splits"] += 1
                waiting[(e, int(child))] = want
                kof[want] = int(k)

        # An episode reset renumbers everything, so forget what we were tracking
        for row in eps.rows:
            e = int(row[bcsim.EpisodeStats.COLUMNS.index("env")])
            seen[e].clear()
            for key in [key for key in waiting if key[0] == e]:
                waiting.pop(key)

    env.close()
    acc["inbox_max"] = inbox_max
    print("  delivery by the parent's length when it split:")
    for L in sorted(by_len):
        c, h = by_len[L]
        if c >= 30:
            print(f"    length {L:3d}  children {c:5d}  heard {h:5d}  {h/c:6.1%}")
    print("  delivery by how many segments the child took:")
    for k in sorted(by_k):
        c, h = by_k[k]
        if c >= 30:
            print(f"    k {k:3d}  children {c:5d}  heard {h:5d}  {h/c:6.1%}")
    return dict(acc)


def inbox_highwater(maps: list[str], num_envs: int = 64, steps: int = 6000,
                    seed: int = 5) -> int:
    """The worst case for MAX_MSGS: every dragon broadcasting all four ways.

    This is what `sonar=True` does, so it is the densest message traffic the
    simulator can actually produce, and it is the regime PPO would train in.
    """
    env = bcsim.BattlecodeVecEnv(maps, num_envs=num_envs, num_threads=8, seed=seed,
                                 closure_capacity=max(8192, num_envs * 160),
                                 sonar=True)
    obs = env.reset()
    rng = np.random.default_rng(seed)
    worst = 0
    hist = collections.Counter()
    for _ in range(steps):
        worst = max(worst, int(obs.num_msgs.max()))
        for v in obs.num_msgs:
            hist[int(v)] += 1
        acts = np.zeros(env.num_envs, np.int32)
        for e in range(env.num_envs):
            legal = np.nonzero(obs.mask[e])[0]
            if not len(legal):
                continue
            # split sometimes, to build the crowd that makes inboxes deep
            if rng.random() < 0.06 and obs.mask[e][SPLIT_HALF]:
                acts[e] = SPLIT_HALF
            else:
                steps_only = legal[legal < 3]
                acts[e] = int(rng.choice(steps_only if len(steps_only) else legal))
        obs, _, _ = env.step(acts)
    env.close()
    total = sum(hist.values())
    print(f"  inbox depth over {total:,} turns, sonar broadcast on:")
    for k in sorted(hist):
        print(f"    {k:2d} messages  {hist[k]:9,}  {hist[k]/total:6.2%}")
    return worst



def sustained(maps: list[str], num_envs: int = 64, steps: int = 4000,
              seed: int = 23, horizon: int = 6) -> dict:
    """The realistic protocol: every dragon broadcasts its own id, every turn.

    Casting a sonar is free, so a real bot speaks on every turn rather than only
    the turn it splits. The question that matters for a codec is therefore not
    "did the child hear its parent at birth" but "how soon does it hear it", and
    whether it can tell WHICH dragon spoke -- the engine tells a receiver only
    the payload, never the sender, so identity has to be carried in the bits.

    Here the payload is just the sender's dragon id under a tag, which is what
    the codec's SONAR_TAG field is for.
    """
    env = bcsim.BattlecodeVecEnv(maps, num_envs=num_envs, num_threads=8, seed=seed,
                                 closure_capacity=max(8192, num_envs * 160))
    obs = env.reset()
    rng = np.random.default_rng(seed)
    n = env.num_envs
    proto = np.full(n, 3, np.int8)

    seen: list[set[int]] = [set() for _ in range(n)]
    # (env, child) -> [parent id, turns taken so far, heard yet]
    track: dict[tuple[int, int], list] = {}
    first_heard = collections.Counter()
    done = collections.Counter()

    for _ in range(steps):
        acts = np.zeros(n, np.int32)
        send = np.full(n, 0b1111, np.uint8)
        value = np.zeros((n, bcsim.SONAR_DIRS), np.uint64)

        for e in range(n):
            did = int(obs.dragon_id[e])
            seen[e].add(did)
            key = (e, did)
            if key in track:
                parent, turns, heard = track[key]
                if not heard:
                    got = [int(v) for v in obs.msgs[e, :int(obs.num_msgs[e])]]
                    if any(seq_of(v) == parent for v in got):
                        track[key][2] = True
                        first_heard[turns] += 1
                        done["heard"] += 1
                    elif turns + 1 >= horizon:
                        done["never"] += 1
                        track.pop(key)
                    else:
                        track[key][1] = turns + 1

            # Every dragon says who it is, in every direction, every turn.
            value[e, :] = np.uint64(payload(did))

            legal = np.nonzero(obs.mask[e])[0]
            if not len(legal):
                continue
            if obs.mask[e][SPLIT_HALF] and rng.random() < 0.25:
                acts[e] = SPLIT_HALF
            else:
                steps_only = legal[legal < 3]
                acts[e] = int(rng.choice(steps_only if len(steps_only) else legal))

        obs, _, eps = env.step(acts, send_dirs=send, sonar=value, protocol=proto)
        for e in range(n):
            for parent, child, k in env.last_splits(e):
                track[(e, int(child))] = [int(parent), 0, False]
        for row in eps.rows:
            e = int(row[bcsim.EpisodeStats.COLUMNS.index("env")])
            seen[e].clear()
            for key in [key for key in track if key[0] == e]:
                track.pop(key)
    env.close()

    total = done["heard"] + done["never"]
    out = {"tracked": total, "heard": done["heard"]}
    if total:
        print(f"  children tracked        {total:,}")
        run = 0
        for t in range(horizon):
            run += first_heard[t]
            label = "at birth" if t == 0 else f"by turn {t + 1}"
            print(f"    {label:<12s} {run:6,}  {run / total:6.1%}")
    return out

def main() -> int:
    maps_dir = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "maps")
    maps = bcsim.load_maps(maps_dir)
    print(f"{len(maps)} maps, MAX_MSGS {bcsim.MAX_MSGS}, "
          f"SONAR_DIRS {bcsim.SONAR_DIRS}")

    print("\n=== parent -> child delivery ===")
    r = transfer(maps)
    if not r.get("splits"):
        print("FAIL: no split ever became legal, so nothing was tested")
        return 1
    if not r.get("children"):
        print("FAIL: splits happened but no child was ever observed acting")
        return 1
    heard = r.get("children_heard_parent", 0)
    print(f"  splits                  {r['splits']:,}")
    print(f"  children that acted     {r['children']:,}")
    print(f"  heard their parent      {heard:,}  "
          f"({heard / r['children']:.1%} of children)")
    print(f"  payload exact           {r.get('exact', 0):,}")
    print(f"  of those, over 2^32     {r.get('exact_wide', 0):,}")
    print(f"  payload mangled         {r.get('mangled', 0):,}")

    print("\n=== broadcasting every turn, how soon does a child hear its parent ===")
    sus = sustained(maps)

    print("\n=== inbox high-water mark ===")
    worst = inbox_highwater(maps)
    print(f"  deepest inbox seen      {worst}   (MAX_MSGS {bcsim.MAX_MSGS})")

    ok = True
    if r.get("mangled"):
        print("\nFAIL: a payload arrived different from the one cast")
        ok = False
    if not heard:
        print("\nFAIL: no child ever received its parent's payload, so parent to "
              "child transfer does not work")
        ok = False
    if not r.get("exact_wide"):
        print("\nFAIL: no payload above 2^32 ever arrived intact, so the wide "
              "path is unproven")
        ok = False
    if worst > bcsim.MAX_MSGS:
        print(f"\nFAIL: an inbox reached {worst}, over MAX_MSGS {bcsim.MAX_MSGS}; "
              "messages were dropped")
        ok = False
    print("\nparent to child transfer works" if ok else "\ntransfer is broken")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
