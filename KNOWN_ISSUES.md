# Known issues

Found 2026-09-21 while fixing the C++ submission. Logged here and deliberately
**not fixed yet**. None of them touch the training loop: the trainer's
observations and masks come from the simulator (`bcsim/`), which is the
reference that each of these was measured against.

## Fixed: C++ submissions v3 and v4 never ran their network

The judge runs C++ bots as wasm with no filesystem, so `mybot/weights.bin` was
never found and every turn used the heuristic fallback. Local `unswbc run`
hid it because it runs C++ natively beside the file. Fixed in v5: the weights
are compiled in (`mybot/weights_data.hpp`, written by
`bcsim/train/export_cpp.py`). Recorded here because the fallback made the
failure invisible. **Run `wasmprobe/check_bot.sh <ckpt>` before every upload.**

## 1. Deployed bot: body order is lost when the body leaves the window

- **Where:** `mybot/obs.hpp`, `Snapshot::trace_body()`.
- **What:** the protocol lists only the body segments inside the 7x7 window.
  The bot numbers its segments (`self_index`) and finds its tail
  (`self_tail`) by following the visible chain from the head. When the body
  leaves the window and comes back, the chain breaks, and segments after the
  gap get no index. The simulator numbers the whole body, so training saw
  indices the deployed bot cannot reconstruct.
- **How often:** in `wasmprobe/parity_obs.py`, 32 of 1,049 turns (3%), only on
  `arena` (small, wraps) and `help`. Every other channel, scalar and mask
  entry matched exactly.
- **Fix options:** (a) the bot remembers its body across turns, since it knows
  its own moves, pearls (LENGTH) and splits (a split child would have to
  start from what it can see); or (b) the simulator marks only the visible
  chain. (b) is the exact fix but changes training observations, so it needs
  a new run.

## 2. Deployed bot: some legal sprints are masked out (same cause)

- **Where:** `mybot/obs.hpp`, the mask.
- **What:** a sprint may step onto a cell the tail vacates during the move.
  That's only legal if the cell is known to be the tail. When the chain is
  broken (issue 1), the bot can't know, so it forbids the move; the
  simulator allows it.
- **Direction:** conservative only. The bot never allows a move the simulator
  forbids; `parity_obs.py` fails if it ever does.
- **How often:** 1 turn in 1,049 (`help`).
- **Fix:** comes with issue 1.

## Fixed: every dragon's first turn traded itself away

- **Where:** `mybot/main.cpp`, `fallback()`.
- **What:** the turn that widens the weights cannot also afford a forward pass,
  so every dragon's first turn is played by a heuristic. It stepped straight on
  without looking at what was there, and entering a head's cell is legal and
  kills both dragons. A split child's first turn is exactly when a friendly head
  is one step away.
- **How bad:** on Default all four dragons split on round 1 and each child's
  blind step killed an ally: **six of our eight dragons gone before round 2**,
  in every game, on both sides of a mirror match, which is why local evaluation
  never saw it (`bcsim` has no first-turn fallback -- every turn there runs the
  network, so the round robin cannot see this class of bug at all). On the server
  it lost a game 0-12 in two rounds to an opponent that did not reciprocate
  (battle 58343, submission v11).
- **Fix:** the heuristic now refuses any step onto a cell holding a head, of
  either team, and prefers straight on among what is left. Round 0-3 deaths on
  Default: 12 -> 0. Against the identical weights with the old heuristic it
  scores **0.875 over 16 games** (both sides of the 8 live maps). On the server,
  the same opponent and map that beat v11 0-12 in two rounds loses to v12
  **109-0** over a full game (battles 58343 and 58665).
- **Present in v10 and every submission before it.**

## 3. `archive/dbgbot/obs.py` marks a pearl on every cell

- **What:** its pearl channel is 1 everywhere, even when the protocol says
  pearl 0. The field is likely read as the string `"0"`, which is truthy.
  Its mask is then wrong too, because it counts a pearl as growth.
- **Impact:** none on training or on the C++ bot. It only matters if the
  Python bot is used as a submission or as a parity reference. It misled a
  parity check once, and `wasmprobe/parity_net.py` no longer uses it.

## 4. Yardstick: a mirror match scored 0.41

`anchor v1_96x4_747M` playing itself scored 0.41 over 88 games, where 0.5 is
expected. That's within sampling noise at this size (two standard errors is
about ±0.11), but if later mirror matches sit clearly below 0.5, look for a
side (team A/B) bias in the eval setup or the maps.

## 5. A split child's facing can differ from the engine on some geometry

- **What:** `bcsim/tests/stress.py` reports 2 block mismatches, both a body
  line's **facing character** for a freshly split dragon:
  `gen23/splitter_policy` dragon 134 turn 1622 (engine `N`, ours `S`) and
  `gen78/splitter_policy` dragon 25 turn 337 (engine `E`, ours `S`).
- **Not sonar.** Verified identical at commit `8f4376ad` before the sonar work,
  so it predates it. Sonar parity is exact (SONAR.md).
- **Impact:** the facing is one character of one body line in the observation, on
  generated maps only; no official map has reproduced it. It would matter for
  observation parity if it happens on a live map, so it is worth fixing before
  relying on `seg_dir` for a split child.
- **Lead, not yet verified:** `Game::split` in `bcsim/cpp/bc_core.hpp` derives the
  child's facing from `opposite(direction_between(...))` over the first two
  segments, and `direction_between` searches N, E, S, W and returns the *first*
  direction that connects. That is ambiguous whenever two directions connect the
  same pair of tiles. Wrap cannot cause it here (generated maps are at least 10 in
  each dimension), but a **portal** can, and these maps carry several. Confirming
  that means dumping the child's two segments at the failing turn.

## Checks that exist now

- `wasmprobe/check_bot.sh <ckpt>`: the pre-upload gate. It checks zip size, a
  no-filesystem wasm run under the judge's metering, observation parity with
  the simulator, parity of the remembered inputs (`mem`, `memfar`) with
  `clone_features.MemoryTracker`, forward parity with the checkpoint, and a
  full real-engine game.
- `bcsim/tests/test_vecenv.py [maps_dir]`: simulator vs engine, turn by turn;
  also run on the augmented maps.
