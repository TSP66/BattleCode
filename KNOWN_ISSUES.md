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

## Checks that exist now

- `wasmprobe/check_bot.sh <ckpt>`: the pre-upload gate. It checks zip size, a
  no-filesystem wasm run under the judge's metering, observation parity with
  the simulator, parity of the remembered inputs (`mem`, `memfar`) with
  `clone_features.MemoryTracker`, forward parity with the checkpoint, and a
  full real-engine game.
- `bcsim/tests/test_vecenv.py [maps_dir]`: simulator vs engine, turn by turn;
  also run on the augmented maps.
