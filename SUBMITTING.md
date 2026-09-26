# Submitting a bot

## The LSTM bot (lstmbot/, v15 onward)

LSTM checkpoints (`train/lstm_net.py`, `args.arch == "lstm"`) ship as `lstmbot/`, not `mybot/`. One command:

```bash
wasmprobe/submit_lstm.sh runs/ratchet_v8_sab/anchors/gen3.pt "what changed"
CHECK_ONLY=1 wasmprobe/submit_lstm.sh <ckpt> x      # every gate, no upload
```

It exports into `lstmbot/weights.hpp` (`train.export_lstm`), then gates: zip < 3.9 MB; `parity_lstm.py` (grid,
legal-move mask, logits and argmax against the simulator and the checkpoint, int32 accumulator headroom);
three full games under `unswbc run --sandbox`, which since unswbc 1.0.0 builds C++ with the **judge's own clang
and flags** and meters every turn with the judge's cost table (so `meter_bot.sh`/zig are no longer the
reference for this bot). Then it uploads and records the version in `runs/submitted/`.

Numerics: int16 weights (12 bits per output row), grid Q12, activations between layers **Q10 (+-32)**. The first
port used Q12 activations, whose +-8 clip hit residual activations of up to ~18 and moved logits by up to 3.4
(p99 1.5); Q10 brings the bot to p99 0.067 against PyTorch, below the bf16 noise the evaluations ran with.
Cost for gen3 (2026-09-25), 140k turns over 11 maps in the judge sandbox: p50 79.4M, max 82.7M points a turn.
`train/quant_eval.py` measures what the int16 numerics cost in play (fp32 vs the emulated bot, CPU).

## The flat bot (mybot/)

Submissions are **C++ only**. The bot is `mybot/` (`main.cpp`, `obs.hpp`, `memory.hpp`, `net.hpp`, `helper.hpp`, `weights_data.hpp`).
Every step runs from the repo root. `wasmprobe/submit.sh` wraps steps 1-4 with the full `check_bot.sh` gate — prefer it.

## 1. Export the weights

```bash
cd bcsim && ../.venv-train/bin/python -m train.export_cpp --ckpt ../runs/v1/latest.pt && cd ..
```

Writes `mybot/weights_data.hpp` (compiled in: the judge has no filesystem) and prints the parameter count, the
scalar count and MACs.
The checkpoint must be `train.net.ActorCritic`, because that's the only architecture `net.hpp` runs.

A clone trained with the remembered features (`train.imitate2 --features mem,memfar`, see `DISTILL_DEVTEST.md`)
wants **708** scalars, not 14: the window's 14 plus `mem` (676) and `memfar` (18), which `mybot/memory.hpp`
computes as it plays, one memory per dragon process. The exporter writes that count into the header and the bot
feeds exactly as many as the blob asks for, so plain 14-scalar checkpoints still ship unchanged. What the bot
remembers has to equal `clone_features.MemoryTracker` exactly, because that is the tracker the checkpoint's win
rates were measured through -- `check_bot.sh` step 4 is that check, and it compares to the last bit.
`--blocks N` ships only the first N residual blocks. It's a stopgap that damages the policy, so train at a size that fits instead.

## 2. Check that it plays

```bash
unswbc run maps/default.map mybot mybot --no-replay
```

It should play all 500 rounds with no protocol warnings.

## 3. Check the CPU budget (don't skip this)

```bash
wasmprobe/meter_bot.sh
```

This compiles the bot to wasm with the judge's flags and prices every turn using the judge's own cost table.
**Every turn must stay under 100M, and aim for under ~80M.** Turn 0 includes start-up and loading the weights.
Nothing else gives you this number: `unswbc run --sandbox` only prices Python, and the server reports nothing back.

## 4. Submit

```bash
unswbc submit mybot -n "short-name" -d "what changed"
```

The zip limit is 4 MiB.
Wait for the status to turn `active` at <https://game.battlecode.au/submissions>.

## 5. Confirm it runs on the server

```bash
wasmprobe/scrim.sh 7 4      # one unranked game vs team 7 on map 4 (Default)
```

Then check the line `first-turn NO_VALID_ACTION (CPU overrun signature): 0 (ok)`.
Anything other than 0 means dragons are running out of CPU on the judge.
Any other death reason (HIT_SELF, HIT_WALL, …) comes from bad play, not the budget.
Unranked games don't change ELO. `scrim.sh --battle <id>` re-reads a battle that has already finished.

## Traps that already cost us submissions

- **Never write output with `std::cout <<`.** Every write is charged 2.5M points plus 4000 per byte, and each `<<` is a separate write.
  `main.cpp` builds each turn's output into a string and writes it once. Leave it that way.
  One debug LOG line cost ~105M and killed v1 and v2 on turn 0. The server strips LOG from replays anyway.
- **Don't trust MAC estimates.** Measured costs are 3.3 points per MAC for the convolutions, 23 for NumPy, and ~15 for dot-product loops.
  Linear layers are exported transposed so they compile to loops the compiler can vectorise.
- **A small replay doesn't mean the bot crashed.** Arena starts with one dragon per side. Read the death reasons instead.
- **Never run `unswbc update mybot`.** It is new in 1.0.0 and replaces a bot's helper files with the ones the toolkit
  ships, which would throw away `helper.hpp`, `obs.hpp`, `memory.hpp` and `net.hpp`. `submit` is safe: it only checks
  PyPI for a newer toolkit. Set `UNSWBC_NO_UPDATE=1` to silence that check in scripts.

More detail: the `battlecode-judge-cost-model` memory, and `wasmprobe/`
(`kernel.rs` + `meter.py` benchmark individual kernels).

## unswbc 1.0.0 (2026-09-23)

Upgraded from 0.3.0 with `uv tool install unswbc@latest`. `metering.py` and its cost table are byte-identical, and
`MAX_TURN_POINTS` (100M) and `MAX_MEMORY_PAGES` (768) are unchanged, so nothing about the budget moved. Three things did:

- **Work after `ENDTURN` is no longer free.** `sandbox.py refill()` computes `gap = spent() - reported` and deducts it
  from the turn starting next. `mybot` is unaffected: `main.cpp` updates memory at line 292, before it picks an action,
  and `flush_turn()` is the last thing in the loop, so the bot blocks on the next read immediately. Re-metered after the
  upgrade at the same **69M** max turn. Keep it that way -- do not move memory work after the write.
- **File reads cost 6 points/byte, up from ~0.18.** Irrelevant here because `export_cpp.py` compiles the weights in,
  which this makes correct by another 33x.
- **The protocol is at major 3, and sonar changed.** `send_sonar(Direction, uint64)` is new alongside the old
  facing-only uint32 form, and a dragon that cast one gets an `ECHOES kelp ally ally_head enemy enemy_head` line in its
  next observation. **Sonar is now sensing, not just messaging**, and `bc_vec.hpp:561` casts it alongside the action, so
  it is free. We use none of it: `mybot/helper.hpp` has the old API and no `ECHOES` parsing, `cpp/bc_core.hpp
  cast_sonar` is facing-only with no echoes, and nothing in training ever sets `send_sonar`. That is the next capability
  on the table, and it needs the sim rule, the bot protocol and new parity checks together.

The nine maps in `maps-live/` already match 1.0.0 byte-for-byte. It also ships `arena` and `colosseum`, which
`train.maps_fetch` does not list as active, so they are deliberately not in the rotation.
