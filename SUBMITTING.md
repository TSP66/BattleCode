# Submitting a bot

Submissions are **C++ only**. The bot is `mybot/` (`main.cpp`, `obs.hpp`, `net.hpp`, `helper.hpp`, `weights_data.hpp`).
Every step runs from the repo root. `wasmprobe/submit.sh` wraps steps 1-4 with the full `check_bot.sh` gate — prefer it.

## 1. Export the weights

```bash
cd bcsim && ../.venv-train/bin/python -m train.export_cpp --ckpt ../runs/v1/latest.pt && cd ..
```

Writes `mybot/weights_data.hpp` (compiled in: the judge has no filesystem) and prints the parameter count and MACs.
The checkpoint must be `train.net.ActorCritic`, because that's the only architecture `net.hpp` runs.
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

More detail: the `battlecode-judge-cost-model` memory, and `wasmprobe/`
(`kernel.rs` + `meter.py` benchmark individual kernels).
