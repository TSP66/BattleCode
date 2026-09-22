# Distillation experiments on a second machine

The main machine's GPU runs the ratchet (`bcsim/train/ratchet.py`), so distillation
runs somewhere else. This file covers what is in the repo for that, how to set up,
and the one hard constraint: **only `train.net.ActorCritic` can be shipped** (see
"What can be submitted").

## 1. Setup

```bash
git clone git@github.com:TSP66/BattleCode.git BattleCode && cd BattleCode
scripts/setup.sh           # .venv-train (torch cu128 + numpy), builds bcsim/*.so, smoke test
# older GPU/driver:  TORCH_INDEX=https://download.pytorch.org/whl/cu126 scripts/setup.sh
```

`make -C bcsim` builds four libraries. Which one a script loads is chosen by
`BCSIM_LIB` before `import bcsim`, and the scripts set it themselves:

| library | used by |
|---|---|
| `libbcvec.so` | train, distill, yardstick, imitate |
| `libbcvec_priv.so` | anything needing the 8 global features (finetune_team, ratchet_train, team_critic) |
| `libbcvec_replay.so` | `replay_dataset` (allows the 4+ step sprints real games contain) |
| `libbctext.so` | map parsing |

Run every command from `bcsim/` (`cd bcsim`). Use `../.venv-train/bin/python`.

## 2. What is committed

| path | what |
|---|---|
| `runs/replays/<team>/games/` | every scraped game: `<id>.replay` (gzipped Cap'n Proto) + `<id>.json` (match metadata), and `series.jsonl` |
| `runs/ft3/maps/` | the 8 maps the rotation held until 2026-09-22; the ratchet run and every eval so far use these |
| `maps-live/` | the rotation as of 2026-09-23: Arena and Colloseum out, **Devil** (32x16) in, 7 maps. Grade on these |
| `maps/` | all 12 maps, all active for training: the live 7 plus arena, Colloseum, help, small, queen_of_spades_but_she_ages |
| `runs/anchors/*.pt` | frozen clones used as eval/training opponents |
| `runs/submitted/v*.pt` + `submitted.jsonl` | every uploaded network |
| `runs/imitate_*/best.pt`, `log.jsonl` | every behaviour clone and its training curve |
| `runs/ft6/snapshots/turns_112721920.pt` | ft6 at 113M turns: the strongest RL policy so far, and the ratchet's start |
| `runs/team_critic/pretrained.pt` | the replay-pretrained win/draw/loss critic |
| `runs/ft*/CHANGES.md` | why each fine-tune existed and what it showed |
| `runs/clone_compare*/`, `runs/server_tests/`, `runs/critic_probe/results.json` | head-to-head and server results |

**Not committed** (derived, large): `runs/replays/<team>/dataset/` (6.2 GB of per-game
`.npz`) and `runs/team_critic/data/` (4.9 GB). Rebuild them as in section 3.

### Scraped games (as of 2026-09-22 ~21:20)

| folder | team | team id | games |
|---|---|---|---|
| `dev_test_1_p` | dev test 1 :P (#1 on the ladder) | 545 | ~1380 |
| `vibing` | Vibing++ | 306 | ~2330 |
| `sss` | SSS | 91 | ~1200 |
| `sponge_albert_and_bob` | Sponge(Albert and Bob) | 213 | ~800 |
| `shink_ai_6500` | SHINK AI 6500 | 280 | ~500 |
| `matcha_latte` | Matcha Latte | 417 | ~550 |
| `vom` | vom | 419 | ~450 |
| `sabotage_d` | Sabotage-d | 46 | ~400 |
| `team` | team | 33 | ~290 |

A game between two tracked teams can be in both folders. Every game re-simulates
exactly in bcsim (fixed pearl seed), so observations rebuilt from a replay are the
ones the training loop would produce.

## 3. Rebuild the imitation datasets

```bash
cd bcsim
../.venv-train/bin/python -m train.replay_dataset --games ../runs/replays/dev_test_1_p --team-id 545 --workers 12
# one per folder, with the team id from the table above
```

This writes `<folder>/dataset/<game id>.npz` (that team's turns only: local 7x7
window, scalars, mask, action codec id, the game's result) plus `index.jsonl`, which
has one row per game with `submission` (the team's submission id), `samples`,
`diverged` and more. Clones are usually trained on the newest submission only;
`imitate.py --submission-weights "1302:1,1273:0.5"` weights by submission.

## 4. Tools

**`train/distill.py`**: on-policy distillation. The teacher plays self-play in the
simulator and the student fits the teacher's full masked action distribution
(KL(teacher‖student) over legal actions) on exactly the states the teacher visits.

```bash
../.venv-train/bin/python -m train.distill --teacher ../runs/ft6/snapshots/turns_112721920.pt \
    --width 48 --blocks 3 --iters 400 --out ../runs/distill_48x3
```

- `--arch actorcritic` (default) is shippable. `--arch student` trains
  `train/student.py` (ReLU, no GroupNorm, cheaper per MAC), which has **no C++
  implementation** (`mybot/net.hpp` only runs ActorCritic).
- The teacher can be any policy checkpoint (train, finetune, imitate, ratchet):
  width, blocks and hidden are read from the file.
- It logs `kl` and `agree` (student argmax = teacher argmax) to `<out>/log.jsonl` and
  saves `<out>/latest.pt` in train.py's layout.
- Limitation: the states come only from teacher self-play. A student that
  deviates visits states it never trained on. A DAgger variant (the student drives,
  the teacher labels) or teacher-vs-league games are obvious experiments.

**`train/imitate.py`**: supervised behaviour cloning from the replay datasets
(offline, no simulator in the loop). It keeps `best.pt` by held-out NLL. It is the
alternative when the teacher is another team's replays rather than a network.

```bash
../.venv-train/bin/python -m train.imitate --data ../runs/replays/dev_test_1_p/dataset \
    --submission-weights "1302:1" --width 64 --blocks 4 --out ../runs/imitate_x
```

**`train/yardstick.py`**: the eval. Greedy play, both sides of every map.

```bash
../.venv-train/bin/python -m train.yardstick --run ../runs/distill_48x3 \
    --ckpt ../runs/distill_48x3/latest.pt --maps ../runs/ft3/maps --games 12 --threads 16 \
    --lags "" --anchors "../runs/submitted/v10.pt,../runs/anchors/sss_r3_bc_64x4.pt,../runs/anchors/sabotage_bc_64x4.pt,../runs/anchors/shink_r1_bc_64x4.pt,../runs/anchors/vibing_r4_bc_64x4.pt,../runs/ft6/snapshots/turns_112721920.pt" \
    --dump ../runs/distill_48x3/eval.jsonl --max-seconds 5400
```

`--games 12` means 96 games per opponent. **Use at least 84–96 games for any
decision.** Identical nets have scored 0.29–0.75 over 32 games.

## 5. Teachers and how strong they are

The 96-game head-to-heads were on the 8 live maps, greedy play
(`runs/clone_compare6/results.jsonl`). The dev test clone r3 (= v10) scored:

| opponent | v10's score |
|---|---|
| v9 (SSS r2 clone) | 0.59–0.63 |
| SSS r3 clone | 0.65 |
| Sabotage clone | 0.60 |
| SHINK r1 clone | 0.52 |
| Vibing r4 clone | 0.80 |
| **ft6 113M** | **0.48** |

ft6 113M (RL from the SSS r3 clone) is the best local policy by a small margin,
but its 0.52 edge over v10 is within noise. The ratchet's baseline gate
(`runs/ratchet/gates/gen0_baseline.json` on the main machine) will give 96-game
numbers for it against the whole league.

Clone accuracies (held-out top-1): dev test r3 82.5%, SHINK r1 89.4%, SSS r3 80.3%,
Sabotage 79.4%, Vibing r4 90.5%.

## 6. What can be submitted (the hard constraint)

- The judge runs the bot as wasm with **no filesystem**, so the weights are compiled
  in (`export_cpp` writes `mybot/weights_data.hpp`).
- The budget is **100M CPU points per dragon per turn**. Measured on the judge's
  meter: 64x4 (hidden 512) costs **56M on turn 0 and ~69M per network turn**, which
  is the largest ActorCritic that fits with margin. Width 96 x 2 blocks cost ~75M;
  each extra 96-wide block adds ~30M. The mid-case C++ cost is ~3.3 points/MAC
  (not NumPy's 23).
- So distillation can go **bigger teacher → 64x4**, or **64x4 → smaller** to buy
  CPU headroom (for example for search at runtime). A student bigger than 64x4
  won't fit.
- Always upload through `wasmprobe/submit.sh <ckpt> "description"`. It runs
  `check_bot.sh` (zip size, metered no-FS wasm run, observation and forward
  parity, a real-engine game) and records the upload in `runs/submitted/`. See
  `SUBMITTING.md`.
- Submission bar the user set: **> 0.60 against every opponent, confirmed on 96
  games**.

## 7. Bringing results back

Copy `runs/<experiment>/` (checkpoint, `log.jsonl`, `eval.jsonl`) to the main
machine, or commit the chosen `best.pt`/`latest.pt` only.

## Rules of the game worth knowing

- A game ends when a team is wiped out (both wiped in the same round = draw), or
  at round 500. At round 500 the longer **longest dragon** wins; if those are
  equal, the greater **total length** wins; only an exact tie on both is a draw.
  `settle()` in `bcsim/cpp/bc_core.hpp` implements exactly this.
- The policy sees a 7x7 window around the dragon's head plus 14 scalars, and
  picks one of 48 codec actions (moves, sprints up to 3 steps, splits).
