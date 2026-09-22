# Cloning dev test r3: features, super-sprints, self-traps, loss weighting

Overnight 2026-09-22/23, on the M3 Pro (MPS, 18 GB). Branch
`distill-devtest-features`. Target: the dev test clone (submission 1302, 1201
games, 10.8M turns), ~90% held-out accuracy, then win rates to decide.

**Win-rate results: see section 7** (filled in last).

## 1. What the data says about dev test

- **It is deterministic.** On repeated identical observations its actions agree
  99.95% of the time. So the ~17% the r3 clone misses is information the network
  lacks or cannot learn, not noise.
- **It is nearly memoryless, but not quite.** On identical observations with a
  *different* previous observation, actions still agree 98.5%.
- **The errors are left vs right.** 98% of its moves are single steps (straight 47%,
  left 25%, right 25%). r3-style clones get straight ~87% right but left/right only
  ~70%. On the ~1.5k windows that are exactly left-right mirror-symmetric, it turns
  toward world north/west 62-68% of the time, which suggests a world-frame planner
  with a fixed neighbour order.
- **It plays a swarm.** 78% of its turns are dragons of length < 6 (41% length 2);
  an ally is in view 85% of the time.
- **Sonar messages** reach 3.5% of its turns; 55% carry a `0xbca` tag with a
  position (x = bits 14-19, y = bits 8-13; median 11 cells from the receiver vs 30 at
  random). Too rare to matter for accuracy.
- **The old held-out number leaked.** Deterministic bots replay identical games:
  33% of held-out rows have an exact twin observation in training games. And r3's
  own split was drawn over a smaller game list, so on today's split v10 scores
  85.0%, well above its logged 82.5%. The clean comparison is the **22 held-out
  games played after r3 was trained** (166k turns). The cutoff assumes the
  submission's 19:28 was ACST.

## 2. Features (all computable by a deployed bot from its own past turns)

`bcsim/train/clone_features.py`. Ablation: 300 training games, 3 epochs, 64x4,
same 300k held-out rows (`clone_report.py`). *novel* = rows whose observation
never occurs in a training game.

| input | held-out | novel | left | right |
|---|---|---|---|---|
| baseline (7x7 + 14 scalars) | 78.1 | 76.2 | 70.1 | 70.1 |
| + hist3 (last 3 moves, age, length change) | 79.1 | 77.3 | 72.3 | 72.1 |
| + hist8 | 79.3 | 77.3 | 72.4 | 72.4 |
| + moves (per-move BFS / area / heads) | 78.0 | 76.2 | 69.9 | 69.9 |
| **+ mem** (13x13 remembered map) | **80.4** | **78.3** | 75.0 | 74.7 |
| + hist8 + mem | 80.4 | 78.3 | 74.7 | 74.8 |
| **+ mem + memfar** (whole-map memory summary) | **80.6** | **78.5** | 75.1 | 75.0 |
| 128x6, no features (capacity probe, stopped after epoch 0) | = baseline at epoch 0 | | | |

- **mem** (676 scalars): for the 13x13 around the head, in the dragon's own frame:
  pearls it expects (seen there, or a respawn timer that has since run out), how
  recently it saw each cell, where its own head has been, kelp it has seen.
- **memfar** (18): whole-map memory summarised: the 3 nearest expected pearls
  (ego dx, dy, distance), 1/(1+d)-weighted expected pearls ahead/left/right/behind,
  share of the map seen, number of pearls expected.
- Memory starts empty for every dragon (a split child is a new process).
- **Dropped:** action history (redundant once memory is in), hand-made search
  features (the conv already computes them), a bigger net (no gain at this data
  size, and too expensive to ship).

**Full data, mem + memfar** (1081 games, 6 epochs, `imitate2.py --chunk-games 270`):

| model | all held-out | novel | clean 22 games | clean novel |
|---|---|---|---|---|
| v10 (= r3) | (85.0, leaked) | (82.3, leaked) | 81.3 | 81.0 |
| **mm: mem + memfar** | 85.2 | 82.3 | **85.6** | **84.6** |
| ft_control (+2 epochs) | 85.4 | 82.1 | 85.7 | 84.5 |

**+4.3 points of clean out-of-sample accuracy over r3. 90% was not reached.** The
duplicate-free ceiling looks well short of it without better learning of the
left/right rule. See section 8.

## 3. Super-sprints

`bcsim/train/super_sprint.py`, with `VecEnv::probe` (`cpp/bc_vec.hpp`): on every
dev test turn the replay is paused, and each codec move is tried on a copy of the
game. A 2- or 3-step sprint is a **super-sprint** when, right after it:

- **last:** every enemy dragon is dead or trapped, and we still have a dragon; or
- **longest:** the enemy's longest dragon (length >= 4) is dead or trapped, and we
  still have a dragon;

and neither dev test's own move nor any single step does the same. *Trapped* is
conservative: every way forward is kelp or a body that cannot move out of the way
first, and the dragon is too short to split (or its team is full).

- 10,751 labels in 10.8M turns (0.10%): 1,984 last-dragon, 8,767 longest-dragon;
  3,747 two-step and 7,004 three-step. All are inside the deployed mask.
- Dev test's own move achieved the same 674 (last) / 4,219 (longest) times.
- **ft_sprint** (fine-tune from mm, 2 epochs, labelled rows relabelled at weight 10):
  on the 1,076 held-out labels it picks the exact sprint 41.7% of the time
  (control 2.0%) and some sprint 42.8% (48.5% on game-ending ones). Elsewhere it
  sprints on 0.42% of turns (dev test 0.28%, control 0.24%), and its agreement
  with dev test there is unchanged (85.1% vs 85.4%). The sprint outputs are no longer
  all zeros.

## 4. Self-traps (counted, dropped as agreed)

`bcsim/train/self_trap.py`: a single step into a pocket smaller than the dragon
(flood fill on the full board, every body a wall), when another single step had
at least max(2 x length, that + 10) cells, after which the dragon really died
(never acted again within area + 3 rounds, with no split in between). A case is
**dropped as a sacrifice if the dragon killed an enemy** (head-on or an enemy
running into it) between the move and its death. The simulator's death
events name the killer.

- 209,660 steps into too-small pockets; 24,978 of those avoidable and fatal;
  **16,029 (64%) were sacrifices** (the swarm dives into enemy pockets to trade),
  leaving **8,949 true self-traps (0.083% of turns)**. That's fewer than
  super-sprints (0.099%), so they were not used for training.
- 63% are length-2 dragons, and they are *less* common at the unit limit than
  turns in general (22% vs 35%), so they are not unit recycling.

## 5. Loss weighting

- **Toss-ups** (`bcsim/train/tossup.py`): dev test turned left/right; the full mm clone
  is split (P(L)+P(R) >= 0.7, |P(L)-P(R)| <= 0.3); the two steps look alike in the
  window (same pearl, BFS distance, reachability, heads nearby, area within 5); and
  memory does not favour a side (remembered-pearl cones within 0.1).
  143,170 rows = 2.5% of left/right turns. A well trained clone is rarely split
  (2.9% of turns at 0.8/0.2), so true toss-ups are few. **ft_tossup:** weight 0.25.
- **Aggressive** (**ft_aggr**): dev test's own moves that directly won or
  killed/trapped the longest enemy get weight **7** (4,893 rows), super-sprint
  relabels **10**, toss-ups **0.25**.

## 6. Evaluation

`bcsim/train/round_robin.py`: every pair plays 12 games per map on the 8 live maps
(`runs/eval_maps` = maps-official minus help), half on each side, greedy.
Memory models play through `clone_eval.Policy`, whose `MemoryTracker` matches the
offline features exactly (checked row by row). `yardstick.evaluate` gained a hook
for such stateful policies. A draw counts half. At 96 games the standard error of
a score is ~0.05.

## 7. Win rates

(filled in below)

## 8. Next steps

(filled in below)

## Reproduce

```bash
cd bcsim
make CXX=clang++ CXXFLAGS="-O3 -mcpu=native -std=c++17 -shared -fPIC -pthread"   # mac
../.venv-train/bin/python -m train.replay_dataset --games ../runs/replays/dev_test_1_p --team-id 545
../.venv-train/bin/python -m train.clone_cache --data ../runs/replays/dev_test_1_p/dataset --submission 1302 --out ../runs/clone_cache/devtest_1302
../.venv-train/bin/python -m train.clone_features --cache ../runs/clone_cache/devtest_1302 --build mem,memfar,moves
../.venv-train/bin/python -m train.imitate2 --cache ../runs/clone_cache/devtest_1302 --features mem,memfar --chunk-games 270 --epochs 6 --out ../runs/i2/full_mem_memfar
../.venv-train/bin/python -m train.super_sprint --games ../runs/replays/dev_test_1_p --team-id 545 --only 1302
../.venv-train/bin/python -m train.self_trap --games ../runs/replays/dev_test_1_p --team-id 545 --only 1302
../.venv-train/bin/python -m train.tossup --cache ../runs/clone_cache/devtest_1302 --ckpt ../runs/i2/full_mem_memfar/best.pt --weight 0.25
../.venv-train/bin/python -m train.round_robin --out ../runs/evals/rr.jsonl name=ckpt ...
```

`clone_cache.py` also writes `obs_hash.npy` (`--hash` alone rebuilds it), the
per-row observation hash behind the "novel" numbers.
