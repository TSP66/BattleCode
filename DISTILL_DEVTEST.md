# Cloning dev test r3: features, super-sprints, self-traps, loss weighting

Overnight 2026-09-22/23 on the M3 Pro (MPS, 18 GB), branch
`distill-devtest-features`. Target: the dev test clone (team 545, submission
1302, 1201 games, 10.8M turns). Goal was ~90% held-out accuracy, then win
rates to decide.

## Verdict

| question | answer |
|---|---|
| Better features? | **Yes.** Remembered-map inputs: **+2.9 points** of held-out accuracy over the same pipeline without them, **+4.3** over r3 on games r3 never saw. Round robin: the memory clones beat r3 (r3 last at 0.460 +- 0.016 over 173-192 games per pair); against the frozen anchor field 0.607 vs r3's 0.577 on identical games. |
| 90% accuracy? | **No: 85.2%** (84.6% on unseen games). See "why not 90". |
| Super-sprints labelled? | **Yes, 10,751** (0.10% of turns). The clone learns them (42% recall on held-out labels, from 2%) and the sprint outputs are no longer all zeros. |
| Do super-sprints win more? | **No - they lose.** ft_sprint 0.484 and ft_aggr 0.481 in the round robin (control 0.532), and 0.510 / 0.530 vs the anchors against mm's 0.607. **Cause found:** 99% of super-sprints are head-on, which kills our own dragon too, so weight 10 taught the clone to trade heads: 14x more draws, games 25 rounds shorter. Section 3 has the fix to try. |
| Self-traps? | **Counted, not used:** 8,949 (0.083% of turns), below super-sprints (0.099%), as agreed. 64% of candidates turned out to be sacrifices. |
| Toss-up downweighting? | **Neutral:** 0.524 vs control 0.532, inside noise. |

Nothing here clears the >0.60-against-everything submission bar, so **no upload
is recommended yet**. The memory features are worth keeping; the sprint
relabelling needs the fix below before it is worth another run.

## 1. What the data says about dev test

- **It is deterministic.** On repeated identical observations its actions agree
  99.95%. The ~17% r3 missed is missing information or unlearned rule, not noise.
- **It is nearly memoryless, but not quite.** On identical observations with a
  *different* previous observation, actions still agree 98.5%. So memory pays,
  but only a little - which is what the ablation found.
- **The errors are left vs right.** 98% of its moves are single steps (straight
  47%, left 25%, right 25%). The r3-style baseline gets straight 87% right and
  left/right only 70%. On the ~1.5k windows that are exactly left-right
  mirror-symmetric it turns toward world north or west 62-68% of the time, so
  there is a world-frame planner with a fixed neighbour order underneath.
- **It plays a swarm.** 78% of its turns are dragons shorter than 6 (41% are
  length 2); an ally is in view 85% of the time. Its small dragons dive into
  enemy pockets to trade - see section 4.
- **Sonar** reaches 3.5% of its turns; 55% of those messages carry a `0xbca` tag
  and a position (x = bits 14-19, y = bits 8-13, a median 11 cells from the
  receiver against 30 at random). Too rare to matter; feature built (`msg`), untrained.
- **The old held-out number leaked.** Deterministic bots replay whole games
  move for move, so 33% of held-out rows have an exact twin observation in a
  training game. On top of that, r3's split was drawn over a smaller game list,
  so on today's split v10 scores 85.0% against the 82.5% in its own log. Two
  honest numbers replace it: *novel* rows (no twin in training) and the **22
  held-out games played after r3 was trained** (166k turns; the cutoff reads the
  submission's 19:28 as ACST).

## 2. Features

`bcsim/train/clone_features.py`. Everything here is computable by a deployed
bot from its own past turns, since each dragon is a long-lived process.
Ablation: 300 training games, 3 epochs, 64x4, the same 300k held-out rows
(`clone_report.py`).

| input | held-out | novel | left | right |
|---|---|---|---|---|
| baseline (7x7 window + 14 scalars) | 78.1 | 76.2 | 70.1 | 70.1 |
| + hist3 (last 3 moves, age, length change) | 79.1 | 77.3 | 72.3 | 72.1 |
| + hist8 | 79.3 | 77.3 | 72.4 | 72.4 |
| + moves (per-move BFS, area, heads) | 78.0 | 76.2 | 69.9 | 69.9 |
| **+ mem** (13x13 remembered map) | **80.4** | **78.3** | 75.0 | 74.7 |
| + hist8 + mem | 80.4 | 78.3 | 74.7 | 74.8 |
| **+ mem + memfar** | **80.6** | **78.5** | 75.1 | 75.0 |

- **mem** (676 scalars): the 13x13 around the head in the dragon's own frame -
  pearls it expects (seen there, or a respawn timer that has run out), how
  recently it saw each cell, where its own head has been, kelp it has seen.
- **memfar** (18): the whole map as remembered, summarised - the 3 nearest
  expected pearls (ego dx, dy, distance), 1/(1+d)-weighted expected pearls
  ahead/left/right/behind, share of map seen, pearls expected.
- Memory starts empty per dragon (a split child is a new process).
- **Dropped:** action history (adds nothing once memory is in), hand-made search
  features (the conv trunk already computes them - `moves` actually cost 0.1),
  and a bigger net (128x6 matched the 64x4 baseline after epoch 0, and would not
  fit the judge's 100M budget anyway).

Full data, 1081 training games, 6 epochs, `imitate2.py --chunk-games 270`:

| model | held-out | novel | 22 unseen games | novel there |
|---|---|---|---|---|
| v10 (= r3, `imitate.py`) | (85.0 leaked) | (82.3 leaked) | 81.3 | 81.0 |
| full_base (no extra features, same pipeline) | 82.3 | 79.4 | - | - |
| **mm = mem + memfar** | **85.2** | **82.3** | **85.6** | **84.6** |
| ft_control (mm + 2 more epochs) | 85.4 | 82.1 | 85.7 | 84.5 |

**+2.9 points over the same pipeline without the features, +4.3 over r3 on games
r3 never saw.**

### Why not 90%

Left/right is still 82% (from 70%), and it is the whole gap. The data says the
rule is there to be found - the bot is deterministic and 98.5% of decisions do
not depend on history - but a memoryless 64x4 conv on a 7x7 window is the wrong
shape for what looks like a world-frame planner with a fixed neighbour order.
The next things to try are in section 7, and none of them is a feature.

## 3. Super-sprints

`bcsim/train/super_sprint.py`, on top of `VecEnv::probe` (`cpp/bc_vec.hpp`),
which tries every codec move on a copy of the live game. A 2- or 3-step sprint
is a **super-sprint** when, right after it:

- **last:** every enemy dragon is dead or trapped and our team still has a
  dragon (the game is won), or
- **longest:** the enemy's longest dragon (length >= 4) is dead or trapped and
  our team still has a dragon,

and neither dev test's own move nor any single step achieves the same.
*Trapped* is conservative: every way forward is kelp or a body that cannot move
out of the way first (not a head, not another dragon's tail; its own tail still
kills it), and it is too short to split or its team is full.

- **10,751 labels in 10.8M turns (0.10%)**: 1,984 last-dragon, 8,767
  longest-dragon; 3,747 two-step, 7,004 three-step; all inside the deployed mask.
  Dev test's own move already did it 674 / 4,219 times (those turns are not labelled).
- **ft_sprint** (fine-tune from mm, 2 epochs, labels relabelled at weight 10)
  learns them: on the 1,076 held-out labels it picks the exact sprint **41.7%**
  of the time (control 2.0%), some sprint 42.8% (48.5% on the game-winning
  ones). Elsewhere it sprints on 0.42% of turns (dev test 0.28%, control 0.24%)
  and its agreement with dev test is unchanged (85.1% vs 85.4%). **The sprint
  half of the action space is no longer all zeros**, which was the point.
- **But it loses games** (section 6). The reason is structural, and I only found
  it by re-probing: **a moving head can only kill by entering the enemy's head
  cell, which kills both dragons.** So 99% of super-sprints (10,696 of 10,751)
  are sacrifices; only 55 - the pure trapping ones - leave our dragon alive.
  Weight 10 on 10.7k head-on trades taught the clone to trade heads in general:
  4.2% draws against 0.0-0.3% for mm and r3, and games 25 rounds shorter.
- Whether a trade is *good* depends on the lengths, and often it isn't: the
  last-dragon sprints trade a median length 3 dragon for their median length 2
  (only 11% trade up), while the longest-dragon ones trade 6 for 7 (53% trade up).

**Fix to try next** (labels already built): `sprint_label_good.npy` keeps the
game-winning sprints plus the longest-dragon ones that trade up in length -
6,594 rows - and a weight of 2-3 rather than 10. `sprint_label_surv.npy` keeps
only the 55 non-suicidal ones, which is too few to train on but is the right
class if the aim is pressure without losses.

## 4. Self-traps (counted, then dropped, as agreed)

`bcsim/train/self_trap.py`: a single step into a pocket smaller than the dragon
(flood fill over the whole board, every body a wall - `PROBE_AREA`), when
another single step had at least max(2 x length, that + 10) free cells, after
which the dragon really did die (it never acts again within area + 3 rounds,
with no split in between). A candidate is **dropped as a sacrifice when that
dragon killed an enemy** between the move and one round after its death; the
simulator's death events name the killer (`VecEnv::last_deaths`).

- 209,660 single steps into a too-small pocket; 24,978 avoidable and fatal; of
  those **16,029 (64%) were sacrifices**, leaving **8,949 true self-traps
  (0.083% of turns)** - below the super-sprint rate (0.099%), so per the rule
  they were not trained on.
- They are mostly cheap dragons: 63% length 2, 26% length 3. They are *less*
  common at the unit limit than turns in general (22% vs 35%), so they are not
  unit-slot recycling - they look like genuine dead-end mistakes.
- The 64% sacrifice share is itself a finding about dev test: the swarm spends
  small dragons diving into enemy pockets to trade.

## 5. Loss weighting

- **Toss-ups** (`bcsim/train/tossup.py`): dev test turned left or right; the mm
  clone is split (P(L)+P(R) >= 0.7, |P(L)-P(R)| <= 0.3); the two steps look the
  same in the window (same pearl, BFS distance, reachability, heads next to the
  cell, areas within 5 cells); and memory does not favour a side (remembered-pearl
  cones within 0.1). **143,170 rows = 2.5% of left/right turns**, weight **0.25**
  (ft_tossup). A well trained clone is rarely split - at 0.8/0.2 only 2.9% of
  turns qualify - so true toss-ups are a thin slice, which is also why the effect
  is small.
- **Aggressive** (ft_aggr): dev test's own moves that directly won or
  killed/trapped the longest enemy at weight **7** (4,893 rows), super-sprint
  relabels at **10**, toss-ups at **0.25**.

## 6. Win rates

Method: `bcsim/train/round_robin.py`, 12 games per map on the 8 live maps
(`runs/eval_maps` = maps-official minus help), both sides, greedy, a draw counts
half. Memory clones play through `clone_eval.Policy`, whose `MemoryTracker`
reproduces the offline features exactly (verified row by row: mem exact, memfar
to float16 rounding). Two seeds, so 173-192 games per pair.

| | ft_aggr | ft_tossup | ft_sprint | ft_control | mm | v10 | **mean** | se |
|---|---|---|---|---|---|---|---|---|
| ft_control | 0.540 | 0.484 | 0.562 | - | 0.516 | 0.557 | **0.532** | 0.016 |
| ft_tossup | 0.509 | - | 0.518 | 0.516 | 0.490 | 0.586 | **0.524** | 0.016 |
| mm | 0.557 | 0.510 | 0.539 | 0.484 | - | 0.505 | **0.519** | 0.016 |
| ft_sprint | 0.506 | 0.482 | - | 0.438 | 0.461 | 0.536 | **0.484** | 0.016 |
| ft_aggr | - | 0.491 | 0.494 | 0.460 | 0.443 | 0.514 | **0.481** | 0.016 |
| v10 (r3) | 0.486 | 0.414 | 0.464 | 0.443 | 0.495 | - | **0.460** | 0.016 |

Against the frozen anchors (96 games per opponent; the ft_sprint and ft_aggr
runs hit a 3h limit, so the comparison below uses only the 32 (opponent, map)
cells every run finished - 384 games each - because the dropped games are the
long ones):

| model | anchor field | draws | kills | deaths | longest | rounds |
|---|---|---|---|---|---|---|
| **mm** | **0.607** | 0.000 | 41.6 | 104.3 | 5.36 | 266 |
| v10 (r3) | 0.577 | 0.003 | 41.6 | 104.5 | 5.43 | 274 |
| ft_aggr | 0.530 | 0.049 | 36.6 | 80.5 | 4.76 | 241 |
| ft_sprint | 0.510 | 0.042 | 36.5 | 82.6 | 4.56 | 243 |

Full 480-game runs, for the two that finished: mm 0.648 (SSS r3 0.646, Vibing r4
0.823, SHINK r1 0.625, Sabotage 0.562, ft6 0.583) against v10's 0.609 (0.604 /
0.719 / 0.599 / 0.573 / 0.552).

Reading it:

- **r3 is last in the round robin** (0.460 +- 0.016, ~2.5 se under even) and mm
  beats it by +0.030 against the same anchors. The features help, modestly.
- **mm vs r3 head to head is a tie** (0.505 over 192 games). The gain shows up
  against the wider field, not against the team whose replays both were trained on.
- **ft_control vs mm is within noise** (0.516), so the extra 2 epochs are not the
  story.
- **Super-sprint and aggressive weighting hurt**: -0.05 in the round robin and
  -0.08 to -0.10 against the anchors, with the draw rate and shorter games
  pointing straight at head-trading.
- **Toss-up downweighting is neutral** (0.524 vs 0.532), which is what a 1.3%
  slice of rows at weight 0.25 should do.
- Accuracy did not predict any of this, exactly as expected: mm is +4.3 points on
  r3 and only +0.030 against the field, and ft_sprint is level on accuracy while
  clearly worse in games.

Caveats: `full_base`'s round-robin pass was cut to protect the rest (the machine
was running 8 evaluations at once), so the no-feature model has accuracy numbers
but no win rates; the two truncated anchor runs are only compared on complete
cells; every number here is greedy play on 8 maps.

## 7. What I would do next, in order

1. **Retry super-sprints properly**: `sprint_label_good.npy` (6,594 rows: wins,
   plus longest-kills that trade up) at weight 2-3, against a fresh control.
   Keeping the sprint head non-zero is valuable for later RL even if the clone
   itself does not improve, which was the original reason for asking.
2. **Attack left/right, not features.** It is the whole remaining gap and no
   input I added moves it much past 82%. Options: predict the *world-frame*
   direction instead of the ego-frame turn (the symmetric-window evidence says
   the rule lives there); add a small recurrent or frame-stacked state; or train
   with a mirror-augmented pair loss so the tie-break is the only thing left to
   learn.
3. **Settle mm vs r3 with more games** - 192 games is +-0.036 at 1 se, and the
   anchors say +0.030. A 400-game head-to-head, or the ratchet's gate on the main
   machine, would call it.
4. **Then the deployment cost**: `mem` is 676 extra scalars into the first fuse
   layer (~0.5M extra MACs, small against the 64x4 trunk), but the bot needs a
   per-dragon memory in `mybot/` (issue 1 in KNOWN_ISSUES.md wants that anyway)
   and the C++ features must match `MemoryTracker` exactly - parity test first.
5. Self-traps as an auxiliary *penalty* rather than a relabel, if the 8,949 rows
   are ever worth it.

## Files

| path | what |
|---|---|
| `bcsim/train/clone_cache.py` | replay dataset -> flat memmaps (+ `obs_hash.npy` for "novel") |
| `bcsim/train/clone_features.py` | mem, memfar, hist, moves, msg; `MemoryTracker` for play |
| `bcsim/train/imitate2.py` | cloning with extra inputs, chunked loading, relabels, per-row weights |
| `bcsim/train/super_sprint.py`, `self_trap.py` | replay labelling |
| `bcsim/train/tossup.py` | toss-up detection and weights |
| `bcsim/train/clone_eval.py`, `round_robin.py`, `clone_report.py`, `sprint_report.py`, `collect_results.py` | evaluation |
| `bcsim/cpp/bc_vec.hpp` | `VecEnv::probe`, `free_area`, `trapped`, `last_deaths` |
| `runs/i2/*/best.pt`, `log.jsonl` | the clones |
| `runs/evals/*.jsonl`, `summary.json` | every game result quoted here |

## Reproduce

```bash
cd bcsim
make CXX=clang++ CXXFLAGS="-O3 -mcpu=native -std=c++17 -shared -fPIC -pthread"   # mac
P=../.venv-train/bin/python
C=../runs/clone_cache/devtest_1302
$P -m train.replay_dataset --games ../runs/replays/dev_test_1_p --team-id 545
$P -m train.clone_cache --data ../runs/replays/dev_test_1_p/dataset --submission 1302 --out $C
$P -m train.clone_features --cache $C --build mem,memfar,moves,hist3,hist8,msg
$P -m train.imitate2 --cache $C --features mem,memfar --chunk-games 270 --epochs 6 --out ../runs/i2/full_mem_memfar
$P -m train.super_sprint --games ../runs/replays/dev_test_1_p --team-id 545 --only 1302
$P -m train.self_trap   --games ../runs/replays/dev_test_1_p --team-id 545 --only 1302
$P -m train.tossup --cache $C --ckpt ../runs/i2/full_mem_memfar/best.pt --weight 0.25
mkdir -p ../runs/eval_maps   # the 8 live maps: maps-official minus help
for m in arena big_empty colosseum default default_small queen_of_spades schooltime trophy; do
  cp ../maps-official/$m.map ../runs/eval_maps/; done
$P -m train.round_robin --out ../runs/evals/rr.jsonl --device mps mm=../runs/i2/full_mem_memfar/best.pt v10=../runs/submitted/v10.pt
```

On this machine a 6-epoch full-data run is ~55 min and 96 games take 20-80 min
depending on load; `--learner NAME` runs one round-robin pass, so passes can run
as parallel processes into the same file.
