# Handoff — 2026-09-23 ~13:15: ratchet3 RUNNING (memory, 23 maps, fresh critic)

## Running now
```
runs/ratchet3/launch.sh   (setsid, supervisor.out / ratchet.log)
  seed        runs/i2/sponge_2110/best.pt  (= v13)
  train maps  maps-all   23 maps, live NINE held at 60% (--live-share 0.6)
  gate maps   maps-live   9 maps -- a gate score still means ladder strength
  league      gen4 (frozen), devtest_2050, v12, v10, sabotage, shink_r1
  critic      runs/team_critic/pretrained.pt  (refit, see below)
```
`maps-all/` is derived and gitignored: `cp maps/*.map maps-gen/*.map maps-all/`.
Stop with `touch runs/ratchet3/STOP` (exits at the next decision point, i.e. end of a
150M-turn segment) or `kill -- -$(cat runs/ratchet3/supervisor.pid)` for immediate.
Dashboard: `.venv-train/bin/python -m train.dash --run runs/ratchet3 --port 8770 --host 0.0.0.0`.

**ratchet2 was stopped 15 min into gen 1** (kept at `runs/ratchet2_abandoned`) because two
new maps were published and an env fixes its map list at construction. Nothing was lost.

## THE MAZE MAPS (stronghold, trauma -- published 2026-09-23, both 48x24)
These are a game mode the old rotation did not contain. Path distance from a team-0 spawn
to the nearest team-1 spawn vs the straight-line torus distance:
```
  stronghold  165 steps / 14 straight = x11.8 detour
  trauma       98        / 14         = x7.0
  devil        34        /  7         = x4.9   <- previous worst
  schooltime    7        /  7         = x1.0
```
In the ratchet3 gen0 baseline, **all 72 stronghold games ran the full 500 rounds with
exactly 0.0 kills** for all six opponents: the teams are fully connected (BFS mirroring
`tile_after_step` says ~100% reachable) but never meet, so the result is the
longest-dragon tiebreak. Pearl density is NOT the cause (18%/12% of tiles spawn, vs
schooltime 13%).

gen0 scored **0.13 on stronghold and 0.72 on trauma** -- they nearly cancel, so the
summary number hides a total failure. **Read the per-map `cells` in the gate JSON.**
Next-run candidate (NOT applied): a maze generator in `mapgen.py` so the 40% invented
share teaches long-detour navigation.

## gen0 baseline moved when the maps did
```
              7 maps   9 maps
  shink_r1     0.702    0.602
  sabotage     0.655    0.556
  v12          0.607    0.528
  gen4         0.583    0.565
  v10          0.548    0.569
  devtest_2050 0.452    0.500
  mean         0.591    0.553   (108 games each, se 0.048)
```
Sponge still leads the league (0.553 over 648 games is ~2.7 se) but by about half as much.

## v13 is on the ladder
Sponge sub-2110 clone, uploaded 2026-09-23, all six checks passed (69M of the judge's
100M on the worst turn; 228 first turns, 0 fallbacks). It replaces **v12, which the round
robin put last of four** -- beaten 0.661 by devtest and 0.630 by sponge, the only results
outside noise at se 0.029.

## THE CRITIC FINDING (the user's idea, and it was load-bearing)
A refit alone cannot give a "clean" critic: same data + same seed reproduces the old
weights **bit for bit** (both hash to 1aca05f429283f45). Only new data changes it.
Scored on the SAME held-out games, split by source (`train.critic_compare`):
```
                      replay (real play)      invented maps
  old (replay-only)   ev 0.507  ll 0.3866    ev -0.0801  ll 0.7825
  new (combined)      ev 0.5135 ll 0.3773    ev  0.2199  ll 0.5842
```
**The old critic had NEGATIVE explained variance on invented maps** -- worse than predicting
the mean, on 40% of training. The new one is better on both subsets. Note the raw `ev` a
pretrain prints fell 0.5068 -> 0.4615 between the runs: that is the held-out SET changing
as self-play joins the split, NOT the critic. Always compare with `critic_compare`.
Old critic kept as `runs/team_critic/pretrained_pre20260923.pt`.

Invented maps are still much harder (0.22 vs 0.51), so more self-play games there would
likely pay. Refit between generations if `calib_ev` drifts.

## Memory is in the simulator (bcsim.N_SCALARS = 708)
- `cpp/bc_memory.hpp`; gate `tests/parity_memory.py` = 144,000 turns, max diff **0** vs the
  Python MemoryTracker. `wasmprobe` check 3 now compares all 708 and passes, so
  `mybot/memory.hpp` and `bc_memory.hpp` agree independently.
- Cost: the raw-rollout microbenchmark said 235,190 -> 188,945 turns/s (-20%). **End to end
  that did not materialise**: ratchet3 runs at 37.8k t/s against the pre-memory run's 34.5k.
  Do not quote the -16%/-20% figure as a PPO cost; measure the run. The Python tracker was
  22.8k turns/s and 26 MB an env, i.e. 8.3x worse and ~26 GB at 1024 envs.
- **Any 14-scalar checkpoint must be widened first**: `train.migrate_scalars` zero-pads the
  scalar layer, which leaves play identical. `runs/anchors708/` holds the migrated league.
- **gen4 is preserved exactly**, as the user required: 192,000 turns in the 708 env,
  **0 action mismatches, max logit diff 0**.

## Lessons worth keeping
- **Imitation accuracy does not predict playing strength.** Sponge clones its teacher 6
  points worse than devtest (0.8169 vs 0.8788) and plays at least as well. Rank clones by
  play, never by val_acc. Ladder position does not predict it either -- cheji is #1 and
  was never cloned; 1,402 games are scraped and ready if that is worth testing.
- **`val_acc_novel` is a composition artefact.** Repeats concentrate in openings (96% of
  rounds 0-10), which are the EASIEST rows (0.976 accuracy), and the effect reverses late
  (round 350+: repeats 0.8345 vs novel 0.8836). Do not headline it.
- **4 epochs, not 6**, for clones: three runs where the last epoch gained nothing and
  val_nll turned. Both new clones' best.pt came from epoch 3.
- **Gate maps with a trained policy, not random play.** `wormhole` looked fine at 42 rounds
  under random play and collapsed to 5-round games under trained play. It is dropped, with
  the diagnosis in its docstring: portals that deliver you into the enemy's boxed-in strip.
- `pgrep -f "<pattern>"` matches your own watcher's command line. A wait loop built that way
  deadlocks on itself; it cost ~10 minutes here.

## Known rough edges
- `team_critic selfplay` picks a policy **per step, not per team**, so it is a flickering
  mixture playing itself rather than two policies contesting. Data is valid, the design is
  not what it should be. Fix before the next recording.
- `imitate2.py:218` throws a benign `.item()` UserWarning; one `.detach()` fixes it.
- The dashboard shows the anchor as **"gen0 (ft6 113M)"**, hardcoded in `ratchet.py`'s
  `init_state`. ratchet3's seed is the sponge 2110 clone, so the label is wrong. It lives in
  `state.json`, which the running supervisor rewrites, so it cannot be corrected without
  restarting the run. Make it derive from `--start` before the next new run.
- ~~`finetune_team.py` full-width scalars~~ **fixed** (commit 2f90404): both the trainable
  critic and `--freeze-team` now slice to their own `scalar[0].weight.shape[1]`, and
  `--resume` sizes the rebuild from the checkpoint instead of `bcsim.N_SCALARS`.
- **While a run is live, `ratchet.py`, `ratchet_train.py` and anything they import
  (`team_critic.py`, `augment.py`, `net.py`, `rollout.py`) are OFF LIMITS**: the supervisor
  spawns them as fresh subprocesses each segment and each gate, so an edit lands mid-run.

---
# Handoff — 2026-09-23 ~11:55: memory is in the simulator; PPO ready to restart

## Everything the restart needs is built and gated
| piece | state |
|---|---|
| `cpp/bc_memory.hpp` + env | **done**. `tests/parity_memory.py`: 144,000 turns, 6 maps, max diff **0** vs the Python MemoryTracker |
| `.so` rebuilt | done. `bcsim.N_SCALARS` is now **708** (14 base + 676 mem + 18 memfar) |
| League migrated | `runs/anchors708/`: gen0-gen4, v9, v10, sss_r3, sabotage, shink_r1, vibing_r4, all "max logit diff 0" |
| **gen4 preserved** | 192,000 turns live in the 708 env: **0 action mismatches, max logit diff 0** |
| 60/40 weighting | `--live-maps/--live-share`, verified 0.5833 -> 0.6000 exactly |
| Ten new maps | `maps-gen/`, gated; `train.mapgen` rebuilds them |
| Throughput cost | rollout 235,190 -> 188,945 turns/s = **-20%** (~16% on a PPO iteration). The Python tracker would have been 22.8k, i.e. 8.3x worse |

**Launch line for the next run:**
```
cd bcsim && setsid nohup /usr/bin/python3 -u -m train.ratchet run --run ../runs/ratchet3 \
    --start ../runs/anchors708/gen4.pt \
    --train-maps ../maps --live-maps ../maps-live --live-share 0.6 --gate-maps ../maps-live \
    --segment-turns 150000000 >> ../runs/ratchet3/supervisor.out 2>&1 < /dev/null &
```
`--train-maps ../maps` is the 12; add `maps-gen` by copying both into one directory (build_pool
takes a single dir). Every league path must come from `runs/anchors708/`, not the originals:
a 14-scalar checkpoint will now fail to load, loudly, which is the intended behaviour.

## Round robin, 2026-09-23 (96 games a pair, all 12 maps, se 0.029)
```
              devtest     sponge   ppo_gen4        v12       mean
devtest             -      0.500      0.490      0.661      0.550
sponge          0.500          -      0.573      0.630      0.568
ppo_gen4        0.510      0.427          -      0.552      0.497
v12             0.339      0.370      0.448          -      0.385
```
**Only one conclusion is significant**: v12, the submission live on the ladder, is clearly the
weakest of the four (beaten 0.661 and 0.630, 3.2 and 2.5 se). sponge / devtest / ppo_gen4 are
within ~2 se of each other -- a single pair over 96 games has se 0.051, so sponge's 0.573 over
gen4 is only 1.4 se and is NOT a win. **The user's "restart from scratch if a distillation is
significantly better" condition is therefore NOT met: seed from gen4.**

By map bucket (live 7 / retired / never played on the server):
```
sponge     0.619   0.552   0.458      <- best on live, collapses on unseen (~2.3 se)
devtest    0.536   0.531   0.597      <- improves on unseen
ppo_gen4   0.500   0.500   0.486      <- flat: the most map-agnostic of the four
v12        0.345   0.417   0.458
```
The user's "clones suck on unseen maps" holds for the strongest clone, not as a law. These are
relative scores in a round robin, so the defensible claim is that PPO is the most map-agnostic --
which is the argument for the wider training pool.

## Clones built (recipe: 64x4, hidden 512, lr 1e-3, chunk-games 270, features mem,memfar)
- `runs/i2/devtest_2050/best.pt` val_acc **0.8788** (PR's mm was 0.8520). Cache 2050:1 + 1302:0.5.
- `runs/i2/sponge_2110/best.pt` val_acc 0.8169.
- **Use 4 epochs, not 5 or 6.** Three runs now (mm's 6th, devtest's 5th, sponge's 5th) where the
  last epoch gained nothing and val_nll turned. Both clones' best.pt came from epoch 3.
- **Imitation accuracy does not predict strength**: sponge is 6 points worse at copying and plays
  at least as well. Rank clones by play, never by val_acc.
- `val_acc_novel` is a composition artefact, not generalisation: repeats concentrate in openings
  (96% of rounds 0-10) which are the EASIEST rows, and the effect reverses late (round 350+:
  repeats 0.8345 vs novel 0.8836). Don't headline it.

## Not done / open
- **v12 is live and is the weakest model we have.** Replacing it is worth considering separately
  from the restart. Nothing clears the >0.60-against-everything bar, so it is a judgement call.
- cheji bt: 1,402 games scraped and ready; never cloned (the user cut it for time). Ladder
  position does not predict clone strength, so it is worth measuring rather than assuming.
- `wormhole` is the weakest of the new maps (42-round games); watch it in training.
- `imitate2.py:218` throws a benign `.item()` UserWarning; one `.detach()` fixes it.
- The dashboard shows the anchor as **"gen0 (ft6 113M)"**, hardcoded in `ratchet.py`'s
  `init_state`. ratchet3's seed is the sponge 2110 clone, so the label is wrong. It lives in
  `state.json`, which the running supervisor rewrites, so it cannot be corrected without
  restarting the run. Make it derive from `--start` before the next new run.

---
# Handoff — 2026-09-23 ~10:05: ratchet PAUSED, cloning the top 3, round robin queued

## The ratchet is paused, not finished
Stopped at **gen 5, 68.7M candidate turns** (experiment total 1.035B). `STOP` was placed and the
process group killed rather than waiting ~2h for the clean decision point; `latest.pt` is written
every 10 iterations (~76s) so almost nothing was lost.
**To resume: `rm runs/ratchet/STOP`, then the launch line in `runs/ratchet/CHANGES.md`.**
It restarts gen 5 segment 0 from `runs/ratchet/cands/g005_s7/latest.pt`.
Record: gens 1-4 all promoted. gen4 vs the league: v10 0.625, v9 0.813, sss_r3 0.760,
sabotage 0.578, shink_r1 0.646, vibing_r4 0.802. **gen4 is the best PPO policy.**

## `make -C bcsim` is DONE (the merge's pending step)
Verified: privileged build still exposes the 8 globals, and the new `probe()`/`last_deaths()`
work. The C++ diff removed zero lines, so the simulator core is untouched and the ratchet can
resume on the new `.so`.

## The harness question, answered
The first-turn bug **never existed in bcsim** -- there is no first-turn fallback there, every
simulated turn runs the network, which is exactly why the round robin was blind to it. So a bcsim
head-to-head was always bug-free; what the fix buys is that local numbers now transfer to the
server. The real constraint is that v12 is a **708-scalar** clone (mem+memfar) and gen4 is a plain
14-scalar net: they can only meet through `clone_eval.py`, which attaches a `MemoryTracker` per
(env, dragon) to the checkpoints that declare those features. Smoke-tested, works.

## Running (all unattended)
- `runs/assess/` -- gen4 vs v12 and v10, 98 games/opponent on `maps-live`, then 96 on all 12.
  Note it is CPU-bound in the Python feature builder (GPU ~5%), so it is slow.
- `runs/cheji_fetch.out` -- cheji bt scrape, 1381 replays.
- `runs/distill3/run.sh` -> `runs/distill3/chain.out` -- waits for the GPU, then devtest clone,
  sponge clone, cheji clone (after its scrape + dataset), then the round robin.
  Recipe is the PR's winner (`full_mem_memfar`): 64x4, hidden 512, 6 epochs, lr 1e-3,
  `--chunk-games 270`, features mem,memfar.

## Decisions the user made (2026-09-23)
- dev test 1 :P: clone **2050 at 1.0 + 1302 at 0.5** (2050 is current with only 595 games;
  1302 has 1361). Needed a new `clone_cache --submission "2050:1,1302:0.5"` (commit e9a37a9).
- Sponge: submission **2110**. cheji bt: submission **2490** (team id 70, was barely scraped).
- Round robin: **5 entrants** -- the 3 new clones, ppo_gen4, v12 -- on **all 12 maps**, because
  `evaluate()` dumps per-map cells, so the live-7 and the unseen-map scores come from one run.

## Unseen maps: which are genuinely unseen
`help`, `small` and `queen_of_spades_but_she_ages` were **never played on the server**, so no
replay-trained clone has ever seen them. `arena` and `Colloseum` were in the rotation until
2026-09-22 12:50, so older clones did see those. That is the clean 3-map novel bucket.

## Map design (NOT to be built yet -- the user asked for a proposal first)
Feature table over all 12 maps shows what the official set never exercises:
- **UNIT_LIMIT is never set on any map at all.** Completely untouched axis.
- **Dragons per team**: live 7 has only 2, 3, 4. No 1 (Colloseum) and no 5-6 (help has 6).
- **Portals**: 0, 0, 0, 2, 4, 24, 24 -- nothing in between, and no map built around portals.
- **Pearl coverage**: 14% (schooltime), 34% (devil), 51% (queen), then four at 100%. Nothing
  below 14%, nothing between 51 and 100.
- **Symmetry**: every live map is x/y/xy symmetric. The engine allows asymmetric maps (arena,
  help and small have no SYMMETRY line). Symmetric maps may let a policy lean on mirror priors --
  note the PR found dev test has a world-frame planner and that left/right is where clones err.
- **Aspect**: only devil (2:1) and small (2:1) are non-square. Nothing like 64x8.
- **Topology**: no ring, no single-chokepoint, no maze.

---
# Handoff — 2026-09-23 ~10:20: PR #1 merged (first-turn fix + remembered-map inputs)

`origin/distill-devtest-features` is merged into local `main` (merge commit, nothing pushed).
Full review notes are in the merge commit message. The headline: **`mybot`'s `fallback()` played
every dragon's first turn and stepped straight on blind. Entering a head's cell is legal and kills
both dragons, so six of eight dragons died before round 2 in every game, in v10 and everything
before it.** Confirmed independently against the API: battle 58343 (v11, sub 2498) lost 0-12 on
map 4; battle 58665 (v12, sub 2520, the same opponent submission 1371 on the same map) won 109-0.
Sub 2520 is active on the server.

Checked against the rules rather than taken on trust: `fill_mask` (cpp/bc_obs.hpp) already excludes
own body and another dragon's body as certain death, and deliberately allows a head-on because it is
a mutual kill. A head is the only legal-but-suicidal step, so refusing heads is the complete fix.

## TWO REBUILD STEPS ARE PENDING (both deliberately not done, the ratchet is running)
1. **`make -C bcsim`** — `cpp/bc_vec.hpp` and `cpp/bc_capi_vec.cpp` gained `probe()` and
   `last_deaths()`, and `bcsim/env.py` binds them. The `.so` files were NOT rebuilt, so the live
   ratchet is untouched, but `env.probe()` / `env.last_deaths()` (used by `clone_eval.py` and
   `self_trap.py`) will fail until the rebuild. **Do it once the run is over, not before:** a new
   segment or gate dlopens the `.so`.
2. **Re-export `mybot/weights_data.hpp`** — it is generated and gitignored, and the copy on disk
   predates the merge, so it has no `SCALARS` and `mybot` will not compile until:
   `cd bcsim && python -m train.export_cpp --ckpt <ckpt> --header ../mybot/weights_data.hpp`.
   Verified both ways: with the stale header the build fails on `embedded::SCALARS`; re-exported
   from `runs/submitted/v10.pt` (a plain 14-scalar checkpoint) it compiles with and without
   `-DBC_DUMP`, which is the PR's backward-compatibility claim holding.

## Verified safe for the live run before merging
- `bcsim/env.py` only adds two methods, each binding its ctypes symbols lazily, so the stale `.so`
  still imports (checked with the privileged build the ratchet uses).
- `bcsim/train/yardstick.py` is imported by the gate. The changes are inert for non-stateful
  policies (`getattr(fn, "stateful", False)` is False, `progress=0.0` silences the new print);
  `evaluate()` was run end-to-end on the merged code against the current `.so` and returned a
  well-formed summary.
- `runs/ft3/maps`, `runs/ratchet` and the league's `v9`/`v10` paths are untouched.

## One correction made on top of the PR
`wasmprobe/submit.sh`'s new naming (`basename $(dirname $CKPT)`) fixed `runs/i2/ft_control/best.pt`
-> `ft_control-best` but broke the RL layout: `runs/ratchet/anchors/gen5.pt` would have uploaded as
`anchors-gen5` and `runs/ft6/snapshots/...` as `snapshots-...`. Bucket directory names
(snapshots, anchors, cands, ckpt, checkpoints) now fall through to the run above.

## Research merged, in one line each (DISTILL_DEVTEST.md has the numbers)
- Remembered-map inputs (`mem` 676 + `memfar` 18) are the real win: +2.9 points of held-out
  accuracy over the same pipeline, +4.3 over r3 on games r3 never saw, and r3 comes last in a
  6-way round robin. Best clone `mm` scores 0.607 against the anchor field.
- Super-sprint relabelling at weight 10 **loses** (0.484/0.481 vs 0.532 control): 99% of
  super-sprints are head-on, so it taught head-trading -- 14x more draws, games 25 rounds shorter.
  Retried with good trades only at weight 3, back to level and it still plays them.
- Self-traps counted (8,949, 0.083% of turns), not used; 64% of candidates were sacrifices.
- Toss-up downweighting: neutral (0.524 vs 0.532).
- Two accuracy traps found: 33% of held-out rows are exact repeats of training rows (deterministic
  teams replay whole games), and an older split inflated v10's reported accuracy 82.5% -> 85.0%.
- Nothing here clears the >0.60-against-everything bar, so no upload is recommended from it.

---
# Handoff — 2026-09-23 ~09:40: rotation changed, and `maps/` is now the full local pool

## The server swapped maps on 2026-09-22 12:54
Out: **Arena**, **Colloseum**. In: **Devil** (32x16, symmetry y, 3 dragons a side of length 4,
heavy vertical kelp corridors, 128 pearl-rich tiles). The live rotation is 7 maps: big_empty,
default, default_small, devil, queen_of_spades, schooltime, trophy. Confirmed two ways:
`GET /api/v1/maps`, and the mapIds of the scraped ladder games (1 and 2 stop at 12:50, 13 starts
at 12:54, all seven uniform at ~1/7 since). Re-pull with `python -m train.maps_fetch [--check]`.

## Map directories now (the user, 2026-09-23: "locally we want all maps to be active")
| dir | what | use |
|---|---|---|
| `maps/` | all 12: the live 7 + arena, Colloseum, help, small, queen_of_spades_but_she_ages | TRAIN on these |
| `maps-live/` | the server's current 7 | GRADE on these |
| `maps-official/` | official maps as downloaded, current and retired | archive |
| `runs/ft3/maps/` | the old 8; what the live ratchet is using | leave alone |

**Next run:** `--maps ../maps` to train on all 12. Keep gates on `../maps-live`, so gate scores
still mean "ladder strength". Old gate numbers were on the old 8, so gen-to-gen comparisons
across the switch are not exact.

**Weighting to decide:** `build_pool` weights every base map equally, so all 12 active puts
only 58% of training on the live 7 and 42% on maps we are never graded on. If that is too much,
the fix is a per-map weight multiplier in `augment.build_pool` (it has no such knob today) --
not written yet, because editing `augment.py` would land in the LIVE ratchet at its next segment.

## Verified (2026-09-23, all 12 maps)
`augment.build_pool('../maps', 48)` -> 588 entries, 12 base maps, every variant parses; random
self-play plays clean on all 12. Per map, random play: Colloseum 22 rounds mean, arena 36,
small 35, default_small 96, queen_of_spades 111, she_ages 113, trophy 125, schooltime 133,
default 146, devil 212, big_empty 474, **help finishes no game inside 500 rounds** (64x64,
6 dragons a side, 4096 pearl tiles, ~77 agent-steps per round). help and big_empty are not
broken, just long -- but with the ratchet's result-only reward they yield few terminal results
per turn, which is another reason to consider down-weighting them.

## The live ratchet was NOT touched
Still on `runs/ft3/maps` (the old 8), gen 4, 966M turns, ~34.5k turns/s, 10h40m in.
Swapping its maps or its code mid-run would make gate scores incomparable across generations --
and note the supervisor re-spawns `train.ratchet_train` (so it re-imports `augment.py`) at every
segment, so an edit to those files DOES reach a running experiment.

## Two footnotes
- A **private map** (id 12) appears in ~3% of the even-hour ladder rounds. Its replays carry a
  placeholder ("INTERNAL_PRIVATE_TESTING_MAP", 64x64), so it cannot be mirrored: we play it blind.
- Not drift: the server's big_empty and default now omit 128/64 redundant `EDGE ... 0 -1`
  (open, no portal) lines. Same maps; `maps/` and `maps-official/` refreshed to the server text.
- `train.py`'s `small_map_frac` metric lists small maps by name and does not include devil.
  Left as is on purpose: devil averages 212 rounds, nothing like arena (36) or Colloseum (22).

---
# Handoff — 2026-09-22 ~21:30 (latest): THE RATCHET is running

- **What:** `bcsim/train/ratchet.py` (supervisor + gate) and `bcsim/train/ratchet_train.py` (segments).
  The design and its reasons are in `runs/ratchet/CHANGES.md`. In short: 50M-turn candidates from
  a gated anchor (start = ft6 112.7M); team-result-only reward; frozen pretrained critic; KL 0.5 to the anchor;
  gate = 384 games vs the anchor + 96 per league member; promote at >= 0.55 with no league drop over 0.10.
- **The user's brief (2026-09-22 21:00):** run ~8 hours with only Claude watching; rock-solid; slow
  improvement is fine; keep the dashboard current (port 8770, same token, now `--run runs/ratchet`).
- **Stop:** `touch runs/ratchet/STOP` (clean), or `kill -- -$(cat runs/ratchet/supervisor.pid)`.
  Restart with the launch line in CHANGES.md; it resumes from state.json.
- **Watch:** `bash runs/ratchet/watch.sh` prints only events (gates, aborts, stalls, dead supervisor, critic calibration < 0.3).
- **Submission:** the user asked (fork) to submit ft6 113M "if we are confident it is the best". Not done: 0.52 vs v10
  over 96 games and 0.45 vs Sabotage. The ratchet baseline gate (`runs/ratchet/gates/gen0_baseline.json`) settles it.
  Promoted anchors are `runs/ratchet/anchors/genN.pt` (export_cpp-compatible). Submit only via wasmprobe/submit.sh
  at > 0.60 vs every opponent over 96 games.
- **Rules (the user):** draws only on a mutual wipe-out; at round 500 an equal longest dragon goes to the greater total length.
  bcsim's `settle()` (cpp/bc_core.hpp) already matches this.

---
# Handoff — 2026-09-22 ~17:20 (latest)

## Running now
- **ft6** (`train/finetune_team.py`): ft5's policy at 98.6M + pretrained team critic, team head
  anchored to REAL results (MC buffer, `--mc-coef 1 --td-coef 0.25`). 10M critic-only warmup
  (to 108.6M), self-EV gate 0.12 (cap 128.6M). alpha 0.75, ent 0.001, KL 0.5, LR 3e-5.
  Why: see `runs/ft6/CHANGES.md` (ft5's TD-only team head drifted: EV vs real results 0.6 -> 0.13).
- Yardstick on ft6 (`--gpu-frac 0.2 --max-seconds 2700`); dashboard port 8770 -> runs/ft6;
  close watch `runs/ft6/watch.py` (critic health, policy summaries, alerts).
- Replay watcher: all versions for Vibing++ only; newest submission for dev test 1 :P (#1),
  SSS, SHINK AI, vom, team, Sabotage-d, Matcha Latte, Sponge.

## Key lessons today
- Watch the team head's EV against REAL results (calib_ev), never the TD EV (~0.99, self-referential).
- ft5 greedy yardsticks (32 games, 7 anchors): mean 0.553 @50M (frozen) -> 0.567 @62.5M -> 0.593 @75M.
- dev test 1 :P clones: r2 (265 games of 1302) 83.5% acc, 0.56 vs v9 (96 games) — not submitted.
  1302 now has 409+ games. Clones in runs/clone_compare4/5.
- The user prefers ONE job at a time on the GPU: side jobs make everything slow.

---
# Handoff — 2026-09-22 midday (update)

## Running right now (13:45)
- **ft4** (`train/finetune_team.py`, team-level critic, see `runs/ft4/CHANGES.md`): start/teacher SSS r3
  clone, critic from `runs/team_critic/pretrained.pt`, opponents self 50% + SSS r3, Sabotage-d,
  SHINK AI r1, Vibing++ r4. Log `runs/ft4.out`, `runs/ft4/log.jsonl`. ~19k turns/s, 25M warmup.
- **Yardstick** on ft4 snapshots (`runs/ft4_yardstick.out`, `--gpu-frac 0.2`).
- **Dashboard** on port 8770 (same token), now `--run runs/ft4`, with team-critic charts
  (team/self EV, calibration vs real results and by round, A_team/A_self correlation).
- Replay watcher unchanged (top 3 + SSS + Vibing++ all). "team" (#1) has too few games to clone yet.
- GPU jobs: use `/usr/bin/python3` (miniconda python3 has a torch the driver can't run).

## Clone round-robin 2026-09-22 (`runs/clone_compare3/`, 96 games per pair)
SSS r3 (712x1, 1079x0.5, 568x0.25; 80.3% acc): 0.53 vs v9, beats SHINK r1 0.58, Vibing r5 0.72.
SHINK AI r1 (89.4%): 0.51 vs v9. Vibing++ r5: weak (0.40 vs r4). Old Sabotage-d clone beats all
new ones (~0.6) but that team left the top 10. Nothing submitted (no clear gain over v9). The user
chose SSS r3 as ft4's base. New anchors: `sss_r3_bc_64x4.pt`, `shink_r1_bc_64x4.pt`.

---
(earlier handoff below)

# Handoff — 2026-09-22 morning

Read this first after a `/clear`. It records where things stand and the next task.

## Running right now
- **Replay watcher**: `python3 -m train.replay_watch --teams "Vibing++:all,SSS:latest" --top 3 --every 30`
  (a setsid process started from `bcsim/`). Every 30 minutes it fetches and converts new games
  for Vibing++ (every version), SSS (newest submission only) and any top-3 team (newest only),
  into `runs/replays/<team>/`. Log: `runs/replay_watch.log`.
- **Dashboard**: `train.dash` on port 8770 (token as before), still pointed at `runs/ft3`.
- **Nothing is training.** The GPU is free.

## Live submission
- **v9** = the SSS r2 behaviour clone (`runs/submitted/v9.pt`), submitted 2026-09-22 04:40 and
  still active. It beats v8 (the Vibing++ clone) ~0.72 locally. Server, unranked, 8 maps each:
  8-0 vs Sabotage-d (their v890), 1-7 vs SSS (712), 1-7 vs SHINK AI (841), 1-7 vs Vibing++ (847).
  Results are in `runs/server_tests/v9_vs_top4.json`.

## Clones (behaviour cloning, one team each, never mixed)
| clone | file | games | held-out acc |
|---|---|---|---|
| Sabotage-d (546) | `runs/anchors/sabotage_bc_64x4.pt` | 232 | 79.4% |
| SSS r2 (712 x1, 568 x0.25) | `runs/anchors/sss_r2_bc_64x4.pt` | 549 | 80.5% |
| Vibing++ r4 (847 x1, 615+ x0.5, 242 x0.25, 98 x0.05) | `runs/anchors/vibing_r4_bc_64x4.pt` | 1445 | 90.5% (3 held-out games) |
| older: Vibing++ r2 = v8, SSS r1 | `vibing_bc_64x4.pt`, `sss_bc_64x4.pt` | | |

Tournament (64 games per pair, `runs/clone_compare2/`): Sabotage-d > SSS r2 (0.56-0.62) >> Vibing++ r2
(SSS r2 wins 0.72-0.77). Train a clone with: `train/imitate.py --submission-weights "id:w,..."`
(it saves `best.pt` by held-out NLL).

Data on disk (all replays re-simulate exactly): Vibing++ 1510 games/12.6M samples, SSS 805/7.8M,
SHINK AI 319/2.7M, Sabotage-d 294/2.4M, "team" (new #3) a few.

## Fine-tune runs (`train/finetune.py`), all stopped and resumable
| run | setup | outcome |
|---|---|---|
| ft1 | Vibing r2 clone, KL 0.1, LR 1e-4 | got worse once the policy trained (scores vs the clone 0.355→0.269) |
| ft2 | rolled back to 50M; KL 0.5, LR 3e-5, 1 policy epoch | stable but flat over ~107M training turns |
| ft3 | SSS r2 clone + `--privileged` critic + ft2 settings | flat: its 137M snapshot scores 0.50 vs its own start (84 games). Stopped at 184M |
Reasons for each: `runs/ft2/CHANGES.md`, `runs/ft3/CHANGES.md`. v6 (from-scratch PPO) is stopped at
577M and can be resumed with `train.train --resume runs/v6/latest.pt`.

**Diagnosis: the critic can't predict per-dragon returns.** Explained variance was ~0.07 (plain) and
~0.12 (privileged).

## Offline critic probe (`train/critic_probe.py`, results in `runs/critic_probe/results.json`)
Predicting the TEAM's final result from replay positions (476 games, both teams, balanced), EV:
8 global features 0.37 | local window 0.26 | local+global 0.37 | full board 0.30 (probably
overfitting on only 476 game results). The global features carry the signal, and the RL problem is the
per-dragon return target: win/lose are paid only to survivors, so dead dragons' returns hide the result.

## NEXT TASK (agreed direction, not started): team-level critic
The user likes team-level advantages (worried they may train slower). Plan:
1. Critic target = the team's final result as win/draw/loss classes, plus a small head for the
   shaped part. Inputs: the 8 global features (`--privileged`, `libbcvec_priv.so`) plus the local view
   plus the opponent slot.
2. Pretrain it on replay positions (re-extract the datasets with the global features; `critic_probe
   extract` is the template).
3. Advantage A = alpha * A_team + (1 - alpha) * A_self, alpha ~0.5 to start, where A_self comes
   from the dense potential terms.
4. Keep ft3's safety net: KL 0.5 to the clone, LR 3e-5, 1 policy epoch. Start from the strongest
   clone (the SSS r2 / Sabotage-d question: check the tournament and the user's preference).
5. Track calibration by round as well as EV (in self-play mirrors EV has a low ceiling).

## Gotchas learned the hard way
- Never kill with `pgrep -f`/pattern loops whose own command line contains the pattern: it
  killed my own shell twice. Match on `/proc/<pid>/cmdline`, and check the process is gone afterwards.
  One process ignored SIGTERM, so follow up with `kill -9` and check.
- Any side GPU job next to a training run must cap its memory
  (`torch.cuda.set_per_process_memory_fraction`), or it can OOM the run (it killed ft2 once).
- Upload only through `wasmprobe/submit.sh` (it runs every check_bot gate and records to
  `runs/submitted/`). It overwrites `mybot/weights_data.hpp`.
- The server removed the Help map; local evals use `runs/ft3/maps` (the 8 live maps).
- Yardstick evals with 32 games per opponent are too noisy (identical nets scored 0.29-0.75).
  Use at least 84 games for a decision.
- All the new code is uncommitted (see `git status`): finetune.py, imitate.py, replay_*.py,
  critic_probe.py, the Critic in net.py, the learn mask in rollout.py, the priv/board buffers in env.py and cpp,
  the BC_MAX_STEPS option, the dash charts, Makefile targets.
