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
