# Proposal: creative maps for PPO training

**Status: proposal only. Nothing is built and nothing is in any rotation.**
Written 2026-09-23 after the user observed that distilled clones do badly on unseen maps.

## Why this is worth doing, in one measurement

`runs/assess/gen4_live.jsonl`, PPO gen4 against v12 (the dev-test clone), 98 games, per map:

| map | gen4 vs v12 |
|---|---|
| trophy | 1.000 |
| **devil** | **0.929** |
| default | 0.571 |
| queen_of_spades | 0.429 |
| schooltime | 0.357 |
| default_small | 0.214 |
| big_empty | 0.143 |

Devil entered the rotation on 2026-09-22 12:54, so it is barely present in the replays the clone
learned from, and the clone is helpless there while PPO takes 93%. That is the whole argument: a
replay-trained clone can only know maps its teacher played, and PPO can be trained on anything we
can write down. The 0.143 on `big_empty` is a separate, real PPO weakness (it scores 0.143 there
against v10 as well) and is not a map-supply problem.

## What the official set never exercises

Every map in `maps/`, tabulated:

| map | size | drg/team | kelp | portals | pearl tiles | UNIT_LIMIT |
|---|---|---|---|---|---|---|
| Colloseum | 16x16 | 1 | 4.7% | 8 | 100% | unset |
| arena | 11x11 | 1 | 18.2% | 0 | 67% | unset |
| big_empty | 64x64 | 3 | 0.0% | 0 | 100% | unset |
| default | 32x32 | 4 | 10.5% | 24 | 100% | unset |
| default_small | 16x16 | 2 | 14.1% | 0 | 100% | unset |
| devil | 32x16 | 3 | 23.8% | 0 | 34% | unset |
| help | 64x64 | 6 | 1.5% | 12 | 100% | unset |
| queen_of_spades | 25x35 | 2 | 16.9% | 4 | 51% | unset |
| schooltime | 60x40 | 3 | 19.0% | 24 | 14% | unset |
| small | 16x8 | 3 | 47.7% | 2 | 100% | unset |
| trophy | 25x25 | 2 | 7.0% | 2 | 100% | unset |

The gaps, for the **live 7** specifically (what PPO actually trains on):

1. **`UNIT_LIMIT` is never set on any map in the set.** Completely unexercised rule.
2. **Dragons per team is only 2, 3 or 4.** No 1 (a duel) and no 5-6 (a swarm; only `help`, which
   is not live, has 6).
3. **Portals are 0, 0, 0, 2, 4, 24, 24.** Nothing in between, and no map whose *structure* is
   portals rather than walls.
4. **Pearl coverage is 14%, 34%, 51%, then four at 100%.** Nothing below 14% (starvation) and
   nothing between 51 and 100.
5. **Every live map is mirror-symmetric** (x, y or xy). The engine does not require it --
   `arena`, `help` and `small` carry no `SYMMETRY` line. A policy on an all-symmetric diet can
   lean on mirror priors, and the PR found left/right is exactly where clones err.
6. **Aspect ratio never exceeds 2:1** (devil 32x16, small 16x8).
7. **No topology**: no ring, no single chokepoint, no maze.

## The proposed maps

Ten, each aimed at one gap, none a variant of an existing map (`augment.py` already makes those).

| name | size | sym | drg/team | what it is for |
|---|---|---|---|---|
| `famine` | 40x40 | xy | 3 | ~5% pearl tiles, slow respawn (200-400). Gap 4: starvation |
| `glut` | 24x24 | xy | 2 | 100% pearls, respawn 1-5. The opposite extreme |
| `duel` | 20x20 | xy | **1** | One dragon a side, length 6. Gap 2: no split partner |
| `hive` | 48x48 | xy | **6** | Six short dragons a side. Gap 2: swarm at the unit cap |
| `corridor` | 64x16 | y | 3 | 4:1 aspect, double the current maximum. Gap 6 |
| `wormhole` | 32x32 | xy | 2 | ~60 portals as the main structure. Gap 3 |
| `choke` | 40x20 | y | 3 | One kelp wall, two gaps. Gap 7: forced contention |
| `ring` | 36x36 | xy | 3 | Solid kelp donut; play circulates. Gap 7 |
| `drift` | 30x30 | **none** | 3 | Asymmetric, uneven pearls. Gap 5 |
| `capped` | 32x32 | xy | 4 | `UNIT_LIMIT 8`. Gap 1, the untouched rule |

`drift` being asymmetric is fair in aggregate because every eval plays both sides of every map.

### Hard constraint: both dimensions above 10

The user, 2026-09-23: official maps always exceed 10 in every dimension. The data agrees --
`arena` at 11x11 is the smallest official map, and the **live** seven bottom out at 16 (`devil`
32x16, `default_small` and `colosseum` 16x16). `corridor` was 64x8 in the first draft, which broke
this; it is now 64x16, which keeps 4:1 aspect (double the current maximum) while staying inside
the shapes the server actually serves. Every other proposed map was already inside the rule.

**This also indicts a map we are training on right now.** `maps/small.map` is **16x8** -- a local
invention, not an official map, and the only map in `maps/` that breaks the rule. As things stand
it is 1/12 of the PPO training pool, i.e. ~8% of training spent on a shape the server will never
serve. Recommend dropping it from the pool, or at minimum giving it a low weight once the
weighting knob exists.

## How to build it

1. `bcsim/train/mapgen.py` writes the ten `.map` files into **`maps-gen/`**, leaving `maps/`,
   `maps-live/` and `maps-official/` untouched.
2. Gate each one before use, the same way the new rotation was checked on 2026-09-23:
   - loads via `bcsim.load_maps`;
   - random self-play finishes games (watch for a map where no game ends inside 500 rounds, as
     `help` does -- with result-only reward that map yields almost no learning signal);
   - `augment.build_pool` produces valid variants of it;
   - both teams can actually reach each other (a map that boxes a side in is a bug unless
     intended, as in `help`).
3. Train on `maps/` + `maps-gen/`, **grade on `maps-live/`**. Never grade on invented maps: the
   ladder is played on the official seven.

## The open problem this depends on

`augment.build_pool` weights every base map equally. Adding ten maps to the twelve we have would
put the live 7 at **7/22 = 32%** of training. That is almost certainly too little of the maps we
are actually scored on. The fix is a per-map weight multiplier in `build_pool`, which it does not
have today (flagged on 2026-09-23, deliberately not written while the ratchet was running).

**Build the weighting knob first, then the maps.** The user approved **60/40** on 2026-09-23: the
live 7 take 60% of sampling, everything else shares 40%. Without the knob, adding these maps is
likely to cost ladder strength rather than add robustness.
