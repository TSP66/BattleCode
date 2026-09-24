# Reward v8 and the pyramid — 2026-09-24

Nothing is training. Nothing has been launched. The full reward spec is
`REWARDS.md`; this is only what state the repo is in.

## The parent-to-child channel now works and is measured (c8f409fb)

The simulator can carry a parent's full 64 bits to the child it just split off.
Four narrowings were in the way, all in the vec layer -- `Action` could cast only
ONE ray along the dragon's facing with a 32-bit payload, the observation's message
buffer was `uint32`, `Action` had no protocol field (so protocol 3 could only be
had through `cfg_.sonar`, which also forces a four-way broadcast), and `MAX_MSGS`
was 4 against a protocol with no cap, which dropped messages constantly.

Run `python3 tests/test_msg_transfer.py` for the numbers. What it establishes:

* payloads arrive byte-exact at full width, 0 mangled;
* **delivery is not reliable.** 57% of children hear their parent at birth, rising
  with the parent's length (56% at length 4, 94% at 14). Broadcasting every turn
  reaches 65% at birth and 89% by the child's sixth turn. **A codec must not
  assume the bits arrive.**
* the engine tells a receiver only the payload, never the sender, so a child that
  hears several sonars identifies its parent from the bits alone -- that is what
  memcodec's `SONAR_TAG` is for;
* inboxes reach 24 deep, so `MAX_MSGS` is 64 and `Observation.num_msgs` carries
  the true uncapped count. Check it, never `len(msgs)`.

`env.last_splits(env_index)` gives (parent, child, k): a child is otherwise
indistinguishable from a dragon that spawned, and an encoder has to be trained
against the specific child its parent seeded.

## The bits are now features: the team's shared map (train/memfeat.py)

`--memchan` on `train.ratchet` / `train.ratchet_train` turns it on. Every dragon
declares protocol 3, broadcasts a 64-bit packet four ways, and what it hears is
unioned into its own **memfar** features. Run `python3 tests/test_memfeat.py`.

**Why the packet is hand-written and `memcodec.py` is not deployed.** memcodec's
measurements are real, but `wide` planes are centred on the sender's HEAD and
rotated to its FACING. A child's head is the parent's old segment `len - k`,
several cells away and usually facing elsewhere, so planes decoded from a
parent's code describe the world around the PARENT. Injecting them into the
child's frame is an offset-plus-rotation error of several cells. An
absolute-coordinate packet has no frame to get wrong -- `SC_HEAD_X` is `hx / w`
and is not mirrored per team, so a cell means the same thing to every dragon --
and the geometry becomes exact arithmetic instead of something a decoder has to
learn. It also costs ~0 judge points instead of 18.1M.

What it carries, and why those fields: 3 expected-pearl cells in absolute
coordinates, plus memfar's 4 directional pearl weights as compass bearings.
memfar ALREADY means "the nearest expected pearls I know about" and "how much
pearl weight lies that way", so the receiver's block keeps its meaning and is
merely computed over the team's knowledge. **No column moves and no checkpoint
is invalidated** -- the pyramid reads memfar as part of its 32 scalars and the
flat 708-scalar nets read the same numbers, so both benefit untrained.

The cones are in there because of a measurement, not a guess. Pearl cells alone
did almost nothing for children: a child hears a packet 77.8% of the time, so
delivery is fine, but the parent is ADJACENT, so its nearest pearls are ones the
child can already see (a child is not blind at birth -- its first look fills its
fresh memory, and 86% already know a pearl). What a child lacks is everything
further out. With the cones, a **child's total cone weight nearly doubles,
1.473 -> 2.779**, and a weight rises on 64.3% of children's first turns.

With a *trained* policy it does considerably more than under random play, as
expected -- a ray is dragged the length of a body before it flies, so longer
dragons reach further: **0.77 pearls merged per turn against 0.19**, heard 63%
against 49%, a cone lifted on 55% of turns against 39%.

Cost 1.6ms a step at 1,024 envs, ~5% of a rollout iteration. `MAX_READ = 4` was
measured, not picked: it gets 99.9% of the cone gain and 94% of the pearl rescues
for 54% of the work, and nothing beyond 8 messages ever changed a number.

The one case it cannot help is a dragon that knows no pearl anywhere: it heard a
pearl on only 7.8% of such turns, and on **100% of those it was given one**. That
is delivery-bound, not payload-bound -- it is alone, and a ray stops at the first
dragon it meets.

A packet states only what its sender has itself observed, never what it was told.
A cone is measured from the SENDER'S position, so relaying compounds that error
until every dragon claims pearls in every direction, and a relayed pearl cell
never expires. That costs about half the raw volume (0.09 pearls merged a turn
against 0.19) and `relay=True` is left available to measure against.

**Bolted onto a policy trained without it, the channel is neutral.** 96 games a
side with `runs/ratchet4/anchors/gen1.pt`:

| opponent | channel off | channel on | delta |
| --- | --- | --- | --- |
| gen4 | 0.5417 | 0.5260 | -0.016 (-0.2 sigma) |
| v12 | 0.6510 | 0.6250 | -0.026 (-0.4 sigma) |

Both within noise, both slightly down -- which is what an unadapted policy
reading a shifted feature should look like, and it is NOT evidence the channel
helps. It has to earn that by being trained with. `runs/ratchet5/launch.sh` is
written and **not launched**; it needs a new `--run` directory, because memfar
moves for every net in the league and the state file refuses to flip the flag
mid-experiment.

Not yet done for deployment: `mybot/` needs the same 60 lines of integer math
(declare protocol 3, cast the packet, merge the inbox into memfar before the
forward pass), and `distill.py` should collect with the channel on if PPO trains
with it. `train.py` and `finetune.py` have no `--memchan`; the ratchet is the
live PPO path.

`tests/stress.py` is green for the first time (336,105 turns, 0 mismatches):
KNOWN_ISSUES #5, a split child's facing, was a split-path bug and is fixed.

## The channel A/B: +5.5 points that are NOT the shared map (2026-09-24 ~22:40)

Commit 86416711 asked for one measurement before trusting `--memchan`: play the
teacher against a fixed league with the channel on and off, because a four-way
broadcast from every dragon takes SC_NUM_MSGS (scalar 12, inside the 14 that
every architecture reads) from 11% non-zero to 78%, and a clone's calibration on
that input does not survive the shift.

Run twice, once per clone of Sabotage-d 3952, against `gen4, v12, sab546`, 12
games a map over the 8 `runs/ft3/maps`, gate seed 12345 in both arms. Raw cells
and the two scripts are in `runs/ab_memchan/` (gitignored like every `runs/`):
`{sonar,mixclone}_{off,on}.json`.

| candidate | channel off | channel on | d |
|---|---|---|---|
| `runs/imitate_sab3952_only` (3952 only, 6.1M samples) | 0.4722 | 0.5209 | +0.0487 |
| `runs/imitate_sab_3952` (3952 x1, older x0.25, 10.9M) | 0.4740 | 0.5347 | +0.0608 |
| **pooled, 576 games/arm** | **0.4731** | **0.5278** | **+0.0547 +-0.0295 (z 1.86)** |

**The base-rate worry is refuted: the channel does not damage the teacher.** It
does the opposite. But read the next paragraph before believing the +5.5.

**Neither candidate can see the shared map.** `train.imitate` clones on the
cache's 14 scalars, and `ActorCritic.forward` slices `scalar[..., :n_scalars]`,
so both clones read base scalars only -- memfar (690..707) does not reach them.
Worse, two of the three league members are blind too: `gen4` and `sab546` were
widened by `migrate_scalars` and never PPO-trained since, so their memfar columns
are still **exactly zero** (checked; `v12` and `ratchet4/gen1` are non-zero). In 2
of 3 matchups *neither side* can read a merged cone, and the only thing the flag
changes for either of them is the SC_NUM_MSGS base rate.

So the +5.5 is a base-rate artefact, not shared knowledge. Per opponent, averaged
over the two clones: gen4 +0.010, v12 +0.073, sab546 +0.081 (+-0.051 each, so the
spread is noise). The earlier A/B on `ratchet4/gen1`, the one candidate here that
*does* read memfar, went the other way: 0.5964 -> 0.5755, d = -0.0209 +-0.051 over
192 games/arm (`runs/ab_memchan/ab_{off,on}.json`).

**What this means for ratchet5.** Keep `--memchan` -- nothing here argues against
it, and 86416711 is right that only PPO can teach the map. But the gate hands the
candidate roughly +5 points for free while the league cannot read memfar, so:

* do not read an early vs-league gain as the channel working. Watch `vs_anchor`
  inside the run, where both sides carry the flag;
* the bias shrinks as promoted anchors (memchan-trained) replace the frozen
  league, and it is worst at gen1;
* `runs/ft3/maps` gate scores from before and after the flag are not comparable
  in either direction. This quantifies the warning already in `ratchet5/launch.sh`.

Power, for the next time this comes up: 288 games/arm resolves d >= 0.08 at 80%
power, 576 resolves 0.06. Anything smaller than that needs ~1,600 games/arm.

## ratchet5 cannot launch yet: its `--start` does not exist

`runs/ratchet5/launch.sh` points at `runs/pyr_sab3952/latest.pt`. **There is no
such file.** The distill that would have written it is
`runs/pyr_sab3952_memchan_aborted/` -- pyramid 48x3, 0.959M params, teacher iter
42842 (the mixed clone), `--memchan` on, stopped at iter 1250 / 41M turns with
top-1 agreement 0.921 and still improving. It was aborted by the `--memchan`
refusal that commit 86416711 then reverted as wrong, so **the checkpoint is
sound**; either point `--start` at it or re-run the distill longer.

Which clone to distil from does not matter: they are a dead heat in play (league
mean 0.4722 vs 0.4740 with the channel off) even though the mixed one is ahead on
held-out accuracy, 0.8035 vs 0.7978 -- and that comparison is rigged, since the
held-out games come from the same mixture the mixed clone trained on.

## ratchet_train was broken, and is fixed

It died on its first step with a shape mismatch: reward v8 appended five Phi
components to the privileged features (`PRIV_COUNT` 8 -> 13) while the frozen
team critic is pretrained on the base 8. Nothing to do with the channel -- the
live PPO path simply could not start. `priv_buf` now takes `obs.priv[:, :N_PRIV]`,
the same slicing its scalar row already did.

## Adopted: the pyramid

On a 1,080-game round robin (`runs/roundrobin_arch.json`) the pyramid tops the
Bradley-Terry fit but is **exactly even head-to-head with the two strongest flat
nets (0.500 against both)**, and its lead over them is 0.8σ and 0.7σ. The only
architecture difference the data supports is **pyramid > ConvLSTM** (2.5σ). So the
case for adopting it is **cost, not strength**: 0.96M parameters against 1.58M and
51.2M judge points against 69.1M, leaving 29.8M spare. Read the older section
below before quoting the ranking at anyone.

**It cannot be submitted yet.** `export_cpp.py` has no arch dispatch at all.

## Reward v8 is built, tested and wired — not run

`cpp/bc_reward8.hpp` is the potential as a pure function of two team shapes, free
of Game and Env. `bc_vec.hpp` supplies the shapes and banks the fully weighted
`kappa*lambda_i*Phi_i/sum(lambda)` per component. Off by default and verified so:
worst |diff| on every v1-v7 component between v8 on and off is **0.000e+00**.

    cd bcsim && python -m train.train --reward v8 --v8-kappa 1.0

The critic is `V(s) = -Phi(s) + f_theta(s)`, with Phi supplied by the ENGINE
through the privileged row (`PRIV_COUNT` 8 -> 13) rather than reimplemented in
torch. `f_theta` is a tanh head, zero-initialised, so at init the critic's
prediction *is* the analytic -Phi.

**Four bugs were found while building it, all in the spec rather than the code.**
They are written up in REWARDS.md with numbers; the short version:

* `V = +Phi` was the wrong sign — shaping is a loan, so a high-Phi state has
  *less* return left. `+Phi` would have doubled the critic's starting error.
* a finisher over unit count pays **+0.019 for a free split of a non-leader**.
  Now over total length, which a split conserves: exactly 0.00000.
* the v1-v7 `primed` flag meant **a dragon's first action got no reward at all**.
* the telescoping invariant is per *agent*, not per episode, and asserting the
  wrong one was actively hiding the bug above.

## The one number to know before launching

**Only 3.3% of the variance in the reward a dragon receives is explained by what
that dragon personally did** (253,906 transitions, random play, median 11 turns
per dragon). The rest is teammates and enemies moving in between. That is the
price of a pure team reward.

`set_reward_v8(..., credit=1)` makes attribution 100% and lowers variance, but it
is **biased** — Phi is per-team-signed, so enemy-turn changes are paid to nobody
(team total -364.8 vs -571.3). It is not the default. **The fix belongs in the
critic as a counterfactual baseline, and that is the main piece of open work.**

Two settings are unmeasured guesses: `kappa = 1.0` (now the only strength knob, so
sweep it first) and `eps = 0.005`. And expect early learning to look slower than
v6's — that is the team reward working as designed.

## Tests

All green: `test_ego`, `test_vecenv`, `test_seam`, `test_rewards`, `test_reward8`
(400k random shapes), `test_reward8_env`, `parity_memory` (144,000 turns, max
|diff| 0), `parity_wide`.

`test_rewards.py` was **already failing at HEAD** before any of this, for an
unrelated reason: its premise 1 is untestable under masked play, because a step is
one tile so the 7x7 window always contains the target and the mask always forbids
moving onto a body (396 masked games: 372 kills, all 372 head-ons). It now also
runs unmasked and reaches 296 non-head-on kills, all 296 paid.

## SONAR IS EXACT — 2026-09-24

Sonar now matches the reference engine **byte for byte**. `tests/parity_sonar.py`
drives engine and simulator in lockstep across three protocol regimes on all ten
official maps: **0 blocks differing out of 423,669 turns carrying 1,159,721
messages.**

**What broke the deadlock: the engine ships a Cap'n Proto replay that records
every single ray** — sender, direction, payload, origin tile, **end tile**, the
dragon hit and its echo classification. `tests/replay.py` reads it (schema taken
from the accessors in the `replay-viewer.vsix` unswbc ships). Months of hypotheses
had been inferred from inside a bot, where `ECHOES` is a direction-less aggregate
and messages carry no sender, so a targeting bug and a classification bug are
indistinguishable. **Read the debug artefact a black box ships before theorising
about the black box.**

The rule we had wrong: **a ray whose first step enters the segment immediately
behind the head is dragged the length of the body and leaves from the TAIL, along
the last body link** — so a curled dragon can cast west and have the ray leave
south. Entering a deeper own segment is an ordinary self-hit. `tests/sonar_truth.py`
predicts every ray independently of our simulator: **176,704 rays, 100.00%, all
ten maps.** Three protocol rules also had to be fixed: the protocol is per dragon
(not per team), a split child inherits its parent's (so it must be applied before
the action), and a payload wider than 32 bits is dropped for a legacy receiver.
The directed `SONAR <dir> <u64>` form is accepted regardless of protocol — gating
it, and gating the ray geometry, were both bugs the mixed regime caught.

Full detail and the retracted claims are in SONAR.md.

What is and is not done against the spec in NIGHT_OBJECTIVES.md:

| asked for | state |
|---|---|
| 64-bit messages | done, exact |
| a different message per cardinal direction each turn | done, exact |
| hearing echoes of what sonars hit | done, **exact on all ten maps** |
| broadcast in every direction every turn | done, metered at 0.51M points/turn |
| **56-bit parent→child memory code after a split** ("the most important thing") | `train/memcodec.py` trains standalone; **still wired to nothing** — but the message path it needs is now verified, so it is unblocked |
| phase 2: inline-head messaging, one-way-street sharing | not started (you marked it low priority) |

**Both ends are now wired, and the deployment path is verified end to end:**

    cd bcsim && python -m train.train --sonar --reward v8      # trains on echoes
    python -m train.export_cpp --ckpt <ck> --header ../mybot/weights_data.hpp
    wasmprobe/check_bot.sh <ck>                                # ALL CHECKS PASSED

* `train.py --sonar` broadcasts in all four directions every turn and feeds the
  five echo counts to the policy (scalars 708-712). The flag is recorded in the
  checkpoint's `args`, and `eval.py`/`yardstick.py` default to `--sonar auto`,
  taking it from the checkpoint so a sonar policy is never evaluated on zeros.
* `mybot` speaks protocol 3 — `PROTOCOL 3` plus four `SONAR <dir> 0` lines joined
  to the single write it already makes (0.51M points, 0.6% of the cap) — **but only
  when the embedded net is 713-scalar.** A 708-scalar net keeps byte-identical
  legacy behaviour, because broadcasting would make `num_msgs` non-zero for a net
  that only ever saw zero there.
* `parity_obs.py` now checks BOTH block formats (with and without the `ECHOES`
  line, which shifts every offset after it): 754 turns, 0 differ.

Before launching, know this: sonar is symmetric, so every dragon in the env
broadcasts and the **frozen league opponents also see a non-zero `num_msgs`** they
were never trained on. A league measured with sonar on is therefore not strictly
comparable with the existing numbers; if the scores jump, suspect that before
believing it is skill.

## Still open

* the counterfactual baseline (above) — the only piece that might need real work
* pyramid deployment: `export_cpp.py` arch dispatch, a second conv trunk in
  `mybot/net.hpp`, wide planes from `mybot/memory.hpp`, new wasmprobe parity
* a pre-existing split-child **facing** mismatch on 2 generated maps
  (`tests/stress.py`); not sonar, present before this work — see KNOWN_ISSUES.md #5
* real truncated BPTT for the ConvLSTM; the current result is a one-step floor
* ratchet4's gate numbers were produced through the broken CUDA-graph harness and
  need remeasuring

---

# Handoff — 2026-09-24 morning: every earlier league number is suspect

## READ THIS FIRST: the evaluation harness was lying

`yardstick.greedy` captured a CUDA graph per network. Capturing a *second* graph
corrupts the *first*: it replays against memory it no longer owns. Same pair, 72
games on maps-live:

| | graphs on | graphs off |
|---|---|---|
| gen1 vs gen4 | **0.0000** | **0.5972 ± 0.058** |

In the broken runs the learner acts on 46k rows against its opponent's 513k,
because its swarm never grows -- it plays badly from the first turn. A mirror
match (one net, one callable on both sides, so only one graph is ever captured)
scores exactly 0.5000 per map; the corruption needs a second capture.

It surfaced because a distilled net "scored" a 0.941 league mean, beating six
opponents 1.0000. The control that settled it: **gen1 scored 1.0000 against
itself.**

Nothing cheap fixed it -- not a shared memory pool, not thread_local capture, not
a static output tensor, not warming up on the capture stream, not disabling the
autocast weight cache -- so the graphs are gone (commit 618b9da6). Eager is 154s
for 72 games on nine maps.

**Consequence: every gate and league number in the rest of this file, and every
`gates/*.json` in runs/ratchet4, came through that path and cannot be trusted.**
That includes the seg0/seg1 league table below and the promotion of gen1. Re-measure
before relying on any of it.

## Architecture: FINAL matrix, 1,080 games (runs/roundrobin_arch.json)

```
score matrix (row against column, both halves pooled, 72 games a pair)
              pyramid devtest    gen1     v12 convlst    gen4
pyramid           -     0.500   0.500   0.542   0.646   0.583
devtest_2050    0.500     -     0.431   0.528   0.583   0.597
gen1            0.500   0.569     -     0.583   0.403   0.569
v12             0.458   0.472   0.417     -     0.514   0.583
convlstm        0.354   0.417   0.597   0.486     -     0.514
gen4            0.417   0.403   0.431   0.417   0.486     -

Bradley-Terry            mean over 360 games
  pyramid       +0.182   0.5542 +/- 0.0262   [pyramid]
  devtest_2050  +0.093   0.5278 +/- 0.0263   [flat]
  gen1          +0.084   0.5250 +/- 0.0263   [flat]
  v12           -0.037   0.4889 +/- 0.0263   [flat]
  convlstm      -0.088   0.4736 +/- 0.0263   [convlstm]
  gen4          -0.233   0.4306 +/- 0.0261   [flat]
```

**Read this carefully: ranking first is not the same as being better.** The
pyramid's lead over the two strongest flat nets is not significant, and head to
head it is exactly even with both:

```
  pyramid - gen1       +0.029   0.8 sigma      head to head 0.500
  pyramid - devtest    +0.026   0.7 sigma      head to head 0.500
  pyramid - convlstm   +0.081   2.2 sigma      head to head 0.646  (2.5 sigma)
```

So the only architecture difference the data supports is **pyramid > ConvLSTM**.
The pyramid's top rank comes from beating the weaker members (convlstm 0.646,
gen4 0.583, v12 0.542) while drawing the stronger ones, not from beating anything
strong.

What that is still worth: the pyramid **matches the best flat nets at 0.96M
parameters against 1.58M and 51.2M judge points against 69.1M**, leaving 29.8M
spare for sonar features, a codec or a bigger trunk. Same strength, 74% of the
cost. That is the case for switching, and it is a cost argument rather than a
strength argument.

Note the non-transitivity: convlstm beats gen1 0.597 but loses to pyramid 0.354
and devtest 0.417, which is why the matrix and the fit are worth more than any
single gate.

## Architecture: planes beat recurrence, and both are level with the flat net

`train/roundrobin.py` (new) plays every saved agent against every other and fits
Bradley-Terry to the whole matrix, because a gate against one anchor cannot tell
"better" from "better against that anchor". Pooled over both halves, 72 games a
pair, maps-live:

```
  pyramid  vs convlstm   0.646 +/- 0.056     planes beat recurrence, 2.6 se
  pyramid  vs gen1       0.500               level with the flat teacher
  convlstm vs gen1       0.597               beats the teacher the pyramid draws
```
Non-transitive, so read the Bradley-Terry fit in runs/roundrobin_arch.json.

Distillation from the same teacher (gen1), matched turns, near-identical params:

```
  pyramid   0.96M params   6.1M turns   KL 0.088   top-1 agree 0.872
  convlstm  0.97M params   6.1M turns   KL 0.113   top-1 agree 0.852
```

**Caveat on the ConvLSTM:** its state is detached between turns, so gradients say
how the state it already has affects this turn's logits, not how a turn's input
should shape the state for later. That is a one-step objective and a floor on
what recurrence can do. Real truncated BPTT needs the graph held across several
of a dragon's turns, which interleave with ~30 others in the same env.

## The judge charges by the loop, not by the MAC

3.3 points per MAC in a convolution, ~15 in a dot-product loop. Pricing per kind
reproduces the meter to 0.14% (deployed 64x4: 69,099,514 predicted vs 69,000,000
measured). `train/budget.py` prices any real module by forward hook.

```
  flat 64x4 (deployed)   69.1M points   69% of cap
  pyramid 48x3/24x2      51.2M points   51% of cap,  29.8M spare
  ConvLSTM 24ch@15       60.9M points   61% of cap,  20.1M spare
  the same state dense    2,916M points  29x OVER the cap
```
**A structured LSTM is 114x cheaper than a flat one of the same capacity.** That
is the only reason recurrence fits at all.

## Sonar under 1.0.0 — see SONAR.md

- **`PROTOCOL 3` gates everything and is printed every turn**, not once. `mybot/`
  prints no PROTOCOL line, so every version submitted so far speaks the legacy
  protocol.
- **Echoes carry no bearing.** The five counts sum to the number of sonars sent,
  so each ray stops on one thing. Broadcasting all four directions gives a
  histogram and throws the direction away; one direction gives a clean reading.
- **The enemy receives our messages** (own 18, ally 255, enemy 234 in one game).
- Our simulator: **byte-identical to the engine on all ten official maps, under
  legacy, protocol 3 and mixed protocols** (fixed 2026-09-24; see the SONAR IS
  EXACT section above and SONAR.md).
- Echo features are wired in behind `BattlecodeVecEnv(sonar=True)`, off by default
  because broadcasting changes what opponents see through SC_NUM_MSGS.

## The 64-bit message: the bits are not the bottleneck

`train/memcodec.py`, 120k planes from real play, variance recovered beyond the mean:

```
  56 bits, decoder sees the child's window   62.2%
  56 bits, blind to it                       55.8%
   8 bits, decoder sees the child's window   50.9%
```
**Eight bits get halfway; the other 48 buy 11 points.** Per channel the code
carries terrain and history (remembered-age 0.93, pearl countdown 0.86, kelp 0.74)
and **not enemies** (0.20 near, 0.09 far). A message should say where the parent
has been, not where it saw an enemy.

## Critic rebuilt, not yet trained

`train/critic_net.py`: board trunk (11 planes of 64x64, 64->32->16->8) plus the
7x7 window, learnable team embeddings for both sides, sinusoidal round and
training-iteration encodings, EMA weights and dropout against the
overfit-then-collapse we have already seen. 4.27M params, 56.5M MAC a row.
Data: 19,247 games across 17 team folders. Cap with `--max-games` (the full set is
~1.08M positions, ~12.5 GB in memory).

## What is NOT done

- Message *timing* does not match the engine, so the codec cannot be deployed as
  measured. The echo path is fine.
- The critic is built and its path verified end to end, but **not pretrained**.
- No PPO run has been started (the user asked for none).
- **Deployment for the pyramid is a substantial job, not a tweak.**
  `export_cpp.py` has no architecture dispatch at all: it walks
  `blocks.{i}.c1.weight` and reads `cfg["width"]` / `cfg["blocks"]` directly, so
  it only knows ActorCritic. Shipping a pyramid needs (a) a tensor order for the
  two branches, the pooled wide head and the wider fuse, (b) a second conv trunk
  in `mybot/net.hpp` over 12x15x15 with the AvgPool, (c) `mybot/memory.hpp` to
  emit the `wide` planes rather than the flat 708 features, and (d) new
  `wasmprobe/parity_net.py` and `parity_mem.py` checks. Until that exists nothing
  on the new architecture can be submitted, whatever the round robin says.
- Adopting `PROTOCOL 3` in mybot changes the block it parses (ECHOES is inserted
  before the tiles), so `mybot/obs.hpp` needs it and `check_bot.sh` must pass
  before any submission. The broadcast itself is cheap: **0.51M points a turn**,
  metered.
- Real BPTT for the ConvLSTM.

---

# Handoff — 2026-09-23 ~18:15: ratchet4 RUNNING, gen1 PROMOTED

## Running now
```
runs/ratchet4/launch.sh   (setsid, supervisor.out / ratchet.log)
  seed        runs/i2/sponge_2110/best.pt  (= v13)
  lr 1e-4   kl-coef 0.4   extend-min 0.45   max-drop 0.15   segment 150M
  train maps  maps-all   23 maps, live NINE held at 60%
  gate maps   maps-live   9 maps
  league      gen4, devtest_2050, v12, v10, sabotage, shink_r1, + gen0 after promotion
  critic      runs/team_critic/pretrained.pt
```
Stop: `touch runs/ratchet4/STOP` (end of segment) or `kill -- -$(cat runs/ratchet4/supervisor.pid)`.
A supervisor restart is LOSSLESS mid-segment: `train_segment` resumes from
`cands/<gen>/latest.pt` using the candidate's own turn counter (cost ~0.3M turns).
That is the only way to change promote/extend thresholds, which are parsed once at launch.

## gen1 history
```
  seg0  150M  vs_anchor 0.539  -> EXTEND
  seg1  300M  vs_anchor 0.569  -> PROMOTE   (anchors/gen1.pt)
```
Earlier runs for comparison: ratchet3 (lr 3e-5, KL 0.5) sat at vs_anchor 0.50 for
128M turns and was abandoned; ratchet2 was stopped for the new maps.

## THE GATE DOES NOT MEASURE THE LADDER
gen1 seg1 beat the anchor better than seg0 and beat the LEAGUE worse:
```
                 seg0     seg1(promoted)
  gen4          0.676     0.574
  devtest_2050  0.556     0.435
  shink_r1      0.574     0.537
  v12           0.537     0.556
  sabotage      0.620     0.639
  v10           0.657     0.657
  league mean   0.603     0.566     (-0.037, ~1.9 se over 648 games)
  vs_anchor     0.539     0.569
```
Promotion is on `vs_anchor` + a per-member `--max-drop`; the league mean is not
checked. **Read the league mean from every gate JSON.** Both checkpoints survive as
`cands/g001_s1/seg0.pt` and `seg1.pt`. Before any submission, round-robin rather than
trusting the gate. Candidate fix: add a "league mean must not fall" condition.

## Next-run queue (NOT applied)
1. **Maze generator in mapgen.py.** stronghold is 0.083 and did not move in 300M
   turns; it is the largest pool of unclaimed ladder points. See [[maze maps]] below.
2. **Critic refit** on a corpus including long maze-type games. `calib_ev` decays to
   ~0.01 by the end of every segment.
3. **Retune glut and famine.** Pearl supply per 1000 tile-rounds: glut 333 vs the
   richest official map (devil) at 38; famine 0.17 vs the sparsest official (default)
   at 1.54. Nothing is randomised - every value is a hardcoded constant - but those two
   are outside anything the organisers ship. Suggest glut ~40, famine ~1.0.
4. **Log the 500-round-game fraction** per iteration, so `calib_ev` decay can be
   attributed instead of guessed.

## calib_ev is not readable mid-segment
It resets with the resolved-game pool at every segment boundary AND every supervisor
restart, then decays the same way each time (0.50 -> ~0.01). Verified three times,
including an accidental controlled repeat at the 15:01 restart. Always quote
`calib_n` next to it. The decay is composition (long near-coin-flip games fill the
pool and a 500-round game deposits into EVERY round bucket), not policy drift -
`klT` is only 0.007, so the policy has barely moved.

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
