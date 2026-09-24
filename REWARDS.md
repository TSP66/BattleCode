# Reward v8 — team potential, hand-set weights

Supersedes v1–v7 (`train/train.py`). Written 2026-09-24 from the proposal kept
below as v8-draft, with four corrections that are load-bearing: one term was
sign-inverted, one rebuilt a failure we have already had, the torus makes the
dispersion term undefined as written, and the time-varying weights have to sit
*inside* the banked potential or the shaping stops telescoping.

## Why we are moving

A binary win/loss reward is too sparse for the critic: it memorises games and
fails to generalise, so the advantages PPO sees are noise. v1–v7 answered that
with six separate per-dragon components, which produced its own pathologies —
v0 shredded itself into ~34 tiny dragons, v3's first launch converged on mutual
kamikaze in 30 iterations. Two rules come out of that history and both are kept
here:

* rewards are written over **team** state, not the dragon's own body, so a
  sacrifice can pay;
* nothing is paid for a raw event (a death, a pearl, a kill) that is already
  priced by a change in team state, or it gets charged twice.

v8 goes further: a **single bounded zero-sum team potential**, plus the true
game result, plus one small non-rivalrous individual term. No per-dragon death,
pearl or kill rewards at all.

## Three design questions, answered

### Potential or plain deltas? Keep the potential. It is not a nil difference.

`γΦ(s′) − Φ(s)` and `Φ(s′) − Φ(s)` differ by `(1 − γ)Φ(s′)` every turn. At
γ = 0.997 that looks like nothing per turn, but it accumulates to almost exactly
`Φ` over the discounted horizon:

```
  phi held at +0.25: plain-delta injects +0.1943 of extra discounted return
  phi held at +0.50: plain-delta injects +0.3887   (the win term is 1.00)
  phi held at +1.00: plain-delta injects +0.7774
```

So the plain-delta form silently adds a **rate reward for occupying a good
position, worth up to ±0.78 — the same order as winning.** Its sign pays the
policy to *hold* a lead rather than convert it, which fights the finisher term
below. Worse, it is no longer shaping: the optimal policy changes, and it
changes by an amount we cannot see on the dashboard.

The discounted form is policy-invariant (Ng, Harada & Russell). That is exactly
the property we want here — dense credit assignment, identical objective. Use
`γΦ(s′) − Φ(s)` with `potential_gamma` set to the PPO `gamma`, as
`bc_vec.hpp:616` already does.

A third option — paying the levels directly each turn, `r_t = λ·tanh(r)` — is the
plain-delta artifact without even the telescoping part. Rejected.

### Potential shaping alone cannot specify the task

`Σ γ^t (γΦ_{t+1} − Φ_t)` telescopes to `γ^T Φ_T − Φ_0`. If Φ is the only reward,
**every policy has the same return and there is nothing to learn.** v8 therefore
keeps a genuine terminal outcome as the one non-telescoping term:

```
non-terminal   R_t = γ·λ(t+1)·Φ(s_{t+1}) − λ(t)·Φ(s_t)
terminal       R_T = W·outcome − λ(T)·Φ(s_T)      i.e. Φ(terminal) := W·outcome
```

with `W = 1` and `outcome ∈ {+1, 0, −1}`. This is not a retreat to binary
rewards — it is what makes the dense part legitimate. Because `λ_win(t)` rises
and `Φ_win → tanh(r)`, the shaping hands over to the real objective smoothly,
which is the phase plan v8-draft described.

### The weights are hand-set, and must be banked *scaled*

The λ are constants of the round number, chosen by hand (below). They are not
learned and not tuned by search.

The trap: computing `λ(t)·[γΦ(s′) − Φ(s)]` does **not** telescope. The residual
is `Σ γΦ_{t+1}(λ(t) − λ(t+1))`, a real injected reward proportional to `−Φ·λ̇`.
With `λ_win` rising that pays the policy **not to be ahead early**; with
`λ_len` falling it pays the policy to hold length for its own sake. Both are
artifacts and neither is visible in any logged component.

**Bank `λ(t)·Φ`, not `Φ`.** Then `Φ̃(s,t) = λ(t)Φ(s)` is a potential over the
extended state `(s,t)` and invariance holds exactly. This matters extra because
`add_team_delta` banks at *each agent's own last turn*, so `t` and `t′` differ
per dragon: the λ must be evaluated at the round stored alongside the bank, not
at the current round.

## v8, exactly

Team 0 is "us", team E the opponent, over **living** dragons only. Sorted
descending length vector per team; `L = L1` the longest, `T` the total, `N` the
count, `A3 = L1 + L2 + L3` (missing dragons contribute 0).

```
r  = 2(L_0 − L_E) / (L_0 + L_E)          ∈ [−2, 2]
z  = 2(T_0 − T_E) / (T_0 + T_E)
a  = 2(A3_0 − A3_E) / (A3_0 + A3_E)
```

```
Φ = λ_win(t)  · tanh(r + λ_z · tanh(z))                 win condition + tie-break
  + λ_len(t)  · tanh(z)                                 dense total-length signal
  + λ_top3(t) · tanh(a)                                 eggs in more than one basket
  + λ_kill(t) · (exp(−N_E/c) − exp(−N_0/c))             finish them off
  + λ_exp(t)  · tanh(ε · (C_0 − C_E))                   explore and fan out
```

`λ_z = 0.4`, `c = 4`, `ε = 0.005`. Every term is antisymmetric under team swap,
so `Φ_A = −Φ_B` exactly and self-play sees a true zero-sum game. Every term is
bounded, so `|Φ| ≤ Σλᵢ` at compile time — the thing v6 did not have, since its
`team_max` was raw segment count.

### `tanh(r + λ_z tanh z)`, not `tanh(r) + λ_z sech²(r) tanh(z)`

The draft's bracket is the first-order Taylor expansion of the closed form,
since `sech²` is `tanh′`. The closed form is gated identically (the gate *is* the
derivative) and cannot be non-monotone. The additive form can:

```
 monotonicity of tanh(r) + L2·sech^2(r)·tanh(z) in r
   L2=0.2 monotone | L2=0.4 monotone | L2=0.5 monotone
   L2=0.7 NON-monotone (r=-2, z=-2) | L2=0.9 NON-monotone
```

`d/dr = sech²r·(1 − 2λ_z·tanh r·tanh z)`, which goes negative once
`2λ_z tanh r tanh z > 1` — a region where *growing your leader lowers your
reward*. The draft's `λ_2 < 1` is not tight enough; it needs `≤ 0.5`. The closed
form removes the question. The gate is well motivated either way: the real
tie-break is longest, then total length, so `z` should only matter when `r ≈ 0`.

### The finisher term, corrected — the draft's was sign-inverted

Draft: `(N_E·tanh(N_0/c) − N_0·tanh(N_E/c)) / (N_0 + N_E)`. Because `tanh` is
concave, `tanh(x)/x` is decreasing, so `N_E·tanh(N_0/c) < N_0·tanh(N_E/c)`
whenever `N_0 > N_E`. **The term is ≤ 0 for every numerical advantage, at every
c**, and it is *zero* at `N_E = 0`, the moment you have won:

```
  c=2   10v10:+0.0000 10v5:-0.3244 10v2:-0.4680 10v1:-0.3292 20v1:-0.3925
  c=5   10v10:+0.0000 10v5:-0.1864 10v2:-0.1560 10v1:-0.0918 20v1:-0.1404
  c=20  10v10:+0.0000 10v5:-0.0092 10v2:-0.0060 10v1:-0.0034 20v1:-0.0113
```

v8 uses a convex function of the enemy count instead, so the last kills are
worth the most, which is the stated intent:

```
  Phi_kill = exp(-N_E/c) - exp(-N_0/c),  c = 4
  10v10:+0.0000 10v5:+0.2044 10v2:+0.5244 10v1:+0.6967 10v0:+0.9179
  20v1: +0.7721  3v1:+0.3064  3v0:+0.5276
```

Antisymmetric, zero at parity, maximal at a wipe-out, and diminishing in `N_0`
so mass-producing dragons cannot farm it. "Ideally kill the biggest ones" needs
no extra term: killing a big enemy already drops `L_E` and `T_E` through `r`
and `z`.

### Dropped: the numerical term and the concentration penalty

The draft's `λ_4·tanh(n)` over `n = 2(N_0−N_E)/(N_0+N_E)` **rebuilds v0's
shredding.** A split of a non-leader conserves `T_0`, leaves `L_0` untouched,
and raises `N_0` by one — free in every other term. Net ΔΦ:

```
 free split of a non-leader (l4=0.3, l6=0.3, l7=0.3)
  N 2v2,   mean len 20:  n +0.1140  killer +0.0402  frag -0.0075  NET +0.1468
  N 4v4,   mean len 15:  n +0.0656  killer +0.0244  frag -0.0050  NET +0.0850
  N 8v8,   mean len 10:  n +0.0351  killer +0.0090  frag -0.0037  NET +0.0404
  N 16v16, mean len  6:  n +0.0182  killer +0.0012  frag -0.0030  NET +0.0163
 with l7 raised to 1.0:  NET +0.1293 / +0.0734 / +0.0318 / +0.0093
```

Positive everywhere. The draft's `λ_7` cannot stop it, and not because it is
mistuned — because it is the wrong measurement. `d/dN tanh(n) ≈ 2/N` while
`d/dN tanh(N/T) ≈ 1/T ≈ 0.01`: `N/T` is a mean-length proxy and barely moves.
Closing a 10× gap would need `λ_7` large enough to dominate the whole reward.

`λ_7` is also described as "a small penalty for being overtly concentrated" but
the formula penalises *fragmentation* (`N/T` large means many short dragons),
and neither `N` nor `T` can tell a healthy spread of lengths from one giant plus
a million tiny — which is the thing the draft wanted to prevent.

**`λ_top3·tanh(a)` replaces both.** The game is decided on `L1` and tie-broken on
`T`, so the free parameter worth pricing is `L2` and `L3`: a second and third
real dragon. It rewards what `λ_4` was reaching for (more useful units) and
protects what `λ_7` was protecting (not one basket), without paying for a free
split — splitting a non-leader leaves `A3` unchanged unless the child enters the
top three, in which case it earned its place.

## Exploring and fanning out, on a torus

The draft's `S` is "avg distance from dragon centre point squared". **Every
official map wraps, and the centroid of points on a torus is not well defined**,
so that quantity has no value to compute.

The correct torus form of the draft's intent is per-axis **circular variance**:
map each head to `θ = 2πx/W`, take `R = |mean(e^{iθ})|`, dispersion `= 1 − R`.
Centroid-free, torus-native, O(N). But it has a fatal degeneracy for our purpose:

```
 circular variance of 8 heads, W=32
   all on one tile        0.0000
   tight convoy           0.0982
   two clumps opposite    1.0000
   uniform                1.0000
```

**Two clumps parked on opposite sides score a perfect 1.0, the same as a uniform
spread.** That is exactly the degenerate policy a static dispersion reward
invites — fan out once on turn 30, stop, collect forever. It also rewards a
*configuration* rather than any activity, so it contributes no gradient after it
saturates.

v8 prices **coverage** instead. `C_0` is the number of distinct walkable tiles
any of our heads has stood on, cumulative over the game; `Φ_exp = tanh(ε(C_0 −
C_E))` with `ε = 0.005`.

Why this is the better answer to "explore and spread out early":

* **Spreading becomes instrumental, not paid.** Two dragons in opposite corners
  doing nothing earn zero. Two dragons sweeping different halves earn double. A
  convoy of N dragons covers tiles at the rate of one dragon, so convoying is
  penalised relative to fanning out — without ever naming dispersion.
* **It decays by itself.** `C` is monotone, so the per-tile payment stops when
  the map is covered, with no λ schedule needed. `λ_exp(t)` is belt and braces.
* **Map-size independent.** Paying `ε` per tile rather than `κ/area` keeps the
  per-tile value identical on an 11×11 and a 64×64; `tanh` bounds the term at ±1
  once the gap passes ~200 tiles. (Normalising by area instead would vary the
  per-tile payment 34× across the official map set.)
* **Torus-trivial.** It is a set count. No distances, no centre, no wrapping.
* **It is a reason to explore that is not a pearl reward.** Coverage is how
  pearls get found, so the team gets paid to look without any dragon being paid
  to eat.
* **Non-rivalrous but ground-rivalrous.** A teammate covering a tile costs me
  nothing (we share the team delta) but does use up that tile, which is the
  pressure that pushes us apart. This is the distinction the whole spec turns
  on: teammates compete for *ground*, never for *reward*.

Implementation: `uint8_t visited[2][area]` in `Env` plus a running count, set
when a head enters a tile. 8 KB per env at the 64×64 maximum, 8 MB at 1024 envs,
O(1) per turn.

**Follow-up, not in v8:** a windowed variant where a tile stops counting `W`
rounds after it was last visited, restoring the real value of re-checking stale
ground (pearls respawn — `MemoryTracker` already tracks the respawn timer).
Maintain it with a ring buffer of expiries for O(1) amortised. Kept out of v8
because cumulative coverage is one fewer knob and cannot oscillate.

## Individual terms: the admissibility rule

The constraint is not "no individual rewards" but **no dragon of ours may be
rewarded at the expense of another of ours.** Formally, an individual term is
admissible iff it is *non-rivalrous*: no teammate's gain may lower another
teammate's reward.

* **Forbidden**: any intra-team normalisation or ranking — share of team length,
  being the longest on our own team, a per-dragon slice of coverage, anything
  divided by a team aggregate. These make dragons compete directly and are the
  reason to be suspicious of individual rewards in the first place.
* **Admissible**: `own_length_delta` (a teammate growing costs me nothing),
  `own_died`. The shared team potential Φ is non-rivalrous by construction,
  since every teammate receives the identical delta.

v8 keeps exactly one individual term, `own_length_delta = 0.01`, annealed to 0
by round 150 — a third of v3's weight, and only during the phase where the
policy is learning to eat and not die.

The reason it is there at all: with a pure team reward, **every dragon receives
the identical reward stream, so there is no credit assignment within a turn.**
A dragon's advantage is dominated by what its ~30 teammates did while it was not
acting. The reward becomes dense in *time* but not local in *space*: a pearl that
paid `+0.03` in v3 now pays `2/T ≈ 0.025` of a `tanh`, diluted across the team —
call it 100× weaker. Early learning gets much slower.

Honest cost: `own_length_delta` does introduce mild *behavioural* rivalry, since
two dragons will race for the same pearl rather than being indifferent about who
takes it. Under a pure team reward they could coordinate. That is the trade, and
it is why the weight is small and temporary.

The non-rivalrous way to get the same variance reduction is on the critic side —
a counterfactual baseline `V(s, i)` conditioned on the acting dragon, so the
advantage subtracts teammate noise (COMA/VDN). That is the principled fix and it
is a requirement on the critic refactor, not on the reward.

## Hand-set weights

Round `t ∈ [0, 500]`, `s = t/500`.

| λ | schedule | 0 | 125 | 250 | 375 | 500 | why |
|---|---|---|---|---|---|---|---|
| `λ_win` | `0.3 + 1.2 s²` | 0.30 | 0.38 | 0.60 | 0.98 | 1.50 | irrelevant early, the whole game late |
| `λ_len` | `0.2 + 0.8(1 − s)` | 1.00 | 0.80 | 0.60 | 0.40 | 0.20 | the dense early signal; hands over to `λ_win` |
| `λ_top3` | `0.6 · clamp((s − 0.4)/0.6, 0, 1)` | 0 | 0 | 0.10 | 0.35 | 0.60 | "don't put all eggs in one basket" only matters once there is something to lose |
| `λ_kill` | `0.8 (1 − s³)` | 0.80 | 0.79 | 0.70 | 0.46 | 0 | decent for most of the game, out at the end where `λ_win` says the same thing |
| `λ_exp` | `1.0 · max(0, 1 − t/150)` | 1.00 | 0.17 | 0 | 0 | 0 | opening only |
| `own_length_delta` | `0.01 · max(0, 1 − t/150)` | 0.010 | 0.0017 | 0 | 0 | 0 | phase-1 credit assignment |
| `W` (terminal) | constant | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | the only non-telescoping term |

Two schedules in substance — one shaping ramp down (`λ_len`, `λ_exp`), one
outcome ramp up (`λ_win`, `λ_top3`) — with `λ_kill` following the outcome ramp
inverted. The intra-group ratios are fixed.

`Σλᵢ(t)` peaks at **3.10 on round 0** and sits between 2.00 and 2.30 from round
250 on (`[0.3, 1.0, 0, 0.8, 1.0]` at the peak). So `|Φ| ≤ 3.1` everywhere, and
the critic's output scale should be chosen for roughly ±3, not ±1.

## Invariants and guards

* Antisymmetry: `Φ(swap(s)) = −Φ(s)` exactly, for every term, at every λ. Assert
  it in a test over random team states — it is the cheapest possible check that
  self-play stays zero-sum.
* Boundedness: `|Φ| ≤ Σλᵢ(t)`. Assert.
* A living dragon has at least a head, so `T ≥ N ≥ 1` and every denominator is
  ≥ 1 while the game runs. A team on 0 dragons ends the game, which is paid by
  the terminal branch, so `0/0` never arises — but adopt `0/0 = 0` anyway and
  clamp `r, z, a` to `±2`.
* Bank once per turn (`acc.banked_at == e.turn` guard at `bc_vec.hpp:623`),
  otherwise a discounted potential gains a spurious `(γ − 1)Φ` per double-bank.
* Φ is **not** zeroed when a dragon dies — a dead dragon keeps the team position
  it left behind, which is what pays a sacrifice for its trade. Carried over
  from v3 unchanged, and it is the reason a sacrifice is learnable at all.
* Bank each `λᵢ(t)Φᵢ` **separately** even though they sum to one scalar. It is
  free and it preserves the per-component dashboard attribution that caught v3's
  kamikaze collapse inside 30 iterations. Without it, one number moves and we
  cannot tell which term moved it.

## The critic follows from this

Most of the return is now `λ(t)Φ(s)`, and **Φ is an analytic function of
privileged team state that we can compute exactly.** So build the critic as a
residual around it:

```
V(s) = λ(t)·Φ(s) + f_θ(s)
```

`f_θ` learns only what Φ does not already explain. The critic starts holding the
right answer for the shaped component instead of rediscovering it from returns,
which is precisely where the memorisation was happening. This is a bigger lever
on the collapse than any λ choice, and it drops into `train/critic_net.py`
alongside the counterfactual baseline above.

## Where it goes in the code

| piece | file | note |
|---|---|---|
| sorted lengths, `A3` | `bc_vec.hpp` `team_stats` | extend to fill a sorted top-3; currently returns total/longest/units |
| coverage counts | `bc_vec.hpp` `Env` | `visited[2][area]` + count, set on head entry |
| Φ and the λ schedules | `bc_vec.hpp` `add_team_delta` | bank `λ(t)Φᵢ` per component, with the round |
| terminal handoff | `bc_vec.hpp` finish path (~`:810`) | `Φ(terminal) := W·outcome`, replacing `RW_WIN/LOSE/DRAW/ELIMINATED` |
| weights, version table | `train/train.py` | `REWARD_V8`; keep v1–v7 for reproducing old runs |
| antisymmetry + bound tests | `bcsim/tests/test_rewards.py` | new |

The v1–v7 components (`RW_PEARLS`, `RW_KILLS`, `RW_DIED`, `RW_SPLITS`,
`RW_ELIMINATED`, `RW_TEAM_*`, `RW_FOE_*`) stay in the enum and keep being
emitted — they are useful dashboard diagnostics even at weight 0, and removing
them would break every saved run's log schema.

## Rejected, with numbers, so they are not tried again

* `λ_4·tanh(n)` on dragon count — pays for a free split of a non-leader in every
  state tried (+0.147 to +0.016), which is v0's shredding.
* `λ_7` as a fragmentation penalty — the counter-derivative is 10× too small
  (`1/T ≈ 0.01` against `2/N`), and `N/T` cannot see the length distribution it
  is supposed to police.
* The draft's finisher term — negative for every numerical advantage at every
  `c` (table above), and zero at the moment of victory.
* Plain (undiscounted) deltas — injects up to ±0.78 of extra return, comparable
  to the win term, and pays the policy to stall while ahead.
* Static dispersion, any form — circular variance scores two parked clumps 1.000,
  identical to a uniform spread, and pays for a configuration rather than an
  activity.
* Euclidean variance about a centroid — undefined on a torus. Not a tuning
  problem; there is no number to compute.

## Open

* `ε = 0.005` per new tile is a guess. It should be set so one new tile is worth
  roughly a third of a pearl-equivalent to the team; measure the actual coverage
  rate in a rollout first.
* Whether `λ_len` and the gated `z` inside `Φ_win` double-count enough to matter.
  They are deliberately redundant — dense early, gated late — but if the policy
  over-values total length in the midgame, `λ_len` is the knob.
* Whether `own_length_delta` is needed at all, or whether the counterfactual
  baseline alone closes the credit-assignment gap. Testable: same run, weight
  0.01 vs 0.
