# BattleCode

A PPO self-play bot for UNSW Battlecode 2026 ("dragons"). It was trained from random weights in a fast C++
copy of the game engine (about 13.5B dragon-turns) and shipped as a C++ bot with its weights compiled in.

## How it was trained

### The policy and its inputs

Each dragon decides on its own, from what it can see, what it remembers and what its teammates told it
over sonar. There is no central controller. The network is a small CNN followed by an MLP:

```
15x15x54 grid ─ conv 48 ─ resblock ─ stride-2 conv 112 ─ 2 resblocks ─ 1x1 squeeze 16 ─ dense 256 ┐
previous action (76-way embedding, 48) ──────────────────────────────────────────────────────────┤
scalars 5 + identity 3 + action history 75 + temperature 2 ─────────────────────────────────────┴─ MLP 128 ─ 128 ─ 75 actions
```

**CNN input:** 54 planes on a 15×15 grid centred on the head and rotated to the dragon's facing.

| Group | Planes |
|---|---|
| Terrain (remembered once seen) | Kelp on each of the 4 edges, portal on each of the 4 edges, tiles that never spawn pearls |
| Live 7×7 view | Pearl, pearl timer, ally head/body, enemy head/body, which way each segment points (4) |
| Self (exact across the grid) | Own body, segment index, tail |
| Decaying memory | Tile last seen (e^(−age/32)), pearl expected, projected pearl timer, enemies and allies last seen (e^(−age/8)), tiles visited (e^(−age/16)) |
| Global (constant planes) | Round, own length, units alive, map width and height |
| Sonar echoes | Fraction of rays hitting kelp, ally, ally head, enemy, enemy head |
| Teammate reports (sonar v2) | Each nearby ally's length, drawn at its head |
| Portal reports | What lies through each known portal: known, closed room, room size, pearls (fading with age) |
| Queens | Am I the queen; ally/enemy queen in view; both queens' last known place (decaying); each queen's offset and freshness |

The board wraps around (maps are tori), and every plane is drawn that way.

**Straight into the MLP, not the CNN.** Some inputs are single numbers, and painting them as constant
planes made the CNN do pointless work. These go into the first dense layer instead. Each was added mid-run
as zero-initialised columns, so the network's output didn't change at the moment it was added:

| Input | What |
|---|---|
| Previous action | Learned 48-dim embedding of the last action (75 actions + "none") |
| Scalars (5) | round/500, (round/500)², length/64, (units alive/limit)², is-queen |
| Identity (3) | Birth round/500, sin(id/7), sin(id/43), so otherwise identical dragons can take on different roles |
| Action history (75) | Decayed count of each own action before the last one (½, ¼, …) |
| Temperature (2) | T and √T, see below |

**Actions (75):** moves, 2- and 3-step sprints, splits, self-kill, and 24 "far sprints" that reach any
tile 4–6 steps away in the 7×7 window in one action.

### Ratchet: training in gated generations

Plain PPO kept finding gains and then losing them, so training runs as a **ratchet**:

- The learner trains in segments of 250–500M turns. After each segment it plays a **gate** of greedy
  games against the current anchor and the four newest past versions. Gate maps are held out from training.
- **Promote** if it scores ≥ 0.6 against every one of them. It becomes the next generation, joins the
  opponent pool and becomes the new KL teacher.
- **Extend** otherwise: it keeps training from its own weights and is never thrown away.
  A candidate that regresses against an old version can't be promoted, even with a good score against the anchor.
- Opponents in training are mostly self-play plus past generations. A past version the learner already
  beats 95% of the time leaves training but stays in the gate.

Version 0 is a uniformly random player. Twelve generations were promoted, and every promotion was
submitted automatically.

### KL leash and entropy

- **KL(teacher ‖ student)** to the last promoted version. It is off until the first promotion,
  because there is no point leashing the learner to the random player.
  Tuning it took several restarts: 0.15 froze learning, 0.005 drifted, 0.1 worked early, and 0.025
  was used for most of the run. Late on it was frozen at **0.01**. A gradient probe showed the KL pull
  and PPO's steady push roughly cancel each other, so the leash was loosened, not removed.
  A segment is aborted if KL goes above 0.5.
- **Entropy** bonus 0.015 → 0 over 6B turns, later reset to 0.005 → 0. It is weighted by sampling
  temperature like the policy loss. Without that weighting, cold rows got up to 4× the bonus and entropy
  stopped falling.

### Temperature: one policy, trained at many temperatures

The policy is **conditioned on its own sampling temperature**. T and √T are fed into its first dense
layer, and it samples from softmax(logits / T). Every game draws a fresh temperature for each team, so
one network learns to play across the whole range instead of one fixed noise level:

| Who | Temperature per game |
|---|---|
| Learner | T ~ U[0.1, top], with top 0.5 → 0.35 over the run (mean ≈ 0.23 at the end) |
| League opponents (past versions) | 10% fully greedy (T = 0), otherwise T ~ U[0, 0.4] (mean ≈ 0.18) |
| Gate and the shipped bot | Greedy (temperature folded in at 0.1) |

**The learner deliberately runs warmer than its opponents.** Its average temperature is higher, so it
explores more, while the opponents play sharper and closer to their best. It is trained against harder
versions of each opponent than it would meet at equal temperature. Every greedy gate game is the fair
comparison.

The PPO policy loss and the entropy bonus are both weighted by T / 0.4. At temperature T, the gradient with
respect to the logits scales as 1/T. Without the weight, cold rows would dominate the update and get up to
4× the entropy bonus.

### Reward: team potential first, then pure win/loss

Earlier attempts each failed in a different way:

- **Per-dragon rewards** (v1–v7) taught shredding and kamikaze runs.
- **Pure win/loss from the start** was too sparse: the critic memorised games and collapsed.
- **Cloning the top teams** and fine-tuning the clones with PPO either stalled or got worse.
  Distilling the top three teams' replays into the PPO policy only helped on one map.

What worked was a curriculum from a dense team reward to the sparse result:

1. **Team potential Φ (reward v8).** One bounded, zero-sum potential per team with **no per-dragon terms**
   (see the table below). Dragons compete for ground, never for reward. The reward is potential-based
   shaping (ΔΦ), so it cannot be gamed.
2. **A win/loss critic head** trained alongside Φ: undiscounted, TD(λ = 0.98 per round), target ±1.
   At first it only learned and never steered. Its AUC was tracked against Φ's at rounds 25/100/200/350.
3. **Sparse is blended in slowly.** The policy's advantage is
   `A = (1 − b)·std(A_Φ) + b·std(A_WL)`, with b ramped **0 → 1 over 1.2B turns**.
   Each part is standardised separately, so neither scale wins by default.
   The final ~6B turns trained on **win/loss only**.

#### The Φ terms

Φ = Σ λᵢ(t)·Φᵢ / Σ λᵢ(t), with κ = 1, so |Φ| ≤ 1. s = round / 500. Each term is computed from our
team's side minus the enemy's side, so swapping the teams flips Φ's sign exactly. `nd(a, b)` = 2(a − b)/(a + b).

| Term | Φᵢ (each in [−1, 1]) | Weight λᵢ(t) | How the weight decays |
|---|---|---|---|
| **WIN** | The round-500 verdict in its own order: tanh(nd(queen) + g·tanh(nd(longest) + g·tanh(nd(total))))<br>Each inner tiebreak is scaled by g so it can never outvote the level above | 1.5·s² | 0 at the start, grows quadratically, 1.5 at round 500 |
| **LEN** | tanh(nd(total length)) | 1 − s | Linear 1 → 0 |
| **QUEEN** | Our living queens − theirs | min(1, (500 − round)/125) | Flat 1 until round 375, then linear to 0, so a late queen kill isn't paid twice |
| **KILL** | e^(−their total/15) − e^(−our total/15): how close each team is to being wiped out | 0.8·(1 − s³) | 0.8, holds, then drops off late |
| **EXP** | tanh(0.005·(our tiles explored − theirs)) | (1 − map known)·clamp((75 − round)/50) | Opening only: gone by round 75, and sooner on small maps |

Each term's share of Φ by round. EXP assumes about 30% of the map is explored, so its exact share varies by map:

| Round | WIN | LEN | QUEEN | KILL | EXP |
|---|---|---|---|---|---|
| 0 | 0% | 29% | 29% | 23% | 20% |
| 50 | 0% | 29% | 33% | 26% | 11% |
| 100 | 2% | 30% | 38% | 30% | 0% |
| 250 | 15% | 19% | 39% | 27% | 0% |
| 375 | 33% | 10% | 39% | 18% | 0% |
| 450 | 63% | 5% | 21% | 11% | 0% |
| 500 | 100% | 0% | 0% | 0% | 0% |

So Φ rewards material and safety early and turns into the actual verdict by the end. A single ΔΦ is paid
to the whole team, and the critic's horizon is short (discount 0.8 per round, about 5 rounds).
Lengths are conserved by a split, so splitting is never paid for its own sake.

The critic is separate from the policy. It sees the **true board**: a wrapped 27×27 crop on the toroidal
map plus a pooled view of the whole board. It also knows which opponent it is playing (each past
generation has its own identity slot) and both teams' temperatures. It never reads policy features.

### Supervised "perfect play" mix

After every gate, a short supervised pass pushes the policy toward moves that are provably right. The
positions are **generated on fresh random maps**, never taken from the training maps, so the lesson has
to generalise rather than be memorised:

- kill the enemy queen when it is in reach, including with far sprints
- self-kill a queen that is trapped
- keep a walled-in enemy queen shut in
- eat an adjacent pearl
- late-game suicide when it decides the tiebreak
- never dive the queen into a dead end she can see

Ordinary self-play turns are mixed in under a KL penalty. The pass stops itself once it moves the policy
by KL 0.02, so it nudges rather than overwrites. It was switched off for the final endgame segments.
The dead-end rule is also a hard action mask in the shipped bot.

### Maps

70% of games are on ~4,000 generated maps (a port of the loong map generator, checked for symmetry and
verified turn by turn against the real engine). The other 30% are on the official maps, including the
server's hidden variants and pearl layouts reconstructed from replays.

## Layout

| Path | What |
|---|---|
| `bcsim/cpp/` | C++ simulator, observations, reward, sonar |
| `bcsim/train/` | `ratchet.py` (gates and promotion), `ratchet_ff_train.py` (PPO segment), `perfect_play.py` (SL pass) |
| `lstmbot/` | The C++ submission (`wasmprobe/submit_lstm.sh` builds, checks parity and uploads) |
| `runs/submitted/` | Every submitted checkpoint |
| `maps-*` | Training, gate and server map sets |

Setup: `scripts/setup.sh` (needs `uv`, `g++` and an NVIDIA driver). Reward details are in `REWARDS.md`,
submission details in `SUBMITTING.md`.
