# BattleCode

A PPO self-play bot for UNSW Battlecode 2026 ("dragons"). It was trained from random weights in a fast C++
copy of the game engine (about 13.5B dragon-turns) and shipped as a C++ bot with its weights compiled in.

## How it was trained

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
- The policy is **temperature-conditioned**. In training it samples at T ~ U[0.1, top], with top
  0.5 → 0.35, and the gate plays greedy.

### Reward: team potential first, then pure win/loss

Earlier attempts each failed in a different way:

- **Per-dragon rewards** (v1–v7) taught shredding and kamikaze runs.
- **Pure win/loss from the start** was too sparse: the critic memorised games and collapsed.
- **Cloning the top teams** and fine-tuning the clones with PPO either stalled or got worse.
  Distilling the top three teams' replays into the PPO policy only helped on one map.

What worked was a curriculum from a dense team reward to the sparse result:

1. **Team potential Φ (reward v8).** One bounded, zero-sum potential per team with **no per-dragon terms**:
   win condition, total length, top-3 length, a finisher term and board coverage. The weights are
   time-varying, so the win condition is ~10% of Φ early and ~90% by round 500. Dragons compete for
   ground, never for reward. The reward is potential-based shaping (ΔΦ), so it cannot be gamed.
2. **A win/loss critic head** trained alongside Φ: undiscounted, TD(λ = 0.98 per round), target ±1.
   At first it only learned and never steered. Its AUC was tracked against Φ's at rounds 25/100/200/350.
3. **Sparse is blended in slowly.** The policy's advantage is
   `A = (1 − b)·std(A_Φ) + b·std(A_WL)`, with b ramped **0 → 1 over 1.2B turns**.
   Each part is standardised separately, so neither scale wins by default.
   The final ~6B turns trained on **win/loss only**.

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
