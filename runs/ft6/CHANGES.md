# ft6: why it exists (started 2026-09-22 ~17:15)

ft5's team head drifted: trained only on its own bootstrapped TD targets (real results
rarely fall inside a rollout), its EV against actual game results decayed
0.6 -> 0.13 over the frozen phase and 0.75 -> 0.41 again after the gate resume, while
its TD EV stayed ~0.99. ft5's greedy yardsticks were drifting up slightly
(mean vs 7 anchors 0.553 @50M -> 0.593 @75M), so its policy is kept.

ft6 = ft5's policy at iter 375 (98.6M turns, with its optimiser) + a fresh critic
from runs/team_critic/pretrained.pt, and the team head anchored to REAL results
(finetune_team.py, pre-change copy in runs/ft5/finetune_team.pre_mc.py):
  * sampled positions (--calib-share 1/32) keep their critic inputs until their game
    ends, then go into a 200k ring buffer with the team's real result;
  * team loss = 1.0 x CE(real result, buffer minibatch of 4096) + 0.25 x CE(TD target);
  * the advantage is unchanged (TD(lambda) of the team value, alpha 0.75).
Critic-only warm-up for 10M turns (to 108.6M), then the self-EV gate 0.12, capped at
128.6M. Everything else as ft5: KL 0.5, LR 3e-5, 1 policy epoch, ent 0.001,
opponents self 50% + SSS r3, Sabotage-d, SHINK AI r1, Vibing++ r4.

## 2026-09-22 ~17:50: less reuse of the real-result buffer (resumed from iter 425, 111.7M)
The real-result loss fell to 0.036 (pretraining's held-out CE was ~0.39): the buffer holds
only a few hundred distinct game results, and 4096 draws per critic step reused each
position ~100x, so the head memorised games. Its EV against real results on fresh positions
held at 0.62 (ece 0.033), but a memorising head generalises worse over time.
Now --mc-batch 512 --mc-coef 0.5 (TD 0.25 unchanged). The policy was still frozen, so nothing
is lost there; the buffer isn't checkpointed and refills in ~20 iterations.
Submission bar (the user): > 0.60 vs every opponent, 96-game confirmation, then submit.
