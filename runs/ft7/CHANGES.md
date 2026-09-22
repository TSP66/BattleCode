# ft7: why it exists (started 2026-09-22 ~19:30)

ft6's team head (anchored to real results) still drifted: EV against real results
0.63 -> 0.30 by 154M, and its greedy yardstick average fell 0.630 (112.5M) -> 0.556
(125M). Likely cause: the self head, now at full loss scale, reshapes the trunk it
shares with the team head. Sabotage-d stayed at 0.41-0.45 on every snapshot, the one
opponent under the user's 0.60 bar.

ft7 = ft6's 112.5M snapshot (ft5's policy + ~1 ft6 policy iteration; best yardstick average)
  * team value from a FROZEN copy of runs/team_critic/pretrained.pt (--freeze-team):
    trunk and head can't move, so it can't drift;
  * the trainable critic (fresh from the pretrained trunk) fits the self head only;
  * Sabotage-d gets 60% of frozen-opponent games = 30% of all games
    (--opp-weights 1,4.5,1,1; the other three clones ~7% each, self-play 50%).
5M critic-only warm-up (to 117.7M), then the 0.12 self-EV gate (cap 137.7M).
Otherwise as ft6: alpha 0.75, KL 0.5 to SSS r3, LR 3e-5 (fresh optimisers: the snapshot
has none), 1 policy epoch, ent 0.001.
Submission bar (the user): > 0.60 vs every opponent, confirmed on 96 games, then submit.
