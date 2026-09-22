# ft3: why it exists (started 2026-09-22 ~04:05)

ft2 (Vibing++ r2 start, KL 0.5, LR 3e-5, 1 policy epoch) held level but did not
learn over ~107M turns of training (pooled training-game scores vs the frozen
clone 0.356 / 0.389 / 0.344 in successive 30M windows). Greedy head-to-head, its
own starting clone beat its 112M snapshot 0.59. It is stopped at iter 600
(157.5M turns) and can be resumed from runs/ft2/latest.pt.

The overnight clone tournament (runs/clone_compare2, 64 games per pair):
SSS r2 beats Vibing++ r2 0.72 (0.77 from the other side), which meets the user's
"much better, go for SSS" rule. Sabotage-d (the new #1) beats SSS r2 0.56-0.62
with less than half the data.

ft3 = SSS r2 clone start and teacher, plus the privileged critic (both teams'
total length, longest dragon, unit counts, round, and the longest margin),
since the diagnosis for ft1/ft2 was a critic that explains ~5-7% of the return.
Otherwise ft2's stable settings: KL 0.5, LR 3e-5, 1 policy epoch, a 50M-turn
warmup, shaping 0.25. Opponents: self 50%, frozen SSS r2, Sabotage-d, Vibing++ r4
(v6 dropped: everything beats it ~0.98).

Confound: start clone and critic both changed. The morning question is
whether the privileged critic's explained variance is clearly above ft1/ft2's
~0.07. That can be read during the warmup alone, before the policy moves.
