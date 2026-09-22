# ft4: why it exists (started 2026-09-22)

ft1-ft3 fitted per-dragon returns (critic EV 0.07-0.12) and did not improve.
ft4 = team-level critic (train/finetune_team.py): A = 0.5 A_team + 0.5 A_self,
the team head predicts win/draw/loss and is pretrained on 3011 replay games
(runs/team_critic/pretrained.pt, held-out EV 0.51, acc 83%).

Start and KL teacher: SSS r3 clone (712 x1, 1079 x0.5, 568 x0.25; 860 games),
the user's choice after runs/clone_compare3 (SSS r3 ~= v9 at 0.53; the
Sabotage-d clone is stronger locally but from an off-ladder team).
Opponents: self 50%, frozen SSS r3, Sabotage-d, SHINK AI r1, Vibing++ r4.
ft3's safety net: KL 0.5, LR 3e-5, 1 policy epoch; 25M-turn warmup.

## 2026-09-22 ~14:10: resumed from iter 50 (13.4M) with a scaled self head
The self head's explained variance sat at ~0 through the first 13M turns: its
target (the dragon's shaped return) spreads ~0.03, so its MSE (~3e-4) was
~1000x smaller than the team head's cross-entropy in the shared trunk and
under one gradient clip; it only ever learned the mean. Fix (finetune_team.py,
old copy in finetune_team.pre_selfscale.py): the self head predicts
return / self_scale (EMA of the return's std, saved in the checkpoint),
values are multiplied back for GAE. Warmup extended to 40M, and after it the
policy stays frozen until the self head's EV EMA >= 0.3 (--self-ev-gate),
capped at 80M (--gate-max). The first launch's rows past iter 50 are in
log.pre_resume.jsonl.
