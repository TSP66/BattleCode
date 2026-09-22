# ft5: why it exists (started 2026-09-22 ~14:40)

A fresh restart of ft4 (the user's call), with the self-head fix from ft4's resume
(self head trained on return / running std) plus:
  * alpha 0.75: A = 0.75 A_team + 0.25 A_self. The self head's EV in ft4
    reached only 0.1-0.2 (its target is dominated by kills, deaths and pearl races),
    while the team head's EV against real results was 0.57-0.70
    (calibration error ~0.04).
  * self-EV gate 0.15 (was 0.3, which it never reached); cap 80M.
  * entropy bonus 0.001, set explicitly; entropy is now also logged while the
    policy is frozen (ft4 showed 0.000 then only because it wasn't measured).
Same as ft4 otherwise: SSS r3 start and teacher, pretrained team critic, KL 0.5,
LR 3e-5, 1 policy epoch, 25M warmup; opponents self 50%, SSS r3, Sabotage-d,
SHINK AI r1, Vibing++ r4. ft4 (stopped at ~35M, policy never trained) is kept
in runs/ft4.

## 2026-09-22 ~16:15: gate lowered to 0.12 (resumed from iter 200, 52.7M)
The self head's EV EMA plateaued at 0.11-0.13 and never reached 0.15, so the
policy was still frozen at 52M. The user asked to lower the gate to 0.12 (EMA at
the checkpoint: 0.129), so the policy trains from here. Same settings
otherwise (alpha 0.75, ent 0.001, KL 0.5, LR 3e-5). Close watch: runs/ft5/watch.py.
