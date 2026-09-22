# ft2: why it exists

ft1 (KL 0.1, policy LR 1e-4, 2 policy epochs) degraded once the policy began
training at 50M turns. Over ~25M turns of training, pooled training-game scores
(n ≈ 270 per opponent) went:

| opponent          | warmup (policy = clone) | training |
|-------------------|-------------------------|----------|
| frozen r2 clone   | 0.355                   | 0.269    |
| SSS clone         | 0.237                   | 0.149    |
| v6 550M           | 0.983                   | 0.969    |

The yardstick agreed (vs v8, the same network as the start: 0.39 at 50M, 0.25
at 63M). Entropy rose from ~0.39 to ~0.50 and games got longer (205 → 311
rounds). Critic explained variance stays around 0.07 (v6 ran at ~0.10), so the
advantages are mostly noise, and from a strong starting policy, noisy updates
cost more than the weak signal gains.

ft2 resumes from ft1's 50M snapshot (critic warmed, policy still the clone),
with fresh optimisers and:
  * --kl-coef 0.5 (was 0.1)
  * --lr 3e-5 (was 1e-4)
  * --policy-epochs 1 (critic keeps 2)
Reward and opponents are unchanged, so the warmed critic stays valid.

Longer-term fix, for the user to decide: give the critic privileged global
state (both teams' lengths, unit counts, round) so it can actually predict
the result.
OOM crash at iter 199 caused by a concurrent A/B test I ran; relaunched identically at 01:04. Lesson: no other GPU jobs next to the fine-tune.
