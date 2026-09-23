#!/bin/bash
# 2026-09-23 ~14:20. ratchet3 was stopped at 128.7M/150M of gen 1 because the
# candidate never left its starting point: vs_anchor sat at 0.499 over 572
# games (se 0.021) with teacher KL 0.0017 and flat entropy, so the 0.55 gate
# was out of reach. The user's call: more drive, two knobs only.
#   lr        3e-5 -> 1e-4   (--lr)
#   kl-coef   0.5  -> 0.4    (--kl-coef)
# Everything else is ratchet3 unchanged: same seed, same 23 maps with the live
# nine at 60%, same nine-map gate, same league, same frozen critic.
#
# --lr-min stays at the 7.5e-6 default, so the discard ladder is now
# 1e-4 -> 5e-5 -> 2.5e-5 -> 1.25e-5 -> 7.5e-6 rather than two halvings.
set -u
cd /home/thomaspetty/BattleCode/bcsim
exec /usr/bin/python3 -u -m train.ratchet run \
    --run ../runs/ratchet4 \
    --start ../runs/i2/sponge_2110/best.pt \
    --train-maps ../maps-all \
    --live-maps ../maps-live --live-share 0.6 \
    --gate-maps ../maps-live \
    --lr 1e-4 \
    --kl-coef 0.4 \
    --extend-min 0.45 \
    --max-drop 0.15 \
    --segment-turns 150000000
