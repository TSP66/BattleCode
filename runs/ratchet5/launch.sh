#!/bin/bash
# 2026-09-24. ratchet4's settings unchanged, plus the team's shared map.
#
# NOT LAUNCHED YET -- waiting on an explicit go.
#
# The one change is --memchan (bcsim/train/memfeat.py): every dragon declares
# protocol 3, broadcasts its nearest expected pearls and its directional pearl
# weights as 64 bits of sonar, and unions what it hears into its own memfar
# features. The flag goes to BOTH the trainer and the gate, which train.ratchet
# enforces -- a candidate trained with it and gated without would be measured on
# different features from the ones it learned on.
#
# It needs a NEW --run directory. memfar moves for every net in the league, so
# gate scores from before and after are not comparable and the state file refuses
# to flip the flag mid-experiment.
#
# --start is ratchet4's current anchor (gen1, 750M experiment turns) rather than
# the original clone, so this builds on what has already been promoted.
#
# What to expect. Bolted onto a policy trained WITHOUT the channel it is neutral:
# 96 games a side, gen4 0.5417 -> 0.5260 (-0.2 sigma), v12 0.6510 -> 0.6250
# (-0.4 sigma). Both within noise and both slightly down, which is what an
# unadapted policy reading a shifted feature should look like. The channel has to
# earn its place by being TRAINED with, and the first thing to watch is whether
# vs_anchor moves at all -- ratchet3 and ratchet4 both stalled at 0.499, so a
# candidate that again never leaves its starting point is a statement about the
# ratchet, not about the channel.
#
# Throughput: the merge costs ~1.6ms a step at 1024 envs, about 5% of an
# iteration, so expect ~33k turns/s where ratchet4 logged ~34.6k.
set -u
cd /home/thomaspetty/BattleCode/bcsim
exec /usr/bin/python3 -u -m train.ratchet run \
    --run ../runs/ratchet5 \
    --start ../runs/ratchet4/anchors/gen1.pt \
    --memchan \
    --train-maps ../maps-all \
    --live-maps ../maps-live --live-share 0.6 \
    --gate-maps ../maps-live \
    --lr 1e-4 \
    --kl-coef 0.4 \
    --extend-min 0.45 \
    --max-drop 0.15 \
    --segment-turns 150000000
