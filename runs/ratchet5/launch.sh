#!/bin/bash
# 2026-09-24. A fresh start: the pyramid, the shared-map channel, and a clone of
# the team that is actually top of the ladder right now.
#
# Why not continue ratchet4. Its anchor (gen1) has 750M turns of RL on it and
# stalled twice -- ratchet3 and ratchet4 both sat at vs_anchor 0.499 -- so the
# user's call was to start from a freshly distilled model instead of a partially
# trained one. Nothing of that lineage is in here.
#
# Where --start comes from, in three steps, all run today:
#   1. Sabotage-d is rank 1 (elo 2085, ahead of SSS 2037). Their current
#      submission is 3952; the clone in the league was built from 546, their
#      OLDEST, so it was badly stale. Fetched 589 games of 3952 (6.86M samples).
#      The simulator replayed all 1,163 games of theirs exactly, 0 diverged.
#   2. train.imitate -> a flat 64x4 clone of 3952, held-out action accuracy
#      0.8035 (the old stale clone managed 0.7950, on a weaker bot). Still
#      improving at the last epoch, so there is more to have here if wanted.
#   3. train.distill --arch pyramid --memchan -> runs/pyr_sab3952, 0.959M params.
#      Distilled WITH the channel on, so the student's feature distribution is
#      the one PPO will show it and there is no shift at handover.
#
# The one new capability is --memchan (bcsim/train/memfeat.py): every dragon
# broadcasts its nearest expected pearls and its directional pearl weights as 64
# bits of sonar, and unions what it hears into its own memfar features. It goes to
# BOTH the trainer and the gate, which train.ratchet enforces.
#
# This needs a NEW --run directory: memfar moves for every net in the league, so
# gate scores from before and after are not comparable, and the state file refuses
# to flip the flag mid-experiment.
#
# Expect the first gate scores to be LOWER than ratchet4's. The starting policy is
# a clone of another team rather than 750M turns of our own RL, which is the trade
# that was asked for. What to watch is whether vs_anchor moves at all: both
# previous ratchets failed by never leaving the starting point (teacher KL 0.0017,
# flat entropy), not by getting worse.
#
# Throughput: ~29.5k turns/s measured for this configuration (the pyramid's memory
# branch and its plane storage cost ~30% against a flat net; the merge itself is
# ~5%).
set -u
cd /home/thomaspetty/BattleCode/bcsim
exec /usr/bin/python3 -u -m train.ratchet run \
    --run ../runs/ratchet5 \
    --start ../runs/pyr_sab3952/latest.pt \
    --memchan \
    --train-maps ../maps-all \
    --live-maps ../maps-live --live-share 0.6 \
    --gate-maps ../maps-live \
    --lr 1e-4 \
    --kl-coef 0.4 \
    --extend-min 0.45 \
    --max-drop 0.15 \
    --segment-turns 150000000
