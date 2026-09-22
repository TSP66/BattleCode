#!/bin/bash
# Clone round-robin, 2026-09-22: each new clone vs the other new ones, the
# incumbents (v9 = SSS r2 via the submitted manifest, Vibing++ r4, Sabotage-d)
# and the scripted bots. 12 games per (opponent, map) x 8 live maps = 96 per pair.
set -e
cd /home/thomaspetty/BattleCode/bcsim
D=../runs/clone_compare3
cp ../runs/imitate_vibing_r5/best.pt $D/vibing_r5.pt
cp ../runs/imitate_sss_r3/best.pt $D/sss_r3.pt
cp ../runs/imitate_shink_r1/best.pt $D/shink_r1.pt
cp ../runs/anchors/vibing_r4_bc_64x4.pt $D/vibing_r4.pt
cp ../runs/anchors/sabotage_bc_64x4.pt $D/sabotage.pt
for x in sss_r3 vibing_r5 shink_r1; do
  opp=$(ls $D/*.pt | grep -v "/$x.pt" | paste -sd,)
  echo "=== $(date +%H:%M) $x vs $opp"
  /usr/bin/python3 -m train.yardstick --run $D --maps ../runs/ft3/maps --games 12 \
      --threads 12 --lags "" --anchors "$opp" --ckpt $D/$x.pt --dump $D/results.jsonl > $D/$x.out 2>&1
done
echo "=== $(date +%H:%M) done"
