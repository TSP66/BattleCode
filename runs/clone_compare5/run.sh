#!/bin/bash
# dev test 1 :P clones on 265 games of submission 1302: 64x4 (r2) and 64x4 with
# hidden 576 (r2_h576, ~12% more params, ~69.6M judge points vs 68.4M).
# vs v9 (submitted manifest), devtest r1 (223 games), the other clones and each other.
# 12 games per (opponent, map) x 8 maps = 96 per pair; 45 min limit so no game is dropped.
set -e
cd /home/thomaspetty/BattleCode/bcsim
D=../runs/clone_compare5
cp ../runs/imitate_devtest_r2/best.pt $D/devtest_r2.pt
cp ../runs/imitate_devtest_r2_h576/best.pt $D/devtest_r2_h576.pt
BASE=../runs/clone_compare4/devtest_r1.pt,../runs/anchors/sss_r3_bc_64x4.pt,../runs/anchors/shink_r1_bc_64x4.pt,../runs/anchors/sabotage_bc_64x4.pt,../runs/anchors/vibing_r4_bc_64x4.pt
for x in devtest_r2 devtest_r2_h576; do
  other=$D/devtest_r2.pt; [ $x = devtest_r2 ] && other=$D/devtest_r2_h576.pt
  echo "=== $(date +%H:%M) $x"
  /usr/bin/python3 -m train.yardstick --run $D --maps ../runs/ft3/maps --games 12 --threads 8 \
      --lags "" --anchors "$BASE,$other" --gpu-frac 0.15 --max-seconds 2700 \
      --ckpt $D/$x.pt --dump $D/results.jsonl > $D/$x.out 2>&1
done
echo "=== $(date +%H:%M) done"
