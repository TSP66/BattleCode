#!/bin/bash
# dev test 1 :P clone vs v9 (live, via the submitted manifest) and the other
# clones; same setup as clone_compare3: 12 games per (opponent, map) x 8 live
# maps = 96 per pair, greedy. GPU capped: ft5 is training next to it.
set -e
cd /home/thomaspetty/BattleCode/bcsim
D=../runs/clone_compare4
#cp ../runs/imitate_devtest_r1/best.pt $D/devtest_r1.pt
OPP=../runs/anchors/sss_r3_bc_64x4.pt,../runs/anchors/shink_r1_bc_64x4.pt,../runs/anchors/sabotage_bc_64x4.pt,../runs/anchors/vibing_r4_bc_64x4.pt
echo "=== $(date +%H:%M) devtest_r1"
/usr/bin/python3 -m train.yardstick --run $D --maps ../runs/ft3/maps --games 12 --threads 8 \
    --lags "" --anchors "$OPP" --gpu-frac 0.15 --max-seconds 2700 --ckpt $D/devtest_r1.pt --dump $D/results.jsonl > $D/devtest_r1.out 2>&1
echo "=== $(date +%H:%M) done"
