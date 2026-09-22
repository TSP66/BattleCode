#!/bin/bash
# dev test 1 :P r3 (798 games of 1302) full eval, 2026-09-22 evening.
# 12 games per (opponent, map) x 8 live maps = 96 per opponent; v9 is played twice
# (as "submitted v9" and "anchor v9_again") = 192 games against the live bot.
set -e
cd /home/thomaspetty/BattleCode/bcsim
D=../runs/clone_compare6
cp ../runs/imitate_devtest_r3/best.pt $D/devtest_r3.pt
opp=$(ls $D/*.pt | grep -v "/devtest_r3.pt" | paste -sd,)
echo "=== $(date +%H:%M) devtest_r3 vs $opp"
/usr/bin/python3 -m train.yardstick --run $D --maps ../runs/ft3/maps --games 12 --threads 16 \
    --lags "" --anchors "$opp" --ckpt $D/devtest_r3.pt --dump $D/results.jsonl --max-seconds 5400 > $D/devtest_r3.out 2>&1
echo "=== $(date +%H:%M) done"
