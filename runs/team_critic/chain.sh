#!/bin/bash
set -u
cd /home/thomaspetty/BattleCode/bcsim
echo "waiting for selfplay..."
while pgrep -f "team_critic selfplay" > /dev/null; do sleep 15; done
echo "selfplay done: $(ls /home/thomaspetty/BattleCode/runs/team_critic/data/9*.npz 2>/dev/null | wc -l) invented-map games"
echo "=== refit on replays + invented-map games ==="
date -Is
/usr/bin/python3 -u -m train.team_critic pretrain --gpu-frac 0.5
date -Is
