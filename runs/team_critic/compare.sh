#!/bin/bash
set -u
cd /home/thomaspetty/BattleCode/bcsim
while pgrep -f "team_critic pretrain" > /dev/null; do sleep 10; done
echo "=== fit done; same held-out games, split by source ==="
/usr/bin/python3 -u -m train.critic_compare \
  /home/thomaspetty/BattleCode/runs/team_critic/pretrained_pre20260923.pt \
  /home/thomaspetty/BattleCode/runs/team_critic/pretrained.pt
