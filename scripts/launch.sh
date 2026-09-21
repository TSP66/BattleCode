#!/usr/bin/env bash
# Start a training run plus its yardstick evaluator (and optionally the dashboard).
#
#     scripts/launch.sh v5_lr1e-4 --width 64 --blocks 4 --lr 1e-4 --reward v4 --compile
#     DASH_PORT=8770 scripts/launch.sh v5_ent02 --ent 0.02 ...
#
# Everything after the run name goes to train.train (see `python -m train.train -h`).
# Output: runs/<name>/ (log.jsonl, latest.pt, snapshots/, eval.jsonl),
#         runs/<name>.out and runs/<name>_yardstick.out.
# Resume: pass --resume runs/<name>/latest.pt with the same flags.
# Stop:   kill the PIDs printed below (also written to runs/<name>.pids).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?usage: launch.sh run_name [train args...]}"
shift
PY="$ROOT/.venv-train/bin/python"
RUN="$ROOT/runs/$NAME"
mkdir -p "$RUN"
cd "$ROOT/bcsim"

nohup nice "$PY" -m train.train --out "$RUN" "$@" >> "$ROOT/runs/$NAME.out" 2>&1 &
pids="$!"
nohup nice "$PY" -m train.yardstick --run "$RUN" >> "$ROOT/runs/${NAME}_yardstick.out" 2>&1 &
pids="$pids $!"
if [ -n "${DASH_PORT:-}" ]; then
    TOKEN="${DASH_TOKEN:-$("$PY" -c 'import secrets; print(secrets.token_urlsafe(16))')}"
    nohup "$PY" -m train.dash --run "$RUN" --port "$DASH_PORT" --host 0.0.0.0 --token "$TOKEN" \
        >> "$ROOT/runs/dash.log" 2>&1 &
    pids="$pids $!"
    echo "dashboard: http://$(hostname):$DASH_PORT/?token=$TOKEN"
fi
echo "$pids" > "$ROOT/runs/$NAME.pids"
echo "run $NAME started, pids: $pids"
echo "tail -f runs/$NAME.out"
