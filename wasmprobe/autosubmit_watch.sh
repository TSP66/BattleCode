#!/usr/bin/env bash
# Submits every anchor a ratchet run promotes, through submit_lstm.sh (so every
# check still gates the upload). Follows the run's gens.jsonl; lines already
# there when it starts are skipped. Runs until killed.
#
#     setsid nohup wasmprobe/autosubmit_watch.sh runs/ratchet_v8_sab > runs/ratchet_v8_sab/autosubmit.log 2>&1 &
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$(cd "${1:?usage: autosubmit_watch.sh runs/<ratchet run>}" && pwd)"
GENS="$RUN/gens.jsonl"
seen=$( [ -f "$GENS" ] && wc -l < "$GENS" || echo 0)
echo "[$(date '+%F %T')] watching $GENS from line $((seen + 1))"

while true; do
    now=$( [ -f "$GENS" ] && wc -l < "$GENS" || echo 0)
    while [ "$seen" -lt "$now" ]; do
        seen=$((seen + 1))
        line="$(sed -n "${seen}p" "$GENS")"
        read -r verdict gen score tag < <(python3 -c '
import json, sys
g = json.loads(sys.argv[1])
print(g["verdict"], g["gen"], g["vs_anchor"], g["tag"])' "$line")
        [ "$verdict" = "promote" ] || { echo "[$(date '+%F %T')] $tag: $verdict, nothing to submit"; continue; }
        ck="$RUN/anchors/gen$gen.pt"
        echo "[$(date '+%F %T')] $tag promoted to gen$gen ($score vs anchor): submitting $ck"
        if "$ROOT/wasmprobe/submit_lstm.sh" "$ck" \
                "auto: $(basename "$RUN") gen$gen, promoted at $score vs previous anchor ($tag)"; then
            echo "[$(date '+%F %T')] gen$gen submitted"
        else
            echo "[$(date '+%F %T')] gen$gen NOT submitted: submit_lstm.sh failed (see above)"
        fi
    done
    sleep 60
done
