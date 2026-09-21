#!/usr/bin/env bash
# Plays one UNRANKED game against another team, then shows how our dragons died.
#
# The server strips all LOG output from replays, so the decoded event stream
# is the only feedback a submission gives. A CPU overrun shows up as
# NO_VALID_ACTION on a dragon's first turn.
#
#     wasmprobe/scrim.sh 7          # vs team 7, random map
#     wasmprobe/scrim.sh 7 4        # vs team 7 on map 4 (Default)
#     wasmprobe/scrim.sh --battle 1819   # re-decode an existing battle
#
# Map ids: 1 Colosseum, 2 Arena, 3 Big Empty, 4 Default, 5 Default Small,
#          6 Help, 7 Queen Of Spades, 9 Schooltime, 11 Trophy
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
API=https://game.battlecode.au/api/v1
# the key is read from the unswbc key store and never printed
KEY="${UNSWBC_KEY:-$(python3 -c "
import json, pathlib
d = json.loads((pathlib.Path.home() / '.unswbc/keys.json').read_text())
print(next(iter(d.values())) if isinstance(d, dict) else d)")}"

if [[ "${1:-}" == "--battle" ]]; then
    ID="$2"
else
    TEAM="${1:?usage: scrim.sh <teamId> [mapId]  |  scrim.sh --battle <id>}"
    BODY="{\"teamId\": $TEAM, \"ranked\": false}"
    [[ -n "${2:-}" ]] && BODY="{\"teamId\": $TEAM, \"ranked\": false, \"mapIds\": [$2]}"
    ID=$(curl -s -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
            -d "$BODY" "$API/battles" | python3 -c "import sys, json; print(json.load(sys.stdin)['ids'][0])")
    echo "requested battle $ID"
fi

while true; do
    STATUS=$(curl -s -H "Authorization: Bearer $KEY" "$API/battles/$ID" \
             | python3 -c "import sys, json; print(json.load(sys.stdin)['match']['status'])")
    [[ "$STATUS" == "completed" || "$STATUS" == "failed" || "$STATUS" == "errored" ]] && break
    sleep 8
done

curl -s -H "Authorization: Bearer $KEY" "$API/battles/$ID" | python3 -c "
import sys, json
d = json.load(sys.stdin); m = d['match']
print(f\"battle {m['id']}: {m['status']} on {d['mapName']}, {d['teamAName']} vs {d['teamBName']}, winner {m['winner']}\")
if m.get('log'): print('server log:', m['log'])"

REPLAY="$(mktemp --suffix=.replay)"
trap 'rm -f "$REPLAY"' EXIT
curl -sL -o "$REPLAY" -H "Authorization: Bearer $KEY" "$API/battles/$ID/replay"
python3 "$ROOT/wasmprobe/replay.py" "$REPLAY"
