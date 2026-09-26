#!/usr/bin/env bash
# The one way to upload the LSTM bot (lstmbot/, train/lstm_net.py checkpoints):
#
#     wasmprobe/submit_lstm.sh runs/ratchet_v8_sab/anchors/gen3.pt "short description"
#     CHECK_ONLY=1 wasmprobe/submit_lstm.sh ckpt.pt x     # every gate, no upload
#
#   1. exports the checkpoint into lstmbot/weights.hpp (+ silu_table.hpp);
#   2. gates, and stops unless every one passes:
#      a. the zip unswbc would upload is under 3.9 MB and carries the weights;
#      b. parity_lstm.py: grid, legal-move mask, logits and argmax against the
#         simulator and the checkpoint, int32 accumulator headroom;
#      c. full games in the judge's sandbox, built with the judge's own clang
#         (unswbc run --sandbox): every turn is played by the net, no warnings,
#         no turn over 90M points;
#   3. uploads lstmbot with unswbc submit;
#   4. records it in runs/submitted/submitted.jsonl with a copy of the checkpoint.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOT="$ROOT/lstmbot"
CKPT="$(cd "$(dirname "${1:?usage: submit_lstm.sh ckpt.pt \"description\"}")" && pwd)/$(basename "$1")"
DESC="${2:?give a short description of what changed}"
PY=/usr/bin/python3                       # torch; CPU only here (CUDA_VISIBLE_DEVICES empty)
SUB="$ROOT/runs/submitted"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export PATH="$HOME/.local/bin:$PATH" UNSWBC_NO_UPDATE=1 CUDA_VISIBLE_DEVICES=
MAX_POINTS=90000000

echo "== export"
(cd "$ROOT/bcsim" && "$PY" -m train.export_lstm --ckpt "$CKPT")
grep -m1 '^// source:' "$BOT/weights.hpp"

echo "== a. package"
(cd "$BOT" && zip -q -9 "$TMP/bot.zip" *.cpp *.hpp bot.toml)
size=$(stat -c %s "$TMP/bot.zip")
[ "$size" -lt 3900000 ] || { echo "zip $size bytes: over 3.9 MB"; exit 1; }
unzip -l "$TMP/bot.zip" | grep -q " weights.hpp$" || { echo "zip has no weights.hpp"; exit 1; }
echo "zip $((size / 1000)) KB: ok"

echo "== b. parity (simulator + checkpoint)"
g++ -O2 -std=c++20 -DBC_DUMP "$BOT/main.cpp" -o "$TMP/bot_native"
(cd "$ROOT/wasmprobe" && "$PY" parity_lstm.py "$TMP/bot_native" "$CKPT" "$ROOT/maps-live" 150) \
    | tee "$TMP/parity.txt" | tail -6
grep -q "^PARITY OK" "$TMP/parity.txt"

echo "== c. judge sandbox games (judge clang, judge meter)"
for map in default trauma big_empty; do
    out="$TMP/game_$map.txt"
    (cd "$ROOT" && timeout 1500 unswbc run --sandbox -v --no-replay "maps-live/$map.map" lstmbot lstmbot) \
        > "$out" 2>&1
    net=$(grep -c "^INDICATOR lstm" "$out" || true)
    nonet=$(grep -c "^INDICATOR NO NET" "$out" || true)
    warn=$(grep -vE "^(INDICATOR|MOVE|SONAR|SPLIT|PROTOCOL)" "$out" \
        | grep -ciE "warn|error|invalid|exceed|timeout|overrun|trap|panic" || true)
    max=$(grep "points per turn" "$out" | sed -n 's/.*max \([0-9.]*\)M.*/\1/p' | sort -n | tail -1)
    result=$(grep -E "wins|draw" "$out" | tail -1)
    printf '%-10s net %6d  no-net %d  warnings %d  max %sM  %s\n' "$map" "$net" "$nonet" "$warn" "$max" "$result"
    if [ "$net" -eq 0 ] || [ "$nonet" -ne 0 ] || [ "$warn" -ne 0 ] || [ -z "$result" ] || [ -z "$max" ] \
            || ! awk -v m="$max" -v l="$MAX_POINTS" 'BEGIN{exit !(m * 1e6 < l)}'; then
        echo "GATE FAILED on $map: do not submit"; exit 1
    fi
done
echo "ALL CHECKS PASSED"
[ -n "${CHECK_ONLY:-}" ] && exit 0

echo "== upload"
DIR="$(basename "$(dirname "$CKPT")")"
case "$DIR" in
    snapshots|anchors|cands|ckpt|checkpoints)
        DIR="$(basename "$(dirname "$(dirname "$CKPT")")")" ;;
esac
NAME="$DIR-$(basename "$CKPT" .pt)"
OUT="$(unswbc submit "$BOT" -n "$NAME" -d "$DESC" 2>&1)"
echo "$OUT"
VERSION="$(echo "$OUT" | sed -n 's/.* as \(v[0-9][0-9]*\).*/\1/p')"
if [ -z "$VERSION" ]; then
    echo "could not read the version number from the upload; NOT recorded" >&2
    exit 1
fi
cp "$CKPT" "$SUB/$VERSION.pt"
"$PY" - "$SUB" "$VERSION" "$NAME" "$CKPT" <<'EOF'
import json, sys, time
sub, version, name, src = sys.argv[1:5]
with open(f"{sub}/submitted.jsonl", "a") as fh:
    fh.write(json.dumps({"version": version, "name": name, "ckpt": f"runs/submitted/{version}.pt",
                         "source": src, "bot": "lstmbot", "time": time.strftime("%Y-%m-%dT%H:%M")}) + "\n")
EOF
echo "recorded $VERSION"
