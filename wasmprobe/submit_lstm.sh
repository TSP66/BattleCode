#!/usr/bin/env bash
# The one way to upload lstmbot/: LSTM checkpoints (train/lstm_net.py, arch lstm) or, since
# 2026-10-05, the feed-forward FFL ones (train/ff_net.py, arch ffl: scratch_1002's 15x15 portal-report
# policies; export_ffl.py + the closed-loop parity_ffl.py, native AND wasm):
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

ARCH=$("$PY" -c "import sys, torch; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['args'].get('arch'))" "$CKPT")
case "$ARCH" in
    ffl) EXPORT=export_ffl; TAG=ffl ;;
    lstm) EXPORT=export_lstm; TAG=lstm
          grep -q "LSTMPolicy" "$BOT/main.cpp" || { echo "lstmbot/ is the FFL bot now (archive/lstmbot_lstm_pre_ffl_1005 has the LSTM one)"; exit 1; } ;;
    *) echo "arch $ARCH: lstmbot plays lstm or ffl checkpoints"; exit 1 ;;
esac

echo "== export ($ARCH)"
(cd "$ROOT/bcsim" && "$PY" -m train.$EXPORT --ckpt "$CKPT")
grep -m1 '^// source:' "$BOT/weights.hpp"

echo "== a. package"
# unswbc submit zips the whole folder, hidden entries included: a Jupyter
# .ipynb_checkpoints copy of weights.hpp pushed gen10 to 4.5 MB and the upload was refused.
stray=$(cd "$BOT" && find . -mindepth 1 -maxdepth 1 -name '.*')
[ -z "$stray" ] || { echo "hidden entries in $BOT would be uploaded: $stray"; exit 1; }
(cd "$BOT" && zip -q -9 -r "$TMP/bot.zip" .)   # everything, as unswbc submit does
size=$(stat -c %s "$TMP/bot.zip")
[ "$size" -lt 3900000 ] || { echo "zip $size bytes: over 3.9 MB"; exit 1; }
unzip -l "$TMP/bot.zip" | grep -q " weights.hpp$" || { echo "zip has no weights.hpp"; exit 1; }
echo "zip $((size / 1000)) KB: ok"

echo "== b. parity (simulator + checkpoint)"
g++ -O2 -std=c++20 -DBC_DUMP "$BOT/main.cpp" -o "$TMP/bot_native"
if [ "$ARCH" = ffl ]; then
    # closed loop in the run's own simulator (BC_QUEEN_GUARD=1): native (int64 accumulator headroom),
    # then the same bot as wasm with the SIMD kernels (zig clang, under wasmtime) -- native parity
    # cannot catch a wasm-only kernel bug
    (cd "$ROOT/wasmprobe" && nice -n 19 "$PY" parity_ffl.py "$TMP/bot_native" "$CKPT" "$ROOT/maps-gate-1001" 1 100) \
        | tee "$TMP/parity.txt" | tail -12
    grep -q "^PARITY OK" "$TMP/parity.txt"
    "$ROOT/.venv-zig/bin/python" -m ziglang c++ -target wasm32-wasi -O2 -msimd128 -std=c++20 -w -DBC_DUMP \
        "$BOT/main.cpp" "$ROOT/wasmprobe/stubs.cpp" -o "$TMP/bot_dump.wasm"
    printf '#!/bin/bash\nexec %s %s %s\n' "$HOME/.local/share/uv/tools/unswbc/bin/python" \
        "$ROOT/wasmprobe/wasmrun.py" "$TMP/bot_dump.wasm" > "$TMP/wasmbot.sh"
    chmod +x "$TMP/wasmbot.sh"
    (cd "$ROOT/wasmprobe" && PARITY_MAPS=portals.map,devil.map,queen_of_spades.map,islands.map \
        nice -n 19 "$PY" parity_ffl.py "$TMP/wasmbot.sh" "$CKPT" "$ROOT/maps-gate-1001" 2 60) \
        | tee "$TMP/parity_wasm.txt" | tail -12
    grep -q "^PARITY OK" "$TMP/parity_wasm.txt"
else
    (cd "$ROOT/wasmprobe" && "$PY" parity_lstm.py "$TMP/bot_native" "$CKPT" "$ROOT/maps-live" 150) \
        | tee "$TMP/parity.txt" | tail -6
    grep -q "^PARITY OK" "$TMP/parity.txt"
fi

echo "== c. judge sandbox games (judge clang, judge meter)"
# unswbc caches the built bot under a hash of the .c/.cpp sources only, not the headers, so a
# new weights.hpp alone reuses the old wasm (gen4..gen10 were all played with gen3's weights).
rm -f "${XDG_CACHE_HOME:-$HOME/.cache}"/unswbc/wasmbots/lstmbot-*.wasm \
      "${XDG_CACHE_HOME:-$HOME/.cache}"/unswbc/lstmbot-*
for map in default trauma big_empty; do
    out="$TMP/game_$map.txt"
    (cd "$ROOT" && nice -n 19 timeout 1500 unswbc run --sandbox -v --no-replay "maps-live/$map.map" lstmbot lstmbot) \
        > "$out" 2>&1
    net=$(grep -c "^INDICATOR $TAG" "$out" || true)
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
