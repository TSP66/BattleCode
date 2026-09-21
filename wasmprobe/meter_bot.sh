#!/usr/bin/env bash
# Prices the C++ bot in the judge's CPU points, per turn, without submitting.
#
# Compiles mybot/ to wasm32-wasi with the judge's flags, then runs it under the
# toolkit's copy of the judge's metering against a recorded 12-round game.
# Any turn over 100M is flagged OVER. Aim to keep every turn under ~80M.
#
#     wasmprobe/meter_bot.sh            # meter mybot/
#     wasmprobe/meter_bot.sh some/dir   # meter another bot directory
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOT="$(cd "${1:-$ROOT/mybot}" && pwd)"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

echo "building $BOT -> wasm32-wasi (-O2 -msimd128) ..."
if ! "$ROOT/.venv-zig/bin/python" -m ziglang c++ -target wasm32-wasi -O2 -msimd128 \
        -std=c++20 -w ${BOT_CFLAGS:-} "$BOT/main.cpp" "$ROOT/wasmprobe/stubs.cpp" \
        -o "$OUT/bot.wasm" > "$OUT/build.log" 2>&1; then
    grep -E "error:" "$OUT/build.log" | head -20
    exit 1
fi

"$HOME/.local/share/uv/tools/unswbc/bin/python" "$ROOT/wasmprobe/runbot.py" \
    "$OUT/bot.wasm" "$ROOT/wasmprobe/transcript.txt" "$BOT"
