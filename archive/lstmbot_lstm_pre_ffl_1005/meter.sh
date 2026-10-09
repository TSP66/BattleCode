#!/usr/bin/env bash
# Generates weights for one config, builds as the judge does, zips, meters.
#   ./meter.sh --c1 64 --c2 128 --hidden 128
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$HERE")"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
/usr/bin/python3 "$HERE/gen_weights.py" "$@"
(cd "$HERE" && zip -q -9 "$OUT/bot.zip" main.cpp helper.hpp obs.hpp lstmnet.hpp weights.hpp bot.toml)
sz=$(stat -c %s "$OUT/bot.zip")
echo "zip $sz bytes = $(python3 -c "print(f'{$sz/2**20:.2f}')") MiB of 4.00"
"$ROOT/.venv-zig/bin/python" -m ziglang c++ -target wasm32-wasi -O2 -msimd128 -std=c++20 -w \
    "$HERE/main.cpp" "$ROOT/wasmprobe/stubs.cpp" -o "$OUT/bot.wasm" > "$OUT/build.log" 2>&1 \
    || { grep -E "error" "$OUT/build.log" | head -20; exit 1; }
"$HOME/.local/share/uv/tools/unswbc/bin/python" "$ROOT/wasmprobe/runbot.py" \
    "$OUT/bot.wasm" "$ROOT/wasmprobe/transcript.txt" "$HERE"
