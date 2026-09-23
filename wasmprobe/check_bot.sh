#!/usr/bin/env bash
# Pre-submission gate for the C++ bot. Run it before every upload:
#
#     wasmprobe/check_bot.sh runs/v2/snapshots/turns_X.pt   # the checkpoint embedded
#
# Fails (exit 1) unless all of these hold:
#   1. the zip is under the 4MB limit and ships no weights.bin;
#   2. built for wasm with the judge's flags and run with NO filesystem, as the
#      judge runs it, the network loads ("net 1 (ok)") and drives every turn
#      but the first, and no turn comes within 10% of the 100M point limit;
#   3. the bot's observation equals the training simulator's on every map
#      (parity_obs.py);
#   4. the remembered inputs it computes as it plays equal the trainer's
#      (parity_mem.py: mem and memfar against clone_features.MemoryTracker);
#   5. the bot's logits and moves are the checkpoint's (parity_net.py);
#   6. a full game in the real engine runs with no warnings, and every turn
#      after a dragon's first is played by the network.
# v3 and v4 were uploaded with a network that never ran on the judge; 2 and 6
# are the checks that would have caught it.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOT="$ROOT/mybot"
CKPT="${1:?usage: check_bot.sh path/to/checkpoint.pt}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
PY="$ROOT/.venv-train/bin/python"
fail=0
say() { printf '%-58s %s\n' "$1" "$2"; }
bad() { say "$1" "FAIL"; fail=1; }
# macOS ships neither timeout nor gtimeout unless coreutils is installed
if command -v timeout > /dev/null; then limit() { timeout "$@"; }
elif command -v gtimeout > /dev/null; then limit() { gtimeout "$@"; }
else limit() { shift; "$@"; }
fi

echo "embedded: $(grep -m1 'SOURCE\[\]' "$BOT/weights_data.hpp" | cut -d'"' -f2)"
echo "checking: $(basename "$CKPT")"
grep -q "\"$(basename "$CKPT")" "$BOT/weights_data.hpp" || \
    echo "  warning: weights_data.hpp was not exported from this checkpoint"

# 1. package
(cd "$BOT" && zip -q -9 "$TMP/bot.zip" *.cpp *.hpp bot.toml)
size=$(stat -c %s "$TMP/bot.zip" 2>/dev/null || stat -f %z "$TMP/bot.zip")
if [ "$size" -lt 3900000 ] && ! grep -q "weights.bin" "$BOT/bot.toml"; then
    say "1. zip $((size / 1000)) KB, no data files" ok
else bad "1. zip $((size / 1000)) KB / bot.toml includes weights.bin"; fi

# 2. wasm, metered, no filesystem
if BOT_CFLAGS=-DBC_LOG OUT_FILE="$TMP/wasm_out.txt" NO_FS=1 "$ROOT/wasmprobe/meter_bot.sh" \
        > "$TMP/meter.txt" 2>&1; then
    max=$(awk '/^ *[0-9]+ +[0-9,]+ /{gsub(",","",$2); if ($2>m) m=$2} END{print m+0}' "$TMP/meter.txt")
    nets=$(grep -c "^INDICATOR net" "$TMP/wasm_out.txt")
    turns=$(grep -c "^ENDTURN" "$TMP/wasm_out.txt")
    if grep -q "net 1 (ok)" "$TMP/wasm_out.txt" && [ "$nets" -eq $((turns - 1)) ] \
            && [ "$max" -lt 90000000 ]; then
        say "2. wasm, no fs: net ran $nets/$((turns - 1)) turns, max $((max / 1000000))M pts" ok
    else
        bad "2. wasm, no fs: net ran $nets/$((turns - 1)), max $((max / 1000000))M pts"
        grep "^LOG" "$TMP/wasm_out.txt" | head -2
    fi
else bad "2. wasm build or run"; tail -5 "$TMP/meter.txt"; fi

# 3 + 4 + 5. native debug build against the simulator, the tracker and the checkpoint
if g++ -O2 -std=c++20 -DBC_DUMP "$BOT/main.cpp" -o "$TMP/bot_native" 2> "$TMP/gcc.txt"; then
    mkdir -p "$TMP/dumps"
    if DUMP_DIR="$TMP/dumps" "$PY" "$ROOT/wasmprobe/parity_obs.py" "$TMP/bot_native" \
            > "$TMP/obs.txt" 2>&1; then
        say "3. observation == simulator: $(tail -1 "$TMP/obs.txt" | cut -d'(' -f1)" ok
    else bad "3. observation parity"; grep -A3 "differ$" "$TMP/obs.txt" | grep -v " 0 differ" | head; tail -1 "$TMP/obs.txt"; fi
    if "$PY" "$ROOT/wasmprobe/parity_mem.py" "$TMP"/dumps/*.txt > "$TMP/mem.txt" 2>&1; then
        say "4. remembered inputs == MemoryTracker: $(tail -2 "$TMP/mem.txt" | tr -d '\n' | cut -d'(' -f1)" ok
    else bad "4. remembered input parity"; grep -v " 0 differ" "$TMP/mem.txt" | tail -6; fi
    if "$PY" "$ROOT/wasmprobe/parity_net.py" "$CKPT" "$TMP"/dumps/*.txt > "$TMP/net.txt" 2>&1; then
        say "5. forward == checkpoint: $(cut -d: -f1 "$TMP/net.txt")" ok
    else bad "5. forward parity"; cat "$TMP/net.txt" | tail -3; fi
else bad "3/4/5. native build"; head "$TMP/gcc.txt"; fi

# 6. the real engine, full length
export PATH="$ROOT/.venv/bin:$HOME/.local/bin:$PATH"
(cd "$ROOT" && limit 900 unswbc run maps/queen_of_spades.map mybot mybot -v --no-replay) \
    > "$TMP/game.txt" 2>&1
first=$(grep -c "INDICATOR first-turn" "$TMP/game.txt")
netg=$(grep -c "INDICATOR net" "$TMP/game.txt")
other=$(grep -cE "INDICATOR (budget|NO NET)" "$TMP/game.txt")
warn=$(grep -ciE "warn|error|invalid" "$TMP/game.txt")
result=$(grep -E "wins|draw" "$TMP/game.txt" | tail -1)
if [ "$netg" -gt 0 ] && [ "$other" -eq 0 ] && [ "$warn" -eq 0 ] && [ -n "$result" ]; then
    say "6. real engine: $netg net turns, $first first turns, 0 fallbacks" ok
else bad "6. real engine: net $netg, fallback $other, warnings $warn, '$result'"; fi

echo
[ "$fail" -eq 0 ] && echo "ALL CHECKS PASSED" || echo "CHECKS FAILED: do not submit"
exit "$fail"
