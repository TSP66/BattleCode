#!/usr/bin/env bash
# The one way to upload the C++ bot:
#
#     wasmprobe/submit.sh runs/v2/snapshots/turns_X.pt "short description"
#
#   1. exports the checkpoint into mybot/weights_data.hpp;
#   2. runs wasmprobe/check_bot.sh and stops unless every check passes;
#   3. uploads mybot with unswbc submit;
#   4. records the upload in runs/submitted/submitted.jsonl, with a copy of the
#      checkpoint, which makes it the "submitted vN" opponent that
#      train.yardstick plays every new snapshot against from then on.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT="$(cd "$(dirname "${1:?usage: submit.sh ckpt.pt \"description\"}")" && pwd)/$(basename "$1")"
DESC="${2:?give a short description of what changed}"
PY="$ROOT/.venv-train/bin/python"
SUB="$ROOT/runs/submitted"
mkdir -p "$SUB"

echo "== export"
(cd "$ROOT/bcsim" && "$PY" -m train.export_cpp --ckpt "$CKPT" | tail -2)

echo "== checks"
"$ROOT/wasmprobe/check_bot.sh" "$CKPT"

echo "== upload"
NAME="$(basename "$(dirname "$(dirname "$CKPT")")")-$(basename "$CKPT" .pt)"
OUT="$(unswbc submit "$ROOT/mybot" -n "$NAME" -d "$DESC" 2>&1)"
echo "$OUT"
VERSION="$(echo "$OUT" | sed -n 's/.* as \(v[0-9]\+\).*/\1/p')"
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
                         "source": src, "time": time.strftime("%Y-%m-%dT%H:%M")}) + "\n")
EOF
echo "recorded $VERSION: evaluations now play it as \"submitted $VERSION\""
