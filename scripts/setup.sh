#!/usr/bin/env bash
# One-time setup on a fresh machine: training venv + simulator build.
#
#     scripts/setup.sh                 # CUDA 12.8 wheels (RTX 50xx needs >= 12.8)
#     TORCH_INDEX=https://download.pytorch.org/whl/cu126 scripts/setup.sh
#
# Needs: uv, g++ (C++17), an NVIDIA driver.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

uv venv --python 3.13 "$ROOT/.venv-train"
uv pip install --python "$ROOT/.venv-train/bin/python" \
    --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple \
    -r "$ROOT/requirements.txt"
make -C "$ROOT/bcsim"

"$ROOT/.venv-train/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
(cd "$ROOT/bcsim" && "$ROOT/.venv-train/bin/python" tests/test_ego.py)
