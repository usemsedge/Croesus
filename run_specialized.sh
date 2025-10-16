#!/usr/bin/env bash
set -euo pipefail

# run_specialized.sh - wrapper to run specialized_models with the correct library path and venv
# Usage: ./run_specialized.sh [args]

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
  source "$VENV_DIR/bin/activate"
fi

# Ensure ffmpeg@7 libs are visible to torchcodec; adjust if you installed ffmpeg to a different location
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib:/opt/homebrew/lib:$ROOT_DIR/.venv/lib/python3.12/site-packages/torch/lib:${DYLD_LIBRARY_PATH:-}"

python "$ROOT_DIR/specialized_models.py" "$@"
