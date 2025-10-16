#!/usr/bin/env bash
set -euo pipefail

# setup.sh - create venv and install python deps needed by the project
# Usage: ./setup.sh

echo "Setting up Croesus project..."

PYTHON=python3
VENV_DIR=.venv

if ! command -v $PYTHON >/dev/null 2>&1; then
  echo "Error: python3 not found. Install Python 3 and try again." >&2
  exit 1
fi

echo "Creating virtual environment in ${VENV_DIR}..."
$PYTHON -m venv ${VENV_DIR}
source ${VENV_DIR}/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

if [ -f requirements.txt ]; then
  echo "Installing pip requirements..."
  pip install -r requirements.txt
fi

echo "Installing recommended optional packages (soundfile, librosa) for audio fallback..."
pip install soundfile librosa numpy || true

echo "(Optional) If you plan to use torchcodec for fast decoding, install it now:
  pip install --upgrade --force-reinstall torchcodec
Note: torchcodec requires a matching FFmpeg and PyTorch build. See README.md for details."

echo "Setup complete. To run the project, either use ./run_specialized.sh or set DYLD_LIBRARY_PATH as explained in README.md"
