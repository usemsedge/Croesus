#!/bin/bash

# Run Flask app with proper DYLD_LIBRARY_PATH for ffmpeg@7 and torch libraries
# This ensures torchcodec can find native dependencies on macOS

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set DYLD_LIBRARY_PATH for ffmpeg and torch
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib:/opt/homebrew/lib:${SCRIPT_DIR}/.venv/lib/python3.12/site-packages/torch/lib"

# Activate virtual environment
source "${SCRIPT_DIR}/.venv/bin/activate"

# Run the Flask app
python "${SCRIPT_DIR}/app.py"
