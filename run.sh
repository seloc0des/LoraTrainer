#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script: creates venv, installs deps, runs the CLI
# Usage: bash run.sh [--no-cache]

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="python3"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "python3 not found on PATH" >&2
  exit 1
fi

# Create venv if missing
if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Activate venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Upgrade pip/setuptools/wheel for smoother installs
python -m pip install --upgrade pip setuptools wheel

# Optional: --no-cache to avoid pip caches on CI or low-disk systems
PIP_ARGS=()
if [ "${1:-}" = "--no-cache" ]; then
  PIP_ARGS+=("--no-cache-dir")
fi

# Install project requirements
pip install "${PIP_ARGS[@]}" -r "$PROJECT_DIR/requirements.txt"

# Run the CLI
python "$PROJECT_DIR/train_lora.py"
