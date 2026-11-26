#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
"$PYTHON_BIN" -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 12), "Use Python 3.11 or 3.12"'
"$PYTHON_BIN" -m venv .venv
if [[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]]; then
  .venv/bin/python -m pip install 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
