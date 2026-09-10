#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Create virtual environment if it does not exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py --local

echo "Local environment ready."
