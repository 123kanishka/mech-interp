#!/usr/bin/env bash
set -e

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip

echo "Environment created."
echo "Now install project dependencies from requirements.txt."
