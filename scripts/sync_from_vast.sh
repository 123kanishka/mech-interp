#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-vast-mech}"
REMOTE_DIR="${2:-/workspace/mech-interp}"

mkdir -p results

echo "Downloading results from Vast..."

rsync -avz --progress \
    "$HOST:$REMOTE_DIR/results/" \
    ./results/

echo "Results downloaded."
