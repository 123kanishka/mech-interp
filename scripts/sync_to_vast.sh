#!/usr/bin/env bash
set -euo pipefail

HOST="${1:-vast-mech}"
REMOTE_DIR="${2:-/workspace/mech-interp}"

echo "Syncing project to ${HOST}:${REMOTE_DIR}"

ssh "$HOST" "mkdir -p '$REMOTE_DIR'"

rsync -avz --progress \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.env' \
    --exclude '.cache/' \
    --exclude '__pycache__/' \
    --exclude 'activations/' \
    --exclude 'results/logs/' \
    --exclude 'results/data/' \
    ./ "$HOST:$REMOTE_DIR/"

echo "Upload complete."
