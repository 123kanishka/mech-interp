#!/usr/bin/env bash

echo "=== TMUX SESSIONS ==="

tmux ls 2>/dev/null || echo "No tmux sessions."

echo
echo "=== GPU ==="

nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv 2>/dev/null || true

echo
echo "=== RECENT LOGS ==="

ls -lht results/logs 2>/dev/null | head
