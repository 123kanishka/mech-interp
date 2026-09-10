#!/usr/bin/env bash
set -u

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

echo
echo "=== RECENT RUN MARKERS ==="
find results/runs -maxdepth 2 -type f \
    \( -name SUCCESS -o -name FAILED -o -name manifest.json \) \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -20 | cut -d' ' -f2-

echo
echo "=== DISK ==="
df -h .
