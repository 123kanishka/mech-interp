#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage:"
    echo "./scripts/start_experiment.sh experiments/my_experiment.py [args...]"
    exit 1
fi

EXPERIMENT="$1"
shift

SESSION="mech_$(date +%Y%m%d_%H%M%S)"
LOG="results/logs/${SESSION}.log"

mkdir -p results/logs

COMMAND="source .venv/bin/activate && python $EXPERIMENT $* 2>&1 | tee $LOG"

echo "Starting:"
echo "$COMMAND"

tmux new-session -d -s "$SESSION" "$COMMAND"

echo
echo "Experiment started."
echo "Session: $SESSION"
echo "Log:     $LOG"
echo
echo "Laptop may now disconnect."
