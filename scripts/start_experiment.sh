#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
    echo "Usage:"
    echo "./scripts/start_experiment.sh experiments/my_experiment.py [args...]"
    exit 1
fi

EXPERIMENT="$1"; shift

if [[ ! -f "$EXPERIMENT" ]]; then
    echo "Experiment not found: $EXPERIMENT" >&2
    exit 2
fi

SESSION="mech_$(date +%Y%m%d_%H%M%S)"
LOG="results/logs/${SESSION}.log"

mkdir -p results/logs

printf -v PYTHON_COMMAND '%q ' python "$EXPERIMENT" "$@"
printf -v LOGGED_COMMAND \
    'set -o pipefail; source .venv/bin/activate; %s 2>&1 | tee %q; status=${PIPESTATUS[0]}; printf "%%s\\n" "$status" > %q; exit "$status"' \
    "$PYTHON_COMMAND" "$LOG" "${LOG}.exitcode"

echo "Starting:"
printf '%s\n' "$PYTHON_COMMAND"

tmux new-session -d -s "$SESSION" bash -lc "$LOGGED_COMMAND"

echo
echo "Experiment started."
echo "Session: $SESSION"
echo "Log:     $LOG"
echo
echo "Detached session started."
