#!/usr/bin/env bash
# Severity sweep experiments across all error types.
# Factual & semantic: severity 1-4
# Omission: severity 1-3 (no level 4 defined)
#
# Usage:
#   ./run_severity_sweeps.sh                  # lean defaults: 1 model, 10 trials
#   ./run_severity_sweeps.sh --trials 30      # more trials
#   ./run_severity_sweeps.sh --full           # all 3 models, 30 trials
set -euo pipefail

TRIALS=10
MODELS="gpt-4o-mini"

while [[ $# -gt 0 ]]; do
  case $1 in
    --trials) TRIALS="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --full)   MODELS="gpt-4o-mini claude-haiku gemini-flash"; TRIALS=30; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "Config: models=[$MODELS] trials=$TRIALS"

echo "=== Factual error sweeps ==="
for sev in 1 2 3 4; do
  echo "--- factual severity $sev ---"
  python run.py --mode run --use-api --trials "$TRIALS" --models $MODELS --error-type factual --severity "$sev"
done

echo "=== Semantic error sweeps ==="
for sev in 1 2 3 4; do
  echo "--- semantic severity $sev ---"
  python run.py --mode run --use-api --trials "$TRIALS" --models $MODELS --error-type semantic --severity "$sev"
done

echo "=== Omission error sweeps ==="
for sev in 1 2 3; do
  echo "--- omission severity $sev ---"
  python run.py --mode run --use-api --trials "$TRIALS" --models $MODELS --error-type omission --severity "$sev"
done

echo "All severity sweeps complete."
