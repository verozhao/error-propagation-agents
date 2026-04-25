#!/usr/bin/env bash
# Phase 7+ sweep: FAVA taxonomy (entity/invented/unverifiable/contradictory).
#
# Single-step sweep: 4 error types x 4 injection steps x 3 severities x queries x trials
#   Baselines run once per error type at sev=1; sev=2/3 reuse via --skip-baseline.
#
# Compound sweep: 5 step-pairs x 4 error types x queries x trials at sev=2.
#
# Usage:
#   ./run_severity_sweeps.sh              # single-step only, 30 trials
#   ./run_severity_sweeps.sh --trials 10  # fewer trials (pilot)
#   ./run_severity_sweeps.sh --full       # single-step + compound sweep
#   ./run_severity_sweeps.sh --compound-only  # compound sweep only
set -euo pipefail

TRIALS=15
COMPOUND_TRIALS=10
QUERIES=12
MODELS="claude-3-haiku"
RUN_SINGLE=true
RUN_COMPOUND=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --trials) TRIALS="$2"; shift 2 ;;
    --compound-trials) COMPOUND_TRIALS="$2"; shift 2 ;;
    --queries) QUERIES="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --full) RUN_COMPOUND=true; shift ;;
    --compound-only) RUN_SINGLE=false; RUN_COMPOUND=true; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "Config: models=[$MODELS] trials=$TRIALS queries=$QUERIES compound_trials=$COMPOUND_TRIALS"
echo "  single-step=$RUN_SINGLE compound=$RUN_COMPOUND"

if $RUN_SINGLE; then
  for etype in entity invented unverifiable contradictory; do
    echo ""
    echo "=== $etype error sweeps ==="

    # Severity 1: includes baseline (error_step=None) runs
    echo "--- $etype severity 1 (with baselines) ---"
    python run.py --mode run --use-api --trials "$TRIALS" \
      --models $MODELS --error-type "$etype" --severity 1 \
      --queries "$QUERIES"

    # Severity 2 & 3: skip baseline — reuse sev=1 baselines (same seed)
    for sev in 2 3; do
      echo "--- $etype severity $sev (skip baseline) ---"
      python run.py --mode run --use-api --trials "$TRIALS" \
        --models $MODELS --error-type "$etype" --severity "$sev" \
        --skip-baseline --queries "$QUERIES"
    done
  done

  echo ""
  echo "Single-step sweeps complete."
fi

if $RUN_COMPOUND; then
  echo ""
  echo "=== Compound injection sweeps (severity 2) ==="

  for etype in entity invented unverifiable contradictory; do
    echo "--- compound $etype ---"
    python run.py --mode run --use-api --trials "$COMPOUND_TRIALS" \
      --models $MODELS --error-type "$etype" --severity 2 \
      --compound-steps "0,1;0,2;0,3;1,2;1,3" --queries "$QUERIES"
  done

  echo ""
  echo "Compound sweeps complete."
fi

echo ""
echo "All sweeps complete."
