#!/usr/bin/env bash
set -euo pipefail
# 1 model x 1 query x 4 severities x 4 injection steps x 5 trials = ~200 trials + baselines
export DIAGNOSTIC_QUERY="best noise-canceling headphones 2025"
for etype in factual omission semantic; do
  for sev in 1 2 3 4; do
    echo "=== ${etype} severity ${sev} ==="
    python run.py --mode run --use-api --trials 5 \
      --models gpt-4o-mini --error-type "$etype" --severity "$sev" \
      --diagnostic-query "$DIAGNOSTIC_QUERY"
  done
done
