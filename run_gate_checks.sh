#!/bin/bash
echo "========================================"
echo "Sweep finished at $(date)"
echo "========================================"
echo ""

echo "=== GATE CHECKS ==="
cd /Users/test/error-propagation-agents
python check_pilot_gates.py results/ragtruth_weighted_error/ragtruth_weighted_sev1_llama-3.1-8b_15trials.jsonl
echo ""

echo "=== CAUSAL MEDIATION ==="
python -c "
import json
from causal_mediation import compute_mediation

records = [json.loads(l) for l in open('results/ragtruth_weighted_error/ragtruth_weighted_sev1_llama-3.1-8b_15trials.jsonl') if l.strip()]
baselines = [r for r in records if r.get('is_baseline') and 'evaluation' in r]
injected = [r for r in records if not r.get('is_baseline') and 'evaluation' in r]

result = compute_mediation(injected, baselines)
if 'error' in result:
    print(f'Error: {result[\"error\"]}')
else:
    print(f'Total Effect: {result[\"total_effect\"]:.3f}  CI: {result.get(\"total_effect_ci_95\", \"?\")}')
    print(f'NIE:          {result[\"nie\"]:.3f}  CI: {result.get(\"nie_ci_95\", \"?\")}')
    print(f'NDE:          {result[\"nde\"]:.3f}  CI: {result.get(\"nde_ci_95\", \"?\")}')
    print(f'Mediation fraction (NIE/TE): {result[\"mediation_fraction_nie_over_te\"]:.3f}  CI: {result.get(\"mediation_fraction_ci_95\", \"?\")}')
    print(f'Regression p-value: {result[\"persistence_failure_regression\"][\"p_value\"]:.4f}')
"
