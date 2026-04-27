#!/bin/bash
# Overnight analysis pipeline — Phase 2 + Phase 3
# No API calls needed. All local compute.
# Expected total time: ~5-6 hours (hierarchical fit dominates)
set -e
cd /Users/test/error-propagation-agents

echo "=========================================="
echo "OVERNIGHT PIPELINE — $(date)"
echo "=========================================="

# Step 1: Compute persistence curves for any files missing them (~5-10 min each)
echo ""
echo ">>> Step 1: Compute persistence curves for new files"
for f in results/ragtruth_weighted_error/*.jsonl; do
    has_curve=$(python3 -c "
import json
with open('$f') as fh:
    for i, line in enumerate(fh):
        r = json.loads(line)
        if r.get('persistence_curve'): print('yes'); break
        if i > 20: print('no'); break
    else: print('no')
" 2>/dev/null)
    if [ "$has_curve" = "no" ]; then
        echo "  Computing curves for $(basename $f)..."
        python compute_curves_batched.py "$f"
    else
        echo "  Skipping $(basename $f) — already has curves"
    fi
done

# Step 2: Gate checks on all main files
echo ""
echo ">>> Step 2: Gate checks"
for f in \
    results/ragtruth_weighted_error/ragtruth_weighted_sev1_claude-haiku-3_15trials.jsonl \
    results/ragtruth_weighted_error/ragtruth_weighted_sev1_claude-sonnet-3-7_5trials.jsonl \
    results/ragtruth_weighted_error/ragtruth_weighted_sev1_claude-sonnet-4_5trials.jsonl \
    results/ragtruth_weighted_error/ragtruth_weighted_sev1_llama-3.1-8b_15trials.jsonl; do
    if [ -f "$f" ]; then
        echo "  Gate check: $(basename $f)"
        python check_pilot_gates.py "$f" || true
    fi
done

# Step 3: Statistical tests (generates CSVs for figures/tables)
echo ""
echo ">>> Step 3: Statistical tests"
python statistical_tests.py

# Step 4: Causal mediation (re-run with all data)
echo ""
echo ">>> Step 4: Causal mediation"
python -c "
from causal_mediation import compute_mediation_per_model
import json, glob
records = []
for f in glob.glob('results/ragtruth_weighted_error/ragtruth_weighted_sev1_*.jsonl'):
    if '_legacy' in f or '_failed' in f: continue
    with open(f) as fh:
        records.extend(json.loads(l) for l in fh if l.strip())
print(f'Loaded {len(records)} records for mediation')
result = compute_mediation_per_model(records)
with open('results/stats/mediation_main.json', 'w') as f:
    json.dump(result, f, indent=2, default=str)
print('Saved mediation results')
"

# Step 5: Decay analysis (headline finding)
echo ""
echo ">>> Step 5: Decay analysis"
python decay_analysis.py

# Step 6: Multi-encoder validation (~20-30 min for encoding)
echo ""
echo ">>> Step 6: Multi-encoder validation"
python multi_encoder_validation.py

# Step 7: Theorem identifiability verification
echo ""
echo ">>> Step 7: Theorem identifiability"
python theorem_identifiability.py

# Step 8: Hierarchical model refit with ALL data (~3.5 hours)
echo ""
echo ">>> Step 8: Hierarchical model refit (this takes ~3.5 hours)"
python -c "
from hierarchical_model import fit_hierarchical, prepare_data, extract_posteriors, rank1_factorization_test
import json, glob
records = []
for f in glob.glob('results/ragtruth_weighted_error/*.jsonl'):
    if '_legacy' in f or '_failed' in f: continue
    with open(f) as fh:
        records.extend(json.loads(l) for l in fh if l.strip())
print(f'Loaded {len(records)} records for hierarchical fit')
data = prepare_data(records)
samples = fit_hierarchical(data, num_samples=2000)
posteriors = extract_posteriors(samples, data)
rank1 = rank1_factorization_test(samples, data)
posteriors['rank1_factorization'] = rank1
with open('results/stats/hierarchical_posteriors.json', 'w') as f:
    json.dump(posteriors, f, indent=2, default=str)
print('Saved to results/stats/hierarchical_posteriors.json')
" 2>&1 | tee results/stats/hierarchical_fit_log_v2.txt

# Step 9: Generate figures and tables
echo ""
echo ">>> Step 9: Generate figures and tables"
python generate_paper_figures.py
python generate_paper_tables.py

echo ""
echo "=========================================="
echo "OVERNIGHT PIPELINE COMPLETE — $(date)"
echo "=========================================="
