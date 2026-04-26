"""Standalone persistence curve computation with progress bar.

Reads the JSONL, computes curves, writes consolidated JSON.
Does NOT touch the JSONL to avoid conflict with the running experiment process.
"""
import json
import os
from tqdm import tqdm
from persistence import persistence_curve

JSONL_PATH = "results/ragtruth_weighted_error/ragtruth_weighted_sev1_llama-3.1-8b_15trials.jsonl"
OUTPUT_JSON = "results/ragtruth_weighted_error/ragtruth_weighted_sev1_llama-3.1-8b_15trials_consolidated.json"

records = []
with open(JSONL_PATH) as fh:
    for line in fh:
        if line.strip():
            records.append(json.loads(line))

print(f"Loaded {len(records)} records from JSONL")

baselines = [r for r in records if r.get("is_baseline")]
print(f"Baselines: {len(baselines)}")

def find_matched_baseline(query, baselines):
    for b in baselines:
        if b["task_query"] == query:
            return b
    return None

n_need = sum(
    1 for r in records
    if not r.get("is_baseline")
    and r.get("error_step") is not None
    and r.get("injected_content")
    and r.get("step_outputs")
)
print(f"Records needing curves: {n_need}")

n_computed = 0
for r in tqdm(records, desc="Computing persistence curves"):
    if r.get("is_baseline") or r.get("error_step") is None:
        continue
    if not r.get("injected_content") or not r.get("step_outputs"):
        continue
    matched = find_matched_baseline(r["task_query"], baselines)
    if matched and matched.get("step_outputs"):
        curve = persistence_curve(r, matched)
        r["persistence_curve"] = curve
        n_computed += 1

print(f"Computed {n_computed} persistence curves")

with open(OUTPUT_JSON, "w") as f:
    json.dump(records, f, indent=2)
print(f"Wrote consolidated JSON: {OUTPUT_JSON}")
