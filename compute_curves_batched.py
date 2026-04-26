"""Batched persistence curve computation — ~10x faster than per-record encoding."""
import json
import numpy as np
from tqdm import tqdm
from severity import get_encoder

# JSONL_PATH = "results/ragtruth_weighted_error/ragtruth_weighted_sev1_claude-haiku-3_15trials.jsonl"
# OUTPUT_JSON = "results/ragtruth_weighted_error/ragtruth_weighted_sev1_claude-haiku-3_15trials_consolidated.json"

JSONL_PATH = "results/ragtruth_weighted_error/ragtruth_weighted_sev1_claude-sonnet-3-7_5trials.jsonl"
OUTPUT_JSON = "results/ragtruth_weighted_error/ragtruth_weighted_sev1_claude-sonnet-3-7_5trials_consolidated.json"

records = []
with open(JSONL_PATH) as fh:
    for line in fh:
        if line.strip():
            records.append(json.loads(line))
print(f"Loaded {len(records)} records")

baselines_by_query = {}
for r in records:
    if r.get("is_baseline"):
        baselines_by_query[r["task_query"]] = r

# Phase 1: collect all texts that need encoding
print("Phase 1: Collecting texts...")
text_set = {}  # text -> index
jobs = []  # (record_idx, step_idx, step_name, delta_text_id, step_text_id, baseline_text_id)

def get_text_id(text):
    if text not in text_set:
        text_set[text] = len(text_set)
    return text_set[text]

for ri, r in enumerate(records):
    if r.get("is_baseline") or r.get("error_step") is None:
        continue
    if not r.get("injected_content") or not r.get("step_outputs"):
        continue
    matched = baselines_by_query.get(r["task_query"])
    if not matched or not matched.get("step_outputs"):
        continue

    delta = r["injected_content"]
    inject_idx = r["error_step"]
    trial_steps = r.get("step_outputs", [])
    baseline_steps = matched.get("step_outputs", [])

    for i, step in enumerate(trial_steps):
        if i <= inject_idx:
            continue
        if i >= len(baseline_steps):
            break
        step_out = step.get("output_text", "")
        base_out = baseline_steps[i].get("output_text", "")
        if not delta or not step_out or not base_out:
            continue
        jobs.append((
            ri, i, step.get("step", f"step_{i}"),
            get_text_id(delta), get_text_id(step_out), get_text_id(base_out),
        ))

print(f"Unique texts to encode: {len(text_set)}")
print(f"Similarity jobs: {len(jobs)}")

# Phase 2: batch encode all texts
print("Phase 2: Batch encoding...")
texts_list = [""] * len(text_set)
for text, idx in text_set.items():
    texts_list[idx] = text

enc = get_encoder("BAAI/bge-large-en-v1.5")
all_embeddings = enc.encode(texts_list, normalize_embeddings=True, batch_size=128, show_progress_bar=True)
print(f"Encoded {len(all_embeddings)} texts")

# Phase 3: compute persistence scores via dot products
print("Phase 3: Computing persistence scores...")
curves = {}  # record_idx -> list of (step_idx, step_name, score)
for ri, step_i, step_name, did, sid, bid in tqdm(jobs, desc="Dot products"):
    sim_inj = float(np.dot(all_embeddings[did], all_embeddings[sid]))
    sim_base = float(np.dot(all_embeddings[did], all_embeddings[bid]))
    score = round(max(0.0, sim_inj - sim_base), 6)
    curves.setdefault(ri, []).append((step_i, step_name, score))

n_computed = 0
for ri, curve in curves.items():
    curve.sort(key=lambda x: x[0])
    records[ri]["persistence_curve"] = curve
    n_computed += 1

print(f"Computed persistence curves for {n_computed} records")

# Phase 4: write output
with open(OUTPUT_JSON, "w") as f:
    json.dump(records, f, indent=2)
print(f"Wrote consolidated JSON: {OUTPUT_JSON}")

# Also rewrite the JSONL atomically
import os
tmp_path = JSONL_PATH + ".tmp2"
with open(tmp_path, "w") as fh:
    for r in records:
        fh.write(json.dumps(r) + "\n")
os.replace(tmp_path, JSONL_PATH)
print(f"Rewrote JSONL: {JSONL_PATH}")
print("DONE")
