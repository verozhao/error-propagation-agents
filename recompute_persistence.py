"""Recompute persistence curves in bulk using batched encoding."""
import json
import sys
import numpy as np
from collections import defaultdict
from persistence import find_matched_baseline
from severity import get_encoder

path = "results/ragtruth_weighted_error/ragtruth_weighted_sev1_llama-3.1-8b_15trials.jsonl"
encoder_name = "BAAI/bge-large-en-v1.5"

print("Loading records...", flush=True)
records = [json.loads(l) for l in open(path) if l.strip()]
baselines = [r for r in records if r.get("is_baseline") and "step_outputs" in r]
injected = [r for r in records if not r.get("is_baseline") and r.get("injection_valid")]
print(f"Total: {len(records)}, Baselines: {len(baselines)}, Injected: {len(injected)}", flush=True)

print("Loading encoder...", flush=True)
enc = get_encoder(encoder_name)

# Build all (record_idx, step_idx, delta, injected_text, baseline_text) tuples
jobs = []
for ri, r in enumerate(injected):
    bl = find_matched_baseline(r["task_query"], baselines)
    if not bl:
        continue
    inject_idx = r.get("error_step")
    if inject_idx is None:
        continue

    trial_steps = r.get("step_outputs", [])
    baseline_steps = bl.get("step_outputs", [])

    # Build delta from actual corrupted output
    if inject_idx < len(trial_steps):
        step = trial_steps[inject_idx]
        pre = step.get("pre_injection_output", "")
        post = step.get("output_text", "")
        delta = post if (pre and post and pre != post) else r.get("injected_content", "")
    else:
        delta = r.get("injected_content", "")

    if not delta:
        continue

    for i, step in enumerate(trial_steps):
        if i <= inject_idx:
            continue
        if i >= len(baseline_steps):
            break
        jobs.append((ri, i, step.get("step", f"step_{i}"), delta,
                      step.get("output_text", ""),
                      baseline_steps[i].get("output_text", "")))

print(f"Total encode jobs: {len(jobs)}", flush=True)

# Batch-encode all unique texts
all_texts = []
text_to_idx = {}
for _, _, _, delta, inj_text, bl_text in jobs:
    for t in (delta, inj_text, bl_text):
        if t not in text_to_idx:
            text_to_idx[t] = len(all_texts)
            all_texts.append(t)

print(f"Unique texts to encode: {len(all_texts)}", flush=True)

batch_size = 128
all_embs = []
for start in range(0, len(all_texts), batch_size):
    batch = all_texts[start:start + batch_size]
    embs = enc.encode(batch, normalize_embeddings=True, show_progress_bar=False)
    all_embs.append(embs)
    done = min(start + batch_size, len(all_texts))
    print(f"  Encoded {done}/{len(all_texts)} texts ({100*done/len(all_texts):.0f}%)", flush=True)

all_embs = np.vstack(all_embs)
print(f"Embedding matrix: {all_embs.shape}", flush=True)

# Compute persistence from cached embeddings
curves = defaultdict(list)  # ri -> [(step_idx, step_name, score)]
for ri, step_i, step_name, delta, inj_text, bl_text in jobs:
    d_idx = text_to_idx[delta]
    i_idx = text_to_idx[inj_text]
    b_idx = text_to_idx[bl_text]
    sim_inj = float(np.dot(all_embs[d_idx], all_embs[i_idx]))
    sim_bl = float(np.dot(all_embs[d_idx], all_embs[b_idx]))
    p = round(max(0.0, sim_inj - sim_bl), 6)
    curves[ri].append((step_i, step_name, p))

# Write back
n_updated = 0
for ri, curve in curves.items():
    injected[ri]["persistence_curve"] = sorted(curve, key=lambda x: x[0])
    n_updated += 1

print(f"\nUpdated {n_updated} records", flush=True)

with open(path, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"Wrote {len(records)} records to {path}", flush=True)

# Stats
all_vals = [v[2] for c in curves.values() for v in c]
print(f"\nPersistence stats ({len(curves)} curves, {len(all_vals)} points):")
print(f"  mean:   {np.mean(all_vals):.4f}")
print(f"  median: {np.median(all_vals):.4f}")
print(f"  p75:    {np.percentile(all_vals, 75):.4f}")
print(f"  p95:    {np.percentile(all_vals, 95):.4f}")
print(f"  max:    {np.max(all_vals):.4f}")
print(f"  >0:     {sum(1 for v in all_vals if v > 0)}/{len(all_vals)} ({100*sum(1 for v in all_vals if v > 0)/len(all_vals):.1f}%)")

by_step = defaultdict(list)
for c in curves.values():
    for _, name, val in c:
        by_step[name].append(val)
print("\nPer-step:")
for name in ["filter", "summarize", "compose", "verify"]:
    vals = by_step.get(name, [])
    if vals:
        print(f"  {name:10s}  n={len(vals):5d}  mean={np.mean(vals):.4f}  median={np.median(vals):.4f}  >0={sum(1 for v in vals if v > 0)}/{len(vals)}")
