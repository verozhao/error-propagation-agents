"""Phase 2A: Multi-encoder persistence validation.

Recomputes persistence for a random subset using BGE, E5, and mpnet,
then reports pairwise Spearman correlation. Target: rho > 0.85.
"""
import json
import glob
import numpy as np
from scipy.stats import spearmanr
from record_utils import is_baseline
from severity import get_encoder

ENCODERS = [
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-large-v2",
    "sentence-transformers/all-mpnet-base-v2",
]
N_SAMPLE = 500
SEED = 42


def load_injected_with_curves(max_records=None):
    records = []
    for path in glob.glob("results/ragtruth_weighted_error/ragtruth_weighted_sev1_llama-3.1-8b_15trials.jsonl"):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if is_baseline(r) or not r.get("persistence_curve"):
                    continue
                if not r.get("injected_content") or not r.get("step_outputs"):
                    continue
                records.append(r)
                if max_records and len(records) >= max_records * 2:
                    break
    return records


def compute_persistence_score(delta_text, step_text, baseline_text, encoder):
    embs = encoder.encode([delta_text, step_text, baseline_text], normalize_embeddings=True)
    sim_inj = float(np.dot(embs[0], embs[1]))
    sim_base = float(np.dot(embs[0], embs[2]))
    return max(0.0, sim_inj - sim_base)


def main():
    print(f"Loading injected records from Llama sev1...")
    records = load_injected_with_curves()
    rng = np.random.default_rng(SEED)
    if len(records) > N_SAMPLE:
        indices = rng.choice(len(records), size=N_SAMPLE, replace=False)
        records = [records[i] for i in indices]
    print(f"Using {len(records)} records")

    # For each record, pick the first valid persistence step
    items = []
    for r in records:
        delta = r.get("injected_content", "")
        inject_idx = r.get("error_step")
        steps = r.get("step_outputs", [])
        # Find matching baseline
        if not delta or inject_idx is None:
            continue
        for step_idx, step_name, _ in r.get("persistence_curve", []):
            if step_idx <= inject_idx or step_idx >= len(steps):
                continue
            step_out = steps[step_idx].get("output_text", "")
            if not step_out:
                continue
            items.append((delta, step_out, step_idx))
            break
    print(f"Valid items for encoding: {len(items)}")

    # We need baseline texts too — load baselines
    baselines_by_query = {}
    for path in glob.glob("results/ragtruth_weighted_error/ragtruth_weighted_sev1_llama-3.1-8b_15trials.jsonl"):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if is_baseline(r) and r.get("step_outputs"):
                    baselines_by_query[r["task_query"]] = r

    # Rebuild items with baseline text
    final_items = []
    for r, (delta, step_out, step_idx) in zip(records, items):
        bl = baselines_by_query.get(r.get("task_query"))
        if not bl or step_idx >= len(bl.get("step_outputs", [])):
            continue
        base_out = bl["step_outputs"][step_idx].get("output_text", "")
        if not base_out:
            continue
        final_items.append((delta, step_out, base_out))

    print(f"Final items with baselines: {len(final_items)}")
    if len(final_items) < 50:
        print("Too few items, aborting")
        return

    # Compute persistence with each encoder
    scores = {}
    for enc_name in ENCODERS:
        print(f"\nEncoding with {enc_name}...")
        enc = get_encoder(enc_name)
        enc_scores = []
        for delta, step_out, base_out in final_items:
            s = compute_persistence_score(delta, step_out, base_out, enc)
            enc_scores.append(s)
        scores[enc_name] = np.array(enc_scores)
        print(f"  Mean persistence: {np.mean(enc_scores):.4f}")

    # Pairwise Spearman correlation
    print("\n--- Pairwise Spearman correlations ---")
    results = {}
    encoder_names = list(scores.keys())
    for i in range(len(encoder_names)):
        for j in range(i + 1, len(encoder_names)):
            e1, e2 = encoder_names[i], encoder_names[j]
            rho, pval = spearmanr(scores[e1], scores[e2])
            pair = f"{e1.split('/')[-1]} vs {e2.split('/')[-1]}"
            results[pair] = {"spearman_rho": round(rho, 4), "p_value": float(pval)}
            print(f"  {pair}: ρ = {rho:.4f} (p = {pval:.2e})")

    all_rhos = [v["spearman_rho"] for v in results.values()]
    min_rho = min(all_rhos)
    print(f"\nMin ρ = {min_rho:.4f} — {'PASS' if min_rho > 0.85 else 'BELOW 0.85 threshold'}")

    out = {
        "n_items": len(final_items),
        "encoders": ENCODERS,
        "pairwise_correlations": results,
        "min_rho": min_rho,
        "pass_threshold_085": min_rho > 0.85,
    }
    out_path = "results/stats/multi_encoder_validation.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
