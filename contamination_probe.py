"""Contamination probe for benchmark queries.

Checks if models produce verbatim answers without any pipeline context,
indicating training-data memorization.
"""
import json
from models import call_model

def run_contamination_probe(ground_truth_path: str, models: list[str]) -> dict:
    """For each query, prompt each model with JUST the question.
    Record if it produces the exact gold answer.

    Returns: dict mapping query → contamination_score (fraction of models
    that produce verbatim answer).
    """
    with open(ground_truth_path) as f:
        gt = json.load(f)

    results = {}
    for entry in gt.get("queries", []):
        query = entry.get("query", "")
        if not query:
            continue
        gold_answers = []
        for assertion in entry.get("assertions", []):
            gold_answers.extend(assertion.get("keywords", []))
        if not gold_answers:
            continue

        gold_set = {a.lower().strip() for a in gold_answers}
        matches = 0

        for model in models:
            try:
                response = call_model(model, f"Answer this question concisely: {query}",
                                      max_tokens=100, temperature=0.0)
                response_lower = response.lower().strip()
                if any(gold in response_lower for gold in gold_set):
                    matches += 1
            except Exception:
                continue

        contamination_score = matches / len(models) if models else 0
        results[query] = {
            "contamination_score": round(contamination_score, 2),
            "likely_contaminated": contamination_score > 0.5,
            "models_matched": matches,
            "models_tested": len(models),
        }

    return results


ACTIVE_MODELS = ["llama-3.1-8b", "claude-haiku-3", "claude-sonnet-3-7", "claude-sonnet-4"]


def stratify_by_contamination(trial_records: list, contamination_scores: dict) -> dict:
    """Split trial records into 'likely_clean' and 'likely_contaminated' subsets."""
    clean = []
    contaminated = []

    for r in trial_records:
        query = r.get("task_query", "")
        score = contamination_scores.get(query, {}).get("contamination_score", 0)
        if score > 0.5:
            contaminated.append(r)
        else:
            clean.append(r)

    return {
        "clean": clean,
        "contaminated": contaminated,
        "n_clean": len(clean),
        "n_contaminated": len(contaminated),
        "fraction_contaminated": len(contaminated) / max(len(clean) + len(contaminated), 1),
    }


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Run contamination probe on ground truth queries")
    parser.add_argument("--gt", default="ground_truth.json", help="Path to ground_truth.json")
    parser.add_argument("--models", nargs="+", default=ACTIVE_MODELS)
    parser.add_argument("--out", default="results/stats/contamination_probe.json")
    args = parser.parse_args()

    print(f"Running contamination probe on {args.gt} with models: {args.models}")
    results = run_contamination_probe(args.gt, args.models)

    n_contaminated = sum(1 for v in results.values() if v["likely_contaminated"])
    print(f"\nProbed {len(results)} queries across {len(args.models)} models")
    print(f"Likely contaminated: {n_contaminated}/{len(results)} "
          f"({n_contaminated/max(len(results),1):.0%})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {args.out}")
