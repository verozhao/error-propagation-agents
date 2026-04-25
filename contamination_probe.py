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
