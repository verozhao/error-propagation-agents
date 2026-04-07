from workflow import StepResult
from models import call_model
from factual_accuracy import evaluate_factual_accuracy, load_ground_truth


def _llm_quality_score(query: str, output: str, judge_model: str) -> int:
    """Ask `judge_model` to rate `output` 1–10. Returns 5 on parse failure."""
    eval_prompt = f"""Rate the quality of this response to the query "{query}" on a scale of 1-10.

Response: {output}

Return only a number from 1-10."""
    try:
        raw = call_model(judge_model, eval_prompt).strip()
        score = int(raw.split()[0])
        return max(1, min(10, score))
    except Exception:
        return 5


def evaluate_workflow_output(
    results: list[StepResult],
    original_query: str,
    expected_keywords: list[str],
    evaluator_model: str = "gpt-4o-mini",
    injected_error: str | None = None,
    ground_truth: dict | None = None,
    judge_models: list[str] | None = None,
) -> dict:
    """Evaluate a pipeline output.

    Backwards-compatible with the original signature: callers that pass only
    (results, original_query, expected_keywords) get the original metric
    fields plus the new ones.

    New optional args:
        injected_error: text of any injected false claim, for survival scoring.
        ground_truth: pre-loaded ground truth dict (avoid re-reading on hot loops).
        judge_models: extra judges to score with for inter-judge comparison.
            Each judge's score is reported under quality_scores[<model>].
    """
    final_output = results[-1].output_text
    verification_output = final_output

    is_valid = "VALID" in verification_output.upper()

    keyword_matches = sum(1 for kw in expected_keywords if kw.lower() in final_output.lower())
    keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0.0

    # primary judge (preserves old behavior)
    quality_score = _llm_quality_score(original_query, final_output, evaluator_model)

    # optional additional judges for inter-judge agreement
    quality_scores = {evaluator_model: quality_score}
    if judge_models:
        for jm in judge_models:
            if jm == evaluator_model:
                continue
            quality_scores[jm] = _llm_quality_score(original_query, final_output, jm)

    # factual accuracy (Phase 1.1)
    if ground_truth is None:
        try:
            ground_truth = load_ground_truth()
        except Exception:
            ground_truth = {}
    factual = evaluate_factual_accuracy(
        pipeline_output=final_output,
        injected_error=injected_error,
        query=original_query,
        ground_truth=ground_truth,
    )

    # original metric (kept verbatim for backward compatibility)
    combined_score = (
        0.3 * int(is_valid)
        + 0.3 * keyword_score
        + 0.4 * (quality_score / 10)
    )

    # new metric (Phase 1.3)
    combined_score_v2 = (
        0.2 * int(is_valid)
        + 0.2 * keyword_score
        + 0.3 * (quality_score / 10)
        + 0.3 * factual.factual_accuracy_score
    )

    return {
        "is_valid": is_valid,
        "keyword_score": keyword_score,
        "quality_score": quality_score,
        "combined_score": combined_score,
        # additive new fields
        "combined_score_v2": combined_score_v2,
        "judge_model": evaluator_model,
        "quality_scores": quality_scores,
        "factual": factual.to_dict(),
    }
