from workflow import StepResult
from models import call_model
from factual_accuracy import evaluate_factual_accuracy, load_ground_truth
import random


def _llm_quality_score(query: str, output: str, judge_model: str) -> int:
    """v1 single-number score. Kept for backward compat."""
    eval_prompt = f"""Rate the quality of this response to the query "{query}" on a scale of 1-10.

Response: {output}

Return only a number from 1-10."""
    try:
        raw = call_model(judge_model, eval_prompt).strip()
        score = int(raw.split()[0])
        return max(1, min(10, score))
    except Exception:
        return 5


def _llm_quality_rubric(query: str, output: str, judge_model: str) -> dict:
    """Multi-dimensional rubric scoring — more sensitive than a single number."""
    eval_prompt = f"""You are a strict evaluator. Rate this response to the query "{query}" on four dimensions.
For each, give a score from 1-10 where 1=terrible, 5=mediocre, 10=excellent.

Response to evaluate:
\"\"\"{output}\"\"\"

Format your response as exactly 4 lines:
ACCURACY: [number]
COMPLETENESS: [number]
COHERENCE: [number]
USEFULNESS: [number]"""
    defaults = {"accuracy": 5, "completeness": 5, "coherence": 5, "usefulness": 5}
    try:
        raw = call_model(judge_model, eval_prompt).strip()
        scores = {}
        for line in raw.split("\n"):
            for dim in ["ACCURACY", "COMPLETENESS", "COHERENCE", "USEFULNESS"]:
                if line.strip().upper().startswith(dim):
                    digits = ''.join(c for c in line.split(":")[-1].strip() if c.isdigit())[:2]
                    if digits:
                        scores[dim.lower()] = max(1, min(10, int(digits)))
        for k, v in defaults.items():
            scores.setdefault(k, v)
        scores["overall"] = round(sum(scores.values()) / 4, 1)
        return scores
    except Exception:
        return {**defaults, "overall": 5.0}


def _pairwise_comparison(query: str, baseline_output: str, test_output: str, judge_model: str) -> dict:
    """Side-by-side comparison — much more sensitive than absolute scoring."""
    # Randomize order to avoid position bias
    a_is_baseline = random.random() < 0.5
    a_text = baseline_output if a_is_baseline else test_output
    b_text = test_output if a_is_baseline else baseline_output

    eval_prompt = f"""Compare these two responses to the query: "{query}"

Response A:
\"\"\"{a_text[:2000]}\"\"\"

Response B:
\"\"\"{b_text[:2000]}\"\"\"

Which response is better? Consider accuracy, completeness, helpfulness.

Answer in exactly this format:
WINNER: A or B or TIE
CONFIDENCE: 1-5 (1=barely different, 5=clearly better)
REASON: one sentence"""
    try:
        raw = call_model(judge_model, eval_prompt).strip()
        winner, confidence, reason = "TIE", 1, ""
        for line in raw.split("\n"):
            line = line.strip()
            if line.upper().startswith("WINNER"):
                val = line.split(":")[-1].strip().upper()
                if "A" in val and "B" not in val:
                    winner = "A"
                elif "B" in val and "A" not in val:
                    winner = "B"
            elif line.upper().startswith("CONFIDENCE"):
                digits = ''.join(c for c in line.split(":")[-1].strip() if c.isdigit())[:1]
                if digits:
                    confidence = max(1, min(5, int(digits)))
            elif line.upper().startswith("REASON"):
                reason = line.split(":", 1)[-1].strip()

        if winner == "TIE":
            degradation = 0.0
        elif (winner == "A" and a_is_baseline) or (winner == "B" and not a_is_baseline):
            degradation = confidence / 5.0  # baseline won → test is worse
        else:
            degradation = -confidence / 5.0  # test won → improvement

        return {
            "winner": "baseline" if degradation > 0 else ("test" if degradation < 0 else "tie"),
            "confidence": confidence,
            "degradation_score": degradation,
            "reason": reason,
        }
    except Exception as e:
        return {"winner": "error", "confidence": 0, "degradation_score": 0.0, "reason": str(e)}


def evaluate_workflow_output(
    results: list[StepResult],
    original_query: str,
    expected_keywords: list[str],
    evaluator_model: str = "gpt-4o-mini",
    injected_error: str | None = None,
    ground_truth: dict | None = None,
    judge_models: list[str] | None = None,
    baseline_output: str | None = None,
) -> dict:
    """Evaluate a pipeline output.

    Backward-compatible with v1 callers. New optional arg:
        baseline_output: final output from a no-error run, for pairwise comparison.
    """
    final_output = results[-1].output_text
    verification_output = final_output

    is_valid = "VALID" in verification_output.upper()

    keyword_matches = sum(1 for kw in expected_keywords if kw.lower() in final_output.lower())
    keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0.0

    # v1 single-number quality score
    quality_score = _llm_quality_score(original_query, final_output, evaluator_model)

    # v2 rubric quality scores
    rubric = _llm_quality_rubric(original_query, final_output, evaluator_model)

    quality_scores = {evaluator_model: quality_score}
    if judge_models:
        for jm in judge_models:
            if jm == evaluator_model:
                continue
            quality_scores[jm] = _llm_quality_score(original_query, final_output, jm)

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

    # assertion coverage from ground truth
    gt_entry = ground_truth.get(original_query) if ground_truth else None
    assertion_score = 1.0
    if gt_entry:
        out_lower = final_output.lower()
        hits = 0
        total = len(gt_entry.get("assertions", []))
        for a in gt_entry.get("assertions", []):
            kws = [k.lower() for k in a.get("keywords", [])]
            if kws and all(kw in out_lower for kw in kws):
                hits += 1
                continue
            for alias in a.get("aliases", []):
                if alias.lower() in out_lower:
                    hits += 1
                    break
        if total > 0:
            assertion_score = hits / total

    # v1 combined score (unchanged)
    combined_score = (
        0.3 * int(is_valid)
        + 0.3 * keyword_score
        + 0.4 * (quality_score / 10)
    )

    # v2 combined score (unchanged)
    combined_score_v2 = (
        0.2 * int(is_valid)
        + 0.2 * keyword_score
        + 0.3 * (quality_score / 10)
        + 0.3 * factual.factual_accuracy_score
    )

    # v3 combined score: spreads weight across more sensitive sub-scores
    # Note: factual_accuracy_score already incorporates assertion preservation
    # internally (preserved * (1 - survival) - contradiction_penalty), so we
    # don't include assertion_score separately to avoid double-counting.
    combined_score_v3 = (
        0.15 * int(is_valid)
        + 0.15 * keyword_score
        + 0.125 * (rubric["accuracy"] / 10)
        + 0.125 * (rubric["completeness"] / 10)
        + 0.125 * (rubric["coherence"] / 10)
        + 0.125 * (rubric["usefulness"] / 10)
        + 0.20 * factual.factual_accuracy_score
    )

    # pairwise comparison (only if baseline provided)
    pairwise = None
    if baseline_output:
        pairwise = _pairwise_comparison(original_query, baseline_output, final_output, evaluator_model)

    result = {
        "is_valid": is_valid,
        "keyword_score": keyword_score,
        "quality_score": quality_score,
        "rubric": rubric,
        "assertion_score": assertion_score,
        "combined_score": combined_score,
        "combined_score_v2": combined_score_v2,
        "combined_score_v3": combined_score_v3,
        "judge_model": evaluator_model,
        "quality_scores": quality_scores,
        "factual": factual.to_dict(),
    }
    if pairwise:
        result["pairwise"] = pairwise
    return result
