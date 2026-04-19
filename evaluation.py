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
    use_llm_judge: bool = True,
) -> dict:
    """Evaluate a pipeline output.

    When use_llm_judge=False, skips all LLM judge calls (quality_score,
    rubric, pairwise). The primary combined_score uses factual_accuracy_score
    instead, making it fully algorithmic and deterministic.

    P0-18 FIX: Content evaluation (keyword_score, factual accuracy,
    assertion coverage, injected-claim survival) is now performed on the
    `compose` step output (i.e. results[-2]), which is the pipeline's
    actual recommendation. The `verify` step output (results[-1]) only
    produces a short 'VALID/INVALID + one-sentence reason' meta-comment,
    which does not re-state product/language/food names and therefore
    cannot be meaningfully checked against ground-truth assertion
    keywords. Using it for content evaluation made every pipeline look
    almost equally bad regardless of injection.

    P0-17 FIX: is_valid now uses startswith('VALID') instead of
    substring containment, because 'INVALID' trivially contains 'VALID'
    and the old test was silently True for every INVALID response.

    Falls back to results[-1] for content if there's no penultimate
    step (defensive — production pipeline always has 5 steps).
    """
    # P0-18: distinguish the two outputs. `verify_text` is the final
    # step's VALID/INVALID meta-comment; `content_text` is what the
    # pipeline actually produced for the user.
    verify_text = results[-1].output_text if results else ""
    content_text = results[-2].output_text if len(results) >= 2 else verify_text
    # Backward-compat alias: old code and old records use `final_output`
    # to mean "the last step". We keep a name for downstream in case
    # anything needs it, but nothing in this function reads it anymore.
    final_output = verify_text  # noqa: F841 — legacy field

    # P0-17 + P0-19: first-token-only VALID detection. Claude models
    # can be verbose; substring checks are fragile when both "VALID" and
    # "INVALID" appear in prose. Parsing only the first token is robust.
    first_token = ""
    if verify_text.strip():
        first_token = verify_text.strip().split(None, 1)[0].upper().rstrip(".,:;")
    is_valid = first_token == "VALID"

    # P0-18: content-level checks use content_text (compose output)
    keyword_matches = sum(1 for kw in expected_keywords if kw.lower() in content_text.lower())
    keyword_score = keyword_matches / len(expected_keywords) if expected_keywords else 0.0

    if use_llm_judge:
        quality_score = _llm_quality_score(original_query, content_text, evaluator_model)
        rubric = _llm_quality_rubric(original_query, content_text, evaluator_model)
        quality_scores = {evaluator_model: quality_score}
        if judge_models:
            for jm in judge_models:
                if jm == evaluator_model:
                    continue
                quality_scores[jm] = _llm_quality_score(original_query, content_text, jm)
    else:
        quality_score = None
        rubric = None
        quality_scores = {}

    if ground_truth is None:
        try:
            ground_truth = load_ground_truth()
        except Exception:
            ground_truth = {}
    factual = evaluate_factual_accuracy(
        pipeline_output=content_text,
        injected_error=injected_error,
        query=original_query,
        ground_truth=ground_truth,
    )

    # assertion coverage from ground truth (P0-18: on content_text)
    gt_entry = ground_truth.get(original_query) if ground_truth else None
    assertion_score = 1.0
    if gt_entry:
        out_lower = content_text.lower()
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

    # ------------------------------------------------------------------
    # D1 FIX: preserved_component rewrite.
    #
    # The old formula used assertion keyword matching for preserved_component.
    # Problem: for HotpotQA auto-generated entries, assertion keywords are
    # often trivially matchable (e.g. ["yes"], or entity names that appear
    # in the query itself). This made preserved_component ≈ 1.0 for ~70%
    # of queries regardless of injection → 40% of combined_score was
    # dead weight with zero discriminative power.
    #
    # Fix: two-tier preserved_component:
    #   Tier 1 (preferred): answer exact match. If the GT entry has an
    #          "answer" field, check whether it appears in the compose
    #          output. This is the purest preservation signal — did the
    #          pipeline produce the correct answer despite injection?
    #   Tier 2 (fallback):  non-trivial assertion matching. For manual
    #          entries without an "answer" field, use only assertions
    #          whose keywords are NOT already in the query text and are
    #          not degenerate ("yes"/"no"). This filters out the ~14/50
    #          trivially-matching assertions.
    #
    # Weights unchanged: 0.4 preserved + 0.4 (1-survival) + 0.2 is_valid.
    # ------------------------------------------------------------------
    # Compute preserved_component with discriminative power
    _gt_answer = gt_entry.get("answer", "") if gt_entry else ""
    _query_lower = original_query.lower()

    if (_gt_answer
            and _gt_answer.strip().lower() not in ("", "n/a", "yes", "no")
            and len(_gt_answer.strip()) > 2):
        # Tier 1: answer exact match (case-insensitive substring)
        # Excludes yes/no because LLMs answer "Both are American" not "yes"
        preserved_component = 1.0 if _gt_answer.lower() in content_text.lower() else 0.0
    elif factual.assertions_total > 0:
        # Tier 2: non-trivial assertions only
        _nontrivial_present = 0
        _nontrivial_total = 0
        for _a in (gt_entry.get("assertions", []) if gt_entry else []):
            _kws = [k.lower() for k in _a.get("keywords", [])]
            # Skip degenerate assertions
            if not _kws:
                continue
            if _kws == ["yes"] or _kws == ["no"]:
                continue
            if all(kw in _query_lower for kw in _kws):
                continue  # keywords all from query text → trivially matchable
            _nontrivial_total += 1
            _out_lower = content_text.lower()
            if all(kw in _out_lower for kw in _kws):
                _nontrivial_present += 1
                continue
            for _alias in _a.get("aliases", []):
                if _alias.lower() in _out_lower:
                    _nontrivial_present += 1
                    break
        if _nontrivial_total > 0:
            preserved_component = _nontrivial_present / _nontrivial_total
        else:
            # All assertions were trivial → fall back to original
            preserved_component = (
                factual.assertions_present / factual.assertions_total
            )
    else:
        preserved_component = 1.0  # no assertions defined → neutral

    survival_component = factual.error_survival_score  # 0 if no injection

    # Contradiction penalty is reused from factual module (small, bounded)
    contradiction_penalty = 0.0
    if factual.assertions_total > 0 and factual.contradictions_present > 0:
        contradiction_penalty = min(0.5, 0.15 * factual.contradictions_present)

    combined_score = (
        0.40 * preserved_component
        + 0.40 * (1.0 - survival_component)
        + 0.20 * int(is_valid)
        - contradiction_penalty
    )
    combined_score = max(0.0, min(1.0, combined_score))

    # P0-2: keep the pre-fix formula under a named field so we can
    # compare before/after if needed. This is NOT the primary metric.
    combined_score_prefix_formula = (
        0.3 * int(is_valid)
        + 0.3 * keyword_score
        + 0.4 * factual.factual_accuracy_score
    )

    # Legacy v1 score preserved for midterm comparability
    if use_llm_judge and quality_score is not None:
        combined_score_legacy = (
            0.3 * int(is_valid)
            + 0.3 * keyword_score
            + 0.4 * (quality_score / 10)
        )
    else:
        combined_score_legacy = None

    # v2 combined score (requires judge)
    if use_llm_judge and quality_score is not None:
        combined_score_v2 = (
            0.2 * int(is_valid)
            + 0.2 * keyword_score
            + 0.3 * (quality_score / 10)
            + 0.3 * factual.factual_accuracy_score
        )
    else:
        combined_score_v2 = None

    # v3 combined score (requires rubric)
    if use_llm_judge and rubric is not None:
        combined_score_v3 = (
            0.15 * int(is_valid)
            + 0.15 * keyword_score
            + 0.125 * (rubric["accuracy"] / 10)
            + 0.125 * (rubric["completeness"] / 10)
            + 0.125 * (rubric["coherence"] / 10)
            + 0.125 * (rubric["usefulness"] / 10)
            + 0.20 * factual.factual_accuracy_score
        )
    else:
        combined_score_v3 = None

    # pairwise comparison (only if baseline provided and judge enabled)
    # P0-18: compare the main content (compose output), not the verify meta-comment
    pairwise = None
    if baseline_output and use_llm_judge:
        pairwise = _pairwise_comparison(original_query, baseline_output, content_text, evaluator_model)

    result = {
        "is_valid": is_valid,
        "keyword_score": keyword_score,
        "quality_score": quality_score,
        "rubric": rubric,
        "assertion_score": assertion_score,
        "combined_score": combined_score,
        "combined_score_prefix_formula": round(combined_score_prefix_formula, 4),
        # P0-2: expose the score components so analysis can decompose
        "combined_score_components": {
            "preserved": round(preserved_component, 4),
            "one_minus_survival": round(1.0 - survival_component, 4),
            "is_valid": int(is_valid),
            "contradiction_penalty": round(contradiction_penalty, 4),
            "weights": {"preserved": 0.40, "one_minus_survival": 0.40, "is_valid": 0.20},
        },
        "combined_score_legacy": combined_score_legacy,
        "combined_score_v2": combined_score_v2,
        "combined_score_v3": combined_score_v3,
        "judge_model": evaluator_model if use_llm_judge else None,
        "quality_scores": quality_scores,
        "factual": factual.to_dict(),
    }
    if pairwise:
        result["pairwise"] = pairwise
    return result
