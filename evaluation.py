from workflow import StepResult
from models import call_model


def evaluate_workflow_output(results: list[StepResult], original_query: str, expected_keywords: list[str], evaluator_model: str = "gpt-4o-mini") -> dict:
    final_output = results[-1].output_text
    verification_output = final_output
    
    is_valid = "VALID" in verification_output.upper()
    
    keyword_matches = sum(1 for kw in expected_keywords if kw.lower() in final_output.lower())
    keyword_score = keyword_matches / len(expected_keywords)
    
    eval_prompt = f"""Rate the quality of this response to the query "{original_query}" on a scale of 1-10.
    
Response: {final_output}

Return only a number from 1-10."""
    
    try:
        quality_score = int(call_model(evaluator_model, eval_prompt).strip().split()[0])
        quality_score = max(1, min(10, quality_score))
    except:
        quality_score = 5
    
    return {
        "is_valid": is_valid,
        "keyword_score": keyword_score,
        "quality_score": quality_score,
        "combined_score": (0.3 * int(is_valid) + 0.3 * keyword_score + 0.4 * (quality_score / 10)),
    }