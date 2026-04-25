"""Cost estimator for the CMU AI Gateway.

Per-trial token estimates derived from midterm data:
    - 5 step calls x (~2K input + ~500 output) tokens each
    - Plus 1 judge call (~2K input + ~300 output) when use_llm_judge=True
"""

GATEWAY_PRICING = {
    "gpt-4.1-mini": {"in": 0.40, "out": 1.60},
    "gpt-4o-mini-2024-07-18": {"in": 0.18, "out": 0.73},
    "claude-3-haiku-20240307": {"in": 0.25, "out": 1.25},
    "claude-3-5-sonnet-20241022": {"in": 3.00, "out": 15.00},
    "claude-sonnet-4-20250514-v1:0": {"in": 3.00, "out": 15.00},
    "claude-haiku-4-5-20251001-v1:0": {"in": 1.00, "out": 5.00},
    "gemini-2.5-flash": {"in": 0.07, "out": 0.30},
    "gemini-2.5-pro": {"in": 3.50, "out": 10.50},
}

INPUT_TOK_PER_TRIAL = 10_000
OUTPUT_TOK_PER_TRIAL = 2_500
JUDGE_INPUT_TOK = 2_000
JUDGE_OUTPUT_TOK = 300


def _rate(model_gateway_id):
    for k, v in GATEWAY_PRICING.items():
        if k == model_gateway_id or k.startswith(model_gateway_id):
            return v
    return None


def estimate_trial_cost(model_name, use_llm_judge=False, judge_model=None):
    from models import CMU_GATEWAY_MODELS, API_MODELS
    model_id = API_MODELS.get(model_name, {}).get("model", model_name)
    gateway_id = CMU_GATEWAY_MODELS.get(model_id, model_id)
    rate = _rate(gateway_id)
    if not rate:
        return None
    cost = (INPUT_TOK_PER_TRIAL * rate["in"] + OUTPUT_TOK_PER_TRIAL * rate["out"]) / 1e6
    if use_llm_judge and judge_model:
        jg_id = API_MODELS.get(judge_model, {}).get("model", judge_model)
        jg_gw = CMU_GATEWAY_MODELS.get(jg_id, jg_id)
        jg_rate = _rate(jg_gw)
        if jg_rate:
            cost += (JUDGE_INPUT_TOK * jg_rate["in"] + JUDGE_OUTPUT_TOK * jg_rate["out"]) / 1e6
    return cost


def estimate_sweep_cost(models, num_trials, error_steps_per_cell, queries,
                        severity_levels, compound_pairs=None,
                        use_llm_judge=False, judge_model=None,
                        skip_baseline=False):
    n_real_queries = sum(
        1 for q in queries
        if not (isinstance(q, dict) and q.get("_placeholder"))
    )
    total = 0.0
    print("=" * 60)
    print(f"Cost estimate — {len(models)} model(s), n={num_trials} trials/cell")
    print(f"  Queries: {n_real_queries}, Severities: {len(severity_levels)}")
    print("=" * 60)
    for m in models:
        per_trial = estimate_trial_cost(m, use_llm_judge, judge_model)
        if per_trial is None:
            print(f"  {m}: pricing unknown, skipping")
            continue
        if compound_pairs:
            n_cells = len(compound_pairs)
        else:
            n_cells = error_steps_per_cell + (0 if skip_baseline else 1)
        n_trials_model = n_cells * n_real_queries * num_trials * len(severity_levels)
        cost = n_trials_model * per_trial
        total += cost
        print(f"  {m}: {n_trials_model:,} trials × ${per_trial:.4f} = ${cost:.2f}")
    print("-" * 60)
    print(f"TOTAL: ${total:.2f}")
    print("=" * 60)
    return total


if __name__ == "__main__":
    from workflow import TASK_TEMPLATES
    from config import WORKFLOW_STEPS

    error_steps = len(WORKFLOW_STEPS) - 1
    estimate_sweep_cost(
        models=["gpt-4o-mini"],
        num_trials=40,
        error_steps_per_cell=error_steps,
        queries=TASK_TEMPLATES,
        severity_levels=[1, 2, 3],
    )
