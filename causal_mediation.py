"""Causal mediation analysis for error propagation.

Decomposes total effect (TE) of injection on final failure into:
- NIE (Natural Indirect Effect): injection → persistence → failure
- NDE (Natural Direct Effect): injection → failure (bypassing persistence)

Key strength: no-unmeasured-confounders assumption is SATISFIED because
injection is a randomized experimental treatment, not observational.
"""
import numpy as np
from scipy import stats

def compute_mediation(trial_records: list, baseline_records: list,
                      final_failure_key: str = "evaluation") -> dict:
    """Compute NIE, NDE, and total effect for each trial.

    Uses the difference method:
    TE = E[Y(1)] - E[Y(0)]  (injection vs no injection)
    NIE = E[Y(1, M(1))] - E[Y(1, M(0))]  (effect through mediator)
    NDE = E[Y(1, M(0))] - E[Y(0, M(0))]  (direct effect)

    Where Y = final failure, M = persistence, 1 = injected, 0 = baseline.
    """
    # Group by (model, query, injection_step)
    from collections import defaultdict

    # Build baseline lookup
    baseline_by_query = defaultdict(list)
    for r in baseline_records:
        if r.get("is_baseline"):
            baseline_by_query[r["task_query"]].append(r)

    results_by_group = defaultdict(list)

    for r in trial_records:
        if r.get("is_baseline") or r.get("error_step") is None:
            continue

        query = r["task_query"]
        model = r["model"]
        meta = r.get("injection_meta", {})
        error_type = meta.get("error_type", "unknown")

        # Get persistence integral (sum of persistence curve)
        persistence_curve = r.get("persistence_curve", [])
        persistence_integral = sum(p for _, _, p in persistence_curve) if persistence_curve else 0

        # Get final failure (1 - normalized quality score)
        eval_data = r.get("evaluation", {})
        if isinstance(eval_data, dict):
            quality = eval_data.get("quality_score", eval_data.get("overall_score", 5))
        else:
            quality = 5
        final_failure = 1.0 if quality <= 4 else 0.0  # binary

        # Matched baseline
        baselines = baseline_by_query.get(query, [])
        baseline_qualities = []
        baseline_persistences = []
        for b in baselines:
            b_eval = b.get("evaluation", {})
            if isinstance(b_eval, dict):
                bq = b_eval.get("quality_score", b_eval.get("overall_score", 5))
            else:
                bq = 5
            baseline_qualities.append(bq)
            baseline_persistences.append(0.0)  # no injection → no persistence

        if not baseline_qualities:
            continue

        baseline_failure = 1.0 if np.mean(baseline_qualities) <= 4 else 0.0

        results_by_group[(model, error_type, r.get("error_step"))].append({
            "persistence_integral": persistence_integral,
            "final_failure": final_failure,
            "baseline_failure": baseline_failure,
        })

    # Aggregate
    overall_te = []
    overall_persistence = []
    overall_failure = []
    group_results = {}

    for group_key, items in results_by_group.items():
        pers = [x["persistence_integral"] for x in items]
        fail = [x["final_failure"] for x in items]
        base = [x["baseline_failure"] for x in items]

        te = np.mean(fail) - np.mean(base)
        overall_te.extend([f - b for f, b in zip(fail, base)])
        overall_persistence.extend(pers)
        overall_failure.extend(fail)

        group_results[str(group_key)] = {
            "n": len(items),
            "mean_persistence": float(np.mean(pers)),
            "failure_rate": float(np.mean(fail)),
            "baseline_failure_rate": float(np.mean(base)),
            "total_effect": float(te),
        }

    # Compute mediation fraction via regression with bootstrap CIs
    if len(overall_persistence) > 10:
        pers_arr = np.array(overall_persistence)
        fail_arr = np.array(overall_failure)
        te_arr = np.array(overall_te) if overall_te else np.array([0.0])

        def _estimate_mediation(pers, fail, te_vals):
            slope, intercept, r_value, p_value, std_err = stats.linregress(pers, fail)
            mean_pers = np.mean(pers)
            nie = slope * mean_pers
            te = np.mean(te_vals)
            nde = te - nie
            frac = nie / te if abs(te) > 0.01 else 0.0
            return nie, nde, te, frac, slope, r_value ** 2, p_value

        point = _estimate_mediation(pers_arr, fail_arr, te_arr)

        n_boot = 1000
        rng = np.random.default_rng(42)
        boot_nie, boot_nde, boot_te, boot_frac = [], [], [], []
        n = len(pers_arr)
        for _ in range(n_boot):
            idx = rng.choice(n, size=n, replace=True)
            b_pers = pers_arr[idx]
            b_fail = fail_arr[idx]
            b_te = te_arr[idx] if len(te_arr) == n else te_arr
            try:
                nie_b, nde_b, te_b, frac_b, *_ = _estimate_mediation(b_pers, b_fail, b_te)
                boot_nie.append(nie_b)
                boot_nde.append(nde_b)
                boot_te.append(te_b)
                boot_frac.append(frac_b)
            except Exception:
                continue

        def _ci(arr):
            a = np.array(arr)
            return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

        return {
            "total_effect": float(point[2]),
            "total_effect_ci_95": _ci(boot_te),
            "nie": float(point[0]),
            "nie_ci_95": _ci(boot_nie),
            "nde": float(point[1]),
            "nde_ci_95": _ci(boot_nde),
            "mediation_fraction_nie_over_te": float(point[3]),
            "mediation_fraction_ci_95": _ci(boot_frac),
            "n_bootstrap": n_boot,
            "persistence_failure_regression": {
                "slope": float(point[4]),
                "r_squared": float(point[5]),
                "p_value": float(point[6]),
            },
            "group_results": group_results,
            "note": "No-unmeasured-confounders satisfied: injection is randomized experimental treatment.",
        }

    return {"error": "insufficient data for mediation analysis"}
