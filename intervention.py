"""Intervention strategies for mitigating error propagation.

Three strategies compared:
1. Threshold heuristic (75th percentile of baseline severity)
2. Learned logistic gate (train on trial features)
3. Optimal stopping (derived from hazard model posterior)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from severity import severity_semantic

def calibrate_threshold(baseline_records: list, percentile: int = 75) -> float:
    """Set intervention threshold from baseline severity distribution."""
    severities = []
    for r in baseline_records:
        for step in r.get("step_outputs", []):
            if step.get("step") != "verify":
                sev = severity_semantic(
                    step.get("input_text", ""),
                    step.get("output_text", ""),
                )
                severities.append(sev)
    if not severities:
        return 0.15  # fallback
    return float(np.percentile(severities, percentile))


def should_reask_threshold(step_input: str, step_output: str, threshold: float) -> bool:
    """Threshold heuristic: re-ask if severity exceeds threshold."""
    sev = severity_semantic(step_input, step_output)
    return sev > threshold


def train_learned_gate(trial_records: list) -> tuple:
    """Train logistic regression on trial features to predict failure.

    Features: [severity_semantic, step_index, error_type_encoded, persistence_at_prev_step]
    Target: persistence > threshold at next step

    Returns: (fitted model, cv_auc, feature_names)
    """
    X, y = [], []
    for r in trial_records:
        if r.get("is_baseline") or not r.get("persistence_curve"):
            continue
        sev = r.get("severity_semantic", 0)
        meta = r.get("injection_meta", {})
        etype = {"entity": 0, "invented": 1, "unverifiable": 2, "contradictory": 3}.get(
            meta.get("error_type", ""), 0)

        curve = r.get("persistence_curve", [])
        for i, (step_idx, step_name, p_score) in enumerate(curve):
            prev_p = curve[i - 1][2] if i > 0 else 0
            X.append([sev, step_idx, etype, prev_p])
            y.append(int(p_score > 0.1))

    if len(X) < 50:
        return None, 0.0, []

    X = np.array(X)
    y = np.array(y)
    model = LogisticRegression(max_iter=1000)
    cv_auc = np.mean(cross_val_score(model, X, y, cv=5, scoring="roc_auc"))
    model.fit(X, y)

    return model, float(cv_auc), ["severity", "step_idx", "error_type", "prev_persistence"]


def optimal_threshold_from_posterior(hazard_samples: np.ndarray, delta_k: float | np.ndarray,
                                     cost_ratio: float = 0.1) -> dict:
    """Derive optimal re-ask threshold from hazard model posterior.

    Re-ask iff: h_k(epsilon) * delta_k > c/U
    => threshold_k = c / (U * delta_k)

    When delta_k is a scalar, posterior uncertainty comes from hazard_samples.
    When delta_k is an array (posterior samples), both sources of uncertainty
    are propagated.

    Args:
        hazard_samples: posterior samples of hazard rate, shape (n_samples,)
            or (n_samples, n_steps). Used to compute the decision boundary.
        delta_k: downstream quality drop per unit hazard. Scalar or array
            of posterior samples with the same leading dimension.
        cost_ratio: c/U, the cost of re-asking relative to the utility of
            a correct answer.

    Returns: dict with threshold mean, 95% CI, and per-sample distribution.
    """
    hazard = np.atleast_1d(hazard_samples).astype(float)

    if isinstance(delta_k, np.ndarray):
        delta = np.atleast_1d(delta_k).astype(float)
    else:
        delta = np.full(len(hazard), float(delta_k))

    if len(delta) != len(hazard):
        delta = np.full(len(hazard), float(np.mean(delta)))

    thresholds = cost_ratio / (delta * hazard + 1e-8)
    thresholds = np.clip(thresholds, 0, 1)

    mean_thresh = float(np.mean(thresholds))
    ci_lo = float(np.percentile(thresholds, 2.5))
    ci_hi = float(np.percentile(thresholds, 97.5))

    return {
        "optimal_threshold": mean_thresh,
        "ci_95": [ci_lo, ci_hi],
        "std": float(np.std(thresholds)),
        "n_posterior_samples": len(thresholds),
        "interpretation": (
            f"Re-ask when predicted hazard > {mean_thresh:.3f} "
            f"[95% CI: {ci_lo:.3f}, {ci_hi:.3f}]"
        ),
    }
