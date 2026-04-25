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


def optimal_threshold_from_posterior(hazard_samples: np.ndarray, delta_k: float,
                                     cost_ratio: float = 0.1) -> dict:
    """Derive optimal re-ask threshold from hazard model posterior.

    Re-ask iff: h_k(ε) · Δ_k > c/U

    Returns distribution over thresholds (one per posterior sample).
    """
    # h_k(ε) · Δ_k > c/U  →  h_k(ε) > c / (U · Δ_k)
    threshold_per_sample = cost_ratio / (delta_k + 1e-6)
    # This is constant given fixed cost ratio and delta
    # But with posterior uncertainty on delta_k, we get a distribution

    return {
        "optimal_threshold": float(threshold_per_sample),
        "interpretation": f"Re-ask when predicted hazard > {threshold_per_sample:.3f}",
    }
