"""Corruption persistence metric — the paper's primary novel measurement.

Measures how much of the injected error semantically persists at each
downstream pipeline step, relative to a matched baseline.

persistence(k) = max(0, sim(delta, output_k) - sim(delta, baseline_k))
"""
import numpy as np
from scipy.optimize import curve_fit
from severity import get_encoder

def corruption_persistence(
    injected_delta: str,
    step_output: str,
    baseline_output: str,
    encoder_name: str = "BAAI/bge-large-en-v1.5",
) -> float:
    """Excess embedding similarity: how much more does step output
    resemble the injected corruption vs. matched baseline?

    Returns: float in [0, ~1]. 0 = fully attenuated.
    """
    if not injected_delta or not step_output or not baseline_output:
        return 0.0
    enc = get_encoder(encoder_name)
    embs = enc.encode(
        [injected_delta, step_output, baseline_output],
        normalize_embeddings=True,
    )
    sim_injected = float(np.dot(embs[0], embs[1]))
    sim_baseline = float(np.dot(embs[0], embs[2]))
    return round(max(0.0, sim_injected - sim_baseline), 6)


def persistence_curve(trial_record: dict, baseline_record: dict, encoder_name: str = "BAAI/bge-large-en-v1.5") -> list:
    """Compute persistence at every step downstream of injection.

    Returns: list of (step_index, step_name, persistence_score)
    """
    delta = trial_record.get("injected_content", "")
    if not delta:
        return []
    inject_idx = trial_record.get("error_step")
    if inject_idx is None:
        return []

    trial_steps = trial_record.get("step_outputs", [])
    baseline_steps = baseline_record.get("step_outputs", [])

    curve = []
    for i, step in enumerate(trial_steps):
        if i <= inject_idx:
            continue
        if i >= len(baseline_steps):
            break
        baseline_step = baseline_steps[i]
        p = corruption_persistence(
            delta,
            step.get("output_text", ""),
            baseline_step.get("output_text", ""),
            encoder_name=encoder_name,
        )
        curve.append((i, step.get("step", f"step_{i}"), p))
    return curve


def corruption_persistence_multi(
    injected_delta: str,
    step_output: str,
    baseline_output: str,
) -> dict:
    """Compute persistence with all three encoders for robustness check.

    Returns: dict mapping short encoder name to persistence score.
    """
    encoders = [
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-large-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ]
    results = {}
    for enc_name in encoders:
        short_name = enc_name.split("/")[-1]
        results[short_name] = corruption_persistence(
            injected_delta, step_output, baseline_output, encoder_name=enc_name
        )
    return results


def persistence_curve_multi(trial_record: dict, baseline_record: dict) -> dict:
    """Compute persistence curves with all three encoders.

    Returns: dict mapping short encoder name to list of (step_index, step_name, score).
    """
    encoders = [
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-large-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ]
    results = {}
    for enc_name in encoders:
        short_name = enc_name.split("/")[-1]
        results[short_name] = persistence_curve(trial_record, baseline_record, encoder_name=enc_name)
    return results


def find_matched_baseline(query: str, baseline_records: list) -> dict | None:
    """Find a baseline record for the same query."""
    for r in baseline_records:
        if r.get("task_query") == query and r.get("is_baseline"):
            return r
    return None


# --- Decay model fitting ---

def _exp_decay(d, a, b):
    return a * np.exp(-b * d)

def _linear_decay(d, a, b):
    return np.maximum(0, a - b * d)

def _flat(d, a):
    return np.full_like(d, a, dtype=float)

def fit_decay_models(distances: np.ndarray, persistences: np.ndarray) -> dict:
    """Fit exponential, linear, and flat decay. Return AICc comparison.

    Args:
        distances: array of step-distances from injection (1, 2, 3, ...)
        persistences: array of persistence scores at those distances

    Returns: dict with {model_name: {params, aic, aicc, residuals}}
    """
    n = len(distances)
    if n < 3:
        return {}

    results = {}
    d = distances.astype(float)
    p = persistences.astype(float)

    models = {
        "exponential": (_exp_decay, 2, [0.5, 0.5]),
        "linear": (_linear_decay, 2, [0.5, 0.1]),
        "flat": (_flat, 1, [np.mean(p)]),
    }

    for name, (func, k, p0) in models.items():
        try:
            popt, _ = curve_fit(func, d, p, p0=p0, maxfev=5000,
                                bounds=(0, [np.inf] * k))
            predicted = func(d, *popt)
            residuals = p - predicted
            ss_res = np.sum(residuals ** 2)
            # AICc
            if n - k - 1 > 0:
                aic = n * np.log(ss_res / n + 1e-10) + 2 * k
                aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
            else:
                aicc = float("inf")
            results[name] = {
                "params": popt.tolist(),
                "ss_res": float(ss_res),
                "aicc": float(aicc),
            }
        except Exception:
            results[name] = {"params": None, "ss_res": float("inf"), "aicc": float("inf")}

    # Determine best model
    best = min(results, key=lambda x: results[x]["aicc"])
    delta_aicc = {}
    for name in results:
        delta_aicc[name] = results[name]["aicc"] - results[best]["aicc"]
    results["best_model"] = best
    results["delta_aicc"] = delta_aicc
    return results
