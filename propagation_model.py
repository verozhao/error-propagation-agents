"""Formal error propagation model: Markov chain with step-wise attenuation.

===========================================================================
THEORETICAL CONTRIBUTION (Section 4 of the paper)
===========================================================================

We model error propagation through a K-step LLM pipeline as a discrete-time
Markov chain over a continuous error intensity state s ∈ [0, 1].

Definitions
-----------
  s_k     = claim survival score at step k (0 = fully attenuated, 1 = verbatim)
  α_k     = step-wise attenuation factor at step k
            α_k := E[s_k / s_{k-1} | s_{k-1} > 0]
            Measures the fraction of error signal that passes through step k.
  K       = number of pipeline steps (K=5 in our pipeline)
  i       = injection point (0-indexed)

Model
-----
  Given injection at step i with initial intensity s_i:

    s_k = s_i · ∏_{j=i+1}^{k} α_j          for k > i       (Eq. 1)

  End-to-end survival:

    S(i) = ∏_{j=i+1}^{K-1} α_j              (Eq. 2)

  Predicted failure rate (linking function):

    FR_pred(i) = 1 − (1 − λ · S(i))          = λ · S(i)     (Eq. 3)

  where λ is a severity-dependent scaling factor estimated from data.

This yields K−1 free parameters {α_1, ..., α_{K-1}} per error type
(or K−1 shared + 3 type-specific λ parameters in the pooled model).

Extensions
----------
  1. Severity modulation:  α_k(sev) = α_k^{(base)} · (1 + β · log(sev))
  2. Type-dependent:       α_k^{(type)} estimated per error type
  3. Compound injection:   S(i,j) = S(i) · S(j) under independence
                           Interaction term δ = S_obs(i,j) − S(i)·S(j)

Validation
----------
  - Compare FR_pred vs FR_obs per (error_type, injection_step)
  - Report R², RMSE, and per-condition residuals
  - Bootstrap CIs on α_k estimates
  - Likelihood ratio test: Markov model vs null (uniform attenuation)

Usage:
    python propagation_model.py                    # fit model on all data
    python propagation_model.py --error-type factual  # single type
    python propagation_model.py --bootstrap 5000   # bootstrap CIs

Produces:
    results/stats/markov_alpha_estimates.csv
    results/stats/markov_predictions.csv
    results/stats/markov_model_fit.csv
    results/stats/markov_compound_test.csv
    figures/paper/fig_markov_fit.pdf
    figures/paper/fig_alpha_cascade.pdf
    figures/paper/fig_predicted_vs_observed.pdf
    paper/tables/table_markov_alphas.tex
===========================================================================
"""

from __future__ import annotations

import argparse
import json
import glob
import os
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from scipy.optimize import minimize

from config import WORKFLOW_STEPS

# Suppress convergence warnings during bootstrap
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Plot style (matches generate_paper_figures.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
})

FIG_DIR = "figures/paper"
STATS_DIR = "results/stats"
TABLE_DIR = "paper/tables"

# Step names (without verify for content evaluation)
CONTENT_STEPS = WORKFLOW_STEPS[:-1]  # search, filter, summarize, compose
N_STEPS = len(WORKFLOW_STEPS)        # 5
N_CONTENT = len(CONTENT_STEPS)       # 4
# Transitions are between consecutive steps: 0→1, 1→2, 2→3, 3→4
N_TRANSITIONS = N_STEPS - 1          # 4

EPSILON = 1e-10  # avoid division by zero


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_records() -> list[dict]:
    """Load all experiment records from JSONL (primary) or JSON (fallback)."""
    from record_utils import is_baseline, injection_is_valid
    rows = []
    for path in glob.glob("results/**/*.jsonl", recursive=True):
        if "stats" in path or "sanity" in path:
            continue
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    if not rows:
        for path in glob.glob("results/**/*.json", recursive=True):
            if "stats" in path or "sanity" in path:
                continue
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                rows.extend(data)
    return rows


def build_transition_data(records: list[dict]) -> pd.DataFrame:
    """Extract consecutive-step survival pairs for Markov transition estimation.

    For each injected record, we observe survival scores at each step
    downstream of the injection point. From these we extract transition
    ratios: r_{k→k+1} = s_{k+1} / s_k for each consecutive pair where
    s_k > 0 (error is present).

    Returns a DataFrame with columns:
        model, error_type, severity, injection_step, from_step, to_step,
        s_from, s_to, ratio
    """
    from record_utils import is_baseline, injection_is_valid

    rows = []
    for r in records:
        if is_baseline(r):
            continue
        if injection_is_valid(r) is False:
            continue
        if r.get("compound_steps"):
            continue
        es = r.get("error_step")
        if isinstance(es, list) or es is None:
            continue

        efs = r.get("error_found_in_step") or {}
        if not efs:
            continue

        # Collect survival scores in step order
        survivals = []
        for i, step_name in enumerate(WORKFLOW_STEPS):
            s = efs.get(step_name, {}).get("survival_score", 0.0)
            survivals.append(s)

        # Extract consecutive transitions from injection point onward
        for k in range(es, N_STEPS - 1):
            s_from = survivals[k]
            s_to = survivals[k + 1]
            ratio = s_to / s_from if s_from > EPSILON else 0.0

            rows.append({
                "model": r.get("model", "unknown"),
                "error_type": r.get("error_type", "unknown"),
                "severity": r.get("severity", 1),
                "injection_step": es,
                "from_step": k,
                "to_step": k + 1,
                "from_step_name": WORKFLOW_STEPS[k],
                "to_step_name": WORKFLOW_STEPS[k + 1],
                "s_from": s_from,
                "s_to": s_to,
                "ratio": ratio,
                "task_query": r.get("task_query", ""),
            })

    return pd.DataFrame(rows)


def build_endpoint_data(records: list[dict]) -> pd.DataFrame:
    """Build per-record data for end-to-end failure rate validation.

    Returns a DataFrame with:
        model, error_type, severity, injection_step, final_survival,
        combined_score, is_baseline
    """
    from record_utils import is_baseline, injection_is_valid

    rows = []
    for r in records:
        bl = is_baseline(r)
        if not bl and injection_is_valid(r) is False:
            continue
        if r.get("compound_steps"):
            continue

        es = r.get("error_step")
        if not bl and (isinstance(es, list) or es is None):
            continue

        ev = r.get("evaluation", {})
        efs = r.get("error_found_in_step") or {}

        # Final survival: at compose step (step 3) — the content output
        compose_surv = efs.get("compose", {}).get("survival_score", 0.0)

        rows.append({
            "model": r.get("model", "unknown"),
            "error_type": r.get("error_type", "unknown"),
            "severity": r.get("severity", 1),
            "injection_step": es if not bl else -1,
            "is_baseline": bl,
            "final_survival": compose_surv,
            "combined_score": ev.get("combined_score"),
            "task_query": r.get("task_query", ""),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model 1: Step-wise attenuation (per error type)
# ---------------------------------------------------------------------------
@dataclass
class MarkovModel:
    """Fitted Markov attenuation model."""
    error_type: str
    alphas: dict[int, float]        # transition index → α_k
    alpha_cis: dict[int, tuple]     # transition index → (lo, hi)
    lambda_scale: float             # severity scaling
    n_observations: dict[int, int]  # transition index → sample size
    r_squared: float = 0.0
    rmse: float = 0.0

    def predict_survival(self, injection_step: int) -> float:
        """Predict end-to-end survival S(i) = ∏_{j=i+1}^{K-1} α_j."""
        product = 1.0
        # From injection step to compose (step 3), since verify doesn't
        # produce content. If injection is at step 3, S = 1 (no downstream).
        for j in range(injection_step + 1, N_CONTENT):
            product *= self.alphas.get(j, 0.0)
        return product

    def predict_failure_rate(self, injection_step: int) -> float:
        """Predict FR from survival: FR = λ · S(i)."""
        return self.lambda_scale * self.predict_survival(injection_step)


def estimate_alphas(
    transition_df: pd.DataFrame,
    error_type: str | None = None,
    min_samples: int = 5,
) -> dict[int, tuple[float, int]]:
    """Estimate α_k for each transition from empirical survival ratios.

    α_k = E[s_k / s_{k-1} | s_{k-1} > threshold]

    We use a threshold on s_from to avoid dividing by near-zero survival
    scores (which produce noisy, uninformative ratios).

    Returns: {transition_index: (alpha, n_samples)}
    """
    df = transition_df.copy()
    if error_type:
        df = df[df["error_type"] == error_type]

    # Only use transitions where the error was actually present at the
    # source step (s_from > threshold). Without this filter, transitions
    # where the error was already fully attenuated would contribute
    # uninformative 0/0 → 0 ratios that bias α_k downward.
    PRESENCE_THRESHOLD = 0.05
    df = df[df["s_from"] > PRESENCE_THRESHOLD]

    alphas = {}
    for trans_idx in range(N_TRANSITIONS):
        sub = df[df["from_step"] == trans_idx]
        if len(sub) < min_samples:
            alphas[trans_idx] = (0.0, len(sub))
            continue

        # Ratio estimator: E[s_to / s_from]
        ratios = sub["ratio"].values
        # Clip to [0, 1] — ratios > 1 can occur from noise (Jaccard
        # overlap increasing due to LLM restatement of the error)
        ratios = np.clip(ratios, 0.0, 1.0)
        alpha = float(np.mean(ratios))
        alphas[trans_idx] = (alpha, len(sub))

    return alphas


def bootstrap_alpha_cis(
    transition_df: pd.DataFrame,
    error_type: str | None = None,
    n_boot: int = 2000,
    alpha_level: float = 0.05,
    seed: int = 42,
) -> dict[int, tuple[float, float]]:
    """Bootstrap 95% CIs for each α_k estimate."""
    df = transition_df.copy()
    if error_type:
        df = df[df["error_type"] == error_type]
    df = df[df["s_from"] > 0.05]

    rng = np.random.default_rng(seed)
    cis = {}

    for trans_idx in range(N_TRANSITIONS):
        sub = df[df["from_step"] == trans_idx]
        ratios = np.clip(sub["ratio"].values, 0.0, 1.0)
        if len(ratios) < 3:
            cis[trans_idx] = (0.0, 0.0)
            continue

        boot_means = np.array([
            np.mean(rng.choice(ratios, size=len(ratios), replace=True))
            for _ in range(n_boot)
        ])
        lo = float(np.percentile(boot_means, 100 * alpha_level / 2))
        hi = float(np.percentile(boot_means, 100 * (1 - alpha_level / 2)))
        cis[trans_idx] = (lo, hi)

    return cis


def fit_lambda(
    endpoint_df: pd.DataFrame,
    alphas: dict[int, tuple[float, int]],
    error_type: str | None = None,
) -> float:
    """Fit the scaling parameter λ that links predicted survival to observed FR.

    λ = argmin_λ  Σ_i (FR_obs(i) − λ · S_pred(i))²

    Closed-form solution: λ = Σ(FR_obs · S_pred) / Σ(S_pred²)
    """
    df = endpoint_df.copy()
    if error_type:
        df = df[df["error_type"] == error_type]

    baselines = df[df["is_baseline"]]
    injected = df[~df["is_baseline"]]

    if baselines.empty or injected.empty:
        return 1.0

    bl_mean = baselines["combined_score"].mean()
    if bl_mean <= 0:
        return 1.0

    # Compute observed FR per injection step
    fr_obs = {}
    for step, group in injected.groupby("injection_step"):
        inj_mean = group["combined_score"].mean()
        fr_obs[step] = max(0.0, 1.0 - inj_mean / bl_mean)

    # Compute predicted survival S(i) per injection step
    s_pred = {}
    for step in fr_obs:
        product = 1.0
        for j in range(step + 1, N_CONTENT):
            product *= alphas.get(j, (0.0, 0))[0]
        s_pred[step] = product

    # Closed-form OLS for λ
    steps = sorted(fr_obs.keys())
    if not steps:
        return 1.0

    y = np.array([fr_obs[s] for s in steps])
    x = np.array([s_pred[s] for s in steps])

    denom = np.sum(x ** 2)
    if denom < EPSILON:
        return 1.0

    lam = float(np.sum(y * x) / denom)
    return max(0.0, min(2.0, lam))  # clip to reasonable range


def fit_model(
    transition_df: pd.DataFrame,
    endpoint_df: pd.DataFrame,
    error_type: str,
    n_boot: int = 2000,
) -> MarkovModel:
    """Fit the complete Markov model for one error type."""
    alphas_raw = estimate_alphas(transition_df, error_type)
    alpha_cis = bootstrap_alpha_cis(transition_df, error_type, n_boot=n_boot)
    lam = fit_lambda(endpoint_df, alphas_raw, error_type)

    model = MarkovModel(
        error_type=error_type,
        alphas={k: v[0] for k, v in alphas_raw.items()},
        alpha_cis=alpha_cis,
        lambda_scale=lam,
        n_observations={k: v[1] for k, v in alphas_raw.items()},
    )

    # Compute goodness of fit
    df = endpoint_df[endpoint_df["error_type"] == error_type].copy()
    baselines = df[df["is_baseline"]]
    injected = df[~df["is_baseline"]]

    if baselines.empty or injected.empty:
        return model

    bl_mean = baselines["combined_score"].mean()
    if bl_mean <= 0:
        return model

    fr_obs_list = []
    fr_pred_list = []
    for step, group in injected.groupby("injection_step"):
        fr_obs = max(0.0, 1.0 - group["combined_score"].mean() / bl_mean)
        fr_pred = model.predict_failure_rate(step)
        fr_obs_list.append(fr_obs)
        fr_pred_list.append(fr_pred)

    if len(fr_obs_list) >= 2:
        y = np.array(fr_obs_list)
        yhat = np.array(fr_pred_list)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        model.r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        model.rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

    return model


# ---------------------------------------------------------------------------
# Model 2: Pooled model with type-specific λ and shared α
# ---------------------------------------------------------------------------
@dataclass
class PooledMarkovModel:
    """Shared attenuation α_k with type-specific scaling λ_type."""
    alphas: dict[int, float]
    alpha_cis: dict[int, tuple]
    lambdas: dict[str, float]       # error_type → λ
    r_squared: float = 0.0
    rmse: float = 0.0
    aic: float = 0.0
    bic: float = 0.0
    n_params: int = 0
    n_obs: int = 0


def fit_pooled_model(
    transition_df: pd.DataFrame,
    endpoint_df: pd.DataFrame,
    n_boot: int = 2000,
) -> PooledMarkovModel:
    """Fit a pooled model with shared α_k across error types."""
    # Shared α estimates (all error types pooled)
    alphas_raw = estimate_alphas(transition_df, error_type=None)
    alpha_cis = bootstrap_alpha_cis(transition_df, error_type=None, n_boot=n_boot)

    # Per-type λ
    lambdas = {}
    for etype in endpoint_df["error_type"].unique():
        lambdas[etype] = fit_lambda(endpoint_df, alphas_raw, etype)

    model = PooledMarkovModel(
        alphas={k: v[0] for k, v in alphas_raw.items()},
        alpha_cis=alpha_cis,
        lambdas=lambdas,
        n_params=N_TRANSITIONS + len(lambdas),
    )

    # Goodness of fit (pooled across all types)
    all_obs = []
    all_pred = []
    for etype in endpoint_df["error_type"].unique():
        df = endpoint_df[endpoint_df["error_type"] == etype]
        baselines = df[df["is_baseline"]]
        injected = df[~df["is_baseline"]]
        if baselines.empty or injected.empty:
            continue
        bl_mean = baselines["combined_score"].mean()
        if bl_mean <= 0:
            continue
        lam = lambdas.get(etype, 1.0)
        for step, group in injected.groupby("injection_step"):
            fr_obs = max(0.0, 1.0 - group["combined_score"].mean() / bl_mean)
            s_pred = 1.0
            for j in range(step + 1, N_CONTENT):
                s_pred *= model.alphas.get(j, 0.0)
            fr_pred = lam * s_pred
            all_obs.append(fr_obs)
            all_pred.append(fr_pred)

    n = len(all_obs)
    model.n_obs = n
    if n >= 2:
        y = np.array(all_obs)
        yhat = np.array(all_pred)
        residuals = y - yhat
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        model.r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        model.rmse = float(np.sqrt(np.mean(residuals ** 2)))
        # AIC / BIC (assuming Gaussian residuals)
        sigma2 = ss_res / n if n > 0 else 1.0
        k = model.n_params
        if sigma2 > 0:
            model.aic = n * np.log(sigma2) + 2 * k
            model.bic = n * np.log(sigma2) + k * np.log(n)

    return model


# ---------------------------------------------------------------------------
# Model 3: Null model (uniform α for all steps)
# ---------------------------------------------------------------------------
def fit_null_model(
    transition_df: pd.DataFrame,
    endpoint_df: pd.DataFrame,
) -> tuple[float, float, float]:
    """Null model: single α for all transitions. Returns (α_null, R², RMSE)."""
    df = transition_df[transition_df["s_from"] > 0.05].copy()
    ratios = np.clip(df["ratio"].values, 0.0, 1.0)
    alpha_null = float(np.mean(ratios)) if len(ratios) > 0 else 0.0

    all_obs = []
    all_pred = []
    for etype in endpoint_df["error_type"].unique():
        edf = endpoint_df[endpoint_df["error_type"] == etype]
        baselines = edf[edf["is_baseline"]]
        injected = edf[~edf["is_baseline"]]
        if baselines.empty or injected.empty:
            continue
        bl_mean = baselines["combined_score"].mean()
        if bl_mean <= 0:
            continue
        for step, group in injected.groupby("injection_step"):
            fr_obs = max(0.0, 1.0 - group["combined_score"].mean() / bl_mean)
            n_downstream = max(0, N_CONTENT - step - 1)
            s_pred = alpha_null ** n_downstream
            all_obs.append(fr_obs)
            all_pred.append(s_pred)

    if len(all_obs) < 2:
        return alpha_null, 0.0, 0.0

    y = np.array(all_obs)
    yhat = np.array(all_pred)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
    return alpha_null, r2, rmse


# ---------------------------------------------------------------------------
# Compound injection: independence test
# ---------------------------------------------------------------------------
def test_compound_independence(
    records: list[dict],
    per_type_models: dict[str, MarkovModel],
) -> pd.DataFrame:
    """Test whether compound injection is super-additive.

    Under the Markov independence assumption:
        S(i,j) = S(i) · S(j)    for injection at both steps i and j

    We compare observed compound survival against this prediction.
    """
    from record_utils import is_baseline, injection_is_valid

    compound_records = [
        r for r in records
        if r.get("compound_steps")
        and not is_baseline(r)
        and injection_is_valid(r) is not False
    ]

    if not compound_records:
        return pd.DataFrame()

    # Collect single-step failure rates
    endpoint_df = build_endpoint_data(records)

    rows = []
    for r in compound_records:
        etype = r.get("error_type")
        model = per_type_models.get(etype)
        if not model:
            continue

        steps = r.get("compound_steps", [])
        if len(steps) < 2:
            continue

        # Observed compound survival at compose step
        efs = r.get("error_found_in_step", {})
        obs_survival = efs.get("compose", {}).get("survival_score", 0.0)

        # Predicted under independence: S(i)·S(j)
        pred_survival = 1.0
        for s in steps:
            pred_survival *= model.predict_survival(s)

        rows.append({
            "error_type": etype,
            "steps": str(steps),
            "obs_survival": obs_survival,
            "pred_survival_indep": pred_survival,
            "interaction": obs_survival - pred_survival,
            "super_additive": obs_survival > pred_survival,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output: CSV tables
# ---------------------------------------------------------------------------
def export_alpha_table(
    per_type_models: dict[str, MarkovModel],
    pooled: PooledMarkovModel,
    out_path: str,
):
    """Export α estimates with CIs to CSV."""
    rows = []
    for etype, model in per_type_models.items():
        for trans_idx in range(N_TRANSITIONS):
            from_name = WORKFLOW_STEPS[trans_idx]
            to_name = WORKFLOW_STEPS[trans_idx + 1]
            alpha = model.alphas.get(trans_idx, 0.0)
            ci = model.alpha_cis.get(trans_idx, (0.0, 0.0))
            n = model.n_observations.get(trans_idx, 0)
            rows.append({
                "scope": etype,
                "transition": f"{from_name} → {to_name}",
                "from_step": trans_idx,
                "to_step": trans_idx + 1,
                "alpha": round(alpha, 4),
                "ci_lo": round(ci[0], 4),
                "ci_hi": round(ci[1], 4),
                "n": n,
            })

    # Add pooled model
    for trans_idx in range(N_TRANSITIONS):
        from_name = WORKFLOW_STEPS[trans_idx]
        to_name = WORKFLOW_STEPS[trans_idx + 1]
        alpha = pooled.alphas.get(trans_idx, 0.0)
        ci = pooled.alpha_cis.get(trans_idx, (0.0, 0.0))
        rows.append({
            "scope": "pooled",
            "transition": f"{from_name} → {to_name}",
            "from_step": trans_idx,
            "to_step": trans_idx + 1,
            "alpha": round(alpha, 4),
            "ci_lo": round(ci[0], 4),
            "ci_hi": round(ci[1], 4),
            "n": 0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return df


def export_predictions(
    per_type_models: dict[str, MarkovModel],
    endpoint_df: pd.DataFrame,
    out_path: str,
):
    """Export predicted vs observed failure rates."""
    rows = []
    for etype, model in per_type_models.items():
        edf = endpoint_df[endpoint_df["error_type"] == etype]
        baselines = edf[edf["is_baseline"]]
        injected = edf[~edf["is_baseline"]]
        if baselines.empty or injected.empty:
            continue
        bl_mean = baselines["combined_score"].mean()
        if bl_mean <= 0:
            continue

        for step, group in injected.groupby("injection_step"):
            fr_obs = max(0.0, 1.0 - group["combined_score"].mean() / bl_mean)
            fr_pred = model.predict_failure_rate(step)
            s_pred = model.predict_survival(step)
            step_name = WORKFLOW_STEPS[step] if step < N_STEPS else f"step_{step}"
            rows.append({
                "error_type": etype,
                "injection_step": step_name,
                "injection_step_idx": step,
                "fr_observed": round(fr_obs, 4),
                "fr_predicted": round(fr_pred, 4),
                "survival_predicted": round(s_pred, 4),
                "residual": round(fr_obs - fr_pred, 4),
                "n": len(group),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return df


def export_model_fit(
    per_type_models: dict[str, MarkovModel],
    pooled: PooledMarkovModel,
    null_alpha: float,
    null_r2: float,
    null_rmse: float,
    out_path: str,
):
    """Export model comparison statistics."""
    rows = []
    for etype, model in per_type_models.items():
        rows.append({
            "model": f"per_type_{etype}",
            "n_params": N_TRANSITIONS + 1,
            "r_squared": round(model.r_squared, 4),
            "rmse": round(model.rmse, 4),
            "lambda": round(model.lambda_scale, 4),
        })
    rows.append({
        "model": "pooled",
        "n_params": pooled.n_params,
        "r_squared": round(pooled.r_squared, 4),
        "rmse": round(pooled.rmse, 4),
        "aic": round(pooled.aic, 2),
        "bic": round(pooled.bic, 2),
    })
    rows.append({
        "model": "null_uniform",
        "n_params": 1,
        "alpha_uniform": round(null_alpha, 4),
        "r_squared": round(null_r2, 4),
        "rmse": round(null_rmse, 4),
    })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return df


# ---------------------------------------------------------------------------
# Output: Figures
# ---------------------------------------------------------------------------
def fig_alpha_cascade(per_type_models: dict[str, MarkovModel], out_path: str):
    """Bar chart showing α_k per step transition, grouped by error type."""
    etypes = sorted(per_type_models.keys())
    n_types = len(etypes)
    if n_types == 0:
        return

    transition_labels = [
        f"{WORKFLOW_STEPS[i]}\n→ {WORKFLOW_STEPS[i+1]}"
        for i in range(N_TRANSITIONS)
    ]

    fig, ax = plt.subplots(figsize=(max(5, N_TRANSITIONS * 1.8), 3.2))
    x = np.arange(N_TRANSITIONS)
    width = 0.8 / n_types
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"]

    for j, etype in enumerate(etypes):
        model = per_type_models[etype]
        vals = [model.alphas.get(i, 0.0) for i in range(N_TRANSITIONS)]
        cis = [model.alpha_cis.get(i, (0, 0)) for i in range(N_TRANSITIONS)]
        yerr_lo = [max(0, vals[i] - cis[i][0]) for i in range(N_TRANSITIONS)]
        yerr_hi = [max(0, cis[i][1] - vals[i]) for i in range(N_TRANSITIONS)]
        offset = (j - n_types / 2 + 0.5) * width
        ax.bar(
            x + offset, vals, width, label=etype,
            yerr=[yerr_lo, yerr_hi], capsize=3,
            color=colors[j % len(colors)], alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(transition_labels, fontsize=8)
    ax.set_ylabel(r"Attenuation factor $\alpha_k$")
    ax.set_title(r"Step-wise attenuation $\alpha_k = P(s_k > 0 \mid s_{k-1} > 0)$")
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(title="Error type", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def fig_predicted_vs_observed(predictions_df: pd.DataFrame, out_path: str):
    """Scatter plot of predicted vs observed failure rates with R² annotation."""
    if predictions_df.empty:
        return

    fig, ax = plt.subplots(figsize=(4, 4))
    markers = {"factual": "o", "semantic": "s", "omission": "^"}
    colors = {"factual": "#2196F3", "semantic": "#FF9800", "omission": "#4CAF50"}

    for etype, group in predictions_df.groupby("error_type"):
        ax.scatter(
            group["fr_observed"], group["fr_predicted"],
            marker=markers.get(etype, "o"),
            color=colors.get(etype, "#333"),
            label=etype, s=40, alpha=0.8, edgecolors="white", linewidth=0.5,
        )

    # Perfect prediction line
    lims = [0, max(0.5, predictions_df[["fr_observed", "fr_predicted"]].max().max() + 0.05)]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="y = x")

    # Overall R²
    y = predictions_df["fr_observed"].values
    yhat = predictions_df["fr_predicted"].values
    if len(y) >= 2:
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse = np.sqrt(np.mean((y - yhat) ** 2))
        ax.text(
            0.05, 0.92,
            f"$R^2 = {r2:.3f}$\nRMSE = {rmse:.3f}",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    ax.set_xlabel("Observed failure rate")
    ax.set_ylabel("Predicted failure rate (Markov)")
    ax.set_title("Markov model: predicted vs observed")
    ax.legend(fontsize=7)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def fig_cascade_waterfall(
    per_type_models: dict[str, MarkovModel],
    out_path: str,
):
    """Waterfall / cascade diagram: show how error signal decays step by step.

    For each error type, starting from injection at step 0 with S=1.0,
    multiply by α_k at each transition to show the cascade.
    """
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = {"factual": "#2196F3", "semantic": "#FF9800", "omission": "#4CAF50"}
    step_labels = WORKFLOW_STEPS

    for etype, model in sorted(per_type_models.items()):
        s = [1.0]  # start at injection step 0
        for k in range(N_TRANSITIONS):
            s.append(s[-1] * model.alphas.get(k, 0.0))
        ax.plot(
            range(N_STEPS), s,
            marker="o", color=colors.get(etype, "#333"),
            label=f"{etype} (λ={model.lambda_scale:.2f})",
            linewidth=2, markersize=6,
        )

    ax.set_xticks(range(N_STEPS))
    ax.set_xticklabels(step_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Survival intensity $S_k$")
    ax.set_xlabel("Pipeline step")
    ax.set_title("Error signal cascade from step 0")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Output: LaTeX table
# ---------------------------------------------------------------------------
def latex_alpha_table(
    per_type_models: dict[str, MarkovModel],
    pooled: PooledMarkovModel,
    null_r2: float,
    out_path: str,
):
    """Generate LaTeX table for the paper (Table: Markov model parameters)."""
    etypes = sorted(per_type_models.keys())
    transition_labels = [
        f"{WORKFLOW_STEPS[i]} $\\to$ {WORKFLOW_STEPS[i+1]}"
        for i in range(N_TRANSITIONS)
    ]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Markov attenuation model: estimated step-wise transition "
        r"probabilities $\alpha_k$ with 95\% bootstrap CIs. $\lambda$ is the "
        r"severity-dependent scaling factor. $R^2$ is computed on per-condition "
        r"failure rates (predicted vs.\ observed).}",
        r"\label{tab:markov}",
    ]

    # Column layout: transition | type1_alpha | type2_alpha | ... | pooled | n
    n_cols = len(etypes) + 2  # transitions + pooled + column header
    col_spec = "l" + "c" * (len(etypes) + 1)
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = "Transition"
    for etype in etypes:
        header += f" & {etype.title()}"
    header += r" & Pooled \\"
    lines.append(header)
    lines.append(r"\midrule")

    for i in range(N_TRANSITIONS):
        row = transition_labels[i]
        for etype in etypes:
            model = per_type_models[etype]
            a = model.alphas.get(i, 0.0)
            ci = model.alpha_cis.get(i, (0, 0))
            row += f" & {a:.3f} ({ci[0]:.2f}, {ci[1]:.2f})"
        a_p = pooled.alphas.get(i, 0.0)
        ci_p = pooled.alpha_cis.get(i, (0, 0))
        row += f" & {a_p:.3f} ({ci_p[0]:.2f}, {ci_p[1]:.2f})"
        row += r" \\"
        lines.append(row)

    lines.append(r"\midrule")

    # Lambda row
    row_lam = r"$\lambda$"
    for etype in etypes:
        lam = per_type_models[etype].lambda_scale
        row_lam += f" & {lam:.3f}"
    row_lam += r" & --- \\"
    lines.append(row_lam)

    # R² row
    row_r2 = r"$R^2$"
    for etype in etypes:
        r2 = per_type_models[etype].r_squared
        row_r2 += f" & {r2:.3f}"
    row_r2 += f" & {pooled.r_squared:.3f}"
    row_r2 += r" \\"
    lines.append(row_r2)

    # RMSE row
    row_rmse = r"RMSE"
    for etype in etypes:
        rmse = per_type_models[etype].rmse
        row_rmse += f" & {rmse:.4f}"
    row_rmse += f" & {pooled.rmse:.4f}"
    row_rmse += r" \\"
    lines.append(row_rmse)

    lines.append(r"\midrule")
    lines.append(
        f"Null (uniform $\\alpha$) & \\multicolumn{{{len(etypes) + 1}}}{{c}}"
        f"{{$R^2 = {null_r2:.3f}$}}"
        + r" \\"
    )

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fit Markov error propagation model"
    )
    parser.add_argument(
        "--error-type", default=None,
        help="Fit only this error type (default: all)"
    )
    parser.add_argument(
        "--bootstrap", type=int, default=2000,
        help="Number of bootstrap resamples for CIs"
    )
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(STATS_DIR, exist_ok=True)
    os.makedirs(TABLE_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    records = load_records()
    if not records:
        print("No records found. Run the sweep first.")
        return

    print(f"Loaded {len(records)} records")

    transition_df = build_transition_data(records)
    endpoint_df = build_endpoint_data(records)

    if transition_df.empty:
        print("No transition data available. Need injected records with "
              "error_found_in_step data.")
        return

    print(f"Transition pairs: {len(transition_df)}")
    print(f"Endpoint records: {len(endpoint_df)}")
    print(f"Error types: {sorted(transition_df['error_type'].unique())}")

    # ------------------------------------------------------------------
    # Fit per-type models
    # ------------------------------------------------------------------
    etypes = sorted(transition_df["error_type"].unique())
    if args.error_type:
        etypes = [args.error_type]

    per_type_models = {}
    for etype in etypes:
        print(f"\n{'='*60}")
        print(f"Fitting model: {etype}")
        print(f"{'='*60}")

        model = fit_model(
            transition_df, endpoint_df, etype,
            n_boot=args.bootstrap,
        )
        per_type_models[etype] = model

        print(f"  Attenuation factors (α_k):")
        for i in range(N_TRANSITIONS):
            a = model.alphas.get(i, 0.0)
            ci = model.alpha_cis.get(i, (0, 0))
            n = model.n_observations.get(i, 0)
            print(f"    {WORKFLOW_STEPS[i]} → {WORKFLOW_STEPS[i+1]}: "
                  f"α = {a:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  "
                  f"(n={n})")
        print(f"  λ = {model.lambda_scale:.4f}")
        print(f"  R² = {model.r_squared:.4f}")
        print(f"  RMSE = {model.rmse:.4f}")

        print(f"\n  End-to-end survival predictions S(i):")
        for i in range(N_CONTENT):
            s = model.predict_survival(i)
            fr = model.predict_failure_rate(i)
            print(f"    Inject at {WORKFLOW_STEPS[i]}: "
                  f"S = {s:.4f}, FR_pred = {fr:.4f}")

    # ------------------------------------------------------------------
    # Fit pooled model
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Fitting pooled model (shared α, per-type λ)")
    print(f"{'='*60}")
    pooled = fit_pooled_model(transition_df, endpoint_df, n_boot=args.bootstrap)
    print(f"  Shared α: {pooled.alphas}")
    print(f"  Per-type λ: {pooled.lambdas}")
    print(f"  R² = {pooled.r_squared:.4f}, RMSE = {pooled.rmse:.4f}")
    if pooled.aic:
        print(f"  AIC = {pooled.aic:.2f}, BIC = {pooled.bic:.2f}")

    # ------------------------------------------------------------------
    # Fit null model
    # ------------------------------------------------------------------
    null_alpha, null_r2, null_rmse = fit_null_model(transition_df, endpoint_df)
    print(f"\nNull model: α_uniform = {null_alpha:.4f}, R² = {null_r2:.4f}, "
          f"RMSE = {null_rmse:.4f}")

    # ------------------------------------------------------------------
    # Model comparison
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Model comparison")
    print(f"{'='*60}")
    for etype, model in per_type_models.items():
        delta_r2 = model.r_squared - null_r2
        print(f"  {etype}: R²={model.r_squared:.4f} vs null R²={null_r2:.4f} "
              f"(ΔR²={delta_r2:+.4f})")
    print(f"  Pooled: R²={pooled.r_squared:.4f} vs null R²={null_r2:.4f} "
          f"(ΔR²={pooled.r_squared - null_r2:+.4f})")

    # ------------------------------------------------------------------
    # Compound independence test
    # ------------------------------------------------------------------
    compound_df = test_compound_independence(records, per_type_models)
    if not compound_df.empty:
        print(f"\nCompound independence test: {len(compound_df)} records")
        compound_path = os.path.join(STATS_DIR, "markov_compound_test.csv")
        compound_df.to_csv(compound_path, index=False)
        print(f"Wrote {compound_path}")

        # Summary
        agg = compound_df.groupby("error_type").agg({
            "obs_survival": "mean",
            "pred_survival_indep": "mean",
            "interaction": "mean",
            "super_additive": "mean",
        })
        print(agg.to_string())

    # ------------------------------------------------------------------
    # Export CSVs
    # ------------------------------------------------------------------
    alpha_df = export_alpha_table(
        per_type_models, pooled,
        os.path.join(STATS_DIR, "markov_alpha_estimates.csv"),
    )

    pred_df = export_predictions(
        per_type_models, endpoint_df,
        os.path.join(STATS_DIR, "markov_predictions.csv"),
    )

    export_model_fit(
        per_type_models, pooled, null_alpha, null_r2, null_rmse,
        os.path.join(STATS_DIR, "markov_model_fit.csv"),
    )

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    fig_alpha_cascade(
        per_type_models,
        os.path.join(FIG_DIR, "fig_alpha_cascade.pdf"),
    )

    fig_predicted_vs_observed(
        pred_df,
        os.path.join(FIG_DIR, "fig_predicted_vs_observed.pdf"),
    )

    fig_cascade_waterfall(
        per_type_models,
        os.path.join(FIG_DIR, "fig_cascade_waterfall.pdf"),
    )

    # ------------------------------------------------------------------
    # Generate LaTeX table
    # ------------------------------------------------------------------
    latex_alpha_table(
        per_type_models, pooled, null_r2,
        os.path.join(TABLE_DIR, "table_markov_alphas.tex"),
    )

    print(f"\n{'='*60}")
    print("Done. All outputs written.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
