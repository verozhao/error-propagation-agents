"""Phase 4.1 + 4.3: paired statistical tests and bootstrap CIs.

Operates on existing `results/**/*.json` files using the v1 `combined_score`
(no API calls). For each (model, error_type, injection_step), compares
error-injected scores to baseline (error_step=None) scores using a Wilcoxon
signed-rank test on per-trial differences and an independent-samples
fallback (Mann-Whitney U) when trials cannot be paired by index.

Bonferroni correction is applied per error_type (15 conditions = 5 steps
x 3 queries) by default; pass --conditions-per-family to override.

Outputs:
    results/stats/significance.csv
    results/stats/failure_rates_with_ci.csv
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob

import numpy as np
import pandas as pd
from scipy import stats

from config import WORKFLOW_STEPS


def load_all(results_glob: str = "results/**/*.json") -> pd.DataFrame:
    rows = []
    for path in glob(results_glob, recursive=True):
        if "stats" in path or "trace_analysis" in path or "sanity_checks" in path:
            continue
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue
        fname = os.path.basename(path)
        fallback_model = fname.split("_")[0]
        for d in data:
            if "evaluation" not in d:
                continue
            ev = d["evaluation"]
            rows.append(
                {
                    "model": d.get("model") or fallback_model,
                    "task_query": d.get("task_query"),
                    "error_type": d.get("error_type"),
                    "error_step": -1 if d.get("error_step") is None else d.get("error_step"),
                    "trial": d.get("trial"),
                    "combined_score": ev.get("combined_score"),
                    "combined_score_v2": ev.get("combined_score_v2"),
                    "source": path,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def run_significance(df: pd.DataFrame, conditions_per_family: int = 15) -> pd.DataFrame:
    out = []
    for (model, etype), sub in df.groupby(["model", "error_type"]):
        baseline = sub[sub["error_step"] == -1]["combined_score"].dropna().values
        if len(baseline) == 0:
            continue
        for step in range(len(WORKFLOW_STEPS)):
            inj = sub[sub["error_step"] == step]["combined_score"].dropna().values
            if len(inj) == 0:
                continue
            mean_b = float(np.mean(baseline))
            mean_i = float(np.mean(inj))
            mean_diff = mean_b - mean_i

            # paired Wilcoxon when sample sizes match, else Mann-Whitney
            try:
                if len(inj) == len(baseline):
                    stat, p = stats.wilcoxon(baseline, inj, zero_method="zsplit")
                    test = "wilcoxon_paired"
                else:
                    stat, p = stats.mannwhitneyu(baseline, inj, alternative="two-sided")
                    test = "mannwhitney"
            except ValueError:
                stat, p, test = float("nan"), float("nan"), "skipped"

            diffs = baseline.mean() - inj  # baseline mean minus per-trial inj
            ci_lo, ci_hi = bootstrap_ci(inj)
            bonf_p = min(1.0, p * conditions_per_family) if p == p else p

            out.append(
                {
                    "model": model,
                    "error_type": etype,
                    "injection_step": WORKFLOW_STEPS[step],
                    "n_baseline": len(baseline),
                    "n_injected": len(inj),
                    "mean_baseline": mean_b,
                    "mean_injected": mean_i,
                    "mean_diff": mean_diff,
                    "injected_ci95_lo": ci_lo,
                    "injected_ci95_hi": ci_hi,
                    "test": test,
                    "p_value": float(p) if p == p else None,
                    "p_value_bonferroni": float(bonf_p) if bonf_p == bonf_p else None,
                    "significant_after_correction": bool(bonf_p < 0.05) if bonf_p == bonf_p else False,
                }
            )
    return pd.DataFrame(out)


def failure_rates_with_ci(df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    for (model, etype), sub in df.groupby(["model", "error_type"]):
        baseline = sub[sub["error_step"] == -1]["combined_score"].dropna().values
        if len(baseline) == 0:
            continue
        baseline_mean = float(baseline.mean())
        for step in range(len(WORKFLOW_STEPS)):
            inj = sub[sub["error_step"] == step]["combined_score"].dropna().values
            if len(inj) == 0:
                continue
            mean_score = float(inj.mean())
            failure_rate = max(0.0, 1 - (mean_score / baseline_mean)) if baseline_mean else 0.0
            ci_lo, ci_hi = bootstrap_ci(inj, n_boot=n_boot)
            fr_lo = max(0.0, 1 - (ci_hi / baseline_mean)) if baseline_mean else 0.0
            fr_hi = max(0.0, 1 - (ci_lo / baseline_mean)) if baseline_mean else 0.0
            rows.append(
                {
                    "model": model,
                    "error_type": etype,
                    "step_name": WORKFLOW_STEPS[step],
                    "n": len(inj),
                    "baseline_mean": baseline_mean,
                    "mean_score": mean_score,
                    "failure_rate": failure_rate,
                    "failure_rate_ci_lo": fr_lo,
                    "failure_rate_ci_hi": fr_hi,
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glob", default="results/**/*.json")
    parser.add_argument("--out", default="results/stats")
    parser.add_argument("--conditions-per-family", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_all(args.glob)
    if df.empty:
        print(f"No records loaded from {args.glob}")
        return
    print(f"Loaded {len(df)} trial records across {df['model'].nunique()} models, "
          f"{df['error_type'].nunique()} error types.")

    sig = run_significance(df, args.conditions_per_family)
    sig.to_csv(os.path.join(args.out, "significance.csv"), index=False)
    print(f"Wrote {len(sig)} significance rows.")

    fr = failure_rates_with_ci(df)
    fr.to_csv(os.path.join(args.out, "failure_rates_with_ci.csv"), index=False)
    print(f"Wrote {len(fr)} failure-rate rows with 95% CIs.")


if __name__ == "__main__":
    main()
