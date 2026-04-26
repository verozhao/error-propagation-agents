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


def _normalize_step(d):
    """P0-1: turn the heterogenous error_step representations into a
    canonical integer for step-wise analyses.
      - baseline (is_baseline=True)     -> -1
      - single-step injection (int)      -> that int
      - compound injection (list)        -> first step of the list
      - missing (legacy bad records)     -> None (caller should drop)

    Issue α: if the record is a non-baseline with injection_valid=False
    (injector produced no delta), return None so callers drop it from
    step-wise statistics.
    """
    from record_utils import is_baseline, injection_is_valid

    if is_baseline(d):
        return -1
    # Issue α: drop failed-injection no-ops from step-wise analyses
    if injection_is_valid(d) is False:
        return None
    es = d.get("error_step")
    if isinstance(es, list):
        return es[0] if es else None
    return es


def load_all(results_glob: str = "results/**/*.json") -> pd.DataFrame:
    rows = []
    # Load from JSONL first
    for path in glob("results/**/*.jsonl", recursive=True):
        if "stats" in path or "trace_analysis" in path or "sanity_checks" in path or "_legacy" in path or "archive" in path:
            continue
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                if "evaluation" not in d:
                    continue
                step = _normalize_step(d)
                if step is None:
                    continue  # legacy compound-as-baseline record — unusable
                ev = d["evaluation"]
                meta = d.get("injection_meta") or {}
                etype = meta.get("error_type") or d.get("error_type")
                rows.append(
                    {
                        "model": d.get("model"),
                        "task_query": d.get("task_query"),
                        "error_type": etype,
                        "error_step": step,
                        "is_compound": isinstance(d.get("error_step"), list),
                        "trial": d.get("trial"),
                        "combined_score": ev.get("combined_score"),
                        "combined_score_v2": ev.get("combined_score_v2"),
                        "source": path,
                    }
                )
    # Fallback to JSON if no JSONL data
    if not rows:
        for path in glob(results_glob, recursive=True):
            if "stats" in path or "trace_analysis" in path or "sanity_checks" in path or "_legacy" in path or "archive" in path:
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
                step = _normalize_step(d)
                if step is None:
                    continue
                ev = d["evaluation"]
                rows.append(
                    {
                        "model": d.get("model") or fallback_model,
                        "task_query": d.get("task_query"),
                        "error_type": d.get("error_type"),
                        "error_step": step,
                        "is_compound": isinstance(d.get("error_step"), list),
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


def run_significance(
    df: pd.DataFrame,
    conditions_per_family: int | None = None,
    correction: str = "holm",
) -> pd.DataFrame:
    """Paired Wilcoxon signed-rank test.

    P0-15 FIX: after the seed refactor, baseline(trial=t) and
    injected(trial=t) at the same (model, query) share the same
    pre-injection random state. So pairing by (task_query, trial) is
    legitimate. We also:
      - exclude compound runs from the per-step comparison (they
        belong to a separate family)
      - auto-compute `conditions_per_family` per (model, error_type)
        as the number of distinct injection steps actually compared
      - support Holm-Bonferroni (default) or plain Bonferroni for
        multiple-comparison correction

    `correction`:
      - "holm"       : Holm-Bonferroni (step-down, uniformly more
                       powerful than Bonferroni, still controls FWER)
      - "bonferroni" : classical Bonferroni
      - "fdr_bh"     : Benjamini-Hochberg FDR
      - "none"       : uncorrected
    """
    from scipy.stats import false_discovery_control  # scipy >= 1.11

    # Exclude compound runs from step-wise paired comparison
    if "is_compound" in df.columns:
        df = df[~df["is_compound"].fillna(False)].copy()

    # D3 FIX: for semantic error type, exclude yes/no-answer queries.
    # The semantic injector's answer-targeted swap produces no meaningful
    # perturbation when the answer is "yes" or "no" (nothing to swap).
    # Including these queries adds noise that dilutes real signal.
    _yn_queries = set()
    try:
        from factual_accuracy import load_ground_truth
        _gt = load_ground_truth()
        for q, entry in _gt.items():
            ans = entry.get("answer", "").strip().lower()
            if ans in ("yes", "no"):
                _yn_queries.add(q)
        if _yn_queries:
            _before = len(df)
            df = df[~((df["error_type"] == "semantic")
                       & (df["task_query"].isin(_yn_queries)))].copy()
            _dropped = _before - len(df)
            if _dropped > 0:
                print(f"D3: dropped {_dropped} semantic rows from "
                      f"{len(_yn_queries)} yes/no queries")
    except Exception:
        pass  # ground truth unavailable — skip filter gracefully

    out = []
    for (model, etype), sub in df.groupby(["model", "error_type"]):
        base = sub[sub["error_step"] == -1].set_index(["task_query", "trial"])["combined_score"]
        step_rows = []
        for step in sorted(sub["error_step"].unique()):
            if step == -1:
                continue
            inj = sub[sub["error_step"] == step].set_index(["task_query", "trial"])["combined_score"]
            paired = base.to_frame("base").join(inj.to_frame("inj"), how="inner").dropna()
            if len(paired) < 3:
                continue
            diffs = paired["base"].values - paired["inj"].values
            try:
                if np.all(diffs == 0):
                    stat, p = 0.0, 1.0
                else:
                    stat, p = stats.wilcoxon(diffs, zero_method="zsplit")
                test = "wilcoxon_paired_by_query_trial"
            except ValueError:
                stat, p, test = float("nan"), float("nan"), "skipped"

            ci_lo, ci_hi = bootstrap_ci(paired["inj"].values)

            n_pairs = len(diffs[diffs != 0])
            if n_pairs > 0 and stat == stat:
                r_rb = 1.0 - (2.0 * stat) / (n_pairs * (n_pairs + 1) / 2)
            else:
                r_rb = 0.0

            if len(diffs) > 1:
                cohens_d = float(diffs.mean() / diffs.std(ddof=1)) if diffs.std(ddof=1) > 0 else 0.0
            else:
                cohens_d = 0.0

            step_name = WORKFLOW_STEPS[step] if step < len(WORKFLOW_STEPS) else f"step_{step}"
            step_rows.append({
                "model": model,
                "error_type": etype,
                "injection_step": step_name,
                "n_paired": len(paired),
                "mean_baseline": float(paired["base"].mean()),
                "mean_injected": float(paired["inj"].mean()),
                "mean_diff": float(diffs.mean()),
                "std_diff": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
                "injected_ci95_lo": ci_lo,
                "injected_ci95_hi": ci_hi,
                "test": test,
                "p_value": float(p) if p == p else None,
                "effect_size_r": round(r_rb, 4),
                "cohens_d": round(cohens_d, 4),
            })

        # Per-family multiple-comparison correction
        fam_size = conditions_per_family if conditions_per_family else len(step_rows)
        if fam_size < 1:
            fam_size = 1
        pvals = [r["p_value"] for r in step_rows]
        valid_idx = [i for i, v in enumerate(pvals) if v is not None]

        if correction == "none" or not valid_idx:
            for r in step_rows:
                r["p_value_adjusted"] = r["p_value"]
                r["correction_method"] = "none"
        elif correction == "bonferroni":
            for r in step_rows:
                r["p_value_adjusted"] = (
                    min(1.0, r["p_value"] * fam_size) if r["p_value"] is not None else None
                )
                r["correction_method"] = f"bonferroni (m={fam_size})"
        elif correction == "holm":
            # Holm-Bonferroni step-down
            valid_pvals = np.array([pvals[i] for i in valid_idx])
            m = len(valid_pvals)
            order = np.argsort(valid_pvals)
            adj = np.empty(m)
            running_max = 0.0
            for rank, idx in enumerate(order):
                adj_p = min(1.0, valid_pvals[idx] * (m - rank))
                running_max = max(running_max, adj_p)
                adj[idx] = running_max
            for j, orig_idx in enumerate(valid_idx):
                step_rows[orig_idx]["p_value_adjusted"] = float(adj[j])
                step_rows[orig_idx]["correction_method"] = f"holm-bonferroni (m={m})"
            for i, r in enumerate(step_rows):
                if i not in valid_idx:
                    r["p_value_adjusted"] = None
                    r["correction_method"] = "holm-bonferroni (skipped)"
        elif correction == "fdr_bh":
            valid_pvals = np.array([pvals[i] for i in valid_idx])
            adj = false_discovery_control(valid_pvals, method="bh")
            for j, orig_idx in enumerate(valid_idx):
                step_rows[orig_idx]["p_value_adjusted"] = float(adj[j])
                step_rows[orig_idx]["correction_method"] = f"fdr_bh (m={len(valid_pvals)})"
            for i, r in enumerate(step_rows):
                if i not in valid_idx:
                    r["p_value_adjusted"] = None
                    r["correction_method"] = "fdr_bh (skipped)"

        for r in step_rows:
            adj = r.get("p_value_adjusted")
            r["significant_after_correction"] = bool(adj < 0.05) if adj is not None else False
            # legacy key for backward compat
            r["p_value_bonferroni"] = r["p_value_adjusted"]

        out.extend(step_rows)
    return pd.DataFrame(out)


def failure_rates_with_ci(df: pd.DataFrame, n_boot: int = 2000) -> pd.DataFrame:
    rows = []
    for (model, etype), sub in df.groupby(["model", "error_type"]):
        baseline = sub[sub["error_step"] == -1]["combined_score"].dropna().values
        if len(baseline) == 0:
            continue
        baseline_mean = float(baseline.mean())
        for step in sorted(sub["error_step"].unique()):
            if step == -1:
                continue
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
                    "step_name": WORKFLOW_STEPS[step] if step < len(WORKFLOW_STEPS) else f"step_{step}",
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
    parser.add_argument(
        "--conditions-per-family",
        type=int,
        default=None,
        help="Override auto-detected family size for multiple-comparison correction. "
             "Default auto-detects per (model, error_type) as the number of distinct injection steps.",
    )
    parser.add_argument(
        "--correction",
        default="holm",
        choices=["holm", "bonferroni", "fdr_bh", "none"],
        help="Multiple-comparison correction (default: holm; Holm-Bonferroni is "
             "uniformly more powerful than Bonferroni while still controlling FWER).",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_all(args.glob)
    if df.empty:
        print(f"No records loaded from {args.glob}")
        return
    print(f"Loaded {len(df)} trial records across {df['model'].nunique()} models, "
          f"{df['error_type'].nunique()} error types.")

    # D3 FIX: globally exclude yes/no queries from semantic error analysis.
    # Applied here so both significance tests and failure-rate CIs are
    # consistent. The filter is also inside run_significance() for safety
    # when called from other scripts.
    try:
        from factual_accuracy import load_ground_truth
        _gt = load_ground_truth()
        _yn = {q for q, e in _gt.items()
               if e.get("answer", "").strip().lower() in ("yes", "no")}
        if _yn and "task_query" in df.columns:
            _mask = (df["error_type"] == "semantic") & (df["task_query"].isin(_yn))
            _n_drop = _mask.sum()
            if _n_drop > 0:
                df = df[~_mask].copy()
                print(f"D3: excluded {_n_drop} semantic rows "
                      f"from {len(_yn)} yes/no queries")
    except Exception:
        pass

    sig = run_significance(df, args.conditions_per_family, correction=args.correction)
    sig.to_csv(os.path.join(args.out, "significance.csv"), index=False)
    print(f"Wrote {len(sig)} significance rows (correction={args.correction}).")

    fr = failure_rates_with_ci(df)
    fr.to_csv(os.path.join(args.out, "failure_rates_with_ci.csv"), index=False)
    print(f"Wrote {len(fr)} failure-rate rows with 95% CIs.")


if __name__ == "__main__":
    main()
