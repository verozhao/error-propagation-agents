"""Severity-dependent analysis: vulnerability scores, attenuation factors,
and degradation-vs-severity curves.

Expects results organized as results/{error_type}_error_sev{severity}/*.json
or results/{error_type}_error/*.json with severity stored in each record.

Usage:
    python severity_analysis.py                    # analyze all available data
    python severity_analysis.py --error-type factual  # single error type
"""

import json
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import WORKFLOW_STEPS, OUTPUT_DIR


def load_severity_results(results_dir="results") -> pd.DataFrame:
    """Load all experiment results, extracting severity from records or directory names."""
    rows = []
    for path in glob.glob(f"{results_dir}/**/*.json", recursive=True):
        if "stats" in path or "sanity" in path or "_legacy" in path or "archive" in path:
            continue
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            continue

        # try to infer severity from directory name (e.g. factual_error_sev2)
        dir_name = os.path.basename(os.path.dirname(path))
        dir_severity = None
        if "_sev" in dir_name:
            try:
                dir_severity = int(dir_name.split("_sev")[-1])
            except ValueError:
                pass

        for d in data:
            if "evaluation" not in d:
                continue
            from record_utils import is_baseline as _is_baseline, injection_is_valid
            if _is_baseline(d):
                step = -1
            else:
                # Issue α: drop failed-injection no-ops
                if injection_is_valid(d) is False:
                    continue
                es = d.get("error_step")
                if isinstance(es, list):
                    step = es[0] if es else None
                else:
                    step = es
                if step is None:
                    continue  # legacy bad record — skip
            es = d.get("error_step")
            ev = d["evaluation"]
            meta = d.get("injection_meta") or {}
            rows.append({
                "model": d.get("model", os.path.basename(path).split("_")[0]),
                "task_query": d.get("task_query"),
                "error_type": d.get("error_type"),
                "error_step": step,
                "is_compound": isinstance(es, list),
                "severity": d.get("severity", dir_severity or 1),
                "severity_physical": meta.get("severity_physical"),
                "trial": d.get("trial"),
                "combined_score": ev.get("combined_score", 0),
                "combined_score_v2": ev.get("combined_score_v2"),
                "combined_score_v3": ev.get("combined_score_v3"),
            })
    return pd.DataFrame(rows)


def compute_failure_rates_by_severity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute failure rates grouped by (model, error_type, severity, step).

    Baselines are shared across severity levels: the sweep only produces
    baselines at sev=1 (sev2/3 use --skip-baseline) because the
    deterministic seed excludes severity — so sev=1 baselines ARE the
    baselines for all severities.
    """
    results = []

    # Pre-compute baseline per (model, error_type) — shared across severities
    baseline_pool = {}
    for (model, etype), group in df.groupby(["model", "error_type"]):
        bl = group[group["error_step"] == -1]["combined_score"]
        if not bl.empty:
            baseline_pool[(model, etype)] = bl.mean()

    for (model, etype, sev), group in df.groupby(["model", "error_type", "severity"]):
        baseline_mean = baseline_pool.get((model, etype))
        if baseline_mean is None:
            continue

        for step in sorted(group["error_step"].unique()):
            if step == -1:
                continue
            step_df = group[group["error_step"] == step]
            if step_df.empty:
                continue
            mean_score = step_df["combined_score"].mean()
            failure_rate = max(0, (baseline_mean - mean_score) / baseline_mean)
            phys = step_df["severity_physical"].dropna()
            results.append({
                "model": model,
                "error_type": etype,
                "severity": sev,
                "severity_physical_mean": float(phys.mean()) if len(phys) > 0 else None,
                "error_step": step,
                "step_name": WORKFLOW_STEPS[step],
                "baseline_score": baseline_mean,
                "mean_score": mean_score,
                "failure_rate": failure_rate,
                "n": len(step_df),
            })
    return pd.DataFrame(results)


def compute_vulnerability_scores(fr_df: pd.DataFrame) -> pd.DataFrame:
    """V(step) = mean failure rate across all severities for a given (model, error_type, step)."""
    vuln = fr_df.groupby(["model", "error_type", "step_name"]).agg(
        vulnerability=("failure_rate", "mean"),
        max_failure_rate=("failure_rate", "max"),
        n_severities=("severity", "nunique"),
    ).reset_index()
    vuln = vuln.sort_values(["model", "error_type", "vulnerability"], ascending=[True, True, False])
    return vuln


def compute_attenuation_factors(fr_df: pd.DataFrame) -> pd.DataFrame:
    """A(step_i) = 1 - FR(injected at step_i) / FR(injected at step_{i-1}).

    Positive = this step attenuates errors from the previous step.
    Negative = this step amplifies errors.
    """
    results = []
    for (model, etype, sev), group in fr_df.groupby(["model", "error_type", "severity"]):
        group = group.sort_values("error_step")
        rates = group.set_index("error_step")["failure_rate"]

        for step in range(1, len(WORKFLOW_STEPS)):
            prev, curr = step - 1, step
            if prev not in rates.index or curr not in rates.index:
                continue
            fr_prev = rates[prev]
            fr_curr = rates[curr]
            if fr_prev > 0:
                attenuation = 1.0 - (fr_curr / fr_prev)
            else:
                attenuation = 0.0 if fr_curr == 0 else -1.0

            results.append({
                "model": model,
                "error_type": etype,
                "severity": sev,
                "step_name": WORKFLOW_STEPS[step],
                "prev_step": WORKFLOW_STEPS[prev],
                "fr_prev": fr_prev,
                "fr_curr": fr_curr,
                "attenuation": attenuation,
            })
    return pd.DataFrame(results)


def plot_severity_curves(fr_df: pd.DataFrame, model: str = None, output_dir: str = "figures"):
    """One plot per error type: x=step, lines=severity levels."""
    os.makedirs(output_dir, exist_ok=True)
    if model:
        fr_df = fr_df[fr_df["model"] == model]

    for etype in fr_df["error_type"].unique():
        edf = fr_df[fr_df["error_type"] == etype]
        fig, ax = plt.subplots(figsize=(8, 5))

        for sev in sorted(edf["severity"].unique()):
            sdf = edf[edf["severity"] == sev].sort_values("error_step")
            phys = sdf["severity_physical_mean"].dropna()
            label = f"sev={sev} (dose={phys.mean():.2f})" if len(phys) > 0 else f"sev={sev}"
            ax.plot(sdf["error_step"], sdf["failure_rate"],
                    marker="o", label=label, linewidth=2)

        ax.set_xlabel("Error Injection Step")
        ax.set_ylabel("Failure Rate")
        ax.set_title(f"{etype.title()} Error — Degradation by Severity")
        ax.set_xticks(range(len(WORKFLOW_STEPS)))
        ax.set_xticklabels(WORKFLOW_STEPS, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = f"{output_dir}/{etype}_severity_curves.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close()


def plot_severity_heatmaps(fr_df: pd.DataFrame, model: str = None, output_dir: str = "figures"):
    """One heatmap per error type: rows=severity, columns=step."""
    os.makedirs(output_dir, exist_ok=True)
    if model:
        fr_df = fr_df[fr_df["model"] == model]

    for etype in fr_df["error_type"].unique():
        edf = fr_df[fr_df["error_type"] == etype]
        pivot = edf.pivot_table(index="severity", columns="step_name",
                                values="failure_rate", aggfunc="mean")
        pivot = pivot[[s for s in WORKFLOW_STEPS if s in pivot.columns]]

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    cbar_kws={"label": "Failure Rate"})
        ax.set_title(f"{etype.title()} Error — Step × Severity")
        ax.set_ylabel("Severity")
        plt.tight_layout()

        path = f"{output_dir}/{etype}_severity_heatmap.png"
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
        plt.close()


def plot_vulnerability_ranking(vuln_df: pd.DataFrame, model: str = None, output_dir: str = "figures"):
    """Grouped bar chart of vulnerability scores per step, grouped by error type."""
    os.makedirs(output_dir, exist_ok=True)
    if model:
        vuln_df = vuln_df[vuln_df["model"] == model]

    fig, ax = plt.subplots(figsize=(10, 5))
    step_order = WORKFLOW_STEPS
    vuln_pivot = vuln_df.pivot_table(index="step_name", columns="error_type",
                                      values="vulnerability", aggfunc="mean")
    vuln_pivot = vuln_pivot.reindex(step_order)
    vuln_pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Vulnerability Score")
    ax.set_title("Step Vulnerability by Error Type")
    ax.set_xticklabels(step_order, rotation=45)
    ax.legend(title="Error Type")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    path = f"{output_dir}/vulnerability_ranking.png"
    plt.savefig(path, dpi=150)
    print(f"Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-type", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    df = load_severity_results(args.results_dir)
    if df.empty:
        print("No results found.")
        return

    if args.error_type:
        df = df[df["error_type"] == args.error_type]

    print(f"Loaded {len(df)} records: {df['severity'].nunique()} severities, "
          f"{df['model'].nunique()} models, {df['error_type'].nunique()} error types")

    fr_df = compute_failure_rates_by_severity(df)

    # Vulnerability scores
    vuln = compute_vulnerability_scores(fr_df)
    print("\n=== VULNERABILITY SCORES ===")
    print(vuln.to_string(index=False))

    # Attenuation factors
    att = compute_attenuation_factors(fr_df)
    if not att.empty:
        print("\n=== ATTENUATION FACTORS ===")
        att_summary = att.groupby(["error_type", "step_name"]).agg(
            mean_attenuation=("attenuation", "mean"),
        ).reset_index()
        print(att_summary.to_string(index=False))

    # Save CSVs
    os.makedirs(os.path.join(args.results_dir, "stats"), exist_ok=True)
    fr_df.to_csv(f"{args.results_dir}/stats/failure_rates_by_severity.csv", index=False)
    vuln.to_csv(f"{args.results_dir}/stats/vulnerability_scores.csv", index=False)
    if not att.empty:
        att.to_csv(f"{args.results_dir}/stats/attenuation_factors.csv", index=False)
    print(f"\nCSVs saved to {args.results_dir}/stats/")

    # Plots
    plot_severity_curves(fr_df, model=args.model)
    plot_severity_heatmaps(fr_df, model=args.model)
    plot_vulnerability_ranking(vuln, model=args.model)


if __name__ == "__main__":
    main()
