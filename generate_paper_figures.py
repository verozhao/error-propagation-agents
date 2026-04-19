"""Generate all 6 publication-quality figures with consistent styling.

Reads from results/stats/*.csv and produces figures/ output at 300 DPI.

Figures:
  1. Error propagation curves with CIs (one subplot per error_type)
  2. Severity dose-response curves
  3. Claim survival heatmap
  4. Compound interaction (observed vs expected FR)
  5. Attenuation factors by step
  6. Correlation feature importance

Usage:
    python generate_paper_figures.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import WORKFLOW_STEPS

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

OUT_DIR = "figures/paper"


def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


def fig1_propagation_curves():
    """Error propagation curves with CI error bars, one subplot per error_type."""
    path = "results/stats/failure_rates_with_ci.csv"
    if not os.path.exists(path):
        print("Skipping Fig 1: failure_rates_with_ci.csv not found")
        return
    df = pd.read_csv(path)
    df["step_idx"] = df["step_name"].map({s: i for i, s in enumerate(WORKFLOW_STEPS)})

    etypes = sorted(df["error_type"].dropna().unique())
    n = len(etypes)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 2.8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, etype in zip(axes, etypes):
        sub = df[df["error_type"] == etype]
        for model, mdf in sub.groupby("model"):
            mdf = mdf.sort_values("step_idx")
            x = mdf["step_idx"].values
            y = mdf["failure_rate"].values
            lo = mdf["failure_rate_ci_lo"].values
            hi = mdf["failure_rate_ci_hi"].values
            ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=3, label=model)
        ax.set_xticks(range(len(WORKFLOW_STEPS) - 1))
        ax.set_xticklabels(WORKFLOW_STEPS[:-1], rotation=30, ha="right")
        ax.set_title(etype.title())
        ax.set_xlabel("Injection step")
        if ax == axes[0]:
            ax.set_ylabel("Failure rate")

    axes[-1].legend(loc="upper left", fontsize=7)
    fig.suptitle("Error Propagation Curves (95% CI)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig1_propagation_curves.pdf")


def fig2_severity_dose_response():
    """Dose-response: FR vs severity, one subplot per error_type."""
    path = "results/stats/failure_rates_by_severity.csv"
    if not os.path.exists(path):
        print("Skipping Fig 2: failure_rates_by_severity.csv not found")
        return
    df = pd.read_csv(path)
    primary = df[df["model"].isin(["gpt-4o-mini", "claude-3-haiku"])]
    if primary.empty:
        primary = df

    etypes = sorted(primary["error_type"].dropna().unique())
    n = len(etypes)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 2.8), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, etype in zip(axes, etypes):
        sub = primary[primary["error_type"] == etype]
        agg = sub.groupby(["model", "severity"])["failure_rate"].mean().reset_index()
        for model, mdf in agg.groupby("model"):
            mdf = mdf.sort_values("severity")
            ax.plot(mdf["severity"], mdf["failure_rate"], marker="s", label=model)
        ax.set_xlabel("Severity")
        ax.set_xticks([1, 2, 3])
        ax.set_title(etype.title())
        if ax == axes[0]:
            ax.set_ylabel("Failure rate")

    axes[-1].legend(fontsize=7)
    fig.suptitle("Severity Dose-Response", y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_severity_dose_response.pdf")


def fig3_survival_heatmap():
    """Claim survival heatmap: injection_step x downstream_step."""
    path = "results/stats/claim_survival_matrix.csv"
    if not os.path.exists(path):
        path2 = "results/stats/survival_matrix.csv"
        if os.path.exists(path2):
            path = path2
        else:
            print("Skipping Fig 3: no survival matrix CSV found")
            return
    df = pd.read_csv(path)

    if "injection_step" in df.columns and "obs_step" in df.columns:
        pivot = df.pivot_table(
            index="injection_step", columns="obs_step",
            values="survival_score", aggfunc="mean")
    elif "injection_step" in df.columns and "downstream_step" in df.columns:
        pivot = df.pivot_table(
            index="injection_step", columns="downstream_step",
            values="survival_score", aggfunc="mean")
    else:
        pivot = df.set_index(df.columns[0])
        pivot = df.pivot_table(
            index="injection_step", columns="downstream_step",
            values="survival_score", aggfunc="mean")

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                cbar_kws={"label": "Survival score"})
    ax.set_title("Injected Claim Survival by Step")
    ax.set_ylabel("Injection step")
    ax.set_xlabel("Downstream step")
    fig.tight_layout()
    _save(fig, "fig3_survival_heatmap.pdf")


def fig4_compound_interaction():
    """Grouped bar chart: observed vs expected FR for compound injections."""
    path = "results/stats/compound_superadditivity.csv"
    if not os.path.exists(path):
        print("Skipping Fig 4: compound_superadditivity.csv not found")
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    agg = df.groupby(["error_type", "steps"]).agg({
        "fr_compound": "mean",
        "fr_expected_indep": "mean",
    }).reset_index()

    x = np.arange(len(agg))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(5, len(agg) * 0.8), 3))
    ax.bar(x - width / 2, agg["fr_compound"], width, label="Observed", color="#2196F3")
    ax.bar(x + width / 2, agg["fr_expected_indep"], width, label="Expected (indep.)", color="#FF9800")
    ax.set_xticks(x)
    labels = [f"{r['error_type']}\n{r['steps']}" for _, r in agg.iterrows()]
    ax.set_xticklabels(labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Failure rate")
    ax.set_title("Compound Error Interaction")
    ax.legend()
    fig.tight_layout()
    _save(fig, "fig4_compound_interaction.pdf")


def fig5_attenuation_factors():
    """Bar chart of attenuation factors by step and error type."""
    path = "results/stats/attenuation_factors.csv"
    if not os.path.exists(path):
        print("Skipping Fig 5: attenuation_factors.csv not found")
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    agg = df.groupby(["error_type", "step_name"])["attenuation"].mean().reset_index()
    pivot = agg.pivot(index="step_name", columns="error_type", values="attenuation")
    step_order = [s for s in WORKFLOW_STEPS if s in pivot.index]
    pivot = pivot.reindex(step_order)

    fig, ax = plt.subplots(figsize=(5, 3))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Attenuation factor")
    ax.set_title("Step Attenuation (positive = attenuates, negative = amplifies)")
    ax.set_xticklabels(step_order, rotation=30, ha="right")
    ax.legend(title="Error type", fontsize=7)
    fig.tight_layout()
    _save(fig, "fig5_attenuation_factors.pdf")


def fig6_correlation_importance():
    """Horizontal bar chart of feature correlations with failure rate."""
    path = "results/stats/correlation_analysis.csv"
    if not os.path.exists(path):
        path2 = "results/stats/posthoc_correlations.csv"
        if os.path.exists(path2):
            path = path2
        else:
            print("Skipping Fig 6: no correlation CSV found")
            return
    df = pd.read_csv(path)

    if "feature" not in df.columns or "correlation" not in df.columns:
        r_col = [c for c in df.columns if "corr" in c.lower() or "r" == c.lower()]
        f_col = [c for c in df.columns if "feat" in c.lower() or "variable" in c.lower()]
        if r_col and f_col:
            df = df.rename(columns={f_col[0]: "feature", r_col[0]: "correlation"})
        else:
            print("Skipping Fig 6: cannot identify feature/correlation columns")
            return

    df = df.sort_values("correlation")

    fig, ax = plt.subplots(figsize=(4, max(2.5, len(df) * 0.3)))
    colors = ["#E53935" if v < 0 else "#43A047" for v in df["correlation"]]
    ax.barh(df["feature"], df["correlation"], color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Pearson r")
    ax.set_title("Feature Correlations with Failure Rate")
    fig.tight_layout()
    _save(fig, "fig6_correlation_importance.pdf")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig1_propagation_curves()
    fig2_severity_dose_response()
    fig3_survival_heatmap()
    fig4_compound_interaction()
    fig5_attenuation_factors()
    fig6_correlation_importance()
    print(f"\nAll figures saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
