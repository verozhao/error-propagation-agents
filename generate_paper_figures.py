"""Generate publication-quality figures for ICLR/NeurIPS/ACL submission.

Reads from results/stats/*.csv and produces figures/paper/*.pdf at 300 DPI.

Figures:
  1. Error propagation curves with 95% CI bands (RQ1)
  2. Severity dose-response curves (RQ2)
  3. Claim survival heatmap by error type (RQ1 + mechanistic)
  4. Compound error interaction: observed vs expected (RQ3)
  5. Error survival decay across pipeline steps (mechanistic core)
  6. Feature correlation importance (post-hoc)

Usage:
    python generate_paper_figures.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from config import WORKFLOW_STEPS

# ─── Publication style ───────────────────────────────────────────────
STEP_LABELS = ["Search", "Filter", "Summarize", "Compose"]
STEP_ORDER = WORKFLOW_STEPS[:-1]  # exclude verify for injection steps
ALL_STEPS = WORKFLOW_STEPS  # include verify for observation steps
ALL_STEP_LABELS = ["Search", "Filter", "Summarize", "Compose", "Verify"]

# Colorblind-friendly palette (Wong 2011), muted for academic use
C_ENTITY        = "#D55E00"  # vermillion
C_INVENTED      = "#0072B2"  # blue
C_UNVERIFIABLE  = "#009E73"  # bluish green
C_CONTRADICTORY = "#CC79A7"  # reddish purple
ETYPE_COLORS = {
    "entity": C_ENTITY, "invented": C_INVENTED,
    "unverifiable": C_UNVERIFIABLE, "contradictory": C_CONTRADICTORY,
}
ETYPE_ORDER = ["entity", "invented", "unverifiable", "contradictory"]
ETYPE_LABELS = {
    "entity": "Entity", "invented": "Invented",
    "unverifiable": "Unverifiable", "contradictory": "Contradictory",
}

# Step colors for within-panel lines
STEP_CMAP = ["#332288", "#88CCEE", "#44AA99", "#CC6677"]  # Tol muted

# Physical severity doses (FAVA taxonomy, ragtruth_weighted injection)
SEV_DOSES = {
    "entity":        {1: 1, 2: 2, 3: 8},
    "invented":      {1: 1, 2: 2, 3: 8},
    "unverifiable":  {1: 1, 2: 2, 3: 8},
    "contradictory": {1: 1, 2: 2, 3: 8},
}
SEV_LABELS = {
    "entity":        "K (entity swaps)",
    "invented":      "K (invented facts)",
    "unverifiable":  "K (unverifiable claims)",
    "contradictory": "K (contradictions)",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "legend.framealpha": 0.8,
    "legend.edgecolor": "0.8",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.3,
    "lines.markersize": 4,
})

OUT_DIR = "figures/paper"


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Wrote {path}")


# ─── Fig 1: Propagation Curves ──────────────────────────────────────
def fig1_propagation_curves():
    """FR by injection step, one subplot per error type, with 95% CI bands."""
    path = "results/stats/failure_rates_with_ci.csv"
    if not os.path.exists(path):
        print("Skipping Fig 1: failure_rates_with_ci.csv not found")
        return
    df = pd.read_csv(path)
    df["step_idx"] = df["step_name"].map({s: i for i, s in enumerate(STEP_ORDER)})
    df = df.dropna(subset=["step_idx"])

    n_types = len(ETYPE_ORDER)
    fig, axes = plt.subplots(1, n_types, figsize=(2.2 * n_types, 2.2), sharey=True)

    for ax, etype in zip(axes, ETYPE_ORDER):
        sub = df[df["error_type"] == etype].sort_values("step_idx")
        if sub.empty:
            ax.set_title(ETYPE_LABELS[etype])
            continue

        color = ETYPE_COLORS[etype]
        x = sub["step_idx"].values
        y = sub["failure_rate"].values
        lo = sub["failure_rate_ci_lo"].values
        hi = sub["failure_rate_ci_hi"].values

        ax.fill_between(x, lo, hi, alpha=0.2, color=color, linewidth=0)
        ax.plot(x, y, "o-", color=color, markersize=5, markeredgecolor="white",
                markeredgewidth=0.6, zorder=3)

        for xi, yi in zip(x, y):
            if yi > 0.005:
                ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                            xytext=(0, 7), ha="center", fontsize=6, color=color)

        ax.set_xticks(range(len(STEP_LABELS)))
        ax.set_xticklabels(STEP_LABELS, rotation=25, ha="right")
        ax.set_title(ETYPE_LABELS[etype], fontweight="bold")
        ax.set_xlabel("Injection point")
        ax.set_xlim(-0.3, len(STEP_LABELS) - 0.7)

    axes[0].set_ylabel("Failure rate")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig1_propagation_curves.pdf")


# ─── Fig 2: Severity Dose-Response ──────────────────────────────────
def fig2_severity_dose_response():
    """FR vs physical dose, one subplot per error type, lines per injection step."""
    path = "results/stats/failure_rates_by_severity.csv"
    if not os.path.exists(path):
        print("Skipping Fig 2: failure_rates_by_severity.csv not found")
        return
    df = pd.read_csv(path)

    n_types = len(ETYPE_ORDER)
    fig, axes = plt.subplots(1, n_types, figsize=(2.2 * n_types, 2.4), sharey=True)

    for ax, etype in zip(axes, ETYPE_ORDER):
        sub = df[df["error_type"] == etype]
        if sub.empty:
            ax.set_title(ETYPE_LABELS[etype])
            continue

        dose_map = SEV_DOSES.get(etype, {1: 1, 2: 2, 3: 8})
        sub = sub.copy()
        sub["dose"] = sub["severity"].map(dose_map)

        for i, (step, sdf) in enumerate(sub.groupby("step_name")):
            if step not in STEP_ORDER:
                continue
            si = STEP_ORDER.index(step)
            sdf = sdf.sort_values("dose")
            ax.plot(sdf["dose"], sdf["failure_rate"], "o-",
                    color=STEP_CMAP[si], label=STEP_LABELS[si],
                    markersize=4, markeredgecolor="white", markeredgewidth=0.5)

        ax.set_title(ETYPE_LABELS[etype], fontweight="bold")
        ax.set_xlabel(SEV_LABELS.get(etype, "Severity"))

        doses = sorted(dose_map.values())
        ax.set_xticks(doses)

    axes[0].set_ylabel("Failure rate")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4,
                   bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=7)
    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig2_severity_dose_response.pdf")


# ─── Fig 3: Survival Heatmap ────────────────────────────────────────
def fig3_survival_heatmap():
    """Claim survival heatmap: injection_step × obs_step, one per error type."""
    path = "results/stats/claim_survival_matrix.csv"
    if not os.path.exists(path):
        print("Skipping Fig 3: no survival matrix CSV found")
        return
    df = pd.read_csv(path)

    n_types = len(ETYPE_ORDER)
    fig, axes = plt.subplots(1, n_types, figsize=(2.4 * n_types, 2.8),
                              gridspec_kw={"width_ratios": [1] * n_types, "wspace": 0.45})
    fig.subplots_adjust(right=0.87, bottom=0.22)

    for ax, etype in zip(axes, ETYPE_ORDER):
        sub = df[df["error_type"] == etype]
        if sub.empty:
            ax.set_title(ETYPE_LABELS[etype])
            continue

        # Average across severities
        pivot = sub.pivot_table(
            index="injection_step", columns="obs_step",
            values="survival_score", aggfunc="mean")

        # Reorder
        row_order = [s for s in STEP_ORDER if s in pivot.index]
        col_order = [s for s in ALL_STEPS if s in pivot.columns]
        pivot = pivot.reindex(index=row_order, columns=col_order)

        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd",
                    ax=ax, cbar=False, vmin=0, vmax=0.8,
                    annot_kws={"fontsize": 7},
                    linewidths=0.5, linecolor="white")

        ax.set_title(ETYPE_LABELS[etype], fontweight="bold")
        ax.set_yticklabels([STEP_LABELS[STEP_ORDER.index(s)] for s in row_order],
                           rotation=0)
        ax.set_xticklabels([ALL_STEP_LABELS[ALL_STEPS.index(s)] for s in col_order],
                           rotation=35, ha="right")
        ax.set_ylabel("Injection point" if ax == axes[0] else "")
        ax.set_xlabel("Observed at step")
        if ax != axes[0]:
            ax.set_yticklabels([])

    # Shared colorbar
    norm = plt.Normalize(0, 0.8)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.91, 0.18, 0.015, 0.65])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Survival score", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    _save(fig, "fig3_survival_heatmap.pdf")


# ─── Fig 4: Compound Interaction ─────────────────────────────────────
def fig4_compound_interaction():
    """Paired comparison: observed compound FR vs independence expectation."""
    path = "results/stats/compound_superadditivity.csv"
    if not os.path.exists(path):
        print("Skipping Fig 4: compound_superadditivity.csv not found")
        return
    df = pd.read_csv(path)
    if df.empty:
        return

    agg = df.groupby(["error_type", "steps"]).agg(
        observed=("fr_compound", "mean"),
        expected=("fr_expected_indep", "mean"),
    ).reset_index()

    n_types = len(ETYPE_ORDER)
    fig, axes = plt.subplots(1, n_types, figsize=(2.2 * n_types, 2.4), sharey=True)

    for ax, etype in zip(axes, ETYPE_ORDER):
        sub = agg[agg["error_type"] == etype].reset_index(drop=True)
        if sub.empty:
            ax.set_title(ETYPE_LABELS[etype])
            continue

        x = np.arange(len(sub))
        color = ETYPE_COLORS[etype]

        for i, row in sub.iterrows():
            ax.plot([i, i], [row["expected"], row["observed"]],
                    color="0.6", linewidth=1.2, zorder=1)
            ax.plot(i, row["expected"], "o", color="0.5", markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=2)
            ax.plot(i, row["observed"], "D", color=color, markersize=5,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)

        ax.set_xticks(x)
        step_labels = [r["steps"].replace("(", "").replace(")", "").replace(", ", ",")
                       for _, r in sub.iterrows()]
        ax.set_xticklabels(step_labels, fontsize=6.5, rotation=25, ha="right")
        ax.set_title(ETYPE_LABELS[etype], fontweight="bold")
        ax.set_xlabel("Step pair")
        ax.axhline(0, color="0.8", linewidth=0.5, zorder=0)

    axes[0].set_ylabel("Failure rate")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="0.3",
               markersize=5, label="Observed (compound)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="0.5",
               markersize=5, label="Expected (independence)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=7)
    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig4_compound_interaction.pdf")


# ─── Fig 5: Survival Decay Curves ───────────────────────────────────
def fig5_survival_decay():
    """Error signal decay across pipeline steps — the core mechanistic finding.

    Shows mean claim survival score at each downstream step, for errors
    injected at step 0 (search), aggregated across severities.
    """
    path = "results/stats/claim_survival_matrix.csv"
    if not os.path.exists(path):
        print("Skipping Fig 5: no survival matrix CSV found")
        return
    df = pd.read_csv(path)

    # Use injection at search (step 0) to show full decay across all downstream steps
    sub = df[df["injection_step"] == "search"].copy()
    if sub.empty:
        print("Skipping Fig 5: no search injection data")
        return

    step_idx_map = {s: i for i, s in enumerate(ALL_STEPS)}
    sub["step_idx"] = sub["obs_step"].map(step_idx_map)
    sub = sub.dropna(subset=["step_idx", "survival_score"])

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    for etype in ETYPE_ORDER:
        et_sub = sub[sub["error_type"] == etype]
        if et_sub.empty:
            continue
        agg = et_sub.groupby("step_idx")["survival_score"].mean().sort_index()
        ax.plot(agg.index, agg.values, "o-", color=ETYPE_COLORS[etype],
                label=ETYPE_LABELS[etype], markersize=5,
                markeredgecolor="white", markeredgewidth=0.6)

    ax.set_xticks(range(len(ALL_STEP_LABELS)))
    ax.set_xticklabels(ALL_STEP_LABELS, rotation=25, ha="right")
    ax.set_xlabel("Pipeline step")
    ax.set_ylabel("Mean survival score")
    ax.set_ylim(bottom=-0.01)
    ax.legend(frameon=True)
    ax.set_title("Error survival decay (injected at Search)", fontweight="bold")
    fig.tight_layout()
    _save(fig, "fig5_survival_decay.pdf")


# ─── Fig 6: Feature Correlation Importance ───────────────────────────
def fig6_correlation_importance():
    """Horizontal bar chart of top feature correlations with failure rate."""
    path = "results/stats/correlation_analysis.csv"
    if not os.path.exists(path):
        # Fallback: try posthoc features
        print("Skipping Fig 6: no correlation CSV found")
        return
    df = pd.read_csv(path)

    if "feature" not in df.columns or "correlation" not in df.columns:
        print("Skipping Fig 6: cannot identify feature/correlation columns")
        return

    # Filter out derived evaluation metrics (they're circular)
    exclude = {"precision", "recall", "f1", "keyword_score", "combined_score",
               "survival_at_final", "n_steps_propagated",
               "unigram_retention", "bigram_retention"}
    df = df[~df["feature"].isin(exclude)]

    # Top 10 by absolute value
    df["abs_corr"] = df["correlation"].abs()
    df = df.nlargest(10, "abs_corr").sort_values("correlation")

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    colors = [C_ENTITY if v > 0 else C_INVENTED for v in df["correlation"]]
    bars = ax.barh(df["feature"], df["correlation"], color=colors, height=0.6,
                   edgecolor="white", linewidth=0.3)

    ax.axvline(0, color="0.3", linewidth=0.5)
    ax.set_xlabel("Pearson $r$ with failure rate")
    ax.set_title("Feature importance", fontweight="bold")

    # Clean up feature names for display
    label_map = {
        "error_step": "Injection step",
        "severity_physical_normalized": "Severity (normalized)",
        "text_length_before": "Text length (pre-injection)",
        "delta_word_count": "Δ word count",
        "n_words_changed": "Words changed",
        "n_sentences_affected": "Sentences affected",
        "n_verbs": "Verb count",
        "n_nouns": "Noun count",
        "n_adjs": "Adjective count",
        "n_advs": "Adverb count",
        "n_entities": "Entity count",
        "injection_position": "Injection position",
        "length_change_ratio": "Length change ratio",
        "tfidf_similarity": "TF-IDF similarity",
    }
    ax.set_yticklabels([label_map.get(f, f) for f in df["feature"]], fontsize=7)

    fig.tight_layout()
    _save(fig, "fig6_correlation_importance.pdf")


# ─── Appendix: Survival Matrix Detail ────────────────────────────────
def figA_survival_matrices():
    """Detailed dual-panel survival matrices for appendix (sev=3 only)."""
    path = "results/stats/claim_survival_matrix.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)

    os.makedirs("figures/paper/appendix", exist_ok=True)

    for etype in ETYPE_ORDER:
        # Pick highest severity available
        sub = df[df["error_type"] == etype]
        max_sev = sub["severity"].max()
        sub = sub[sub["severity"] == max_sev]
        if sub.empty:
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.75, 2.5))

        for ax, metric, title, cmap in [
            (ax1, "propagation_rate", "Binary propagation", "YlOrRd"),
            (ax2, "survival_score", "Continuous survival", "YlOrRd"),
        ]:
            pivot = sub.pivot_table(
                index="injection_step", columns="obs_step",
                values=metric, aggfunc="mean")

            row_order = [s for s in STEP_ORDER if s in pivot.index]
            col_order = [s for s in ALL_STEPS if s in pivot.columns]
            pivot = pivot.reindex(index=row_order, columns=col_order)

            sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap,
                        ax=ax, cbar=False, vmin=0, vmax=1.0,
                        annot_kws={"fontsize": 7},
                        linewidths=0.5, linecolor="white")

            ax.set_yticklabels([STEP_LABELS[STEP_ORDER.index(s)] for s in row_order],
                               rotation=0)
            ax.set_xticklabels([ALL_STEP_LABELS[ALL_STEPS.index(s)] for s in col_order],
                               rotation=35, ha="right")
            ax.set_title(title, fontsize=8)
            ax.set_ylabel("Injection point" if ax == ax1 else "")
            ax.set_xlabel("Observed at step")

        fig.suptitle(f"{ETYPE_LABELS[etype]} (severity {max_sev})",
                     fontweight="bold", fontsize=9, y=1.02)
        fig.tight_layout(w_pad=1.5)
        _save(fig, f"appendix/survival_matrix_{etype}.pdf")


# ─── Appendix: Verify Detection Rate ────────────────────────────────
def figA_verify_detection():
    """Verify detection rate heatmap: error_type × injection_step."""
    path = "results/stats/verify_detection_rate.csv"
    if not os.path.exists(path):
        return

    df = pd.read_csv(path)
    if df.empty or "detection_rate" not in df.columns:
        return

    # Need per-step detection rates - check if available
    # If only per-error-type aggregates, create a simpler bar chart
    fig, ax = plt.subplots(figsize=(3.5, 2.0))

    x = np.arange(len(df))
    colors = [ETYPE_COLORS.get(et, "0.5") for et in df["error_type"]]
    ax.bar(x, df["detection_rate"], color=colors, width=0.5,
           edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([ETYPE_LABELS.get(et, et) for et in df["error_type"]])
    ax.set_ylabel("Detection rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_title("Verify step detection rate", fontweight="bold")
    fig.tight_layout()
    _save(fig, "appendix/verify_detection_rate.pdf")


# ─── Appendix: Degradation Distribution ─────────────────────────────
def figA_degradation_distribution():
    """KDE of failure rate by severity and error type."""
    path = "results/stats/posthoc_features.csv"
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    if "failure_rate" not in df.columns:
        return

    n_types = len(ETYPE_ORDER)
    fig, axes = plt.subplots(1, n_types, figsize=(2.2 * n_types, 2.2), sharey=True)

    for ax, etype in zip(axes, ETYPE_ORDER):
        sub = df[df["error_type"] == etype]
        if sub.empty:
            continue
        for sev in sorted(sub["severity"].unique()):
            sev_sub = sub[sub["severity"] == sev]["failure_rate"]
            if len(sev_sub) > 5:
                sev_sub.plot.kde(ax=ax, label=f"sev={int(sev)}", linewidth=1.0)
        ax.set_xlim(-0.05, 0.6)
        ax.set_title(ETYPE_LABELS[etype], fontweight="bold")
        ax.set_xlabel("Failure rate")

    axes[0].set_ylabel("Density")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 1.06), frameon=False, fontsize=7)
    fig.tight_layout(w_pad=1.5)
    _save(fig, "appendix/degradation_distribution.pdf")


# ─── Main ────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "appendix"), exist_ok=True)

    print("Generating paper figures...")
    fig1_propagation_curves()
    fig2_severity_dose_response()
    fig3_survival_heatmap()
    fig4_compound_interaction()
    fig5_survival_decay()
    fig6_correlation_importance()

    print("\nGenerating appendix figures...")
    figA_survival_matrices()
    figA_verify_detection()
    figA_degradation_distribution()

    print("\nGenerating causal DAG figure...")
    fig_causal_dag()

    print(f"\nAll figures saved to {OUT_DIR}/")


def fig_causal_dag():
    """Causal DAG: injection -> step outputs -> persistence -> final failure.

    Programmatic figure for Section 3.6.
    """
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1.5, 2.5)
    ax.axis("off")

    nodes = {
        "Injection\n(Treatment)": (0.5, 1.0),
        "Step k\nOutput": (2.0, 1.0),
        "Step k+1\nOutput": (3.5, 1.0),
        "Persistence\n(Mediator)": (3.5, -0.5),
        "Final\nFailure": (5.5, 1.0),
        "Severity": (0.5, -0.5),
        "Model": (2.0, 2.2),
        "Error\nType": (3.5, 2.2),
    }

    for label, (x, y) in nodes.items():
        bbox_style = "round,pad=0.3"
        fc = "#E3F2FD"
        if "Treatment" in label:
            fc = "#FFF3E0"
        elif "Failure" in label:
            fc = "#FFEBEE"
        elif "Mediator" in label:
            fc = "#E8F5E9"
        elif label in ("Severity", "Model", "Error\nType"):
            fc = "#F3E5F5"
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle=bbox_style, facecolor=fc, edgecolor="#666",
                          linewidth=0.8))

    edges = [
        ("Injection\n(Treatment)", "Step k\nOutput", "solid"),
        ("Step k\nOutput", "Step k+1\nOutput", "solid"),
        ("Step k\nOutput", "Persistence\n(Mediator)", "solid"),
        ("Step k+1\nOutput", "Persistence\n(Mediator)", "solid"),
        ("Persistence\n(Mediator)", "Final\nFailure", "solid"),
        ("Injection\n(Treatment)", "Final\nFailure", "dashed"),  # NDE path
        ("Severity", "Injection\n(Treatment)", "solid"),
        ("Model", "Step k\nOutput", "solid"),
        ("Error\nType", "Step k+1\nOutput", "solid"),
    ]

    for src, dst, style in edges:
        x1, y1 = nodes[src]
        x2, y2 = nodes[dst]
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->", color="#333", lw=1.2,
                linestyle=style,
                connectionstyle="arc3,rad=0.1" if abs(y2 - y1) > 1 else "arc3,rad=0.05",
            ),
        )

    ax.text(3.0, -1.2, "Solid = causal path; Dashed = direct effect (NDE)",
            ha="center", fontsize=7, style="italic", color="#666")
    ax.set_title("Causal DAG: Error Propagation in LLM Pipelines", fontsize=10, pad=10)
    fig.tight_layout()
    _save(fig, "fig_causal_dag.pdf")


if __name__ == "__main__":
    main()
