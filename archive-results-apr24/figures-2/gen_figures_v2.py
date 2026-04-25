"""
Publication-quality figures for top-tier conference submission.
Uses pre-computed stats CSVs + raw JSONL for survival data.
"""
import json, os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from scipy import stats as sp_stats

# ═══════════════════════════════════════════════════════════════════
# Style configuration — NeurIPS / ICML single column
# ═══════════════════════════════════════════════════════════════════
COL_W = 5.5

# Tol vibrant (colorblind-safe)
C = {"factual": "#EE7733", "semantic": "#0077BB", "omission": "#33BBEE"}
ETYPE_ORDER = ["factual", "semantic", "omission"]
ETYPE_LABEL = {"factual": "Factual", "semantic": "Semantic", "omission": "Omission"}


STEP_NAMES = ["search", "filter", "summarize", "compose"]
STEP_LABELS = ["Search", "Filter", "Summarize", "Compose"]
ALL_STEPS = ["search", "filter", "summarize", "compose", "verify"]
ALL_LABELS = ["Search", "Filter", "Summarize", "Compose", "Verify"]

STEP_MARKERS = ["o", "s", "D", "^"]
STEP_COLORS = ["#332288", "#88CCEE", "#44AA99", "#CC6677"]

SEV_DOSES = {
    "factual":  {1: 1, 2: 2, 3: 8},
    "semantic": {1: 1, 2: 2, 3: 8},
    "omission": {1: 0.10, 2: 0.25, 3: 0.75},
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "legend.framealpha": 0.0,
    "legend.edgecolor": "none",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "grid.linewidth": 0.3,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
    "lines.markersize": 4.5,
    "patch.linewidth": 0.4,
})

STATS = "/home/claude/repo/error-propagation-agents-main/results/stats"
DATA  = "/home/claude/repo/error-propagation-agents-main/results"
OUT   = "/home/claude/figures_pub"
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════
def load_raw():
    """Load raw JSONL for survival heatmap and score distributions."""
    records = []
    for etype in ETYPE_ORDER:
        for sev in [1, 2, 3]:
            p = f"{DATA}/{etype}_error/{etype}_sev{sev}_claude-3-haiku_15trials.jsonl"
            if not os.path.exists(p): continue
            with open(p) as f:
                for line in f:
                    d = json.loads(line)
                    ev = d.get("evaluation", {})
                    records.append({
                        "error_type": d["error_type"],
                        "severity": d["severity"],
                        "error_step": d["error_step"],
                        "is_baseline": d.get("is_baseline", False),
                        "injection_valid": d.get("injection_valid"),
                        "combined_score": ev.get("combined_score"),
                        "query": d.get("task_query", ""),
                        "error_found": d.get("error_found_in_step", {}),
                    })
    return pd.DataFrame(records)

print("Loading data...")
df_raw = load_raw()
fr_ci = pd.read_csv(f"{STATS}/failure_rates_with_ci.csv")
fr_sev = pd.read_csv(f"{STATS}/failure_rates_by_severity.csv")
alpha_df = pd.read_csv(f"{STATS}/markov_alpha_estimates.csv")
pred_df = pd.read_csv(f"{STATS}/markov_predictions.csv")
print(f"  {len(df_raw)} raw records, {len(fr_ci)} FR conditions, {len(fr_sev)} FR×sev conditions")

# Map step names to indices
step2idx = {s: i for i, s in enumerate(STEP_NAMES)}

# ═══════════════════════════════════════════════════════════════════
# Fig 1: Error propagation curves — the main result
# Aggregated across all severities, with bootstrap 95% CI bands
# ═══════════════════════════════════════════════════════════════════
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(COL_W, 1.85), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for ax, etype in zip(axes, ETYPE_ORDER):
        sub = fr_ci[fr_ci.error_type == etype].copy()
        sub["si"] = sub.step_name.map(step2idx)
        sub = sub.dropna(subset=["si"]).sort_values("si")
        if sub.empty: continue

        x  = sub.si.values
        y  = sub.failure_rate.values
        lo = sub.failure_rate_ci_lo.values
        hi = sub.failure_rate_ci_hi.values
        color = C[etype]

        ax.fill_between(x, lo, hi, alpha=0.15, color=color, linewidth=0)
        ax.plot(x, y, "-", color=color, linewidth=1.5, zorder=3)
        ax.plot(x, y, "o", color=color, markersize=5.5,
                markeredgecolor="white", markeredgewidth=0.9, zorder=4)

        for xi, yi in zip(x, y):
            off = 8 if yi < 0.45 else -11
            ax.annotate(f"{yi:.1%}", (xi, yi), textcoords="offset points",
                        xytext=(0, off), ha="center", fontsize=6.5,
                        color=color, fontweight="medium")

        ax.set_xticks(range(4))
        ax.set_xticklabels(STEP_LABELS, rotation=30, ha="right")
        ax.set_title(ETYPE_LABEL[etype], fontsize=9, pad=4)
        ax.set_xlim(-0.35, 3.35)
        ax.grid(axis="y", linestyle="-", alpha=0.15)

    axes[0].set_ylabel("Failure rate")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    axes[0].set_ylim(-0.02, 0.72)
    axes[1].set_xlabel("Injection point", labelpad=3)

    fig.savefig(f"{OUT}/fig1_propagation_curves.pdf")
    plt.close(fig)
    print("  ✓ Fig 1")

# ═══════════════════════════════════════════════════════════════════
# Fig 2: Severity dose-response
# ═══════════════════════════════════════════════════════════════════
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(COL_W, 2.05), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for ax, etype in zip(axes, ETYPE_ORDER):
        doses = SEV_DOSES[etype]
        # Get baselines (only in sev1 files)
        bl = df_raw[(df_raw.error_type == etype) & (df_raw.is_baseline == True)]
        bl_mean = bl["combined_score"].dropna().mean()
        if bl_mean == 0 or np.isnan(bl_mean): continue

        for si, step_name in enumerate(STEP_NAMES):
            xs, ys = [], []
            for sev in [1, 2, 3]:
                inj = df_raw[(df_raw.error_type == etype) & (df_raw.severity == sev) &
                             (df_raw.error_step == si) & (df_raw.injection_valid == True)]
                scores = inj["combined_score"].dropna().values
                if len(scores) < 10: continue
                # Match original: mean relative degradation
                fr_val = np.mean(np.maximum(0, (bl_mean - scores) / bl_mean))
                xs.append(doses[sev])
                ys.append(fr_val)
            if xs:
                ax.plot(xs, ys, marker=STEP_MARKERS[si], color=STEP_COLORS[si],
                        label=STEP_LABELS[si], markersize=4.5,
                        markeredgecolor="white", markeredgewidth=0.6)

        ax.set_title(ETYPE_LABEL[etype], fontsize=9, pad=4)
        d = sorted(doses.values())
        ax.set_xticks(d)
        if etype == "omission":
            ax.set_xticklabels([f"{v:.0%}" for v in d])
        ax.grid(axis="y", linestyle="-", alpha=0.15)

    axes[0].set_ylabel("Failure rate")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    axes[0].set_ylim(-0.02, 0.82)
    axes[1].set_xlabel("Perturbation dose", labelpad=3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.07), frameon=False, fontsize=6.5)

    fig.savefig(f"{OUT}/fig2_dose_response.pdf")
    plt.close(fig)
    print("  ✓ Fig 2")

# ═══════════════════════════════════════════════════════════════════
# Fig 3: Claim survival heatmap (injection × observation step)
# ═══════════════════════════════════════════════════════════════════
def fig3():
    # Build survival matrix from raw error_found_in_step
    matrices = {}
    for etype in ETYPE_ORDER:
        sub = df_raw[(df_raw.error_type == etype) & (~df_raw.is_baseline) &
                     (df_raw.injection_valid == True) & (df_raw.severity == 1)]
        mat = np.full((4, 5), np.nan)
        for inj_step in range(4):
            inj_sub = sub[sub.error_step == inj_step]
            for obs_idx, obs_step in enumerate(ALL_STEPS):
                vals = []
                for _, row in inj_sub.iterrows():
                    ef = row.get("error_found", {})
                    if isinstance(ef, dict) and obs_step in ef:
                        vals.append(ef[obs_step].get("survival_score", 0))
                if vals:
                    mat[inj_step, obs_idx] = np.mean(vals)
        matrices[etype] = mat

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad("#f5f5f5")

    fig, axes = plt.subplots(1, 3, figsize=(COL_W + 0.6, 2.35))
    fig.subplots_adjust(wspace=0.35, right=0.88)

    for ax, etype in zip(axes, ETYPE_ORDER):
        mat = matrices[etype]
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=0.7, aspect="auto")

        for i in range(4):
            for j in range(5):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                color = "white" if v > 0.4 else "#333333"
                if j < i:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=6, color="#bbbbbb")
                else:
                    ax.text(j, i, f".{int(v*100):02d}" if v < 1 else "1.0",
                            ha="center", va="center", fontsize=6.5,
                            color=color, fontweight="medium")

        ax.set_xticks(range(5))
        ax.set_xticklabels(ALL_LABELS, rotation=35, ha="right", fontsize=6.5)
        ax.set_yticks(range(4))
        ax.set_yticklabels(STEP_LABELS if etype == ETYPE_ORDER[0] else [""] * 4,
                           fontsize=6.5)
        ax.set_title(ETYPE_LABEL[etype], fontsize=9, pad=5)
        ax.tick_params(length=0)

        # Highlight injection diagonal
        for k in range(4):
            if k < 5:
                ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1,
                             fill=False, edgecolor=C[etype], linewidth=1.3, zorder=5))

    axes[0].set_ylabel("Injection point", labelpad=2)
    axes[1].set_xlabel("Observed at step", labelpad=3)

    cax = fig.add_axes([0.90, 0.20, 0.013, 0.58])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Survival score", fontsize=7, labelpad=3)
    cb.ax.tick_params(labelsize=6)

    fig.savefig(f"{OUT}/fig3_survival_heatmap.pdf")
    plt.close(fig)
    print("  ✓ Fig 3")

# ═══════════════════════════════════════════════════════════════════
# Fig 4: Markov model — α cascade + predicted vs observed
# ═══════════════════════════════════════════════════════════════════
def fig4():
    fig = plt.figure(figsize=(COL_W, 2.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.15, 1], wspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # (a) Alpha values
    transitions = ["search → filter", "filter → summarize",
                    "summarize → compose", "compose → verify"]
    trans_short = [r"$\alpha_1$"+"\nS→F", r"$\alpha_2$"+"\nF→Su",
                   r"$\alpha_3$"+"\nSu→C", r"$\alpha_4$"+"\nC→V"]

    x = np.arange(len(transitions))
    w = 0.22
    offsets = {"factual": -w, "semantic": 0, "omission": w}

    for etype in ETYPE_ORDER:
        sub = alpha_df[alpha_df.scope == etype]
        if sub.empty: continue
        alphas, err_lo, err_hi = [], [], []
        for t in transitions:
            row = sub[sub.transition == t]
            if row.empty:
                alphas.append(0); err_lo.append(0); err_hi.append(0)
            else:
                a = row.iloc[0]["alpha"]
                alphas.append(a)
                err_lo.append(a - row.iloc[0]["ci_lo"])
                err_hi.append(row.iloc[0]["ci_hi"] - a)

        ax1.bar(x + offsets[etype], alphas, w, color=C[etype],
                label=ETYPE_LABEL[etype], edgecolor="white", linewidth=0.3)
        ax1.errorbar(x + offsets[etype], alphas, yerr=[err_lo, err_hi],
                     fmt="none", color="#444444", linewidth=0.6,
                     capsize=1.8, capthick=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(trans_short, fontsize=6, linespacing=1.3)
    ax1.set_ylabel(r"Attenuation factor $\alpha_k$", labelpad=2)
    ax1.set_ylim(0, 1.05)
    ax1.axhline(0.5, color="#dddddd", linewidth=0.5, linestyle="--", zorder=0)
    ax1.legend(fontsize=6, loc="upper right", frameon=False,
               handlelength=1.2, handletextpad=0.3)
    ax1.set_title("(a) Step-wise attenuation", fontsize=8.5, pad=4, loc="left")
    ax1.grid(axis="y", alpha=0.15)

    # (b) Predicted vs Observed
    for etype in ETYPE_ORDER:
        sub = pred_df[pred_df.error_type == etype]
        ax2.scatter(sub.fr_predicted, sub.fr_observed, color=C[etype],
                    s=35, edgecolor="white", linewidth=0.6, zorder=3,
                    label=ETYPE_LABEL[etype])

    ax2.plot([0, 0.6], [0, 0.6], "--", color="#cccccc", linewidth=0.8, zorder=1)
    ax2.set_xlim(-0.02, 0.62)
    ax2.set_ylim(-0.02, 0.62)
    ax2.set_xlabel("Predicted FR (Markov)", labelpad=2)
    ax2.set_ylabel("Observed FR", labelpad=2)
    ax2.set_aspect("equal")
    ax2.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    obs = pred_df.fr_observed.values
    pred = pred_df.fr_predicted.values
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    ax2.text(0.05, 0.93, f"$R^2 = {r2:.2f}$\nRMSE$= {rmse:.3f}$",
             transform=ax2.transAxes, fontsize=7, va="top",
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#dddddd", lw=0.4))
    ax2.set_title("(b) Model fit", fontsize=8.5, pad=4, loc="left")
    ax2.grid(alpha=0.15)

    fig.savefig(f"{OUT}/fig4_markov_model.pdf")
    plt.close(fig)
    print("  ✓ Fig 4")

# ═══════════════════════════════════════════════════════════════════
# Fig 5: Score degradation — Δ combined_score by injection point
# Clean paired-difference view: one line per error type
# ═══════════════════════════════════════════════════════════════════
def fig5():
    fig, ax = plt.subplots(figsize=(3.2, 2.2))

    for etype in ETYPE_ORDER:
        bl = df_raw[(df_raw.error_type == etype) & (df_raw.is_baseline == True)]
        bl_mean = bl["combined_score"].dropna().mean()
        if bl_mean == 0 or np.isnan(bl_mean): continue

        # Use sev=1 only for clean comparison
        sub = df_raw[(df_raw.error_type == etype) & (df_raw.severity == 1) &
                     (~df_raw.is_baseline) & (df_raw.injection_valid == True)]
        xs, deltas = [], []
        for si in range(4):
            scores = sub[sub.error_step == si]["combined_score"].dropna().values
            if len(scores) < 10: continue
            xs.append(si)
            deltas.append(bl_mean - np.mean(scores))

        ax.plot(xs, deltas, "o-", color=C[etype], markersize=5,
                markeredgecolor="white", markeredgewidth=0.8,
                label=ETYPE_LABEL[etype], zorder=3)

        for xi, di in zip(xs, deltas):
            if di > 0.05:
                ax.annotate(f"Δ={di:.3f}", (xi, di), textcoords="offset points",
                            xytext=(8, 2), fontsize=5.5, color=C[etype])

    ax.axhline(0, color="#cccccc", linewidth=0.6, zorder=0)
    ax.set_xticks(range(4))
    ax.set_xticklabels(STEP_LABELS, rotation=25, ha="right")
    ax.set_xlabel("Injection point")
    ax.set_ylabel("Score degradation\n(baseline − injected)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.15)
    ax.set_xlim(-0.3, 3.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_score_degradation.pdf")
    plt.close(fig)
    print("  ✓ Fig 5")

# ═══════════════════════════════════════════════════════════════════
# Fig 6: Compound injection — observed vs expected
# ═══════════════════════════════════════════════════════════════════
def fig6():
    comp_csv = f"{STATS}/compound_superadditivity.csv"
    if not os.path.exists(comp_csv):
        print("  ✗ Fig 6: no compound CSV")
        return
    comp = pd.read_csv(comp_csv)

    # Aggregate per (error_type, steps) — the CSV has per-trial rows
    agg = comp.groupby(["error_type", "steps"]).agg(
        fr_obs=("fr_compound", "mean"),
        fr_exp=("fr_expected_indep", "mean"),
        n=("fr_compound", "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(COL_W, 2.1), sharey=True)
    fig.subplots_adjust(wspace=0.12)

    for ax, etype in zip(axes, ETYPE_ORDER):
        sub = agg[agg.error_type == etype].sort_values("steps")
        if sub.empty: continue

        x = np.arange(len(sub))
        w = 0.32
        ax.bar(x - w/2, sub.fr_obs, w, color=C[etype], edgecolor="white",
               linewidth=0.3, label="Observed", zorder=3)
        ax.bar(x + w/2, sub.fr_exp, w, color=C[etype], alpha=0.3,
               edgecolor=C[etype], linewidth=0.5, label="Expected",
               hatch="///", zorder=3)

        # Superadditivity indicators
        for xi, (_, row) in enumerate(sub.iterrows()):
            delta = row.fr_obs - row.fr_exp
            if abs(delta) > 0.02:
                symbol = "▲" if delta > 0 else "▼"
                color_s = C[etype] if delta > 0 else "#999999"
                ymax = max(row.fr_obs, row.fr_exp) + 0.015
                ax.text(xi, ymax, symbol, ha="center", fontsize=5,
                        color=color_s)

        labels = [s.replace("(", "").replace(")", "").replace(", ", ",")
                  for s in sub.steps]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6)
        ax.set_title(ETYPE_LABEL[etype], fontsize=9, pad=4)
        ax.grid(axis="y", alpha=0.15)

    axes[0].set_ylabel("Failure rate")
    axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    axes[1].set_xlabel("Injection step pair", labelpad=3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.07), frameon=False, fontsize=6.5)

    fig.savefig(f"{OUT}/fig6_compound.pdf")
    plt.close(fig)
    print("  ✓ Fig 6")

# ═══════════════════════════════════════════════════════════════════
# Fig 7: Survival decay curve — injected at step 0, tracked through
# ═══════════════════════════════════════════════════════════════════
def fig7():
    fig, ax = plt.subplots(figsize=(3.0, 2.1))

    for etype in ETYPE_ORDER:
        sub = df_raw[(df_raw.error_type == etype) & (df_raw.error_step == 0) &
                     (~df_raw.is_baseline) & (df_raw.injection_valid == True) &
                     (df_raw.severity == 1)]
        means = []
        cis = []
        for obs_step in ALL_STEPS:
            vals = []
            for _, row in sub.iterrows():
                ef = row.get("error_found", {})
                if isinstance(ef, dict) and obs_step in ef:
                    vals.append(ef[obs_step].get("survival_score", 0))
            if vals:
                m = np.mean(vals)
                means.append(m)
                # Bootstrap CI
                boot = [np.mean(np.random.choice(vals, len(vals), replace=True))
                        for _ in range(1000)]
                cis.append((np.percentile(boot, 2.5), np.percentile(boot, 97.5)))
            else:
                means.append(0); cis.append((0, 0))

        x = np.arange(5)
        lo = [c[0] for c in cis]
        hi = [c[1] for c in cis]

        ax.fill_between(x, lo, hi, alpha=0.1, color=C[etype], linewidth=0)
        ax.plot(x, means, "o-", color=C[etype], markersize=4.5,
                markeredgecolor="white", markeredgewidth=0.7,
                label=ETYPE_LABEL[etype])

    ax.set_xticks(range(5))
    ax.set_xticklabels(ALL_LABELS, rotation=30, ha="right")
    ax.set_xlabel("Pipeline step")
    ax.set_ylabel("Survival score")
    ax.set_ylim(-0.01, None)
    ax.legend(fontsize=6.5, frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig7_survival_decay.pdf")
    plt.close(fig)
    print("  ✓ Fig 7")

# ═══════════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════════
print("\nGenerating publication figures...")
fig1()
fig2()
fig3()
fig4()
fig5()
fig6()
fig7()
print(f"\n✓ All figures saved to {OUT}/")
