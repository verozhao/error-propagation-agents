"""Generate LaTeX tables for the paper from analysis outputs."""
import os

import pandas as pd

from config import WORKFLOW_STEPS


def _fr_cell(row):
    """Format failure rate with CI as 'FR (lo, hi)'."""
    return f"{row['failure_rate']:.2f} ({row['failure_rate_ci_lo']:.2f}, {row['failure_rate_ci_hi']:.2f})"


def table1_propagation():
    """Failure rates with CIs - main results table."""
    path = "results/stats/failure_rates_with_ci.csv"
    if not os.path.exists(path):
        print(f"Skipping Table 1: {path} not found")
        return
    df = pd.read_csv(path)
    primary = df[df["model"].isin(["gpt-4o-mini", "claude-3-haiku"])].copy()
    if primary.empty:
        print("Skipping Table 1: no primary model data")
        return

    primary["cell"] = primary.apply(_fr_cell, axis=1)
    pivot = primary.pivot_table(
        index=["error_type", "step_name"],
        columns="model",
        values="cell",
        aggfunc="first",
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Failure rates by injection step and error type (95\% bootstrap CIs).}",
        r"\label{tab:propagation}",
        r"\begin{tabular}{ll" + "c" * len(pivot.columns) + "}",
        r"\toprule",
        r"Error Type & Step & " + " & ".join(pivot.columns) + r" \\",
        r"\midrule",
    ]
    prev_etype = None
    for (etype, step), row in pivot.iterrows():
        prefix = etype if etype != prev_etype else ""
        cells = " & ".join(str(row.get(c, "--")) for c in pivot.columns)
        lines.append(f"{prefix} & {step} & {cells}" + r" \\")
        prev_etype = etype
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out = os.path.join("paper", "tables", "table1_propagation.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")


def table2_significance():
    """Significance test results with effect sizes."""
    path = "results/stats/significance.csv"
    if not os.path.exists(path):
        print(f"Skipping Table 2: {path} not found")
        return
    df = pd.read_csv(path)
    primary = df[df["model"].isin(["gpt-4o-mini", "claude-3-haiku"])].copy()
    if primary.empty:
        print("Skipping Table 2: no primary model data")
        return

    cols = ["model", "error_type", "injection_step", "n_paired",
            "mean_diff", "p_value_adjusted", "effect_size_r", "cohens_d",
            "significant_after_correction"]
    subset = primary[[c for c in cols if c in primary.columns]]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Wilcoxon signed-rank tests with Holm--Bonferroni correction and effect sizes.}",
        r"\label{tab:significance}",
        r"\begin{tabular}{llcrrrrc}",
        r"\toprule",
        r"Model & Error & Step & $n$ & $\Delta$ & $p_{\mathrm{adj}}$ & $r$ & $d$ & Sig. \\",
        r"\midrule",
    ]
    for _, row in subset.iterrows():
        sig = r"\checkmark" if row.get("significant_after_correction") else ""
        p_adj = row.get("p_value_adjusted", float("nan"))
        p_str = f"{p_adj:.4f}" if pd.notna(p_adj) else "--"
        r_val = row.get("effect_size_r", float("nan"))
        r_str = f"{r_val:.3f}" if pd.notna(r_val) else "--"
        d_val = row.get("cohens_d", float("nan"))
        d_str = f"{d_val:.3f}" if pd.notna(d_val) else "--"
        lines.append(
            f"{row['model']} & {row['error_type']} & {row['injection_step']} & "
            f"{row['n_paired']} & {row['mean_diff']:.3f} & {p_str} & {r_str} & {d_str} & {sig}"
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out = os.path.join("paper", "tables", "table2_significance.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")


def table3_compound():
    """Compound super-additivity results."""
    path = "results/stats/compound_superadditivity.csv"
    if not os.path.exists(path):
        print(f"Skipping Table 3: {path} not found")
        return
    df = pd.read_csv(path)
    if df.empty:
        print("Skipping Table 3: empty")
        return

    agg = df.groupby(["model", "error_type", "steps"]).agg({
        "fr_compound": "mean",
        "fr_expected_indep": "mean",
        "interaction_delta": "mean",
        "super_additive": "mean",
    }).reset_index()

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Compound injection: observed vs.\ expected failure rates under independence.}",
        r"\label{tab:compound}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Error & Steps & FR\textsubscript{obs} & FR\textsubscript{exp} & $\Delta$ & Super-add. \\",
        r"\midrule",
    ]
    for _, row in agg.iterrows():
        sa = r"\checkmark" if row["super_additive"] > 0.5 else ""
        lines.append(
            f"{row['model']} & {row['error_type']} & {row['steps']} & "
            f"{row['fr_compound']:.3f} & {row['fr_expected_indep']:.3f} & "
            f"{row['interaction_delta']:.3f} & {sa}"
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    out = os.path.join("paper", "tables", "table3_compound.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")


def main():
    os.makedirs("paper/tables", exist_ok=True)
    table1_propagation()
    table2_significance()
    table3_compound()


if __name__ == "__main__":
    main()
