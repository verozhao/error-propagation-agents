"""Phase 4.3 plotting helpers — additive, do not modify existing plots.py.

Reads results/stats/failure_rates_with_ci.csv (produced by
statistical_tests.py) and emits bar charts and propagation curves with
95% bootstrap CI error bars.

Output: figures/error_bars/<error_type>_<model>.png and
        figures/error_bars/<error_type>_propagation.png
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from config import WORKFLOW_STEPS

OUT_DIR = os.path.join("figures", "error_bars")


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["step_idx"] = df["step_name"].map({s: i for i, s in enumerate(WORKFLOW_STEPS)})
    return df


def plot_propagation_with_ci(df: pd.DataFrame, error_type: str):
    sub = df[df["error_type"] == error_type].sort_values(["model", "step_idx"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model, mdf in sub.groupby("model"):
        mdf = mdf.sort_values("step_idx")
        x = mdf["step_idx"].values
        y = mdf["failure_rate"].values
        lo = mdf["failure_rate_ci_lo"].values
        hi = mdf["failure_rate_ci_hi"].values
        yerr = [y - lo, hi - y]
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=model, linewidth=1.5)
    ax.set_xticks(range(len(WORKFLOW_STEPS)))
    ax.set_xticklabels(WORKFLOW_STEPS, rotation=30)
    ax.set_xlabel("Error injection step")
    ax.set_ylabel("Failure rate (95% bootstrap CI)")
    ax.set_title(f"Error propagation — {error_type}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{error_type}_propagation.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  wrote {out}")


def main(csv_path: str = "results/stats/failure_rates_with_ci.csv"):
    if not os.path.exists(csv_path):
        print(f"{csv_path} missing — run statistical_tests.py first.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    df = _load(csv_path)
    for etype in df["error_type"].dropna().unique():
        plot_propagation_with_ci(df, etype)


if __name__ == "__main__":
    main()
