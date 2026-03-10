import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import WORKFLOW_STEPS

RESULTS_DIR = "results"
MODEL = "gpt-4o-mini"
ERROR_TYPES = ["factual_error", "omission_error", "semantic_error"]
LABEL_MAP = {
    "factual_error": "Factual",
    "omission_error": "Omission",
    "semantic_error": "Semantic",
}


def load_error_type(error_type_dir: str) -> pd.DataFrame:
    files = glob.glob(f"{RESULTS_DIR}/{error_type_dir}/{MODEL}_*.json")
    if not files:
        raise FileNotFoundError(f"No files found for {MODEL} in {RESULTS_DIR}/{error_type_dir}/")
    rows = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        for r in data:
            if "error" in r:
                continue
            rows.append({
                "error_step": r["error_step"] if r["error_step"] is not None else -1,
                "combined_score": r["evaluation"].get("combined_score") or r["evaluation"].get("combined", 0),
                "error_type": error_type_dir,
            })
    return pd.DataFrame(rows)


def compute_failure_rates(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["error_step"] == -1]["combined_score"].mean()
    results = []
    for step in range(len(WORKFLOW_STEPS)):
        step_df = df[df["error_step"] == step]
        if len(step_df) == 0:
            continue
        mean_score = step_df["combined_score"].mean()
        failure_rate = max(0, (baseline - mean_score) / baseline)
        results.append({
            "error_step": step,
            "step_name": WORKFLOW_STEPS[step],
            "failure_rate": failure_rate,
            "error_type": df["error_type"].iloc[0],
        })
    return pd.DataFrame(results)


def plot_comparison(dfs: list[pd.DataFrame], output_path: str = "figures/error_type_comparison.png"):
    import os
    os.makedirs("figures", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "s", "^"]

    for df, marker in zip(dfs, markers):
        label = LABEL_MAP[df["error_type"].iloc[0]]
        ax.plot(df["error_step"], df["failure_rate"], marker=marker, label=label, linewidth=2, markersize=7)

    ax.set_xlabel("Error Injection Step")
    ax.set_ylabel("Failure Rate (Degradation from Baseline)")
    ax.set_title(f"Error Type Comparison — {MODEL}")
    ax.set_xticks(range(len(WORKFLOW_STEPS)))
    ax.set_xticklabels(WORKFLOW_STEPS, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    failure_dfs = []
    for et in ERROR_TYPES:
        df = load_error_type(et)
        fdf = compute_failure_rates(df)
        failure_dfs.append(fdf)
        print(f"{LABEL_MAP[et]}:")
        print(fdf[["step_name", "failure_rate"]].to_string(index=False))
        print()

    plot_comparison(failure_dfs)
