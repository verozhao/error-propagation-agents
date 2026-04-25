import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import WORKFLOW_STEPS
from record_utils import is_baseline, injection_is_valid

RESULTS_DIR = "results"
MODEL = "gpt-4o-mini"
ERROR_TYPES = ["entity", "invented", "unverifiable", "contradictory"]
LABEL_MAP = {
    "entity": "Entity Substitution",
    "invented": "Invented Detail",
    "unverifiable": "Unverifiable Claim",
    "contradictory": "Contradictory",
}


def load_error_type(error_type: str, results_dir: str = None) -> pd.DataFrame:
    """Load records for a specific FAVA error type.

    Supports both per-type directories and ragtruth_weighted runs
    (post-hoc decomposition via injection_meta.error_type).
    """
    results_dir = results_dir or RESULTS_DIR
    # Try per-type directory first, then ragtruth_weighted
    search_dirs = [f"{error_type}_error", "ragtruth_weighted_error"]
    files = []
    for d in search_dirs:
        files.extend(glob.glob(f"{results_dir}/{d}/{MODEL}_*.json"))
        files.extend(glob.glob(f"{results_dir}/{d}/ragtruth_weighted_*.json"))
    if not files:
        raise FileNotFoundError(f"No files found for {MODEL} in {results_dir}")
    rows = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        for r in data:
            if "error" in r:
                continue
            # Filter to matching error type from injection_meta
            meta = r.get("injection_meta") or {}
            record_type = meta.get("error_type", r.get("error_type", ""))
            if is_baseline(r):
                step = -1
            else:
                if record_type != error_type and record_type not in ("", "ragtruth_weighted"):
                    continue
                if injection_is_valid(r) is False:
                    continue
                es = r.get("error_step")
                if isinstance(es, list):
                    step = es[0] if es else -1
                elif es is None:
                    continue
                else:
                    step = es
            rows.append({
                "error_step": step,
                "combined_score": r["evaluation"].get("combined_score") or r["evaluation"].get("combined", 0),
                "error_type": error_type,
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
    markers = ["o", "s", "^", "D"]

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
