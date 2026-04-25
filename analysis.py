import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
from config import OUTPUT_DIR, WORKFLOW_STEPS
import glob


def compute_failure_rates(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["error_step"] == -1].groupby("model")["combined_score"].mean()
    
    results = []
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        baseline_score = baseline[model]
        
        for step in range(len(WORKFLOW_STEPS)):
            step_df = model_df[model_df["error_step"] == step]
            if len(step_df) == 0:
                continue
            
            mean_score = step_df["combined_score"].mean()
            degradation = (baseline_score - mean_score) / baseline_score
            failure_rate = 1 - (mean_score / baseline_score)
            
            results.append({
                "model": model,
                "error_step": step,
                "step_name": WORKFLOW_STEPS[step],
                "mean_score": mean_score,
                "baseline_score": baseline_score,
                "degradation": degradation,
                "failure_rate": max(0, failure_rate),
            })
    
    return pd.DataFrame(results)


def exponential_decay(x, a, b):
    return a * np.exp(-b * x)


def linear_decay(x, a, b):
    return a - b * x


def constant(x, a):
    return np.full_like(x, a, dtype=float)


def fit_propagation_pattern(df: pd.DataFrame, model: str) -> dict:
    model_df = df[df["model"] == model].sort_values("error_step")
    x = model_df["error_step"].values.astype(float)
    y = model_df["failure_rate"].values
    
    fits = {}
    
    try:
        popt, _ = curve_fit(exponential_decay, x, y, p0=[1, 0.5], maxfev=5000)
        y_pred = exponential_decay(x, *popt)
        fits["exponential"] = {"params": popt, "rmse": np.sqrt(np.mean((y - y_pred) ** 2))}
    except:
        fits["exponential"] = {"params": None, "rmse": float("inf")}
    
    try:
        popt, _ = curve_fit(linear_decay, x, y, p0=[1, 0.1], maxfev=5000)
        y_pred = linear_decay(x, *popt)
        fits["linear"] = {"params": popt, "rmse": np.sqrt(np.mean((y - y_pred) ** 2))}
    except:
        fits["linear"] = {"params": None, "rmse": float("inf")}
    
    try:
        popt, _ = curve_fit(constant, x, y, p0=[0.5], maxfev=5000)
        y_pred = constant(x, *popt)
        fits["constant"] = {"params": popt, "rmse": np.sqrt(np.mean((y - y_pred) ** 2))}
    except:
        fits["constant"] = {"params": None, "rmse": float("inf")}
    
    best_pattern = min(fits.keys(), key=lambda k: fits[k]["rmse"])
    
    return {
        "model": model,
        "best_pattern": best_pattern,
        "fits": fits,
        "x": x.tolist(),
        "y": y.tolist(),
    }


def identify_critical_steps(df: pd.DataFrame) -> pd.DataFrame:
    critical = df.loc[df.groupby("model")["failure_rate"].idxmax()]
    return critical[["model", "error_step", "step_name", "failure_rate"]]


def plot_heatmap(df: pd.DataFrame, output_path: str = None):
    pivot = df.pivot(index="model", columns="step_name", values="failure_rate")
    pivot = pivot[WORKFLOW_STEPS]
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Reds", cbar_kws={"label": "Failure Rate"})
    plt.title("Error Impact Heatmap by Model and Step")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_error_propagation(df: pd.DataFrame, output_path: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in df["model"].unique():
        model_df = df[df["model"] == model].sort_values("error_step")
        ax.plot(model_df["error_step"], model_df["failure_rate"], marker="o", label=model)
    
    ax.set_xlabel("Error Injection Step")
    ax.set_ylabel("Failure Rate (Degradation from Baseline)")
    ax.set_title("Error Propagation Across Models")
    ax.set_xticks(range(len(WORKFLOW_STEPS)))
    ax.set_xticklabels(WORKFLOW_STEPS, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_pattern_comparison(pattern_results: list[dict], output_path: str = None):
    n = len(pattern_results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(axes).flatten()
    
    for idx, result in enumerate(pattern_results):
        ax = axes[idx]
        x = np.array(result["x"])
        y = np.array(result["y"])
        
        ax.scatter(x, y, color="black", s=50, zorder=5, label="Observed")
        
        x_smooth = np.linspace(0, max(x), 100)
        
        if result["fits"]["exponential"]["params"] is not None:
            y_exp = exponential_decay(x_smooth, *result["fits"]["exponential"]["params"])
            ax.plot(x_smooth, y_exp, "--", label=f"Exponential (RMSE={result['fits']['exponential']['rmse']:.3f})")
        
        if result["fits"]["linear"]["params"] is not None:
            y_lin = linear_decay(x_smooth, *result["fits"]["linear"]["params"])
            ax.plot(x_smooth, y_lin, "-.", label=f"Linear (RMSE={result['fits']['linear']['rmse']:.3f})")
        
        ax.set_title(f"{result['model']}\nBest: {result['best_pattern']}")
        ax.set_xlabel("Error Step")
        ax.set_ylabel("Failure Rate")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
    for idx in range(len(pattern_results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()

def load_single_result(filepath):
    import os
    from record_utils import is_baseline as _is_baseline, injection_is_valid
    model_name = os.path.basename(filepath).split("_")[0]
    with open(filepath) as f:
        data = json.load(f)
    all_data = []
    for d in data:
        if "error" in d:
            continue
        if "model" not in d:
            d["model"] = model_name
        es = d.get("error_step")
        if _is_baseline(d):
            d["error_step"] = -1
        else:
            # Issue α: drop failed-injection no-ops
            if injection_is_valid(d) is False:
                continue
            if isinstance(es, list):
                # compound: use first step for step-wise roll-ups
                d["error_step"] = es[0] if es else -1
                d["_compound"] = True
            elif es is None:
                # non-baseline but no explicit step — legacy compound; skip.
                continue
        all_data.append(d)
    return pd.DataFrame(all_data)

def load_all_results(results_dir="results"):
    from record_utils import is_baseline as _is_baseline, injection_is_valid
    all_data = []
    for f in glob.glob(f"{results_dir}/**/*.json", recursive=True):
        if "_legacy" in f or "archive" in f:
            continue
        model_name = f.split("/")[-1].split("_")[0]
        with open(f) as file:
            data = json.load(file)
            for d in data:
                if "error" in d:
                    continue
                if "model" not in d:
                    d["model"] = model_name
                es = d.get("error_step")
                if _is_baseline(d):
                    d["error_step"] = -1
                else:
                    if injection_is_valid(d) is False:
                        continue
                    if isinstance(es, list):
                        d["error_step"] = es[0] if es else -1
                        d["_compound"] = True
                    elif es is None:
                        continue
                all_data.append(d)
    return pd.DataFrame(all_data)

def generate_report(results_path=None, error_type=None):
    import os
    os.makedirs("figures", exist_ok=True)

    if results_path and os.path.isfile(results_path):
        df = load_single_result(results_path)
    else:
        default_dir = os.path.join(OUTPUT_DIR, f"{error_type}_error") if error_type else OUTPUT_DIR
        df = load_all_results(results_path or default_dir)
    df["combined_score"] = df["evaluation"].apply(lambda x: x.get("combined") or x.get("combined_score", 0))
    df["error_step"] = df["error_step"].fillna(-1).astype(int)
    failure_df = compute_failure_rates(df)

    prefix = f"{error_type}_" if error_type else ""
    figures_dir = "figures"
    csv_dir = os.path.join(OUTPUT_DIR, f"{error_type}_error") if error_type else OUTPUT_DIR
    os.makedirs(csv_dir, exist_ok=True)

    print("=" * 60)
    print("ERROR PROPAGATION ANALYSIS REPORT")
    print("=" * 60)

    print("\n1. FAILURE RATES BY MODEL AND STEP")
    print("-" * 40)
    pivot = failure_df.pivot(index="step_name", columns="model", values="failure_rate")
    print(pivot.round(3))

    print("\n2. CRITICAL STEPS (HIGHEST IMPACT)")
    print("-" * 40)
    critical = identify_critical_steps(failure_df)
    print(critical.to_string(index=False))

    print("\n3. PROPAGATION PATTERNS")
    print("-" * 40)
    patterns = []
    for model in df["model"].unique():
        result = fit_propagation_pattern(failure_df, model)
        patterns.append(result)
        print(f"{model}: {result['best_pattern']} (RMSE={result['fits'][result['best_pattern']]['rmse']:.4f})")

    plot_error_propagation(failure_df, f"{figures_dir}/{prefix}error_propagation.png")
    plot_heatmap(failure_df, f"{figures_dir}/{prefix}heatmap.png")
    plot_pattern_comparison(patterns, f"{figures_dir}/{prefix}pattern_comparison.png")

    summary_df = pd.DataFrame([{
        "model": p["model"],
        "pattern": p["best_pattern"],
        "rmse": p["fits"][p["best_pattern"]]["rmse"]
    } for p in patterns])
    summary_df.to_csv(f"{csv_dir}/pattern_summary.csv", index=False)

    return failure_df, patterns