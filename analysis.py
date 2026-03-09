import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats
from config import OUTPUT_DIR, WORKFLOW_STEPS


def load_results(filepath: str) -> pd.DataFrame:
    with open(filepath) as f:
        data = json.load(f)
    
    rows = []
    for r in data:
        if "error" in r:
            continue
        rows.append({
            "model": r["model"],
            "task_query": r["task_query"],
            "error_step": r["error_step"] if r["error_step"] is not None else -1,
            "error_type": r["error_type"],
            "trial": r["trial"],
            "is_valid": r["evaluation"]["is_valid"],
            "keyword_score": r["evaluation"]["keyword_score"],
            "quality_score": r["evaluation"]["quality_score"],
            "combined_score": r["evaluation"]["combined_score"],
        })
    
    return pd.DataFrame(rows)


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
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(pattern_results[:6]):
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
    
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()


def generate_report(results_file: str):
    df = load_results(results_file)
    failure_df = compute_failure_rates(df)
    
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
    
    plot_error_propagation(failure_df, f"{OUTPUT_DIR}/error_propagation.png")
    plot_pattern_comparison(patterns, f"{OUTPUT_DIR}/pattern_comparison.png")
    
    summary_df = pd.DataFrame([{
        "model": p["model"],
        "pattern": p["best_pattern"],
        "rmse": p["fits"][p["best_pattern"]]["rmse"]
    } for p in patterns])
    summary_df.to_csv(f"{OUTPUT_DIR}/pattern_summary.csv", index=False)
    
    return failure_df, patterns