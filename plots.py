import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.optimize import curve_fit
import glob

def load_all_results(results_dir="results"):
    all_data = []
    for f in glob.glob(f"{results_dir}/**/*.json", recursive=True):
        model_name = f.split("/")[-1].split("_")[0]
        with open(f) as file:
            data = json.load(file)
            for d in data:
                if "error" not in d:
                    d["model"] = model_name
                    all_data.append(d)
    return pd.DataFrame(all_data)

def compute_failure_rates(df):
    results = []
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        baseline = model_df[model_df["error_step"].isna()]["evaluation"].apply(lambda x: x["combined"]).mean()

        for step in range(5):
            step_df = model_df[model_df["error_step"] == step]
            if len(step_df) == 0:
                continue
            step_score = step_df["evaluation"].apply(lambda x: x["combined"]).mean()
            degradation = (baseline - step_score) / baseline if baseline > 0 else 0
            results.append({
                "model": model,
                "error_step": step,
                "baseline": baseline,
                "score": step_score,
                "degradation": degradation,
            })
    return pd.DataFrame(results)

def plot_error_propagation(failure_df, output_path="figures/error_propagation.png"):
    plt.figure(figsize=(10, 6))
    step_names = ["Search", "Filter", "Summarize", "Compose", "Verify"]

    for model in failure_df["model"].unique():
        model_df = failure_df[failure_df["model"] == model].sort_values("error_step")
        plt.plot(model_df["error_step"], model_df["degradation"], marker="o", label=model, linewidth=2)

    plt.xlabel("Error Injection Step", fontsize=12)
    plt.ylabel("Performance Degradation", fontsize=12)
    plt.title("Error Propagation Across Models", fontsize=14)
    plt.xticks(range(5), step_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

def plot_heatmap(failure_df, output_path="figures/heatmap.png"):
    pivot = failure_df.pivot(index="model", columns="error_step", values="degradation")
    pivot.columns = ["Search", "Filter", "Summarize", "Compose", "Verify"]

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Reds", cbar_kws={"label": "Degradation"})
    plt.title("Error Impact Heatmap by Model and Step", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

def fit_decay_patterns(failure_df):
    def exp_decay(x, a, b):
        return a * np.exp(-b * x)

    def linear_decay(x, a, b):
        return a - b * x

    results = []
    for model in failure_df["model"].unique():
        model_df = failure_df[failure_df["model"] == model].sort_values("error_step")
        x = model_df["error_step"].values.astype(float)
        y = model_df["degradation"].values

        try:
            popt_exp, _ = curve_fit(exp_decay, x, y, p0=[0.5, 0.3], maxfev=5000)
            rmse_exp = np.sqrt(np.mean((y - exp_decay(x, *popt_exp)) ** 2))
        except:
            rmse_exp = float("inf")

        try:
            popt_lin, _ = curve_fit(linear_decay, x, y, p0=[0.5, 0.1], maxfev=5000)
            rmse_lin = np.sqrt(np.mean((y - linear_decay(x, *popt_lin)) ** 2))
        except:
            rmse_lin = float("inf")

        pattern = "exponential" if rmse_exp < rmse_lin else "linear"
        results.append({"model": model, "pattern": pattern, "rmse_exp": rmse_exp, "rmse_lin": rmse_lin})

    return pd.DataFrame(results)

def generate_report():
    import os
    os.makedirs("figures", exist_ok=True)

    df = load_all_results()
    print(f"Loaded {len(df)} results from {df['model'].nunique()} models")

    failure_df = compute_failure_rates(df)
    print("\n=== Failure Rates ===")
    print(failure_df.round(3))

    print("\n=== Generating Plots ===")
    plot_error_propagation(failure_df)
    plot_heatmap(failure_df)

    print("\n=== Pattern Analysis ===")
    patterns = fit_decay_patterns(failure_df)
    print(patterns)

    print("\n=== Critical Steps ===")
    critical = failure_df.loc[failure_df.groupby("model")["degradation"].idxmax()]
    print(critical[["model", "error_step", "degradation"]])

    return df, failure_df, patterns

if __name__ == "__main__":
    df, failure_df, patterns = generate_report()