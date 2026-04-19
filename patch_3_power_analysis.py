"""
PATCH 3: Add post-hoc power analysis to statistical_tests.py

This function should be added after the run_significance() function,
and called in main() after significance results are computed.

It computes achieved power for the Wilcoxon signed-rank test using
the normal approximation, which reviewers at ICLR/NeurIPS expect.
"""


def compute_posthoc_power(sig_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Compute achieved statistical power for each test in sig_df.

    Uses the normal approximation to the Wilcoxon signed-rank test:
        z_observed = effect_size_r * sqrt(n)
        power = P(Z > z_alpha - z_observed)

    where z_alpha is the critical value for the chosen alpha level.

    Appends 'achieved_power' column to sig_df.
    Underpowered tests (power < 0.8) are flagged.
    """
    from scipy.stats import norm

    z_alpha = norm.ppf(1 - alpha / 2)  # two-tailed

    powers = []
    for _, row in sig_df.iterrows():
        n = row.get("n_paired", 0)
        r = abs(row.get("effect_size_r", 0))

        if n < 3 or r == 0 or np.isnan(r):
            powers.append(np.nan)
            continue

        # Normal approximation: z_obs ≈ r * sqrt(n)
        z_obs = r * np.sqrt(n)
        # Power = P(reject H0 | H1 true) = P(Z > z_alpha - z_obs) + P(Z < -z_alpha - z_obs)
        power = norm.cdf(z_obs - z_alpha) + norm.cdf(-z_obs - z_alpha)
        powers.append(round(min(1.0, max(0.0, power)), 4))

    sig_df = sig_df.copy()
    sig_df["achieved_power"] = powers
    sig_df["underpowered"] = sig_df["achieved_power"].apply(
        lambda p: True if (not np.isnan(p) and p < 0.8) else False
    )

    # Print summary
    valid = sig_df.dropna(subset=["achieved_power"])
    n_underpowered = valid["underpowered"].sum()
    mean_power = valid["achieved_power"].mean()
    print(f"\n  Post-hoc power analysis (alpha={alpha}, two-tailed):")
    print(f"    Mean achieved power: {mean_power:.3f}")
    print(f"    Underpowered tests (power < 0.8): {n_underpowered}/{len(valid)}")
    if n_underpowered > 0:
        underpowered = valid[valid["underpowered"]]
        for _, row in underpowered.iterrows():
            print(f"      {row['model']} / {row['error_type']} / {row['injection_step']}: "
                  f"power={row['achieved_power']:.3f}, n={row['n_paired']}, r={row['effect_size_r']:.3f}")
            # Suggest minimum n for power=0.8
            r = abs(row['effect_size_r'])
            if r > 0.01:
                z_beta = norm.ppf(0.8)  # 0.842
                n_needed = int(np.ceil(((z_alpha + z_beta) / r) ** 2))
                print(f"        → need n≈{n_needed} for power=0.8 at this effect size")

    return sig_df


# --- INTEGRATION INSTRUCTIONS ---
#
# In statistical_tests.py::main(), after the line that saves significance.csv:
#
#     sig_df.to_csv(sig_path, index=False)
#
# Add:
#
#     sig_df = compute_posthoc_power(sig_df, alpha=0.05)
#     sig_df.to_csv(sig_path, index=False)  # overwrite with power column
#     print(f"Updated {sig_path} with post-hoc power analysis")
#
# This overwrites the CSV with the extra columns: achieved_power, underpowered.
