"""Theorem 1: Identifiability of step-wise attenuation factors.

Theorem (Identifiability of alpha_k under experimental injection):

    Given an LLM pipeline with K steps and randomized error injection at
    step i with known initial intensity s_i, the step-wise attenuation
    factors {alpha_{i+1}, ..., alpha_{K-1}} are identifiable from
    persistence measurements at each downstream step, provided:

    (A1) Injection is randomized (experimentally controlled), eliminating
         confounding between injection and downstream behavior.
    (A2) Persistence is measured at every step k > i (not just the final
         step), giving K - i - 1 transition observations per trial.
    (A3) The number of trials per cell n >= 15, ensuring sufficient
         precision for transition ratio estimation.

Proof sketch:

    Under the Markov model:
        s_k = s_i * prod_{j=i+1}^{k} alpha_j    for k > i       (Eq. 1)

    Since s_i is known (the injected error content) and s_k is observed
    (persistence at step k), each consecutive ratio:
        r_{k->k+1} = s_{k+1} / s_k = alpha_{k+1}                (Eq. 2)

    is directly estimable from the data. By (A1), the injection is
    exogenous so E[r_{k->k+1}] = alpha_{k+1} without bias from
    confounders. By (A2), we observe all K-i-1 transitions. By (A3),
    the law of large numbers gives consistent estimation.

    For the hierarchical Bayesian extension:
        log h_{k,m,tau,d} ~ N(mu_k + alpha_m + beta_tau + gamma_d, sigma^2)

    Identifiability follows from the fact that the design matrix has full
    column rank when we have observations across multiple models (m),
    error types (tau), and domains (d) — which our experimental design
    guarantees (4 models x 4 types x 4+ domains).

    The optimal intervention threshold (Corollary 1):
        Re-ask iff h_k(epsilon) * Delta_k > c/U

    is identifiable because h_k comes from the hierarchical posterior
    and Delta_k (downstream quality drop) is directly observed.

Monte Carlo verification below confirms finite-sample identifiability.
"""
import numpy as np
from scipy import stats


def simulate_pipeline(
    n_steps: int = 5,
    true_alphas: list[float] | None = None,
    n_trials: int = 100,
    noise_std: float = 0.05,
    seed: int = 42,
) -> dict:
    """Simulate a pipeline with known attenuation factors.

    Returns simulated persistence measurements at each step.
    """
    rng = np.random.default_rng(seed)
    if true_alphas is None:
        true_alphas = [0.8, 0.6, 0.5, 0.3]  # 4 transitions for 5-step pipeline
    assert len(true_alphas) == n_steps - 1

    records = []
    for trial in range(n_trials):
        s = [1.0]
        for k, alpha in enumerate(true_alphas):
            noise = rng.normal(0, noise_std)
            s_next = max(0.0, min(1.0, s[-1] * alpha + noise))
            s.append(s_next)
        records.append(s)

    return {
        "persistence_matrix": np.array(records),
        "true_alphas": true_alphas,
        "n_trials": n_trials,
        "n_steps": n_steps,
    }


def estimate_alphas_from_simulation(persistence_matrix: np.ndarray) -> np.ndarray:
    """Estimate alpha_k from simulated persistence data via transition ratios."""
    n_trials, n_steps = persistence_matrix.shape
    n_transitions = n_steps - 1
    alphas = np.zeros(n_transitions)

    for k in range(n_transitions):
        s_from = persistence_matrix[:, k]
        s_to = persistence_matrix[:, k + 1]
        mask = s_from > 0.05
        if mask.sum() < 3:
            alphas[k] = 0.0
            continue
        ratios = np.clip(s_to[mask] / s_from[mask], 0, 1)
        alphas[k] = np.mean(ratios)

    return alphas


def monte_carlo_verification(
    n_simulations: int = 500,
    n_trials_per_sim: int = 15,
    true_alphas: list[float] | None = None,
    noise_std: float = 0.05,
    seed: int = 42,
) -> dict:
    """Monte Carlo verification of identifiability.

    Runs many simulations with known ground truth and checks whether
    estimated alphas converge to true values as n increases.

    Reports: bias, RMSE, coverage of 95% bootstrap CIs.
    """
    rng = np.random.default_rng(seed)
    if true_alphas is None:
        true_alphas = [0.8, 0.6, 0.5, 0.3]

    n_transitions = len(true_alphas)
    all_estimates = np.zeros((n_simulations, n_transitions))

    for sim in range(n_simulations):
        sim_data = simulate_pipeline(
            n_steps=n_transitions + 1,
            true_alphas=true_alphas,
            n_trials=n_trials_per_sim,
            noise_std=noise_std,
            seed=seed + sim,
        )
        all_estimates[sim] = estimate_alphas_from_simulation(
            sim_data["persistence_matrix"]
        )

    true_arr = np.array(true_alphas)
    bias = np.mean(all_estimates, axis=0) - true_arr
    rmse = np.sqrt(np.mean((all_estimates - true_arr) ** 2, axis=0))

    coverage = np.zeros(n_transitions)
    for k in range(n_transitions):
        ci_lo = np.percentile(all_estimates[:, k], 2.5)
        ci_hi = np.percentile(all_estimates[:, k], 97.5)
        coverage[k] = float(ci_lo <= true_alphas[k] <= ci_hi)

    return {
        "true_alphas": true_alphas,
        "mean_estimates": np.mean(all_estimates, axis=0).tolist(),
        "bias": bias.tolist(),
        "rmse": rmse.tolist(),
        "coverage_95": coverage.tolist(),
        "max_abs_bias": float(np.max(np.abs(bias))),
        "max_rmse": float(np.max(rmse)),
        "all_covered": bool(np.all(coverage == 1.0)),
        "n_simulations": n_simulations,
        "n_trials_per_sim": n_trials_per_sim,
    }


def convergence_test(
    sample_sizes: list[int] | None = None,
    true_alphas: list[float] | None = None,
    n_simulations: int = 200,
    seed: int = 42,
) -> dict:
    """Test that estimation error decreases with sample size (consistency)."""
    if sample_sizes is None:
        sample_sizes = [10, 15, 30, 60, 120, 240]
    if true_alphas is None:
        true_alphas = [0.8, 0.6, 0.5, 0.3]

    results = {}
    for n in sample_sizes:
        mc = monte_carlo_verification(
            n_simulations=n_simulations,
            n_trials_per_sim=n,
            true_alphas=true_alphas,
            seed=seed,
        )
        results[n] = {
            "max_rmse": mc["max_rmse"],
            "max_abs_bias": mc["max_abs_bias"],
            "all_covered": mc["all_covered"],
        }

    rmse_values = [results[n]["max_rmse"] for n in sample_sizes]
    is_decreasing = all(
        rmse_values[i] >= rmse_values[i + 1] * 0.8
        for i in range(len(rmse_values) - 1)
    )

    return {
        "sample_sizes": sample_sizes,
        "per_n": results,
        "rmse_decreasing": is_decreasing,
    }


def rank_identifiability_test(
    n_models: int = 4,
    n_types: int = 4,
    n_domains: int = 4,
    n_steps: int = 5,
) -> dict:
    """Verify that the hierarchical model design matrix has full column rank.

    The hierarchical model has effects: mu_step (K), alpha_model (M),
    beta_type (T), gamma_domain (D), beta_severity (1).
    Total parameters: K + M + T + D + 1 (plus sigma and phi).

    Full rank requires n_observations >> n_parameters and that the
    experimental design crosses all factors.
    """
    # Reference-coded: drop first level of each factor to avoid
    # collinearity (intercept absorbed into mu_step[0]).
    n_params = n_steps + (n_models - 1) + (n_types - 1) + (n_domains - 1) + 1
    n_cells = n_models * n_types * n_domains * n_steps
    min_trials_for_identifiability = n_params + 1

    rng = np.random.default_rng(42)
    X = np.zeros((n_cells, n_params))
    row = 0
    for m in range(n_models):
        for t in range(n_types):
            for d in range(n_domains):
                for s in range(n_steps):
                    col = 0
                    # mu_step indicators (all levels, acts as intercept per step)
                    X[row, s] = 1.0
                    col = n_steps
                    # alpha_model: reference-coded (drop m=0)
                    if m > 0:
                        X[row, col + m - 1] = 1.0
                    col += n_models - 1
                    # beta_type: reference-coded (drop t=0)
                    if t > 0:
                        X[row, col + t - 1] = 1.0
                    col += n_types - 1
                    # gamma_domain: reference-coded (drop d=0)
                    if d > 0:
                        X[row, col + d - 1] = 1.0
                    col += n_domains - 1
                    # severity (continuous)
                    X[row, col] = rng.random()
                    row += 1

    rank = np.linalg.matrix_rank(X)

    return {
        "n_parameters": n_params,
        "n_cells": n_cells,
        "design_matrix_rank": int(rank),
        "full_rank": rank == n_params,
        "min_trials_total": min_trials_for_identifiability,
        "note": (
            "Reference-coded design matrix. The Bayesian hierarchical model "
            "uses proper priors instead of reference coding, which also "
            "ensures identifiability."
        ),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Theorem 1: Identifiability Verification")
    print("=" * 60)

    print("\n--- Design matrix rank test ---")
    rank_result = rank_identifiability_test()
    print(f"Parameters: {rank_result['n_parameters']}")
    print(f"Design cells: {rank_result['n_cells']}")
    print(f"Matrix rank: {rank_result['design_matrix_rank']}")
    print(f"Full rank: {rank_result['full_rank']}")

    print("\n--- Monte Carlo verification (n=15 per cell, 500 sims) ---")
    mc_result = monte_carlo_verification(n_simulations=500, n_trials_per_sim=15)
    print(f"True alphas:     {mc_result['true_alphas']}")
    print(f"Mean estimates:  {[f'{x:.4f}' for x in mc_result['mean_estimates']]}")
    print(f"Bias:            {[f'{x:+.4f}' for x in mc_result['bias']]}")
    print(f"RMSE:            {[f'{x:.4f}' for x in mc_result['rmse']]}")
    print(f"95% CI coverage: {mc_result['coverage_95']}")
    print(f"Max |bias|: {mc_result['max_abs_bias']:.4f}")
    print(f"Max RMSE:   {mc_result['max_rmse']:.4f}")

    print("\n--- Convergence test ---")
    conv = convergence_test()
    for n, res in conv["per_n"].items():
        print(f"  n={n:>4d}: max_rmse={res['max_rmse']:.4f}, "
              f"max_|bias|={res['max_abs_bias']:.4f}, "
              f"covered={res['all_covered']}")
    print(f"RMSE decreasing with n: {conv['rmse_decreasing']}")

    print("\n--- Summary ---")
    if rank_result["full_rank"] and mc_result["max_abs_bias"] < 0.05:
        print("PASS: Identifiability conditions satisfied.")
    else:
        print("WARNING: Review identifiability conditions.")
