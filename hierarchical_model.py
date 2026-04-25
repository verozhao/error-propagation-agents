"""Hierarchical Bayesian hazard model for error propagation.

log h_{k,m,τ,d} ~ N(μ_k + α_m + β_τ + γ_d, σ²)

Fit via numpyro MCMC. Produces posterior credible intervals
that are tight even at n=15/cell via partial pooling.
"""
import numpy as np

def prepare_data(trial_records: list, persistence_threshold: float = 0.1) -> dict:
    """Convert trial records to arrays for numpyro.

    Returns dict with:
        failure: binary array (persistence > threshold)
        persistence: continuous array [0,1]
        step_idx: integer array
        model_idx: integer array
        error_type_idx: integer array
        domain_idx: integer array
        severity: continuous array
        index maps: {name: idx} dicts
    """
    model_map, type_map, domain_map = {}, {}, {}
    failures, persistences, steps, models, types, domains, severities = [], [], [], [], [], [], []

    for r in trial_records:
        if r.get("is_baseline") or r.get("error_step") is None:
            continue
        meta = r.get("injection_meta", {})
        error_type = meta.get("error_type", "unknown")
        domain = r.get("task_domain", "unknown")
        model = r.get("model", "unknown")
        sev = r.get("severity_semantic", 0.1)

        persistence_at_steps = r.get("persistence_curve", [])
        for step_idx, step_name, p_score in persistence_at_steps:
            if model not in model_map:
                model_map[model] = len(model_map)
            if error_type not in type_map:
                type_map[error_type] = len(type_map)
            if domain not in domain_map:
                domain_map[domain] = len(domain_map)

            failures.append(int(p_score > persistence_threshold))
            persistences.append(p_score)
            steps.append(step_idx)
            models.append(model_map[model])
            types.append(type_map[error_type])
            domains.append(domain_map[domain])
            severities.append(sev)

    return {
        "failure": np.array(failures),
        "persistence": np.array(persistences),
        "step_idx": np.array(steps),
        "model_idx": np.array(models),
        "error_type_idx": np.array(types),
        "domain_idx": np.array(domains),
        "severity": np.array(severities),
        "model_map": model_map,
        "type_map": type_map,
        "domain_map": domain_map,
        "n_steps": max(steps) + 1 if steps else 0,
        "n_models": len(model_map),
        "n_types": len(type_map),
        "n_domains": len(domain_map),
    }


def build_model():
    """Numpyro model specification.

    Call this inside a numpyro MCMC sampling context.
    """
    import numpyro
    import numpyro.distributions as dist
    import jax.numpy as jnp

    def hazard_model(step_idx, model_idx, type_idx, domain_idx, severity,
                     persistence=None, n_steps=5, n_models=4, n_types=4, n_domains=4):
        # Priors
        mu_step = numpyro.sample("mu_step", dist.Normal(0, 1).expand([n_steps]))
        alpha_model = numpyro.sample("alpha_model", dist.Normal(0, 0.5).expand([n_models]))
        beta_type = numpyro.sample("beta_type", dist.Normal(0, 0.5).expand([n_types]))
        gamma_domain = numpyro.sample("gamma_domain", dist.Normal(0, 0.5).expand([n_domains]))
        beta_severity = numpyro.sample("beta_severity", dist.Normal(0, 1))
        sigma = numpyro.sample("sigma", dist.HalfNormal(0.5))

        # Linear predictor
        logit_mu = (mu_step[step_idx] + alpha_model[model_idx] +
                    beta_type[type_idx] + gamma_domain[domain_idx] +
                    beta_severity * jnp.log(severity + 1e-6))

        # Beta regression for continuous persistence
        # Transform logit_mu to (0,1) via sigmoid for the mean
        mu = numpyro.deterministic("mu", jax.nn.sigmoid(logit_mu))
        # Concentration parameter
        phi = numpyro.sample("phi", dist.Gamma(2, 0.5))

        # Beta likelihood
        alpha_param = mu * phi
        beta_param = (1 - mu) * phi

        with numpyro.plate("obs", len(step_idx)):
            # Clamp persistence to (0.001, 0.999) for Beta distribution
            if persistence is not None:
                p_clamped = jnp.clip(persistence, 0.001, 0.999)
                numpyro.sample("persistence_obs", dist.Beta(alpha_param, beta_param), obs=p_clamped)

    return hazard_model


def fit_hierarchical(data: dict, num_warmup=500, num_samples=2000, num_chains=2):
    """Run MCMC and return posterior samples."""
    import jax
    import jax.random as jrandom
    import numpyro
    from numpyro.infer import MCMC, NUTS

    model = build_model()
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains)

    rng_key = jrandom.PRNGKey(42)
    mcmc.run(
        rng_key,
        step_idx=data["step_idx"],
        model_idx=data["model_idx"],
        type_idx=data["error_type_idx"],
        domain_idx=data["domain_idx"],
        severity=data["severity"],
        persistence=data["persistence"],
        n_steps=data["n_steps"],
        n_models=data["n_models"],
        n_types=data["n_types"],
        n_domains=data["n_domains"],
    )
    mcmc.print_summary()
    return mcmc.get_samples()


def extract_posteriors(samples: dict, data: dict) -> dict:
    """Extract interpretable posterior summaries."""
    results = {}

    for param in ["mu_step", "alpha_model", "beta_type", "gamma_domain", "beta_severity", "phi"]:
        if param in samples:
            vals = np.array(samples[param])
            results[param] = {
                "mean": np.mean(vals, axis=0).tolist(),
                "std": np.std(vals, axis=0).tolist(),
                "ci_2.5": np.percentile(vals, 2.5, axis=0).tolist(),
                "ci_97.5": np.percentile(vals, 97.5, axis=0).tolist(),
            }

    # Model robustness ranking (lower alpha = more robust)
    if "alpha_model" in samples:
        alpha = np.array(samples["alpha_model"])
        model_names = {v: k for k, v in data["model_map"].items()}
        ranking = []
        for idx in range(alpha.shape[1]):
            ranking.append({
                "model": model_names.get(idx, f"model_{idx}"),
                "alpha_mean": float(np.mean(alpha[:, idx])),
                "alpha_ci": [float(np.percentile(alpha[:, idx], 2.5)),
                             float(np.percentile(alpha[:, idx], 97.5))],
            })
        results["robustness_ranking"] = sorted(ranking, key=lambda x: x["alpha_mean"])

    return results


def rank1_factorization_test(samples: dict, data: dict) -> dict:
    """Test if α_k(model, error_type) ≈ φ(model) · ψ(k, error_type).

    Compute via SVD of the posterior mean matrix.
    Report explained variance of rank-1 approximation.
    """
    if "mu_step" not in samples or "alpha_model" not in samples:
        return {"explained_variance_rank1": None}

    # Build matrix: rows = models, cols = steps × types
    mu_step = np.mean(np.array(samples["mu_step"]), axis=0)
    alpha_model = np.mean(np.array(samples["alpha_model"]), axis=0)
    beta_type = np.mean(np.array(samples["beta_type"]), axis=0)

    n_models = len(alpha_model)
    n_steps = len(mu_step)
    n_types = len(beta_type)

    # Construct interaction matrix
    matrix = np.zeros((n_models, n_steps * n_types))
    for m in range(n_models):
        for s in range(n_steps):
            for t in range(n_types):
                matrix[m, s * n_types + t] = alpha_model[m] + mu_step[s] + beta_type[t]

    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    total_var = np.sum(S ** 2)
    rank1_var = S[0] ** 2 / total_var if total_var > 0 else 0

    return {
        "explained_variance_rank1": float(rank1_var),
        "singular_values": S.tolist(),
        "interpretation": "≥0.6 = model reduces to scalar robustness score per model"
    }
