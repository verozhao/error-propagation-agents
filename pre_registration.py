"""Pre-registration and hypothesis testing framework.

Supports:
- Pre-registered primary hypotheses (Holm-corrected)
- Exploratory secondary hypotheses (FDR-corrected)
- Thesis contingency document generation
"""
import json
import hashlib
from datetime import datetime
from pingouin import tost

def create_registration_template() -> dict:
    """Generate blank pre-registration template."""
    return {
        "timestamp": datetime.now().isoformat(),
        "primary_hypotheses": [
            {"id": "H1", "statement": "", "test": "", "threshold": "", "decision_rule": ""},
            {"id": "H2", "statement": "", "test": "", "threshold": "", "decision_rule": ""},
        ],
        "secondary_hypotheses": [
            {"id": f"S{i}", "statement": "", "test": ""} for i in range(1, 5)
        ],
        "thesis_contingency": {
            "H1_confirmed_H2_confirmed": "",
            "H1_confirmed_H2_rejected": "",
            "H1_rejected_H2_confirmed": "",
            "H1_rejected_H2_rejected": "",
            "partial": "",
        },
        "analysis_hierarchy": {
            "headline_tests": "Holm-corrected across 10 tests",
            "exploratory": "FDR (Benjamini-Hochberg), clearly labeled",
            "model_parameters": "Bootstrap CIs, no multiplicity correction",
        },
        "bayesian_decision_rules": {
            "credible_interval": "Confirm if 95% credible interval excludes null. Partially confirm if 90% CI excludes null but 95% does not.",
            "rope": "Region of practical equivalence: [-0.05, 0.05] for standardized effects. Report posterior mass inside ROPE.",
            "bayes_factor": "BF > 10 strong evidence, 3-10 moderate, 1-3 anecdotal. Report alongside p-values.",
        },
    }


def commit_registration(registration: dict, filepath: str = "pre_registration.json"):
    """Save and compute hash for audit trail."""
    registration["commit_hash"] = hashlib.sha256(
        json.dumps(registration, sort_keys=True).encode()
    ).hexdigest()[:16]
    with open(filepath, "w") as f:
        json.dump(registration, f, indent=2)
    print(f"Pre-registration saved: {filepath}")
    print(f"Hash: {registration['commit_hash']}")
    print("Commit this file to git BEFORE running the main sweep.")
    return registration


def run_tost_equivalence(group1: list, group2: list,
                         bound: float = 0.10, alpha: float = 0.05) -> dict:
    """Two One-Sided Tests for equivalence."""
    import numpy as np
    from scipy import stats

    g1 = np.array(group1)
    g2 = np.array(group2)

    # TOST: test if difference is within [-bound, +bound]
    diff = np.mean(g1) - np.mean(g2)
    se = np.sqrt(np.var(g1) / len(g1) + np.var(g2) / len(g2))

    # Upper bound test: H0: diff >= bound
    t_upper = (diff - bound) / se
    p_upper = stats.t.cdf(t_upper, df=min(len(g1), len(g2)) - 1)

    # Lower bound test: H0: diff <= -bound
    t_lower = (diff + bound) / se
    p_lower = 1 - stats.t.cdf(t_lower, df=min(len(g1), len(g2)) - 1)

    p_tost = max(p_upper, p_lower)

    return {
        "mean_difference": float(diff),
        "equivalence_bounds": [-bound, bound],
        "p_value_tost": float(p_tost),
        "equivalent": p_tost < alpha,
        "n1": len(g1),
        "n2": len(g2),
    }
