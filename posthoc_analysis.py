"""Post-hoc analysis of error propagation experiments.

Reads existing experiment JSON results and computes:
  1. Injection features (word count, position, POS, sentence count)
  2. TF-IDF cosine similarity between baseline and error outputs
  3. Precision / Recall / F1 against ground truth
  4. Information retention (unigram/bigram overlap with baseline)
  5. Per-error-type regression formulas
  6. Universal combined formula via nonlinear fitting
  7. Step-level attenuation factors

Does NOT require re-running experiments — works entirely on saved JSON data.

Usage:
    python posthoc_analysis.py
    python posthoc_analysis.py --results-dir results --model gpt-4o-mini
"""

import json
import os
import re
import math
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from config import WORKFLOW_STEPS

import spacy
_NLP = None
def _get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def load_ground_truth(path="ground_truth.json"):
    with open(path) as f:
        raw = json.load(f)
    return {entry["query"]: entry for entry in raw["queries"]}

GT = None
def _get_gt():
    global GT
    if GT is None:
        try:
            GT = load_ground_truth()
        except Exception:
            GT = {}
    return GT


# Text utilities
def _tokenize(text: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', (text or "").lower())

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', (text or "").strip()) if s.strip()]

# POS tagging via spaCy
_POS_MAP = {
    "NOUN": "n_nouns", "PROPN": "n_nouns",
    "ADJ": "n_adjs",
    "VERB": "n_verbs", "AUX": "n_verbs",
    "ADV": "n_advs",
}

def _count_pos(text: str) -> dict:
    counts = {"n_nouns": 0, "n_adjs": 0, "n_verbs": 0, "n_advs": 0, "n_entities": 0}
    if not text or not text.strip():
        return counts
    nlp = _get_nlp()
    doc = nlp(text)
    for token in doc:
        key = _POS_MAP.get(token.pos_)
        if key:
            counts[key] += 1
    counts["n_entities"] = len(doc.ents)
    return counts


# TF-IDF similarity
def tfidf_cosine(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0
    tokens_a, tokens_b = _tokenize(text_a), _tokenize(text_b)
    tf_a, tf_b = Counter(tokens_a), Counter(tokens_b)
    vocab = set(tf_a) | set(tf_b)
    doc_freq = {w: (1 if w in tf_a else 0) + (1 if w in tf_b else 0) for w in vocab}

    vec_a, vec_b = {}, {}
    for w in vocab:
        idf = math.log(2 / doc_freq[w]) + 1
        vec_a[w] = tf_a.get(w, 0) * idf
        vec_b[w] = tf_b.get(w, 0) * idf

    dot = sum(vec_a.get(k, 0) * vec_b.get(k, 0) for k in vocab)
    na = math.sqrt(sum(v*v for v in vec_a.values()))
    nb = math.sqrt(sum(v*v for v in vec_b.values()))
    return dot / (na * nb) if na and nb else 0.0


# Precision / Recall / F1 against ground truth
def precision_recall_f1(output: str, query: str) -> dict:
    gt = _get_gt()
    entry = gt.get(query)
    if not entry:
        return {"precision": None, "recall": None, "f1": None}

    out_lower = (output or "").lower()
    assertions = entry.get("assertions", [])
    contradictions = entry.get("contradictions", [])

    tp = 0
    for a in assertions:
        kws = [k.lower() for k in a.get("keywords", [])]
        if kws and all(kw in out_lower for kw in kws):
            tp += 1
            continue
        for alias in a.get("aliases", []):
            if alias.lower() in out_lower:
                tp += 1
                break

    fp = sum(1 for c in contradictions if c.lower() in out_lower)

    recall = tp / len(assertions) if assertions else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp,
        "total_assertions": len(assertions),
    }


# Information retention
def information_retention(baseline: str, test: str) -> dict:
    if not baseline or not test:
        return {"unigram_retention": None, "bigram_retention": None}
    base_tok = _tokenize(baseline)
    test_tok = _tokenize(test)
    base_set, test_set = set(base_tok), set(test_tok)

    uni = len(base_set & test_set) / len(base_set) if base_set else 0
    base_bi = set(zip(base_tok, base_tok[1:]))
    test_bi = set(zip(test_tok, test_tok[1:]))
    bi = len(base_bi & test_bi) / len(base_bi) if base_bi else 0

    return {"unigram_retention": round(uni, 4), "bigram_retention": round(bi, 4)}


# Feature extraction from a single record
def extract_record_features(record: dict, baselines: dict) -> dict:
    """Extract all features from one experiment record.
    baselines: dict of (model, query) -> baseline final output text."""
    es = record.get("error_step")
    if es is None or es == -1:
        return None
    if "evaluation" not in record:
        return None

    ev = record["evaluation"]
    step_outputs = record.get("step_outputs", [])
    has_traces = step_outputs and isinstance(step_outputs[0], dict) and "output_text" in step_outputs[0]
    injected = record.get("injected_content", "")
    query = record.get("task_query", "")
    model = record.get("model", "")

    # injection features from delta text
    pos = _count_pos(injected) if injected else {"n_nouns": 0, "n_adjs": 0, "n_verbs": 0, "n_advs": 0, "n_entities": 0}
    delta_words = len(injected.split()) if injected else 0

    # features from traces
    pre_inj_text = ""
    post_inj_text = ""
    final_text = ""
    injection_position = 0.5
    n_sentences_affected = 0

    if has_traces:
        final_text = step_outputs[-1].get("output_text", "")
        for so in step_outputs:
            if so.get("error_injected"):
                pre_inj_text = so.get("pre_injection_output", "") or ""
                post_inj_text = so.get("output_text", "")
                # estimate injection position via first divergence point
                if pre_inj_text and post_inj_text:
                    # find where pre and post first differ
                    min_len = min(len(pre_inj_text), len(post_inj_text))
                    diverge_idx = min_len  # default: end
                    for ci in range(min_len):
                        if pre_inj_text[ci] != post_inj_text[ci]:
                            diverge_idx = ci
                            break
                    injection_position = diverge_idx / max(len(post_inj_text), 1)
                # sentences affected
                pre_sents = len(_sentences(pre_inj_text))
                post_sents = len(_sentences(post_inj_text))
                n_sentences_affected = abs(post_sents - pre_sents) if pre_sents else 1
                break

    pre_words = len(pre_inj_text.split()) if pre_inj_text else 0
    post_words = len(post_inj_text.split()) if post_inj_text else 0

    # baseline for TF-IDF and retention
    bl_key = (model, query)
    bl_text = baselines.get(bl_key, "")

    tfidf_sim = tfidf_cosine(bl_text, final_text) if bl_text and final_text else None
    retention = information_retention(bl_text, final_text) if bl_text and final_text else {}
    prf = precision_recall_f1(final_text, query) if final_text else {}

    # error survival across steps
    efs = record.get("error_found_in_step", {})
    survival_at_final = 0.0
    if efs:
        last_step = WORKFLOW_STEPS[-1] if WORKFLOW_STEPS else None
        if last_step and last_step in efs:
            survival_at_final = efs[last_step].get("survival_score", 0.0)

    # how many steps does the error survive through
    n_steps_propagated = sum(1 for v in efs.values() if v.get("propagated", False))

    rubric = ev.get("rubric", {})

    return {
        # identifiers
        "model": model,
        "error_type": record.get("error_type"),
        "error_step": es,
        "step_name": WORKFLOW_STEPS[es] if es < len(WORKFLOW_STEPS) else "?",
        "severity": record.get("severity", 1),
        "task_query": query,
        "has_traces": has_traces,
        # injection features (what the professor wants)
        "delta_word_count": delta_words,
        "injection_position": round(injection_position, 3),
        "n_sentences_affected": n_sentences_affected,
        "text_length_before": pre_words,
        "text_length_after": post_words,
        "n_words_changed": abs(post_words - pre_words),
        "length_change_ratio": post_words / max(pre_words, 1),
        **pos,
        # propagation features
        "survival_at_final": survival_at_final,
        "n_steps_propagated": n_steps_propagated,
        # existing evaluation metrics
        "combined_score": ev.get("combined_score", 0),
        "combined_score_v2": ev.get("combined_score_v2"),
        "combined_score_v3": ev.get("combined_score_v3"),
        "quality_score": ev.get("quality_score"),
        "keyword_score": ev.get("keyword_score"),
        "assertion_score": ev.get("assertion_score"),
        "is_valid": int(ev.get("is_valid", False)),
        "rubric_accuracy": rubric.get("accuracy"),
        "rubric_completeness": rubric.get("completeness"),
        "rubric_coherence": rubric.get("coherence"),
        "rubric_usefulness": rubric.get("usefulness"),
        # new metrics computed post-hoc
        "tfidf_similarity": round(tfidf_sim, 4) if tfidf_sim is not None else None,
        "precision": prf.get("precision"),
        "recall": prf.get("recall"),
        "f1": prf.get("f1"),
        "unigram_retention": retention.get("unigram_retention"),
        "bigram_retention": retention.get("bigram_retention"),
    }


# Data loading
def load_all_records(results_dir="results"):
    records = []
    baselines = {}  # (model, query) -> final output text

    for path in sorted(glob.glob(f"{results_dir}/**/*.json", recursive=True)):
        if any(skip in path for skip in ["stats", "sanity", "trace_analysis"]):
            continue
        with open(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        if not isinstance(data, list):
            continue

        for d in data:
            if "evaluation" not in d:
                continue
            records.append(d)

            # collect baselines
            es = d.get("error_step")
            if es is None or es == -1:
                so = d.get("step_outputs", [])
                if so and isinstance(so[0], dict) and "output_text" in so[0]:
                    key = (d.get("model", ""), d.get("task_query", ""))
                    if key not in baselines:
                        baselines[key] = so[-1].get("output_text", "")

    return records, baselines


def build_feature_df(records, baselines) -> pd.DataFrame:
    rows = []
    for r in records:
        feat = extract_record_features(r, baselines)
        if feat:
            rows.append(feat)
    return pd.DataFrame(rows)


def add_failure_rates(df: pd.DataFrame, records: list) -> pd.DataFrame:
    bl_scores = {}
    for r in records:
        es = r.get("error_step")
        if es is not None and es != -1:
            continue
        ev = r.get("evaluation", {})
        key = (r.get("model", ""), r.get("error_type", ""))
        bl_scores.setdefault(key, []).append(ev.get("combined_score", 0))

    bl_means = {k: np.mean(v) for k, v in bl_scores.items()}

    def calc_fr(row):
        key = (row["model"], row["error_type"])
        bl = bl_means.get(key, 0)
        return max(0, (bl - row["combined_score"]) / bl) if bl > 0 else 0

    df["failure_rate"] = df.apply(calc_fr, axis=1)
    return df


# Analysis functions
def correlation_analysis(df: pd.DataFrame, target: str = "failure_rate"):
    injection_features = [
        "delta_word_count", "injection_position", "n_sentences_affected",
        "n_words_changed", "length_change_ratio", "text_length_before",
        "n_nouns", "n_adjs", "n_verbs", "n_advs", "n_entities",
        "severity", "error_step",
    ]
    eval_metrics = [
        "tfidf_similarity", "precision", "recall", "f1",
        "unigram_retention", "bigram_retention",
        "survival_at_final", "n_steps_propagated",
    ]
    all_cols = injection_features + eval_metrics
    available = [c for c in all_cols if c in df.columns and df[c].notna().sum() > 5]

    print(f"\n{'='*60}")
    print(f"CORRELATION ANALYSIS (target: {target})")
    print(f"{'='*60}")

    corrs = {}
    for col in available:
        valid = df[[col, target]].dropna()
        if len(valid) < 5:
            continue
        r = valid[col].corr(valid[target])
        corrs[col] = r

    # print sorted by magnitude
    for col, r in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True):
        tag = "[INJ]" if col in injection_features else "[EVAL]"
        print(f"  {tag} {col:30s}  r = {r:+.3f}")

    # plot
    if corrs:
        corr_s = pd.Series(corrs).sort_values(key=abs, ascending=True)
        fig, ax = plt.subplots(figsize=(9, 6))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in corr_s]
        corr_s.plot(kind="barh", ax=ax, color=colors)
        ax.set_xlabel(f"Pearson r with {target}")
        ax.set_title("Feature Correlations with Error Degradation")
        ax.axvline(0, color="black", linewidth=0.5)
        plt.tight_layout()
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/posthoc_correlations.png", dpi=150)
        print(f"\nSaved: figures/posthoc_correlations.png")
        plt.close()

    return corrs


def per_error_type_regression(df: pd.DataFrame):
    """Fit failure_rate ~ injection features per error type. This produces the
    'formula' the professor asked for."""
    try:
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.model_selection import cross_val_score
    except ImportError:
        print("pip install scikit-learn")
        return {}

    feature_cols = [
        "delta_word_count", "injection_position", "n_sentences_affected",
        "n_words_changed", "n_nouns", "n_adjs", "n_verbs", "n_advs", "n_entities",
        "severity", "error_step",
    ]

    print(f"\n{'='*60}")
    print("PER-ERROR-TYPE REGRESSION FORMULAS")
    print(f"{'='*60}")

    formulas = {}
    for etype in sorted(df["error_type"].dropna().unique()):
        edf = df[df["error_type"] == etype]
        available = [c for c in feature_cols if c in edf.columns and edf[c].notna().sum() > 10]
        subset = edf[available + ["failure_rate"]].dropna()

        if len(subset) < 15:
            print(f"\n{etype}: insufficient data ({len(subset)} rows)")
            continue

        X = subset[available].values
        y = subset["failure_rate"].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        # linear fit with 5-fold CV
        lin = LinearRegression()
        cv_scores_lin = cross_val_score(lin, X_s, y, cv=5, scoring="r2")
        lin.fit(X_s, y)
        r2_lin_train = lin.score(X_s, y)

        # polynomial (degree 2) fit with 5-fold CV
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X_s)
        ridge = Ridge(alpha=1.0)
        cv_scores_poly = cross_val_score(ridge, X_poly, y, cv=5, scoring="r2")
        ridge.fit(X_poly, y)
        r2_poly_train = ridge.score(X_poly, y)

        print(f"\n--- {etype} (n={len(subset)}) ---")
        print(f"  Linear   R²_train={r2_lin_train:.3f}  R²_cv={cv_scores_lin.mean():.3f} ± {cv_scores_lin.std():.3f}")
        print(f"  Poly+int R²_train={r2_poly_train:.3f}  R²_cv={cv_scores_poly.mean():.3f} ± {cv_scores_poly.std():.3f}")

        # print linear formula with significance via bootstrap
        coeffs = sorted(zip(available, lin.coef_), key=lambda x: abs(x[1]), reverse=True)
        formula_str = f"  FR = {lin.intercept_:.4f}"
        for name, coef in coeffs:
            if abs(coef) < 0.001:
                continue
            sign = "+" if coef >= 0 else "-"
            formula_str += f" {sign} {abs(coef):.4f}·{name}"
        print(formula_str)

        # bootstrap CIs for coefficient significance
        rng = np.random.default_rng(42)
        n_boot = 500
        boot_coefs = np.zeros((n_boot, len(available)))
        for b in range(n_boot):
            idx = rng.choice(len(X_s), size=len(X_s), replace=True)
            lr = LinearRegression().fit(X_s[idx], y[idx])
            boot_coefs[b] = lr.coef_
        ci_lo = np.percentile(boot_coefs, 2.5, axis=0)
        ci_hi = np.percentile(boot_coefs, 97.5, axis=0)

        print(f"  Coefficients (95% CI):")
        for j, (name, coef) in enumerate(coeffs):
            idx_j = available.index(name)
            sig = "*" if (ci_lo[idx_j] > 0 or ci_hi[idx_j] < 0) else " "
            print(f"    {sig} {name:25s}  β={coef:+.4f}  [{ci_lo[idx_j]:+.4f}, {ci_hi[idx_j]:+.4f}]")

        formulas[etype] = {
            "linear_r2_train": r2_lin_train,
            "linear_r2_cv": float(cv_scores_lin.mean()),
            "linear_r2_cv_std": float(cv_scores_lin.std()),
            "poly_r2_train": r2_poly_train,
            "poly_r2_cv": float(cv_scores_poly.mean()),
            "poly_r2_cv_std": float(cv_scores_poly.std()),
            "intercept": float(lin.intercept_),
            "coefficients": {n: {"beta": float(c), "ci_lo": float(ci_lo[available.index(n)]),
                                  "ci_hi": float(ci_hi[available.index(n)])}
                             for n, c in coeffs},
            "n": len(subset),
        }

    return formulas


def universal_formula(df: pd.DataFrame):
    """Fit a single formula across all error types — the 'universal law' the professor wants."""
    try:
        from sklearn.linear_model import Ridge, LinearRegression
        from sklearn.preprocessing import StandardScaler, PolynomialFeatures
        from sklearn.model_selection import cross_val_score
    except ImportError:
        return

    feature_cols = [
        "delta_word_count", "injection_position", "n_sentences_affected",
        "n_words_changed", "n_nouns", "n_adjs", "n_verbs", "n_advs", "n_entities",
        "severity", "error_step",
    ]
    available = [c for c in feature_cols if c in df.columns and df[c].notna().sum() > 10]
    subset = df[available + ["failure_rate", "error_type"]].dropna()

    if len(subset) < 30:
        print(f"\nUniversal formula: insufficient data ({len(subset)} rows)")
        return

    # add error_type as dummy variables
    dummies = pd.get_dummies(subset["error_type"], prefix="etype", drop_first=True)
    X_df = pd.concat([subset[available].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    all_features = list(X_df.columns)
    X = X_df.values
    y = subset["failure_rate"].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    # linear with CV
    lin = LinearRegression()
    cv_lin = cross_val_score(lin, X_s, y, cv=5, scoring="r2")
    lin.fit(X_s, y)
    r2_lin_train = lin.score(X_s, y)

    # polynomial with interactions + CV
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = poly.fit_transform(X_s)
    ridge = Ridge(alpha=1.0)
    cv_poly = cross_val_score(ridge, X_poly, y, cv=5, scoring="r2")
    ridge.fit(X_poly, y)
    r2_poly_train = ridge.score(X_poly, y)

    print(f"\n{'='*60}")
    print("UNIVERSAL FORMULA (all error types combined)")
    print(f"{'='*60}")
    print(f"  n = {len(subset)}")
    print(f"  Linear   R²_train={r2_lin_train:.3f}  R²_cv={cv_lin.mean():.3f} ± {cv_lin.std():.3f}")
    print(f"  Poly+int R²_train={r2_poly_train:.3f}  R²_cv={cv_poly.mean():.3f} ± {cv_poly.std():.3f}")

    coeffs = sorted(zip(all_features, lin.coef_), key=lambda x: abs(x[1]), reverse=True)
    formula_str = f"  FR = {lin.intercept_:.4f}"
    for name, coef in coeffs:
        if abs(coef) < 0.001:
            continue
        sign = "+" if coef >= 0 else "-"
        formula_str += f" {sign} {abs(coef):.4f}·{name}"
    print(formula_str)

    print(f"\n  Top 5 predictors:")
    for name, coef in coeffs[:5]:
        print(f"    {name:30s}  β = {coef:+.4f}")


def attenuation_analysis(df: pd.DataFrame):
    """Per-step attenuation: does step_i reduce or amplify errors from step_{i-1}?"""
    print(f"\n{'='*60}")
    print("STEP ATTENUATION FACTORS")
    print(f"{'='*60}")
    print("  A > 0 = error buffer (good), A < 0 = error amplifier (bad)")

    results = []
    for (model, etype, sev), group in df.groupby(["model", "error_type", "severity"]):
        step_means = group.groupby("error_step")["failure_rate"].mean()
        for step in range(1, len(WORKFLOW_STEPS)):
            if step - 1 not in step_means.index or step not in step_means.index:
                continue
            fr_prev = step_means[step - 1]
            fr_curr = step_means[step]
            if fr_prev > 0.001:
                att = 1.0 - (fr_curr / fr_prev)
            else:
                att = 0.0 if fr_curr < 0.001 else -1.0
            results.append({
                "model": model, "error_type": etype, "severity": sev,
                "step": WORKFLOW_STEPS[step],
                "fr_from_prev": round(fr_prev, 4),
                "fr_at_step": round(fr_curr, 4),
                "attenuation": round(att, 3),
            })

    att_df = pd.DataFrame(results)
    if att_df.empty:
        print("  No data for attenuation analysis.")
        return

    summary = att_df.groupby(["error_type", "step"])["attenuation"].mean().unstack(fill_value=0)
    # reorder columns to pipeline order (skip "search" since attenuation is relative to prev step)
    step_order = [s for s in WORKFLOW_STEPS if s in summary.columns]
    summary = summary[step_order]
    print(f"\n  Mean attenuation by error type and step:")
    print(summary.round(3).to_string())

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    summary.T.plot(kind="bar", ax=ax)
    ax.set_ylabel("Attenuation Factor")
    ax.set_title("Step Attenuation (positive = error buffer)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(title="Error Type")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/posthoc_attenuation.png", dpi=150)
    print(f"  Saved: figures/posthoc_attenuation.png")
    plt.close()


def severity_degradation_plot(df: pd.DataFrame):
    """Plot failure rate vs severity per error type."""
    if df["severity"].nunique() < 2:
        print("\n  Only one severity level found, skipping severity plot.")
        return

    etypes = sorted(df["error_type"].unique())
    fig, axes = plt.subplots(1, len(etypes), figsize=(5 * len(etypes), 4),
                              sharey=True, squeeze=False)

    # compute global y range for consistent comparison
    for i, etype in enumerate(etypes):
        ax = axes[0][i]
        edf = df[df["error_type"] == etype]
        sev_step = edf.groupby(["severity", "step_name"])["failure_rate"].mean().reset_index()

        for step in WORKFLOW_STEPS:
            sdf = sev_step[sev_step["step_name"] == step]
            if not sdf.empty:
                ax.plot(sdf["severity"], sdf["failure_rate"], marker="o", label=step)

        ax.set_xlabel("Severity")
        if i == 0:
            ax.set_ylabel("Failure Rate")
        ax.set_title(f"{etype.title()}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Degradation by Severity Level", fontsize=13)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/posthoc_severity_curves.png", dpi=150)
    print(f"\nSaved: figures/posthoc_severity_curves.png")
    plt.close()


def new_metrics_summary(df: pd.DataFrame):
    """Print summary of TF-IDF, precision/recall/F1, retention metrics."""
    print(f"\n{'='*60}")
    print("NEW EVALUATION METRICS SUMMARY")
    print(f"{'='*60}")

    metrics = ["tfidf_similarity", "precision", "recall", "f1", "unigram_retention", "bigram_retention"]
    for etype in sorted(df["error_type"].dropna().unique()):
        edf = df[df["error_type"] == etype]
        print(f"\n  {etype}:")
        for m in metrics:
            if m in edf.columns and edf[m].notna().any():
                vals = edf[m].dropna()
                print(f"    {m:25s}  mean={vals.mean():.3f}  std={vals.std():.3f}  n={len(vals)}")

    # heatmap: mean of each metric by error_type x step
    for m in ["tfidf_similarity", "f1", "recall"]:
        if m not in df.columns or df[m].notna().sum() < 10:
            continue
        pivot = df.pivot_table(index="error_type", columns="step_name", values=m, aggfunc="mean")
        pivot = pivot[[s for s in WORKFLOW_STEPS if s in pivot.columns]]
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, 3))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax)
        ax.set_title(f"{m} by Error Type × Step")
        plt.tight_layout()
        plt.savefig(f"figures/posthoc_{m}_heatmap.png", dpi=150)
        print(f"  Saved: figures/posthoc_{m}_heatmap.png")
        plt.close()



# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--model", default=None, help="Filter to one model")
    args = parser.parse_args()

    print("Loading experiment results...")
    records, baselines = load_all_records(args.results_dir)
    if not records:
        print("No records found.")
        return

    print(f"Found {len(records)} total records, {len(baselines)} baseline outputs")

    print("Extracting features...")
    df = build_feature_df(records, baselines)
    if df.empty:
        print("No error-injected records with traces found.")
        return

    df = add_failure_rates(df, records)
    df = df[df['error_type'].notna() & (df['model'] != '') & (df['model'].notna())]

    if args.model:
        df = df[df["model"] == args.model]

    n_total = len(df)
    n_with_traces = df["has_traces"].sum()
    n_without = n_total - n_with_traces
    if n_without > 0:
        print(f"\n  Dropping {n_without} records without traces (old format)")
        df = df[df["has_traces"]].copy()

    print(f"\nAnalysis dataset: {len(df)} records")
    print(f"  Models: {sorted(df['model'].unique())}")
    print(f"  Error types: {sorted(df['error_type'].unique())}")
    print(f"  Severities: {sorted(df['severity'].unique())}")

    # save full feature CSV
    os.makedirs(f"{args.results_dir}/stats", exist_ok=True)
    csv_path = f"{args.results_dir}/stats/posthoc_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # run all analyses
    correlation_analysis(df)
    new_metrics_summary(df)
    formulas = per_error_type_regression(df)
    universal_formula(df)
    attenuation_analysis(df)
    severity_degradation_plot(df)

    # save formulas
    if formulas:
        formula_path = f"{args.results_dir}/stats/regression_formulas.json"
        with open(formula_path, "w") as f:
            json.dump(formulas, f, indent=2)
        print(f"\n  Saved: {formula_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
