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
from scipy.stats import gaussian_kde
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

    # Use trigram containment (matching factual_accuracy.py logic) to avoid
    # false positives from single-word or short substring matches
    fp = 0
    out_tok = _tokenize(output)
    out_tris = set(zip(out_tok, out_tok[1:], out_tok[2:])) if len(out_tok) >= 3 else set()
    for c in contradictions:
        c_tok = _tokenize(c)
        if len(c_tok) < 3:
            # short phrase: require exact substring (same as factual_accuracy.py)
            if c.lower() in out_lower:
                fp += 1
        else:
            c_tris = set(zip(c_tok, c_tok[1:], c_tok[2:]))
            if c_tris and len(c_tris & out_tris) / len(c_tris) >= 0.5:
                fp += 1

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
    inj_meta = record.get("injection_meta") or {}
    severity_physical = inj_meta.get("severity_physical", record.get("severity", 1))

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

    # self-correction: did verify detect the error?
    verify_detected = False
    if has_traces:
        verify_text = step_outputs[-1].get("output_text", "")
        verify_detected = "INVALID" in verify_text.upper()

    # per-step information compression: how much does each step change its input?
    step_compression = []
    if has_traces:
        for so in step_outputs:
            in_len = len(_tokenize(so.get("input_text", "")))
            out_len = len(_tokenize(so.get("output_text", "")))
            step_compression.append(out_len / max(in_len, 1))

    rubric = ev.get("rubric", {})

    return {
        # identifiers
        "model": model,
        "error_type": record.get("error_type"),
        "error_step": es,
        "step_name": WORKFLOW_STEPS[es] if es < len(WORKFLOW_STEPS) else "?",
        "severity": record.get("severity", 1),
        "severity_physical": severity_physical,
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
        # self-correction features
        "verify_detected": int(verify_detected),
        "step_compression_mean": round(np.mean(step_compression), 4) if step_compression else None,
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
    # Use combined_score_v3 (676 distinct values) instead of v1 (34 values)
    # for much better regression resolution
    score_col = "combined_score_v3"
    fallback_col = "combined_score"

    bl_scores = {}
    for r in records:
        es = r.get("error_step")
        if es is not None and es != -1:
            continue
        ev = r.get("evaluation", {})
        key = (r.get("model", ""), r.get("error_type", ""))
        s = ev.get(score_col) or ev.get(fallback_col, 0)
        bl_scores.setdefault(key, []).append(s)

    bl_means = {k: np.mean(v) for k, v in bl_scores.items()}

    def calc_fr(row):
        key = (row["model"], row["error_type"])
        bl = bl_means.get(key, 0)
        score = row.get(score_col)
        if score is None or np.isnan(score):
            score = row.get(fallback_col, 0)
        return max(0, (bl - score) / bl) if bl > 0 else 0

    df["failure_rate"] = df.apply(calc_fr, axis=1)
    return df


# Analysis functions
def correlation_analysis(df: pd.DataFrame, target: str = "failure_rate"):
    injection_features = [
        "delta_word_count", "injection_position", "n_sentences_affected",
        "n_words_changed", "length_change_ratio", "text_length_before",
        "n_nouns", "n_adjs", "n_verbs", "n_advs", "n_entities",
        "severity", "severity_physical", "error_step",
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
        "severity", "severity_physical", "error_step",
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
        "severity", "severity_physical", "error_step",
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
                att = max(-2.0, min(2.0, 1.0 - (fr_curr / fr_prev)))
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



def self_correction_analysis(df: pd.DataFrame, records: list):
    """Analyze per-step action spaces and self-correction capabilities.

    The professor's framework:
      - Each step has an observation space (what it sees) and action space (what it can do)
      - Some steps can detect errors from prior steps (observation capability)
      - Detection may or may not reduce error propagation (correction effectiveness)
      - These capabilities should factor into the universal formula
    """
    print(f"\n{'='*60}")
    print("SELF-CORRECTION & ACTION SPACE ANALYSIS")
    print(f"{'='*60}")

    # --- Step action space definitions ---
    action_space = {
        "search":    {"observes": "original query only",
                      "can_detect": "none (generates from scratch)",
                      "can_correct": "no prior step to correct"},
        "filter":    {"observes": "search results",
                      "can_detect": "irrelevant/low-quality results",
                      "can_correct": "remove suspicious entries"},
        "summarize": {"observes": "filtered results",
                      "can_detect": "contradictions within results",
                      "can_correct": "omit conflicting claims"},
        "compose":   {"observes": "summary only",
                      "can_detect": "incoherence, off-topic content",
                      "can_correct": "rephrase, reframe"},
        "verify":    {"observes": "recommendation + original query",
                      "can_detect": "factual issues, query mismatch",
                      "can_correct": "flag INVALID (but does not rewrite)"},
    }

    print("\n  Step Action Spaces:")
    for step, space in action_space.items():
        print(f"    {step:12s}  observes: {space['observes']}")
        print(f"    {'':12s}  detects:  {space['can_detect']}")
        print(f"    {'':12s}  corrects: {space['can_correct']}")

    # --- Verify detection rate by injection step and severity ---
    if "verify_detected" not in df.columns:
        print("\n  No verify_detected data available.")
        return

    print(f"\n  Verify Detection Rate (did verify flag INVALID?):")
    det_pivot = df.pivot_table(index="error_type", columns=["error_step", "severity"],
                                values="verify_detected", aggfunc="mean")
    # Simplify: by error_step only
    det_by_step = df.pivot_table(index="error_type", columns="step_name",
                                  values="verify_detected", aggfunc="mean")
    step_order = [s for s in WORKFLOW_STEPS if s in det_by_step.columns]
    det_by_step = det_by_step[step_order]
    print(det_by_step.round(3).to_string())

    # --- Does detection actually help? Compare failure rates when verify detected vs not ---
    print(f"\n  Failure Rate: verify detected vs. not detected:")
    print(f"    {'error_type':12s}  {'detected':>10s}  {'not_detected':>14s}  {'delta':>8s}  {'n_det':>6s}  {'n_not':>6s}")
    for etype in sorted(df["error_type"].dropna().unique()):
        edf = df[df["error_type"] == etype]
        det = edf[edf["verify_detected"] == 1]["failure_rate"]
        ndet = edf[edf["verify_detected"] == 0]["failure_rate"]
        if len(det) > 5 and len(ndet) > 5:
            delta = det.mean() - ndet.mean()
            print(f"    {etype:12s}  {det.mean():10.3f}  {ndet.mean():14.3f}  {delta:+8.3f}  {len(det):6d}  {len(ndet):6d}")

    # --- Detection rate by severity ---
    print(f"\n  Verify Detection Rate by Severity:")
    det_sev = df.pivot_table(index="error_type", columns="severity",
                              values="verify_detected", aggfunc="mean")
    print(det_sev.round(3).to_string())

    # --- Per-step error attenuation: compare survival scores across steps ---
    print(f"\n  Error Survival Decay Across Steps (mean survival_score):")
    survival_data = []
    for r in records:
        es = r.get("error_step")
        if es is None or es == -1:
            continue
        efs = r.get("error_found_in_step", {})
        if not efs:
            continue
        for step_name, info in efs.items():
            survival_data.append({
                "error_type": r.get("error_type"),
                "error_step": es,
                "severity": r.get("severity", 1),
                "observed_at": step_name,
                "survival_score": info.get("survival_score", 0),
            })

    if survival_data:
        surv_df = pd.DataFrame(survival_data)
        surv_pivot = surv_df.pivot_table(index="error_type", columns="observed_at",
                                          values="survival_score", aggfunc="mean")
        surv_pivot = surv_pivot[[s for s in WORKFLOW_STEPS if s in surv_pivot.columns]]
        print(surv_pivot.round(3).to_string())

    # --- Plot: detection rate heatmap ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Detection rate by step
    ax = axes[0]
    sns.heatmap(det_by_step, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("Verify Detection Rate by Injection Step")

    # Detection rate by severity
    ax = axes[1]
    sns.heatmap(det_sev, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax, vmin=0, vmax=1)
    ax.set_title("Verify Detection Rate by Severity")

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/posthoc_self_correction.png", dpi=150)
    print(f"\n  Saved: figures/posthoc_self_correction.png")
    plt.close()

    # --- Plot: survival decay across steps ---
    if survival_data:
        fig, ax = plt.subplots(figsize=(8, 4))
        for etype in sorted(surv_df["error_type"].unique()):
            edata = surv_df[surv_df["error_type"] == etype]
            means = edata.groupby("observed_at")["survival_score"].mean()
            means = means.reindex(WORKFLOW_STEPS).dropna()
            ax.plot(range(len(means)), means.values, marker="o", label=etype)
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(means.index, rotation=45)

        ax.set_ylabel("Mean Error Survival Score")
        ax.set_xlabel("Pipeline Step")
        ax.set_title("Error Survival Decay Across Pipeline")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("figures/posthoc_survival_decay.png", dpi=150)
        print(f"  Saved: figures/posthoc_survival_decay.png")
        plt.close()

    # --- Add verify_detected to regression features ---
    print(f"\n  Regression with self-correction variable:")
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score

        feature_cols = [
            "delta_word_count", "injection_position", "n_sentences_affected",
            "n_words_changed", "n_nouns", "n_adjs", "n_verbs", "n_advs", "n_entities",
            "severity", "severity_physical", "error_step", "verify_detected",
        ]
        available = [c for c in feature_cols if c in df.columns and df[c].notna().sum() > 10]
        subset = df[available + ["failure_rate"]].dropna()

        if len(subset) > 30:
            X = subset[available].values
            y = subset["failure_rate"].values
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            lin = LinearRegression()
            cv = cross_val_score(lin, X_s, y, cv=5, scoring="r2")
            lin.fit(X_s, y)

            print(f"    R²_train={lin.score(X_s, y):.3f}  R²_cv={cv.mean():.3f} ± {cv.std():.3f}")
            coeffs = sorted(zip(available, lin.coef_), key=lambda x: abs(x[1]), reverse=True)
            for name, coef in coeffs[:5]:
                print(f"      {name:25s}  beta={coef:+.4f}")
            print(f"    (verify_detected beta={dict(coeffs).get('verify_detected', 0):+.4f})")
    except ImportError:
        pass


def distributional_analysis(df: pd.DataFrame):
    """Distributional reporting: PDFs, CDFs, percentiles, and P(degradation > X)."""
    print(f"\n{'='*60}")
    print("DISTRIBUTIONAL ANALYSIS")
    print(f"{'='*60}")

    etypes = sorted(df["error_type"].dropna().unique())
    severities = sorted(df["severity"].unique())

    # --- Percentile table ---
    print(f"\n  Percentile table (failure_rate):")
    pcts = [50, 75, 90, 95]
    header = f"  {'condition':35s}" + "".join(f"  P{p:02d}" for p in pcts) + "   mean    n"
    print(header)
    print("  " + "-" * len(header))

    for etype in etypes:
        for sev in severities:
            mask = (df["error_type"] == etype) & (df["severity"] == sev)
            vals = df.loc[mask, "failure_rate"].dropna()
            if len(vals) < 5:
                continue
            pct_vals = np.percentile(vals, pcts)
            label = f"{etype} sev={sev}"
            row = f"  {label:35s}"
            for v in pct_vals:
                row += f"  {v:.3f}"
            row += f"  {vals.mean():.3f}  {len(vals):4d}"
            print(row)

    # --- P(degradation > X%) table ---
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]
    print(f"\n  P(failure_rate > threshold):")
    header = f"  {'condition':35s}" + "".join(f"  >{int(t*100):02d}%" for t in thresholds)
    print(header)
    print("  " + "-" * len(header))

    for etype in etypes:
        for sev in severities:
            mask = (df["error_type"] == etype) & (df["severity"] == sev)
            vals = df.loc[mask, "failure_rate"].dropna()
            if len(vals) < 5:
                continue
            label = f"{etype} sev={sev}"
            row = f"  {label:35s}"
            for t in thresholds:
                prob = (vals > t).mean()
                row += f"  {prob:.2f} "
            print(row)

    # --- KDE density plots per error type ---
    fig, axes = plt.subplots(1, len(etypes), figsize=(5 * len(etypes), 4),
                              sharey=True, sharex=True, squeeze=False)
    x_grid = np.linspace(0, df["failure_rate"].quantile(0.99) + 0.05, 200)

    for i, etype in enumerate(etypes):
        ax = axes[0][i]
        edf = df[df["error_type"] == etype]
        for sev in severities:
            vals = edf.loc[edf["severity"] == sev, "failure_rate"].dropna()
            if len(vals) < 5:
                continue
            kde = gaussian_kde(vals, bw_method=0.3)
            ax.plot(x_grid, kde(x_grid), label=f"sev={sev}")
            ax.fill_between(x_grid, kde(x_grid), alpha=0.15)
        ax.set_xlabel("Failure Rate")
        if i == 0:
            ax.set_ylabel("Density")
        ax.set_title(f"{etype.title()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Degradation Distribution by Severity", fontsize=13)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/posthoc_degradation_pdf.png", dpi=150)
    print(f"\n  Saved: figures/posthoc_degradation_pdf.png")
    plt.close()

    # --- CDF plots per error type ---
    fig, axes = plt.subplots(1, len(etypes), figsize=(5 * len(etypes), 4),
                              sharey=True, sharex=True, squeeze=False)

    for i, etype in enumerate(etypes):
        ax = axes[0][i]
        edf = df[df["error_type"] == etype]
        for sev in severities:
            vals = edf.loc[edf["severity"] == sev, "failure_rate"].dropna().sort_values()
            if len(vals) < 5:
                continue
            cdf_y = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf_y, label=f"sev={sev}")
        ax.set_xlabel("Failure Rate")
        if i == 0:
            ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"{etype.title()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Degradation CDF by Severity", fontsize=13)
    plt.tight_layout()
    plt.savefig("figures/posthoc_degradation_cdf.png", dpi=150)
    print(f"  Saved: figures/posthoc_degradation_cdf.png")
    plt.close()

    # --- Per-step distributional breakdown ---
    print(f"\n  Per-step percentiles (P50 / P90):")
    print(f"  {'error_type':12s} {'sev':>3s}  ", end="")
    for s in WORKFLOW_STEPS:
        print(f"  {s:>14s}", end="")
    print()

    for etype in etypes:
        for sev in severities:
            mask = (df["error_type"] == etype) & (df["severity"] == sev)
            sub = df[mask]
            if len(sub) < 5:
                continue
            print(f"  {etype:12s} {sev:3d}  ", end="")
            for step in WORKFLOW_STEPS:
                vals = sub.loc[sub["step_name"] == step, "failure_rate"].dropna()
                if len(vals) < 3:
                    print(f"  {'--':>14s}", end="")
                else:
                    p50 = np.percentile(vals, 50)
                    p90 = np.percentile(vals, 90)
                    print(f"  {p50:.2f}/{p90:.2f}    ", end="")
            print()


def failure_rate_sanity_check(df: pd.DataFrame):
    """Check that failure_rate has enough spread for meaningful regression."""
    print(f"\n{'='*60}")
    print("FAILURE RATE SANITY CHECK")
    print(f"{'='*60}")

    fr = df["failure_rate"].dropna()
    print(f"  n={len(fr)}  mean={fr.mean():.3f}  std={fr.std():.3f}  "
          f"min={fr.min():.3f}  max={fr.max():.3f}  unique={fr.nunique()}")
    pcts = np.percentile(fr, [10, 25, 50, 75, 90])
    print(f"  P10={pcts[0]:.3f}  P25={pcts[1]:.3f}  P50={pcts[2]:.3f}  "
          f"P75={pcts[3]:.3f}  P90={pcts[4]:.3f}")

    if fr.nunique() < 20:
        print(f"  WARNING: only {fr.nunique()} unique values — target variable has low resolution")
    if fr.std() < 0.05:
        print(f"  WARNING: std={fr.std():.3f} — target variable has very low spread")

    # per error_type breakdown
    print(f"\n  By error type:")
    for etype in sorted(df["error_type"].dropna().unique()):
        efr = df.loc[df["error_type"] == etype, "failure_rate"].dropna()
        print(f"    {etype:12s}  n={len(efr):4d}  mean={efr.mean():.3f}  "
              f"std={efr.std():.3f}  unique={efr.nunique()}")

    # score component analysis: what drives variation?
    print(f"\n  Score component variation:")
    for col in ["is_valid", "keyword_score", "combined_score"]:
        if col in df.columns:
            vals = df[col].dropna()
            print(f"    {col:25s}  mean={vals.mean():.3f}  std={vals.std():.3f}  unique={vals.nunique()}")


def per_query_analysis(df: pd.DataFrame, target: str = "failure_rate"):
    """Per-query breakdown: different domains may have different propagation patterns."""
    if "task_query" not in df.columns or df["task_query"].nunique() < 2:
        print("\n  Only 1 query in data — skipping per-query analysis.")
        return

    print(f"\n{'='*60}")
    print("PER-QUERY ANALYSIS")
    print(f"{'='*60}")

    key_features = [
        "delta_word_count", "error_step", "severity",
        "n_nouns", "n_adjs", "n_words_changed",
    ]
    available = [c for c in key_features if c in df.columns and df[c].notna().sum() > 5]

    for query in sorted(df["task_query"].unique()):
        qdf = df[df["task_query"] == query]
        print(f"\n  Query: \"{query}\" (n={len(qdf)})")
        print(f"    failure_rate: mean={qdf[target].mean():.3f}  std={qdf[target].std():.3f}")

        # top correlations for this query
        corrs = {}
        for col in available:
            valid = qdf[[col, target]].dropna()
            if len(valid) >= 5:
                corrs[col] = valid[col].corr(valid[target])

        top = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        print(f"    Top correlations with {target}:")
        for name, r in top:
            print(f"      {name:25s}  r = {r:+.3f}")

        # per error_type mean failure rate
        print(f"    Mean failure rate by error type:")
        for etype in sorted(qdf["error_type"].dropna().unique()):
            efr = qdf.loc[qdf["error_type"] == etype, target].dropna()
            print(f"      {etype:12s}  mean={efr.mean():.3f}  std={efr.std():.3f}  n={len(efr)}")

        # per severity
        if qdf["severity"].nunique() > 1:
            print(f"    Mean failure rate by severity:")
            for sev in sorted(qdf["severity"].unique()):
                sfr = qdf.loc[qdf["severity"] == sev, target].dropna()
                print(f"      sev={sev}  mean={sfr.mean():.3f}  n={len(sfr)}")

    # cross-query comparison: which query is most vulnerable?
    print(f"\n  Cross-query vulnerability ranking:")
    query_means = df.groupby("task_query")[target].mean().sort_values(ascending=False)
    for q, m in query_means.items():
        n = len(df[df["task_query"] == q])
        print(f"    {m:.3f}  {q}  (n={n})")


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
    failure_rate_sanity_check(df)
    per_query_analysis(df)
    correlation_analysis(df)
    new_metrics_summary(df)
    formulas = per_error_type_regression(df)
    universal_formula(df)
    attenuation_analysis(df)
    severity_degradation_plot(df)
    distributional_analysis(df)
    self_correction_analysis(df, records)

    # save formulas
    if formulas:
        formula_path = f"{args.results_dir}/stats/regression_formulas.json"
        with open(formula_path, "w") as f:
            json.dump(formulas, f, indent=2)
        print(f"\n  Saved: {formula_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
