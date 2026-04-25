"""Analyze chain architecture validation results and compare against the main pipeline.

Loads results/langchain_validation/*.json and results from the main pipeline,
then produces a comparison table showing whether the three propagation modes
(persistence, attenuation, reconstruction) reproduce across architectures.

Step mapping:
    The 3-step chain (retrieval/summarize/answer) maps to the 5-step medium
    pipeline as follows for qualitative comparison:
        retrieval -> search+filter (collapsed)
        summarize -> summarize
        answer    -> compose
    Results are reported as qualitative replication of decay shape, not
    quantitative cell-by-cell comparison, since step counts differ.

Usage:
    python langchain_analysis.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from record_utils import is_baseline, injection_is_valid


def load_langchain_records() -> list[dict]:
    rows = []
    for f in glob.glob("results/langchain_validation/*.json"):
        with open(f) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            rows.extend(data)
    return rows


def load_main_records() -> list[dict]:
    rows = []
    for f in glob.glob("results/**/*.jsonl", recursive=True):
        if "stats" in f or "sanity" in f or "langchain" in f or "_legacy" in f or "archive" in f:
            continue
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _failure_rate_table(records: list[dict], framework: str) -> pd.DataFrame:
    """Compute failure rates per (error_type, injection_step)."""
    baselines = [r for r in records if is_baseline(r)]
    injected = [r for r in records if not is_baseline(r) and injection_is_valid(r)]

    bl_by_etype = {}
    for r in baselines:
        etype = r["error_type"]
        ev = r.get("evaluation", {})
        score = ev.get("combined_score") if "combined_score" in ev else None
        if score is None:
            preserved = ev.get("preserved", 1.0)
            survival = ev.get("survival_score", 0.0)
            score = 0.4 * preserved + 0.4 * (1 - survival) + 0.2
        bl_by_etype.setdefault(etype, []).append(score)

    bl_mean = {k: np.mean(v) for k, v in bl_by_etype.items()}

    rows = []
    cell_scores = {}
    for r in injected:
        etype = r["error_type"]
        step = r.get("error_step")
        if isinstance(step, list):
            continue
        ev = r.get("evaluation", {})
        score = ev.get("combined_score") if "combined_score" in ev else None
        if score is None:
            preserved = ev.get("preserved", 1.0)
            survival = ev.get("survival_score", 0.0)
            score = 0.4 * preserved + 0.4 * (1 - survival) + 0.2
        cell_scores.setdefault((etype, step), []).append(score)

    for (etype, step), scores in cell_scores.items():
        bm = bl_mean.get(etype, 1.0)
        fr = max(0, 1 - np.mean(scores) / bm) if bm > 0 else 0
        rows.append({
            "framework": framework,
            "error_type": etype,
            "injection_step": step,
            "n": len(scores),
            "mean_score": round(np.mean(scores), 4),
            "baseline_mean": round(bm, 4),
            "failure_rate": round(fr, 4),
        })

    return pd.DataFrame(rows)


def compare():
    lc_records = load_langchain_records()
    main_records = load_main_records()

    if not lc_records:
        print("No LangChain validation data found in results/langchain_validation/")
        return

    lc_table = _failure_rate_table(lc_records, "langchain")
    main_table = _failure_rate_table(main_records, "main_pipeline") if main_records else pd.DataFrame()

    os.makedirs("results/stats", exist_ok=True)

    if not lc_table.empty:
        lc_table.to_csv("results/stats/langchain_failure_rates.csv", index=False)
        print("=== LangChain Validation Failure Rates ===")
        print(lc_table.to_string(index=False))

    if not main_table.empty and not lc_table.empty:
        merged = lc_table.merge(
            main_table,
            on=["error_type"],
            suffixes=("_lc", "_main"),
            how="outer",
        )
        comparison = lc_table.groupby("error_type")["failure_rate"].mean().reset_index()
        comparison.columns = ["error_type", "fr_langchain"]
        if not main_table.empty:
            main_agg = main_table.groupby("error_type")["failure_rate"].mean().reset_index()
            main_agg.columns = ["error_type", "fr_main"]
            comparison = comparison.merge(main_agg, on="error_type", how="outer")
            comparison["delta"] = comparison["fr_langchain"] - comparison["fr_main"]
        comparison.to_csv("results/stats/langchain_comparison.csv", index=False)
        print("\n=== Cross-Architecture Comparison (mean FR by error type) ===")
        print(comparison.to_string(index=False))

    print("\n=== Propagation Mode Classification ===")
    for etype in lc_table["error_type"].unique():
        sub = lc_table[lc_table["error_type"] == etype].sort_values("injection_step")
        frs = sub["failure_rate"].values
        if len(frs) < 2:
            continue
        trend = np.polyfit(range(len(frs)), frs, 1)[0] if len(frs) > 1 else 0
        mean_fr = np.mean(frs)
        if mean_fr < 0.05:
            mode = "RECONSTRUCTION"
        elif trend > 0.05:
            mode = "AMPLIFICATION"
        elif trend < -0.05:
            mode = "ATTENUATION"
        else:
            mode = "PERSISTENCE"
        print(f"  {etype}: {mode} (mean_FR={mean_fr:.3f}, trend={trend:.3f})")


if __name__ == "__main__":
    compare()
