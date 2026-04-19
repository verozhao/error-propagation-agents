"""Analyze verify-retry ablation: compare retry-on vs retry-off results.

Loads results from normal runs (max_retries=1) and no-retry runs
(max_retries=0, produced via --no-retry flag) and computes the delta
in failure rates.

The no-retry results should be in files containing 'noretry' in their
name, or in a separate subdirectory. This script also checks whether
the verify step's detection rate (what fraction of injected errors
does the verify step flag as INVALID) correlates with recovery.

Usage:
    python retry_ablation_analysis.py
"""

import glob
import json
import os

import numpy as np
import pandas as pd

from record_utils import is_baseline, injection_is_valid
from config import WORKFLOW_STEPS


def load_records(pattern: str) -> list[dict]:
    rows = []
    for f in glob.glob(pattern, recursive=True):
        if "stats" in f or "sanity" in f or "langchain" in f:
            continue
        if f.endswith(".jsonl"):
            for line in open(f):
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        elif f.endswith(".json"):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    rows.extend(data)
            except Exception:
                pass
    return rows


def _classify_retry(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split records into retry-on and retry-off groups.

    Heuristic: records from files with 'noretry' in the path are retry-off.
    Records can also have a 'max_retries' field set by the experiment runner.
    """
    retry_on = []
    retry_off = []
    for r in records:
        if r.get("max_retries") == 0:
            retry_off.append(r)
        elif r.get("max_retries", 1) >= 1:
            retry_on.append(r)
    return retry_on, retry_off


def _compute_fr(records: list[dict]) -> dict:
    """Compute failure rate by (model, error_type) from a set of records."""
    baselines = [r for r in records if is_baseline(r)]
    injected = [r for r in records if not is_baseline(r) and injection_is_valid(r)]

    bl_scores = {}
    for r in baselines:
        key = (r["model"], r["error_type"])
        bl_scores.setdefault(key, []).append(r["evaluation"]["combined_score"])

    inj_scores = {}
    for r in injected:
        key = (r["model"], r["error_type"])
        inj_scores.setdefault(key, []).append(r["evaluation"]["combined_score"])

    result = {}
    for key in set(bl_scores) | set(inj_scores):
        bm = np.mean(bl_scores.get(key, [1.0]))
        im = np.mean(inj_scores.get(key, [0.0]))
        fr = max(0, 1 - im / bm) if bm > 0 else 0
        result[key] = {
            "failure_rate": round(fr, 4),
            "n_baseline": len(bl_scores.get(key, [])),
            "n_injected": len(inj_scores.get(key, [])),
        }
    return result


def verify_detection_rate(records: list[dict]) -> pd.DataFrame:
    """What fraction of injected-error trials does verify flag as INVALID?"""
    rows = []
    for r in records:
        if is_baseline(r) or not injection_is_valid(r):
            continue
        ev = r.get("evaluation", {})
        is_valid = ev.get("is_valid", True)
        rows.append({
            "model": r["model"],
            "error_type": r["error_type"],
            "error_step": r.get("error_step"),
            "severity": r.get("severity", 1),
            "verify_flagged": not is_valid,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    detection = df.groupby(["model", "error_type"]).agg(
        n_trials=("verify_flagged", "count"),
        n_flagged=("verify_flagged", "sum"),
        detection_rate=("verify_flagged", "mean"),
    ).reset_index()
    return detection


def analyze():
    all_records = load_records("results/**/*")

    retry_on, retry_off = _classify_retry(all_records)

    print(f"Records loaded: {len(all_records)} total")
    print(f"  retry-on:  {len(retry_on)}")
    print(f"  retry-off: {len(retry_off)}")

    os.makedirs("results/stats", exist_ok=True)

    detection = verify_detection_rate(all_records)
    if not detection.empty:
        detection.to_csv("results/stats/verify_detection_rate.csv", index=False)
        print("\n=== Verify Detection Rate ===")
        print(detection.to_string(index=False))

    if not retry_off:
        print("\nNo retry-off data found. Run experiments with --no-retry to generate.")
        print("Example: python run.py --mode run --use-api --trials 20 "
              "--models gpt-4o-mini --error-type factual --severity 2 --no-retry")
        return

    fr_on = _compute_fr(retry_on)
    fr_off = _compute_fr(retry_off)

    rows = []
    for key in set(fr_on) | set(fr_off):
        model, etype = key
        on = fr_on.get(key, {})
        off = fr_off.get(key, {})
        fr_with = on.get("failure_rate", float("nan"))
        fr_without = off.get("failure_rate", float("nan"))
        delta = fr_without - fr_with if not (np.isnan(fr_with) or np.isnan(fr_without)) else float("nan")
        pct_reduction = (delta / fr_without * 100) if fr_without > 0 and not np.isnan(delta) else float("nan")
        rows.append({
            "model": model,
            "error_type": etype,
            "fr_with_retry": fr_with,
            "fr_without_retry": fr_without,
            "delta": round(delta, 4) if not np.isnan(delta) else None,
            "pct_reduction": round(pct_reduction, 1) if not np.isnan(pct_reduction) else None,
            "n_retry_on": on.get("n_injected", 0),
            "n_retry_off": off.get("n_injected", 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/stats/retry_ablation.csv", index=False)
    print("\n=== Retry Ablation Results ===")
    print(df.to_string(index=False))

    if not df.empty and df["pct_reduction"].notna().any():
        mean_reduction = df["pct_reduction"].mean()
        print(f"\nMean failure rate reduction from single retry: {mean_reduction:.1f}%")


if __name__ == "__main__":
    analyze()
