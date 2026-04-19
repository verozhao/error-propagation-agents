"""Compound injection super-additivity analysis.

Tests whether P(fail | inject at i AND j) > P(fail|i) + P(fail|j) - P(fail|i)*P(fail|j)
i.e., whether compound errors interact super-additively.
"""
import json
import glob
import os

import numpy as np
import pandas as pd

from record_utils import is_baseline, injection_is_valid
from config import WORKFLOW_STEPS


def load_all_records():
    rows = []
    for f in glob.glob("results/**/*.jsonl", recursive=True):
        if "stats" in f or "sanity" in f:
            continue
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def analyze_superadditivity():
    records = load_all_records()
    baselines = [r for r in records if is_baseline(r)]
    singles = [r for r in records if not is_baseline(r)
               and not r.get("compound_steps")
               and injection_is_valid(r)]
    compounds = [r for r in records if r.get("compound_steps")
                 and injection_is_valid(r)]

    if not compounds:
        print("No compound injection data found.")
        return

    bl = {}
    for r in baselines:
        key = (r["model"], r["error_type"])
        bl.setdefault(key, []).append(r["evaluation"]["combined_score"])
    bl_mean = {k: np.mean(v) for k, v in bl.items()}

    single_fr = {}
    for r in singles:
        es = r["error_step"]
        if isinstance(es, list):
            continue
        key = (r["model"], r["error_type"], es)
        single_fr.setdefault(key, []).append(r["evaluation"]["combined_score"])
    single_fr_mean = {}
    for k, scores in single_fr.items():
        model, etype, step = k
        bm = bl_mean.get((model, etype), 1.0)
        fr = max(0, 1 - np.mean(scores) / bm) if bm > 0 else 0
        single_fr_mean[k] = fr

    results = []
    for r in compounds:
        steps = tuple(r["compound_steps"])
        model, etype = r["model"], r["error_type"]
        bm = bl_mean.get((model, etype), 1.0)
        score = r["evaluation"]["combined_score"]
        fr_compound = max(0, 1 - score / bm) if bm > 0 else 0

        fr_i = single_fr_mean.get((model, etype, steps[0]), 0)
        fr_j = single_fr_mean.get((model, etype, steps[1]), 0) if len(steps) > 1 else 0
        fr_expected_indep = fr_i + fr_j - fr_i * fr_j

        results.append({
            "model": model, "error_type": etype,
            "steps": str(steps),
            "fr_compound": round(fr_compound, 4),
            "fr_step_i": round(fr_i, 4),
            "fr_step_j": round(fr_j, 4),
            "fr_expected_indep": round(fr_expected_indep, 4),
            "super_additive": fr_compound > fr_expected_indep,
            "interaction_delta": round(fr_compound - fr_expected_indep, 4),
        })

    df = pd.DataFrame(results)
    os.makedirs("results/stats", exist_ok=True)
    df.to_csv("results/stats/compound_superadditivity.csv", index=False)
    print(df.groupby(["error_type", "super_additive"]).size())
    print("\nMean interaction delta by error_type:")
    print(df.groupby("error_type")["interaction_delta"].mean())
    return df


if __name__ == "__main__":
    analyze_superadditivity()
