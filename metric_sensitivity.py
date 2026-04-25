"""Sweep the combined_score weight vector to test metric robustness."""
import json
import glob

import numpy as np
import pandas as pd

from record_utils import is_baseline, injection_is_valid


def recompute_score(ev, w_preserved, w_survival, w_valid):
    preserved = ev.get("factual", {}).get("factual_accuracy_score", 1.0)
    survival = ev.get("factual", {}).get("error_survival_score", 0.0)
    is_valid = 1 if ev.get("is_valid", True) else 0
    contradiction_penalty = ev.get("combined_score_components", {}).get("contradiction_penalty", 0.0)
    score = w_preserved * preserved + w_survival * (1 - survival) + w_valid * is_valid
    score = max(0, min(1, score - contradiction_penalty))
    return score


def sweep():
    records = []
    for f in glob.glob("results/**/*.jsonl", recursive=True):
        if "stats" in f or "sanity" in f or "_legacy" in f or "archive" in f:
            continue
        for line in open(f):
            try:
                r = json.loads(line)
                if "evaluation" in r:
                    records.append(r)
            except Exception:
                continue

    grid = []
    for wp in np.arange(0.2, 0.7, 0.1):
        for ws in np.arange(0.2, 0.7, 0.1):
            wv = round(1.0 - wp - ws, 2)
            if 0.05 <= wv <= 0.5:
                grid.append((round(wp, 2), round(ws, 2), wv))

    results = []
    for wp, ws, wv in grid:
        scores_bl, scores_inj = [], []
        for r in records:
            ev = r["evaluation"]
            s = recompute_score(ev, wp, ws, wv)
            if is_baseline(r):
                scores_bl.append(s)
            elif injection_is_valid(r):
                scores_inj.append(s)
        if scores_bl and scores_inj:
            bl_mean = np.mean(scores_bl)
            inj_mean = np.mean(scores_inj)
            fr = max(0, 1 - inj_mean / bl_mean) if bl_mean > 0 else 0
            results.append({
                "w_preserved": wp, "w_survival": ws, "w_valid": wv,
                "mean_failure_rate": round(fr, 4),
                "bl_mean": round(bl_mean, 4),
                "inj_mean": round(inj_mean, 4),
            })

    df = pd.DataFrame(results)
    df.to_csv("results/stats/metric_sensitivity.csv", index=False)
    print(df.sort_values("mean_failure_rate"))
    return df


if __name__ == "__main__":
    sweep()
