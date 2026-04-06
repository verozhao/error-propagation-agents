"""Step-level analysis of injected-error survival across the pipeline.

Reads result JSON files written by experiment.py with `save_traces=True`
and computes, for each (model, error_type, injection_step) combination,
the fraction of trials in which the injected delta is still detectable
N steps downstream. This is the propagation trace described in Phase 2.2
of the report upgrade plan.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from glob import glob

import pandas as pd

from config import WORKFLOW_STEPS

STEP_INDEX = {s: i for i, s in enumerate(WORKFLOW_STEPS)}


def load_traced_results(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for d in data:
            if "error" in d and "evaluation" not in d:
                continue
            out.append(d)
    return out


def step_survival_table(records: list[dict]) -> pd.DataFrame:
    """Return long-form DF: model, error_type, injection_step, observed_step, n, propagated_rate, mean_score."""
    buckets: dict[tuple, list[tuple[bool, float]]] = defaultdict(list)
    for r in records:
        if not r.get("error_found_in_step") or r.get("error_step") is None:
            continue
        inj = WORKFLOW_STEPS[r["error_step"]] if isinstance(r["error_step"], int) and r["error_step"] >= 0 else None
        if inj is None:
            continue
        for observed_step, info in r["error_found_in_step"].items():
            key = (r.get("model"), r.get("error_type"), inj, observed_step)
            buckets[key].append((bool(info.get("propagated")), float(info.get("survival_score", 0.0))))

    rows = []
    for (model, etype, inj, observed), vals in buckets.items():
        n = len(vals)
        if not n:
            continue
        propagated_rate = sum(1 for p, _ in vals if p) / n
        mean_score = sum(s for _, s in vals) / n
        rows.append({
            "model": model,
            "error_type": etype,
            "injection_step": inj,
            "observed_step": observed,
            "observed_step_idx": STEP_INDEX.get(observed, -1),
            "injection_step_idx": STEP_INDEX.get(inj, -1),
            "downstream_distance": STEP_INDEX.get(observed, -1) - STEP_INDEX.get(inj, -1),
            "n": n,
            "propagated_rate": propagated_rate,
            "mean_survival_score": mean_score,
        })
    return pd.DataFrame(rows).sort_values(
        ["model", "error_type", "injection_step_idx", "observed_step_idx"]
    )


def downstream_decay(df: pd.DataFrame) -> pd.DataFrame:
    """Average propagated_rate by downstream distance from the injection point."""
    sub = df[df["downstream_distance"] >= 0]
    return (
        sub.groupby(["model", "error_type", "downstream_distance"])
        .agg(propagated_rate=("propagated_rate", "mean"), mean_score=("mean_survival_score", "mean"), n=("n", "sum"))
        .reset_index()
    )


def main(results_glob: str = "results/**/*.json", out_dir: str = "results/trace_analysis"):
    os.makedirs(out_dir, exist_ok=True)
    paths = glob(results_glob, recursive=True)
    records = load_traced_results(paths)
    if not records:
        print(f"No records found under {results_glob}")
        return
    df = step_survival_table(records)
    if df.empty:
        print("No traced runs found (these analyses need experiment.py results with save_traces=True).")
        return
    df.to_csv(os.path.join(out_dir, "step_survival.csv"), index=False)
    decay = downstream_decay(df)
    decay.to_csv(os.path.join(out_dir, "downstream_decay.csv"), index=False)
    print(f"Wrote {len(df)} step-survival rows and {len(decay)} decay rows to {out_dir}/")


if __name__ == "__main__":
    main()
