"""Build claim-survival matrices per (error_type, injection_step, severity).

Produces results/stats/claim_survival_matrix.csv and figures/survival_matrices/*.png.
"""

import json
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import WORKFLOW_STEPS


def load_records(results_glob="results/**/*.jsonl"):
    rows = []
    for path in glob.glob(results_glob, recursive=True):
        if "stats" in path or "sanity" in path or "archive" in path:
            continue
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    if not rows:
        for path in glob.glob("results/**/*.json", recursive=True):
            if "stats" in path or "sanity" in path or "archive" in path:
                continue
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                rows.extend(data)
    return rows


def build_survival_matrices(records):
    results = {}
    for r in records:
        # P0-1: skip baselines (explicit flag preferred, legacy fallback)
        is_baseline = r.get("is_baseline")
        if is_baseline is None:
            is_baseline = (r.get("error_step") is None
                           and r.get("compound_steps") is None)
        if is_baseline:
            continue
        # skip compound runs — this analysis is single-injection only
        if r.get("compound_steps"):
            continue
        es = r.get("error_step")
        if isinstance(es, list) or es is None:
            # compound or legacy-bad record; skip
            continue
        efs = r.get("error_found_in_step") or {}
        if not efs:
            continue
        key = (r["error_type"], r.get("severity", 1))
        inj_step_idx = es
        for step_name, d in efs.items():
            try:
                obs_step_idx = WORKFLOW_STEPS.index(step_name)
            except ValueError:
                continue
            if obs_step_idx < inj_step_idx:
                continue
            results.setdefault(key, []).append({
                "injection_step": WORKFLOW_STEPS[inj_step_idx],
                "injection_step_idx": inj_step_idx,
                "obs_step": step_name,
                "obs_step_idx": obs_step_idx,
                "propagated": int(bool(d.get("propagated"))),
                "survival": float(d.get("survival_score", 0.0)),
            })
    mats = {}
    for key, rows in results.items():
        df = pd.DataFrame(rows)
        prop_pivot = df.pivot_table(
            index="injection_step_idx", columns="obs_step_idx",
            values="propagated", aggfunc="mean"
        )
        surv_pivot = df.pivot_table(
            index="injection_step_idx", columns="obs_step_idx",
            values="survival", aggfunc="mean"
        )
        prop_pivot.index = [WORKFLOW_STEPS[i] for i in prop_pivot.index]
        surv_pivot.index = [WORKFLOW_STEPS[i] for i in surv_pivot.index]
        prop_pivot.columns = [WORKFLOW_STEPS[i] for i in prop_pivot.columns]
        surv_pivot.columns = [WORKFLOW_STEPS[i] for i in surv_pivot.columns]
        mats[key] = {"propagation": prop_pivot, "survival": surv_pivot}
    return mats


def plot_survival_heatmaps(mats, out_dir="figures/survival_matrices"):
    os.makedirs(out_dir, exist_ok=True)
    for (etype, sev), m in mats.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        sns.heatmap(m["propagation"], annot=True, fmt=".2f", cmap="Reds",
                    vmin=0, vmax=1, ax=axes[0], cbar_kws={"label": "P(propagated)"})
        axes[0].set_title(f"{etype} sev={sev}: binary propagation")
        axes[0].set_xlabel("Observed at step")
        axes[0].set_ylabel("Injected at step")
        sns.heatmap(m["survival"], annot=True, fmt=".2f", cmap="YlOrRd",
                    vmin=0, vmax=1, ax=axes[1], cbar_kws={"label": "Survival score"})
        axes[1].set_title(f"{etype} sev={sev}: continuous survival")
        axes[1].set_xlabel("Observed at step")
        axes[1].set_ylabel("Injected at step")
        plt.tight_layout()
        path = f"{out_dir}/{etype}_sev{sev}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"Saved: {path}")


def export_csv(mats, out_path="results/stats/claim_survival_matrix.csv"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for (etype, sev), m in mats.items():
        for inj in m["propagation"].index:
            for obs in m["propagation"].columns:
                p = m["propagation"].loc[inj, obs] if obs in m["propagation"].columns else None
                s = m["survival"].loc[inj, obs] if obs in m["survival"].columns else None
                rows.append({
                    "error_type": etype, "severity": sev,
                    "injection_step": inj, "obs_step": obs,
                    "propagation_rate": p, "survival_score": s,
                })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main():
    records = load_records()
    if not records:
        print("No records found.")
        return
    print(f"Loaded {len(records)} records")
    mats = build_survival_matrices(records)
    if not mats:
        print("No survival matrices could be built (no injected records with error_found_in_step).")
        return
    export_csv(mats)
    plot_survival_heatmaps(mats)


if __name__ == "__main__":
    main()
