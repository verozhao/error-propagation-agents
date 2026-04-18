"""Sanity-check pipeline for human evaluation of error propagation.

Workflow
--------
1. `select` — pick the 5 worst-case (error_type, injection_step) conditions
   per error type from traced experiment results, then 3 trials per
   condition (one per query). Writes:
       results/sanity_checks/cases.csv     (one row per case, blank human cols)
       results/sanity_checks/cases.md      (side-by-side viewer)
       results/sanity_checks/cases.json    (full payload incl. step text)

2. (manual) the two authors fill in the human columns of cases.csv:
       human_quality_rating          1-5 integer
       error_visible_in_output       yes / no
       output_factually_correct      yes / no
       notes                         free text

3. `score` — recomputes:
       - Cohen's kappa between the two annotators (if both annotated)
       - Pearson + Spearman correlation between human_quality_rating and
         the automated metrics (combined_score, combined_score_v2,
         factual_accuracy_score)
       - confusion matrix: automated says "no degradation" vs human says
         "clearly degraded" (and vice versa)

Run as:
    python sanity_check.py select
    python sanity_check.py score
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from glob import glob
from collections import defaultdict
from statistics import mean

import pandas as pd

from config import WORKFLOW_STEPS

OUT_DIR = os.path.join("results", "sanity_checks")
ANNOTATION_FIELDS = [
    "human_quality_rating_a",
    "human_quality_rating_b",
    "error_visible_in_output_a",
    "error_visible_in_output_b",
    "output_factually_correct_a",
    "output_factually_correct_b",
    "notes",
]


from record_utils import is_baseline as _is_baseline, injection_is_valid


def _load_records(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        with open(p) as f:
            data = json.load(f)
        for d in data:
            if "evaluation" not in d:
                continue
            if not d.get("step_outputs") or not isinstance(d["step_outputs"][0], dict):
                continue
            if "output_text" not in d["step_outputs"][0]:
                continue  # untraced legacy run, can't show text
            out.append(d)
    return out


def select_cases(results_glob: str = "results/**/*.json", n_per_error: int = 5):
    """Pick the worst-case conditions for human review."""
    os.makedirs(OUT_DIR, exist_ok=True)
    records = _load_records(glob(results_glob, recursive=True))
    if not records:
        print(
            "No traced records found. Re-run experiments with save_traces=True "
            "(the new default in experiment.py) before running sanity_check select."
        )
        return

    df = pd.DataFrame(
        [
            {
                "error_type": r.get("error_type"),
                "error_step": r.get("error_step"),
                "task_query": r.get("task_query"),
                "model": r.get("model"),
                "combined_score": r["evaluation"].get("combined_score", 0.0),
                "combined_score_v2": r["evaluation"].get("combined_score_v2"),
                "_record": r,
            }
            for r in records
            if not _is_baseline(r)
               and not isinstance(r.get("error_step"), list)
               and injection_is_valid(r) is not False  # Issue α: drop no-ops
        ]
    )
    if df.empty:
        print("No error-injected traced records.")
        return

    # baseline mean per (model, error_type)  — P0-1: use is_baseline flag
    base = pd.DataFrame(
        [
            {
                "model": r.get("model"),
                "error_type": r.get("error_type"),
                "combined_score": r["evaluation"].get("combined_score", 0.0),
            }
            for r in records
            if _is_baseline(r)
        ]
    )
    base_mean = base.groupby(["model", "error_type"])["combined_score"].mean().to_dict()

    # rank conditions by mean degradation
    grouped = df.groupby(["error_type", "model", "error_step"]).agg(
        mean_score=("combined_score", "mean"), n=("combined_score", "size")
    ).reset_index()
    grouped["baseline"] = grouped.apply(
        lambda r: base_mean.get((r["model"], r["error_type"]), 0.0), axis=1
    )
    grouped["degradation"] = grouped["baseline"] - grouped["mean_score"]

    selected_cases: list[dict] = []
    for etype in sorted(df["error_type"].dropna().unique()):
        worst = (
            grouped[grouped["error_type"] == etype]
            .sort_values("degradation", ascending=False)
            .head(n_per_error)
        )
        for _, cond in worst.iterrows():
            cond_rows = df[
                (df["error_type"] == etype)
                & (df["model"] == cond["model"])
                & (df["error_step"] == cond["error_step"])
            ]
            for q in cond_rows["task_query"].unique():
                q_rows = cond_rows[cond_rows["task_query"] == q]
                # pick worst-scoring trial for that query within this condition
                worst_trial = q_rows.sort_values("combined_score").iloc[0]
                selected_cases.append(worst_trial["_record"])

    _write_outputs(selected_cases, base_mean)
    print(f"Wrote {len(selected_cases)} cases to {OUT_DIR}/")


def _write_outputs(records: list[dict], base_mean: dict):
    cases_json_path = os.path.join(OUT_DIR, "cases.json")
    cases_csv_path = os.path.join(OUT_DIR, "cases.csv")
    cases_md_path = os.path.join(OUT_DIR, "cases.md")

    with open(cases_json_path, "w") as f:
        json.dump(records, f, indent=2)

    csv_fields = [
        "case_id",
        "model",
        "error_type",
        "error_step",
        "task_query",
        "combined_score",
        "combined_score_v2",
        "factual_accuracy_score",
        "baseline_mean",
    ] + ANNOTATION_FIELDS

    with open(cases_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for i, r in enumerate(records):
            ev = r.get("evaluation", {})
            writer.writerow(
                {
                    "case_id": f"case_{i:03d}",
                    "model": r.get("model"),
                    "error_type": r.get("error_type"),
                    "error_step": r.get("error_step"),
                    "task_query": r.get("task_query"),
                    "combined_score": ev.get("combined_score"),
                    "combined_score_v2": ev.get("combined_score_v2"),
                    "factual_accuracy_score": (ev.get("factual") or {}).get("factual_accuracy_score"),
                    "baseline_mean": base_mean.get((r.get("model"), r.get("error_type"))),
                    **{k: "" for k in ANNOTATION_FIELDS},
                }
            )

    with open(cases_md_path, "w") as f:
        f.write("# Sanity check cases\n\n")
        f.write("Fill in `cases.csv` with your annotations, then run `python sanity_check.py score`.\n\n")
        for i, r in enumerate(records):
            ev = r.get("evaluation", {})
            inj_step = r.get("error_step")
            inj_step_name = WORKFLOW_STEPS[inj_step] if isinstance(inj_step, int) and 0 <= inj_step < len(WORKFLOW_STEPS) else "?"
            f.write(f"---\n\n## case_{i:03d}\n\n")
            f.write(f"- **model**: `{r.get('model')}`\n")
            f.write(f"- **error_type**: `{r.get('error_type')}`  **injected at step**: `{inj_step_name}`\n")
            f.write(f"- **query**: {r.get('task_query')}\n")
            f.write(f"- **combined_score**: {ev.get('combined_score'):.3f}  **v2**: {ev.get('combined_score_v2')}\n")
            fac = ev.get("factual") or {}
            if fac:
                f.write(
                    f"- **factual_accuracy**: {fac.get('factual_accuracy_score'):.3f}  "
                    f"propagated={fac.get('error_propagated')}  "
                    f"survival={fac.get('error_survival_score'):.3f}  "
                    f"assertions={fac.get('assertions_present')}/{fac.get('assertions_total')}  "
                    f"contradictions={fac.get('contradictions_present')}\n"
                )
            inj = r.get("injected_content")
            if inj:
                f.write(f"\n**Injected content**:\n\n> {inj}\n")
            steps = r.get("step_outputs") or []
            if steps and "pre_injection_output" in steps[0]:
                pre = next(
                    (s.get("pre_injection_output") for s in steps if s.get("error_injected")),
                    None,
                )
                if pre:
                    f.write(f"\n**Pre-injection output at injected step**:\n\n```\n{pre[:1500]}\n```\n")
            # P0-18: show compose (index -2, the main recommendation content)
            # and verify (index -1, the VALID/INVALID self-judgment) separately.
            # Automated metrics are computed on compose, not verify.
            if len(steps) >= 2:
                compose_out = steps[-2].get("output_text", "") or ""
                verify_out = steps[-1].get("output_text", "") or ""
                f.write(f"\n**Recommendation (compose step — primary content)**:\n\n```\n{compose_out[:2000]}\n```\n")
                f.write(f"\n**Self-verification (verify step)**:\n\n```\n{verify_out[:500]}\n```\n\n")
            elif steps:
                final = steps[-1].get("output_text", "") or ""
                f.write(f"\n**Final pipeline output**:\n\n```\n{final[:2000]}\n```\n\n")


def _kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two equal-length integer lists."""
    if not a or len(a) != len(b):
        return float("nan")
    cats = sorted(set(a) | set(b))
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pe = sum(
        (a.count(c) / n) * (b.count(c) / n) for c in cats
    )
    if pe == 1:
        return 1.0
    return (po - pe) / (1 - pe)


def _pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    dx = sum((xi - mx) ** 2 for xi in x) ** 0.5
    dy = sum((yi - my) ** 2 for yi in y) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def _spearman(x: list[float], y: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for i, idx in enumerate(order):
            r[idx] = i + 1
        return r

    return _pearson(rank(x), rank(y))


def score_annotations():
    csv_path = os.path.join(OUT_DIR, "cases.csv")
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found — run `select` first.")
        return
    rows = list(csv.DictReader(open(csv_path)))
    annotated = [r for r in rows if r.get("human_quality_rating_a") or r.get("human_quality_rating_b")]
    if not annotated:
        print("No annotated rows found yet — fill in cases.csv and re-run.")
        return

    def parse_int(s):
        try:
            return int(s)
        except (TypeError, ValueError):
            return None

    def parse_yn(s):
        s = (s or "").strip().lower()
        if s in ("y", "yes", "1", "true"):
            return 1
        if s in ("n", "no", "0", "false"):
            return 0
        return None

    a_vals = [parse_int(r["human_quality_rating_a"]) for r in annotated]
    b_vals = [parse_int(r["human_quality_rating_b"]) for r in annotated]
    paired = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
    kappa = _kappa([a for a, _ in paired], [b for _, b in paired]) if paired else float("nan")

    # combined human rating: average of available annotators
    def human_avg(r):
        vals = [parse_int(r["human_quality_rating_a"]), parse_int(r["human_quality_rating_b"])]
        vals = [v for v in vals if v is not None]
        return mean(vals) if vals else None

    points = [
        (human_avg(r), float(r["combined_score"]) if r["combined_score"] else None,
         float(r["combined_score_v2"]) if r["combined_score_v2"] else None,
         float(r["factual_accuracy_score"]) if r["factual_accuracy_score"] else None)
        for r in annotated
    ]
    pts_v1 = [(h, v) for h, v, _, _ in points if h is not None and v is not None]
    pts_v2 = [(h, v) for h, _, v, _ in points if h is not None and v is not None]
    pts_fac = [(h, v) for h, _, _, v in points if h is not None and v is not None]

    def corr_block(name, pts):
        if len(pts) < 2:
            return f"{name}: insufficient data"
        h, m = zip(*pts)
        return f"{name}: pearson={_pearson(list(h), list(m)):.3f}  spearman={_spearman(list(h), list(m)):.3f}  n={len(pts)}"

    # confusion matrix on "degraded" labels
    def auto_degraded(r):
        bm = float(r["baseline_mean"]) if r["baseline_mean"] else None
        cs = float(r["combined_score"]) if r["combined_score"] else None
        if bm is None or cs is None:
            return None
        return (bm - cs) > 0.1  # threshold

    def human_degraded(r):
        h = human_avg(r)
        if h is None:
            return None
        return h <= 3  # 1-5 scale, low = degraded

    confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for r in annotated:
        a, h = auto_degraded(r), human_degraded(r)
        if a is None or h is None:
            continue
        if a and h:
            confusion["tp"] += 1
        elif a and not h:
            confusion["fp"] += 1
        elif not a and not h:
            confusion["tn"] += 1
        else:
            confusion["fn"] += 1

    print("=" * 60)
    print("SANITY CHECK SCORING")
    print("=" * 60)
    print(f"Annotated cases: {len(annotated)} / {len(rows)}")
    print(f"Cohen's kappa (annotators a vs b, n={len(paired)}): {kappa:.3f}")
    print()
    print(corr_block("combined_score (v1)", pts_v1))
    print(corr_block("combined_score_v2", pts_v2))
    print(corr_block("factual_accuracy_score", pts_fac))
    print()
    print("Confusion (auto vs human, threshold=0.1 score drop, human <= 3 = degraded):")
    print(f"  TP={confusion['tp']}  FP={confusion['fp']}  TN={confusion['tn']}  FN={confusion['fn']}")

    out_path = os.path.join(OUT_DIR, "scoring_report.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "n_annotated": len(annotated),
                "n_paired": len(paired),
                "cohens_kappa": kappa,
                "pearson_v1": _pearson(*zip(*pts_v1)) if len(pts_v1) >= 2 else None,
                "pearson_v2": _pearson(*zip(*pts_v2)) if len(pts_v2) >= 2 else None,
                "pearson_fac": _pearson(*zip(*pts_fac)) if len(pts_fac) >= 2 else None,
                "spearman_v1": _spearman(*zip(*pts_v1)) if len(pts_v1) >= 2 else None,
                "spearman_v2": _spearman(*zip(*pts_v2)) if len(pts_v2) >= 2 else None,
                "spearman_fac": _spearman(*zip(*pts_fac)) if len(pts_fac) >= 2 else None,
                "confusion": confusion,
            },
            f,
            indent=2,
        )
    print(f"\nWrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["select", "score"])
    parser.add_argument("--glob", default="results/**/*.json")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    if args.action == "select":
        select_cases(args.glob, args.n)
    else:
        score_annotations()


if __name__ == "__main__":
    main()
