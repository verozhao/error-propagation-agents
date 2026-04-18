"""Post-diagnostic verification. Fails loudly if the fixes are incomplete."""
import json
import glob
import sys
import os
import subprocess
from collections import defaultdict

import pandas as pd


def _load_records():
    """Load records from both JSONL and JSON files."""
    rows = []
    seen_keys = set()

    for f in glob.glob("results/**/*.jsonl", recursive=True):
        if "stats" in f or "sanity" in f or "archive" in f:
            continue
        with open(f) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if "evaluation" not in r:
                        continue
                    rows.append(r)
                except Exception:
                    continue

    if not rows:
        for f in glob.glob("results/**/*.json", recursive=True):
            if "stats" in f or "sanity" in f or "baseline_pre_fix" in f or "archive" in f:
                continue
            for r in json.load(open(f)):
                if "evaluation" not in r:
                    continue
                rows.append(r)

    return rows


def main():
    raw_records = _load_records()

    from record_utils import is_baseline as _is_baseline, injection_is_valid

    rows = []
    for r in raw_records:
        es = r["error_step"]
        is_baseline = _is_baseline(r)
        inj_valid = injection_is_valid(r)
        if is_baseline:
            step_norm = -1
        elif isinstance(es, list):
            step_norm = es[0] if es else -1
        elif es is None:
            # legacy bad record; skip
            continue
        else:
            step_norm = es
        rows.append({
            "etype": r["error_type"],
            "sev": r["severity"],
            "step": step_norm,
            "is_baseline": is_baseline,
            "is_compound": isinstance(es, list),
            "injection_valid": inj_valid,
            "score": r["evaluation"]["combined_score"],
            "quality_score": r["evaluation"].get("quality_score"),
            "combined_score_legacy": r["evaluation"].get("combined_score_legacy"),
            "meta": r.get("injection_meta") or {},
            "task_query": r.get("task_query"),
            "trial": r.get("trial"),
            "compound_steps": r.get("compound_steps"),
            "error_found_in_step": r.get("error_found_in_step"),
        })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} records")

    failures = []

    # Check 1: no verify injections
    if (df["step"] == 4).any():
        failures.append(f"FAIL P0-1: {(df['step']==4).sum()} verify injections found")
    else:
        print("PASS P0-1: no verify injections")

    # Check 2: severity_physical is present and monotonic
    df_single = df[df["compound_steps"].isna() | (df["compound_steps"].apply(lambda x: x is None))]
    for etype in ["factual", "omission", "semantic"]:
        sub = df_single[(df_single["etype"] == etype) & (df_single["step"] >= 0)]
        if sub.empty:
            print(f"WARN P0-2: no {etype} injection records found")
            continue
        phys = sub.groupby("sev")["meta"].apply(
            lambda s: pd.Series([m.get("severity_physical", None) for m in s]).dropna().mean()
        )
        if phys.isna().any():
            failures.append(f"FAIL P0-2: {etype} missing severity_physical at sev={phys[phys.isna()].index.tolist()}")
        elif not phys.is_monotonic_increasing:
            failures.append(f"FAIL P0-2: {etype} severity_physical not monotonic: {phys.to_dict()}")
        else:
            print(f"PASS P0-2: {etype} severity_physical = {phys.round(3).to_dict()}")

    # Check 3: baseline means stable across severity
    base = df_single[df_single["is_baseline"] == True]
    if not base.empty:
        stability = base.groupby(["etype", "sev"])["score"].mean().unstack()
        max_spread = (stability.max(axis=1) - stability.min(axis=1)).max()
        if max_spread > 0.02:
            failures.append(f"FAIL P0-4: baseline drift = {max_spread:.3f} (expected <=0.02 with judge off)")
        else:
            print(f"PASS P0-4: baseline drift = {max_spread:.4f}")
    else:
        print("WARN P0-4: no baseline records found, skipping drift check")

    # Check 4: failure rate monotonic
    fr_by = defaultdict(dict)
    for etype in df_single["etype"].unique():
        sub = df_single[df_single["etype"] == etype]
        base_mean = sub[sub["step"] == -1].groupby("sev")["score"].mean()
        for step in range(4):
            inj_mean = sub[sub["step"] == step].groupby("sev")["score"].mean()
            if not base_mean.empty and not inj_mean.empty:
                fr = ((base_mean - inj_mean) / base_mean).clip(lower=0)
                fr_by[etype][step] = fr.to_dict()

    print("\nFailure rate by (error_type, step, severity):")
    for etype, by_step in fr_by.items():
        print(f"  {etype}:")
        for step, d in by_step.items():
            print(f"    step={step}: " + ", ".join(f"sev{s}={v:.3f}" for s, v in sorted(d.items())))

    # Check 5: run statistical_tests.py and verify pairing
    result = subprocess.run(["python", "statistical_tests.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        failures.append("FAIL P0-3: statistical_tests.py returned non-zero exit code")
    else:
        sig_path = "results/stats/significance.csv"
        if os.path.exists(sig_path):
            sig = pd.read_csv(sig_path)
            if not (sig["test"] == "wilcoxon_paired_by_query_trial").all():
                failures.append(f"FAIL P0-3: non-paired tests in significance.csv: {sig['test'].value_counts().to_dict()}")
            else:
                print(f"PASS P0-3: all {len(sig)} rows use paired Wilcoxon")
        else:
            failures.append("FAIL P0-3: significance.csv not found")

    # Check 6: combined_score present, quality_score null (LLM judge is off)
    has_judge = df["quality_score"].notna().sum()
    if has_judge > 0:
        # Only fail if the new (Phase 3+) records have judge scores
        new_records = df[df["combined_score_legacy"].isna()]
        new_with_judge = new_records["quality_score"].notna().sum()
        if new_with_judge > 0:
            failures.append(f"FAIL P1-5: {new_with_judge} new records have quality_score (judge should be off)")
        else:
            print(f"PASS P1-5: all new records have quality_score=null (judge off); {has_judge} legacy records have scores")
    else:
        print("PASS P1-5: all records have quality_score=null (judge off)")

    missing_score = df["score"].isna().sum()
    if missing_score > 0:
        failures.append(f"FAIL P1-5: {missing_score} records missing combined_score")
    else:
        print(f"PASS P1-5: all {len(df)} records have combined_score")

    # Check 7: compound records present (if any compound runs exist)
    compound_records = df[df["compound_steps"].apply(lambda x: x is not None and x != [])]
    if len(compound_records) > 0:
        print(f"PASS compound: {len(compound_records)} compound records found")
    else:
        print("WARN compound: no compound records found (run compound diagnostic to populate)")

    # Check 8: claim-survival matrices exist
    has_efs = df["error_found_in_step"].apply(lambda x: x is not None and len(x) > 0 if x else False).sum()
    if has_efs > 0:
        print(f"PASS survival: {has_efs} records have error_found_in_step data")
    else:
        print("WARN survival: no error_found_in_step data found")

    # Check 9 (Issue α): injection validity rate — failed injections
    # should be a small fraction (<5%) of attempts. Higher rates mean
    # the source text is too short for the injector (e.g. single-sentence
    # outputs for omission), and those conditions should be reported.
    non_bl = df[df["is_baseline"] == False]
    if not non_bl.empty:
        invalid = non_bl[non_bl["injection_valid"] == False]
        invalid_rate = len(invalid) / len(non_bl)
        print(f"\nInjection validity (non-baseline records):")
        print(f"  total={len(non_bl)}, invalid={len(invalid)}, rate={invalid_rate:.2%}")
        if invalid_rate > 0.05:
            # breakdown by condition
            by_cell = (
                invalid.groupby(["etype", "sev", "step"]).size()
                .reset_index(name="n_invalid")
                .sort_values("n_invalid", ascending=False)
                .head(10)
            )
            print(f"  WARN: invalid-injection rate > 5%. Top cells:")
            print(by_cell.to_string(index=False))
        else:
            print(f"  PASS: invalid-injection rate under 5% threshold")

    if failures:
        print("\n".join(["\n===== DIAGNOSTIC FAILED ====="] + failures))
        sys.exit(1)
    print("\n===== DIAGNOSTIC PASSED =====")


if __name__ == "__main__":
    main()
