"""Re-evaluate legacy experiment records using the current v2 combined_score formula.

GAP 5 FIX: Open-source model runs (llama, deepseek, mistral, qwen) used the
pre-P0-2 formula: 0.3*is_valid + 0.3*keyword + 0.4*factual. The newer runs use
0.4*preserved + 0.4*(1-survival) + 0.2*is_valid. These are not comparable.

This script:
  1. Loads all JSONL/JSON result files
  2. Identifies records that lack combined_score_components (legacy formula)
  3. Re-computes combined_score using the v2 formula from their stored evaluation data
  4. Writes updated records to new files (non-destructive)

Usage:
    python reevaluate_legacy.py              # dry run: report what needs fixing
    python reevaluate_legacy.py --write      # write corrected records
"""

import argparse
import glob
import json
import os
from copy import deepcopy


def _recompute_v2(ev: dict) -> tuple[float, dict]:
    """Recompute combined_score using the v2 formula from evaluation fields."""
    factual = ev.get("factual", {})

    assertions_present = factual.get("assertions_present", 0)
    assertions_total = factual.get("assertions_total", 0)
    survival = factual.get("error_survival_score", 0.0)

    if assertions_total > 0:
        preserved = assertions_present / assertions_total
    else:
        preserved = 1.0

    is_valid_val = 1 if ev.get("is_valid", False) else 0

    contradiction_penalty = 0.0
    contradictions = factual.get("contradictions_present", 0)
    if assertions_total > 0 and contradictions > 0:
        contradiction_penalty = min(0.5, 0.15 * contradictions)

    score = (
        0.40 * preserved
        + 0.40 * (1.0 - survival)
        + 0.20 * is_valid_val
        - contradiction_penalty
    )
    score = max(0.0, min(1.0, score))

    components = {
        "preserved": round(preserved, 4),
        "one_minus_survival": round(1.0 - survival, 4),
        "is_valid": is_valid_val,
        "contradiction_penalty": round(contradiction_penalty, 4),
        "weights": {"preserved": 0.40, "one_minus_survival": 0.40, "is_valid": 0.20},
    }
    return score, components


def process_file(path: str, write: bool) -> dict:
    """Process one result file. Returns stats."""
    records = []
    is_jsonl = path.endswith(".jsonl")

    if is_jsonl:
        with open(path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    else:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            records = data
        else:
            return {"skipped": True}

    n_legacy = 0
    n_already_v2 = 0
    updated_records = []

    for r in records:
        ev = r.get("evaluation")
        if not ev:
            updated_records.append(r)
            continue

        has_components = "combined_score_components" in ev
        if has_components:
            n_already_v2 += 1
            updated_records.append(r)
            continue

        n_legacy += 1
        r_new = deepcopy(r)
        old_score = ev.get("combined_score")
        new_score, components = _recompute_v2(ev)

        r_new["evaluation"]["combined_score_legacy_original"] = old_score
        r_new["evaluation"]["combined_score"] = round(new_score, 4)
        r_new["evaluation"]["combined_score_components"] = components
        r_new["evaluation"]["_reevaluated"] = True
        updated_records.append(r_new)

    stats = {
        "path": path,
        "total": len(records),
        "legacy": n_legacy,
        "already_v2": n_already_v2,
    }

    if write and n_legacy > 0:
        out_dir = os.path.join(os.path.dirname(path), "reevaluated")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, os.path.basename(path))

        if is_jsonl:
            with open(out_path, "w") as f:
                for r in updated_records:
                    f.write(json.dumps(r) + "\n")
        else:
            with open(out_path, "w") as f:
                json.dump(updated_records, f, indent=2)

        stats["output"] = out_path

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Write re-evaluated records to results/*/reevaluated/")
    args = parser.parse_args()

    files = glob.glob("results/**/*.jsonl", recursive=True) + \
            glob.glob("results/**/*.json", recursive=True)
    files = [f for f in files if "stats" not in f and "sanity" not in f and "reevaluated" not in f]

    total_legacy = 0
    total_v2 = 0

    for path in sorted(files):
        stats = process_file(path, args.write)
        if stats.get("skipped"):
            continue
        if stats["legacy"] > 0:
            print(f"  LEGACY  {stats['legacy']:>4} / {stats['total']:>4}  {path}")
            if args.write and "output" in stats:
                print(f"          -> {stats['output']}")
        total_legacy += stats.get("legacy", 0)
        total_v2 += stats.get("already_v2", 0)

    print(f"\nTotal: {total_legacy} legacy records, {total_v2} already v2")
    if total_legacy > 0 and not args.write:
        print("Run with --write to produce re-evaluated copies.")


if __name__ == "__main__":
    main()
