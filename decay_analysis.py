"""Phase 2B: Persistence decay model fitting — headline empirical finding."""
import json
import glob
import numpy as np
from collections import defaultdict
from persistence import fit_decay_models
from record_utils import is_baseline


def load_persistence_curves():
    """Load all records with persistence curves from JSONL files."""
    records = []
    for path in glob.glob("results/ragtruth_weighted_error/*.jsonl"):
        if "_legacy" in path or "_failed" in path:
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("persistence_curve") and not is_baseline(r):
                    records.append(r)
    return records


def aggregate_by_model(records):
    """Group persistence scores by (model, steps_from_injection)."""
    by_model = defaultdict(lambda: defaultdict(list))
    for r in records:
        model = r.get("model", "unknown")
        inject_step = r.get("error_step")
        if inject_step is None or isinstance(inject_step, list):
            continue
        for step_idx, step_name, score in r["persistence_curve"]:
            distance = step_idx - inject_step
            if distance > 0:
                by_model[model][distance].append(score)
    return by_model


def run_decay_analysis():
    print("Loading persistence curves...")
    records = load_persistence_curves()
    print(f"Loaded {len(records)} records with persistence curves")

    by_model = aggregate_by_model(records)

    results = {}
    for model in sorted(by_model.keys()):
        distances_dict = by_model[model]
        distances = sorted(distances_dict.keys())
        mean_persistence = [np.mean(distances_dict[d]) for d in distances]
        std_persistence = [np.std(distances_dict[d]) for d in distances]
        n_per_step = [len(distances_dict[d]) for d in distances]

        d_arr = np.array(distances, dtype=float)
        p_arr = np.array(mean_persistence, dtype=float)

        fit = fit_decay_models(d_arr, p_arr)

        results[model] = {
            "distances": distances,
            "mean_persistence": [round(x, 6) for x in mean_persistence],
            "std_persistence": [round(x, 6) for x in std_persistence],
            "n_per_step": n_per_step,
            "fit": fit,
        }

        best = fit.get("best_model", "?")
        delta = fit.get("delta_aicc", {})
        print(f"\n{model}:")
        print(f"  Steps from injection: {distances}")
        print(f"  Mean persistence:     {[f'{x:.4f}' for x in mean_persistence]}")
        print(f"  N per step:           {n_per_step}")
        print(f"  Best decay model:     {best}")
        print(f"  ΔAICc:                {delta}")

    # Also fit aggregate across all models
    all_distances = defaultdict(list)
    for model_data in by_model.values():
        for d, scores in model_data.items():
            all_distances[d].extend(scores)
    distances = sorted(all_distances.keys())
    mean_p = [np.mean(all_distances[d]) for d in distances]
    d_arr = np.array(distances, dtype=float)
    p_arr = np.array(mean_p, dtype=float)
    agg_fit = fit_decay_models(d_arr, p_arr)
    results["aggregate"] = {
        "distances": distances,
        "mean_persistence": [round(x, 6) for x in mean_p],
        "fit": agg_fit,
    }
    print(f"\nAggregate (all models):")
    print(f"  Best: {agg_fit.get('best_model', '?')}, ΔAICc: {agg_fit.get('delta_aicc', {})}")

    out_path = "results/stats/decay_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    return results


if __name__ == "__main__":
    run_decay_analysis()
