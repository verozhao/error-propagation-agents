"""Pre-registered pilot gate checks. Pass all five before launching main sweep."""
import json
import glob
import numpy as np
from collections import defaultdict
from scipy import stats

# Find the pilot file (most recent 15-trial Llama JSONL)
candidates = sorted(glob.glob("results/ragtruth_weighted_error/*15trials*.jsonl"))
assert candidates, "No 15-trial pilot JSONL found"
path = candidates[-1]
print(f"Inspecting: {path}\n")

records = [json.loads(l) for l in open(path) if l.strip()]
errors = [r for r in records if "error" in r]
ok = [r for r in records if "error" not in r and "evaluation" in r]
baselines = [r for r in ok if r.get("is_baseline")]
injected = [r for r in ok if not r.get("is_baseline")]

print(f"Total: {len(records)}, errors: {len(errors)}, ok: {len(ok)}")
print(f"  Baselines: {len(baselines)}, Injected: {len(injected)}\n")

gates = []

# Gate 1: Run completion (no widespread errors)
err_rate = len(errors) / len(records)
g1 = err_rate < 0.05
gates.append(("Run completion (error rate < 5%)", g1, f"{err_rate*100:.1f}% errors"))

# Gate 2: Baseline failure rate < 0.15
def is_failure(r):
    q = r.get("evaluation", {}).get("quality_score")
    return q is not None and q <= 4
bl_fr = sum(is_failure(r) for r in baselines) / max(len(baselines), 1)
g2 = bl_fr < 0.15
gates.append(("Baseline FR < 0.15", g2, f"baseline_fr={bl_fr:.3f}"))

# Gate 3: At least one (etype, step) cell shows FR > baseline with effect > 0.10
cell_fr = defaultdict(list)
cell_baseline_fr = defaultdict(list)
for r in injected:
    et = r.get("injection_meta", {}).get("error_type", "?")
    step = r.get("error_step")
    cell_fr[(et, step)].append(is_failure(r))
for r in baselines:
    cell_baseline_fr["all"].append(is_failure(r))

bl_mean = np.mean(cell_baseline_fr["all"]) if cell_baseline_fr["all"] else 0
significant_cells = []
for cell, vals in cell_fr.items():
    if len(vals) < 5:
        continue
    cell_mean = np.mean(vals)
    delta = cell_mean - bl_mean
    if delta > 0.10:
        # quick z-test against baseline
        n1, n2 = len(vals), len(cell_baseline_fr["all"])
        if n2 == 0: continue
        p1, p2 = cell_mean, bl_mean
        p_pooled = (sum(vals) + sum(cell_baseline_fr["all"])) / (n1 + n2)
        se = np.sqrt(p_pooled * (1-p_pooled) * (1/n1 + 1/n2)) if p_pooled > 0 else 1
        z = (p1 - p2) / se if se > 0 else 0
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        if p < 0.05:
            significant_cells.append((cell, delta, p))

g3 = len(significant_cells) >= 1
detail = f"{len(significant_cells)} cells with FR>baseline+0.10, p<0.05"
if significant_cells:
    detail += f"; e.g. {significant_cells[0]}"
gates.append(("Signal exists (≥1 cell FR > baseline + 0.10, p<0.05)", g3, detail))

# Gate 4: Persistence curves present for ≥75% of injected
n_curves = sum(1 for r in injected if r.get("persistence_curve"))
pct = n_curves / max(len(injected), 1)
g4 = pct >= 0.75
gates.append(("Persistence curves present for ≥75% injected", g4, f"{pct*100:.1f}%"))

# Gate 5: All FAVA error types sampled (under ragtruth_weighted)
etypes_seen = set()
for r in injected:
    et = r.get("injection_meta", {}).get("error_type")
    if et:
        etypes_seen.add(et)
expected = {"entity", "invented", "unverifiable", "contradictory"}
missing = expected - etypes_seen
g5 = len(missing) == 0
gates.append(("All 4 FAVA types sampled", g5, f"missing: {missing}" if missing else "all 4 present"))

print("=" * 70)
print("PILOT GATE CHECK")
print("=" * 70)
for name, passed, detail in gates:
    mark = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {mark}  {name}")
    print(f"          {detail}")

n_pass = sum(1 for _, p, _ in gates if p)
print()
print(f"  {n_pass}/{len(gates)} gates passed")
if n_pass == len(gates):
    print("  → Proceed to Llama main sweep (165q × 15t × 5c, ~$33.41)")
else:
    print("  → DO NOT launch main sweep. Investigate failures above.")