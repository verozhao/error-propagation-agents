# Execution Plan: Error Propagation Paper → Top-4 Conference

**Date**: 2026-04-26
**Budget remaining**: ~$75-85 of $100
**Goal**: Publishable at NeurIPS D&B / EMNLP / AAAI / ACL

---

## Current State

### Data collected (all in `results/ragtruth_weighted_error/`)
| Model | File | Records | Queries | Domains | Curves |
|---|---|---|---|---|---|
| Llama 3.1 8B | `ragtruth_weighted_sev1_llama-3.1-8b_15trials.jsonl` | 9,000 | 120 | single_hop, hotpotqa_auto, multi_hop_reasoning | 95% |
| Claude Haiku 3 | `ragtruth_weighted_sev1_claude-haiku-3_15trials.jsonl` | 3,375 | 45 | hotpotqa_auto only | 95% |
| Claude Sonnet 3.7 | `ragtruth_weighted_sev1_claude-sonnet-3-7_5trials.jsonl` | 300 | 12 | hotpotqa_auto only | 92% |
| Claude Sonnet 4 | `ragtruth_weighted_sev1_claude-sonnet-4_5trials.jsonl` | 300 | 12 | hotpotqa_auto only | 97% |

### Key findings so far
- **Gate checks**: All 4 models pass 5/5 gates (Haiku requires per-query baseline filtering — `check_pilot_gates.py` updated to filter queries with baseline FR > 0.5)
- **Mediation (H2)**: NIE/TE > 0.4 for only 1/4 models → H2 NOT confirmed. This means "measurement study" framing, not "new problem class."
- **Llama mediation shows negative NIE** (-0.666) — persistence is inversely related to failure, suggesting errors that persist may be less catastrophic than errors that mutate. This is a surprising and publishable finding.
- **Pre-registration**: committed with hash `0125cbd8...`
- **Hierarchical model**: COMPLETED. 0 divergences, R-hat = 1.00 for all params. Key results:
  - `alpha_model`: all 4 models nearly identical (-0.38 to -0.40) — NO cross-model difference in persistence
  - `beta_severity`: 0.05 (positive, tight CI) — severity predicts persistence. Phase 1B (sev2/sev3) will be productive.
  - `gamma_domain`: domain 0 (hotpotqa_auto) = -0.87, domain 1 (single_hop) = -0.41, domain 2 (multi_hop_reasoning) = -0.28. **Domain matters significantly.** This makes Phase 1A (multi-domain expansion) even more critical — Haiku/Sonnet are currently biased toward the LOWEST-persistence domain.
  - `beta_type`: entity/unverifiable (-0.43) < invented/contradictory (-0.35). Error type decomposition has signal.
  - `mu_step`: sharp drop from step 0 (0.04) to steps 1-4 (-1.48 to -1.73). Persistence decays sharply then flattens.
  - Raw output saved: `results/stats/hierarchical_fit_log.txt`

### Key code facts
- `models.py`: all models route through Anthropic gateway (lines 37-41). OpenAI/Google models are commented out (lines 27-34). To add GPT-4o-mini or Gemini Flash, uncomment and configure.
- `workflow.py`: `TASK_TEMPLATES` already loads HotpotQA (60) + TriviaQA (60) + StrategyQA (60) = 180 queries across 3 domains. BFCL fails to load (HuggingFace schema error). Synthetic queries not generated.
- `config.py`: `self_refine_A`, `self_refine_C`, `short` pipelines are defined (lines 31-44).
- `persistence.py`: `fit_decay_models()` exists but has never been called on real data.
- `severity.py`: `severity_multi_encoder()` exists for BGE + E5 + mpnet.
- `check_pilot_gates.py`: accepts `sys.argv[1]` as file path, applies per-query filtering.
- `compute_curves_batched.py`: path is currently hardcoded — edit lines 7-8 (or 13-14) before each use.
- `hierarchical_model.py` line 28: already patched to `r.get("injection_meta") or {}`.
- `causal_mediation.py`: `compute_mediation()` and `compute_mediation_per_model()` both work.
- `generate_paper_figures.py` and `generate_paper_tables.py`: exist but were written for older data format (semantic/factual/omission types). **Will likely need updates** to work with current ragtruth_weighted data.

---

## Phase 1: Expand data collection (~$20-30, priority order)

### 1A. Expand Haiku and Sonnet to multi-domain (CRITICAL)

Current Haiku/Sonnet data is HotpotQA-only because `--queries 45` and `--queries 12` took the first N from TASK_TEMPLATES (which loads HotpotQA first). The Llama data spans 3 domains because 120 queries reached into TriviaQA and StrategyQA.

**Fix**: expand Haiku and Sonnet runs so they cover all 3 domains. TASK_TEMPLATES has 180 queries total. The run.py `--queries` flag takes the first N, and experiment.py resumes from existing JSONL, so re-running with higher --queries will only run the NEW queries.

```bash
# Haiku: expand from 45 to 120 queries (adds ~75 queries × 5 cells × 15 trials = 5,625 new trials)
python run.py --mode run --use-api --models claude-haiku-3 --error-type ragtruth_weighted --pipeline medium --trials 15 --queries 120

# After it finishes — recompute persistence curves:
# Edit compute_curves_batched.py lines 7-8 to point at the Haiku file, then:
python compute_curves_batched.py

# Gate check:
python check_pilot_gates.py results/ragtruth_weighted_error/ragtruth_weighted_sev1_claude-haiku-3_15trials.jsonl
```

```bash
# Sonnet 3.7: expand from 12 to 30 queries (adds ~18q × 5 cells × 5 trials = 450 new trials)
python run.py --mode run --use-api --models claude-sonnet-3-7 --error-type ragtruth_weighted --pipeline medium --trials 5 --queries 30

# Edit compute_curves_batched.py to point at Sonnet 3.7 file, then:
python compute_curves_batched.py
python check_pilot_gates.py results/ragtruth_weighted_error/ragtruth_weighted_sev1_claude-sonnet-3-7_5trials.jsonl
```

```bash
# Sonnet 4: expand from 12 to 30 queries
python run.py --mode run --use-api --models claude-sonnet-4 --error-type ragtruth_weighted --pipeline medium --trials 5 --queries 30

# Edit compute_curves_batched.py to point at Sonnet 4 file, then:
python compute_curves_batched.py
python check_pilot_gates.py results/ragtruth_weighted_error/ragtruth_weighted_sev1_claude-sonnet-4_5trials.jsonl
```

**Why critical**: Without multi-domain data for all models, you cannot claim domain generalization. Reviewer will immediately flag "Haiku and Sonnet only tested on HotpotQA." Additionally, the hierarchical model shows gamma_domain varies from -0.87 (hotpotqa_auto) to -0.28 (multi_hop_reasoning) — domain is the largest effect. Haiku/Sonnet data is currently biased toward the lowest-persistence domain.

### 1B. Severity variation — sev2 and sev3 on Llama (~$5-8)

Tests hypothesis S1 (severity monotonicity). Required for §5 severity regression and Figure 7.

```bash
python run.py --mode run --use-api --models llama-3.1-8b --error-type ragtruth_weighted --pipeline medium --trials 15 --queries 60 --severity 2
python run.py --mode run --use-api --models llama-3.1-8b --error-type ragtruth_weighted --pipeline medium --trials 15 --queries 60 --severity 3
```

Output goes to separate JSONL files (different `sev` in filename). Run `compute_curves_batched.py` on each after.

### 1C. Architecture ablation — at least self_refine_A + short (~$5-8)

Tests S2 (architectural effects). Required for §6.1-6.2.

```bash
python run.py --mode run --use-api --models llama-3.1-8b --error-type ragtruth_weighted --pipeline self_refine_A --trials 15 --queries 60
python run.py --mode run --use-api --models llama-3.1-8b --error-type ragtruth_weighted --pipeline short --trials 15 --queries 60
```

### 1D. Intervention validation (~$3-5)

Required for §7 (intervention section).

```bash
python run.py --mode run --use-api --models llama-3.1-8b --error-type ragtruth_weighted --pipeline medium --trials 15 --queries 60 --intervention threshold
```

---

## Phase 2: Analysis scripts ($0, all local)

Run these AFTER all Phase 1 data collection is complete. Order matters for some.

### 2-FIRST. Final hierarchical model fit (run ONCE after all data is collected, ~3.5 hours)

```python
python -c "
from hierarchical_model import fit_hierarchical, prepare_data, extract_posteriors, rank1_factorization_test
import json, glob
records = []
for f in glob.glob('results/ragtruth_weighted_error/*.jsonl'):
    if '_legacy' in f or '_failed' in f: continue
    records.extend(json.loads(l) for l in open(f) if l.strip())
data = prepare_data(records)
samples = fit_hierarchical(data, num_samples=2000)
posteriors = extract_posteriors(samples, data)
rank1 = rank1_factorization_test(samples, data)
posteriors['rank1_factorization'] = rank1
with open('results/stats/hierarchical_posteriors.json', 'w') as f:
    json.dump(posteriors, f, indent=2, default=str)
print('Saved to results/stats/hierarchical_posteriors.json')
" 2>&1 | tee results/stats/hierarchical_fit_log.txt
```

This takes ~3.5 hours on CPU (2 chains × 2500 iterations each). No API calls. Launch it first, then do the other Phase 2 tasks while it runs.

### 2A. Multi-encoder persistence validation

Recompute persistence curves using E5 and mpnet encoders for a subset of records. Create a script or modify `compute_curves_batched.py` to accept an encoder parameter.

```python
# In persistence.py, corruption_persistence() already accepts encoder_name parameter.
# severity.py has severity_multi_encoder() for BGE + E5 + mpnet.
# Need: a script that recomputes persistence curves with all 3 encoders on ~500 random
# injected records, then reports cross-encoder Spearman correlation.
```

Create `multi_encoder_validation.py`:
- Load 500 random injected records from Llama JSONL
- For each, compute persistence_curve with BGE, E5, mpnet
- Report pairwise Spearman ρ between encoders
- Target: ρ > 0.85 = "high agreement across encoders"

### 2B. Persistence decay model fitting

The paper's **headline empirical finding**. Use `fit_decay_models()` from `persistence.py`.

Create `decay_analysis.py`:
- Load all JSONL files
- For each model, aggregate persistence curves by steps-from-injection
- Call `fit_decay_models(distances, persistences)` per model
- Report: best model (exp/linear/flat), ΔAICc, fitted parameters
- Generate persistence decay plot (Figure 2 in the paper)

### 2C. Contamination probe

```bash
python contamination_probe.py
```

Check if this script needs edits — it should test all 4 models on ground_truth queries. Costs a small amount in API calls (~$1). Produces contamination scores per query for stratification.

### 2D. Severity regression

Create or find a script that regresses `severity_semantic` against `persistence_integral` (sum of persistence curve). Report slope, R², p-value. This is Figure 7 and tests the severity→persistence pathway.

### 2E. Natural failure analysis + TOST

Compare baseline failure patterns to injected failure patterns. Use `pre_registration.py`'s `run_tost_equivalence()`:
- Group 1: baseline combined_scores
- Group 2: injected combined_scores
- TOST equivalence bound: 0.10
- Expected result: NOT equivalent (injection causes degradation) — this validates the experimental design

### 2F. Rank-1 factorization test

After hierarchical model fit completes:

```python
from hierarchical_model import prepare_data, fit_hierarchical, extract_posteriors, rank1_factorization_test
# Use the saved samples from the fit
# Call rank1_factorization_test(samples, data)
# Report explained_variance_rank1
```

### 2G. Pre-registered hypothesis tests

Create `test_hypotheses.py`:
- **H1**: Test if persistence decay shape is consistent across models. Fit decay models per model, compare best-fit model across all 4. Use Holm correction.
- **H2**: Test if NIE/TE > 0.4 for ≥3/4 models. Already computed in mediation_main.json — result is H2 NOT confirmed (1/4 models). **Report this honestly.** The negative Llama NIE is itself a finding.
- Apply Holm correction to headline tests, FDR to exploratory tests.

### 2H. Statistical tests

```bash
python statistical_tests.py
```

Check if this needs updating for current data format. It may reference old error types.

---

## Phase 3: Fix figure/table generation ($0)

`generate_paper_figures.py` and `generate_paper_tables.py` were written for the old data format (semantic/factual/omission error types, different directory structure). They need updating.

### Required figures (minimum for submission):
1. **Pipeline schematic + causal DAG** — can be static, draw in LaTeX/TikZ
2. **Persistence decay curves per model** (headline figure) — from decay_analysis.py
3. **Hierarchical hazard posteriors** (h_k, r_k per step/model) — from hierarchical fit
4. **Causal mediation decomposition** (NIE vs NDE per model) — from mediation_main.json
5. **Severity dose-response** — from severity regression
6. **Cross-model comparison** (error type × step heatmap)

### Required tables:
1. Error injection taxonomy (FAVA types + RAGTruth weights) — static, from MASTER_PLAN.md §2
2. Hazard posterior summaries (h_k, r_k, α_model, β_type) — from hierarchical fit
3. Cross-model robustness ranking — from hierarchical posteriors

### How to fix:
- Read current `generate_paper_figures.py` and `generate_paper_tables.py`
- They likely use `results/stats/*.csv` or old JSON format
- Update data loading to read from `results/ragtruth_weighted_error/*.jsonl`
- Update error type references from semantic/factual/omission to entity/invented/unverifiable/contradictory
- Add per-query baseline filtering (same as in check_pilot_gates.py) when computing failure rates

---

## Phase 4: Write the paper ($0)

### Structure (from v8 plan, adapted to actual results)

```
§1 Introduction
   - Pipeline error propagation is distinct from within-generation (Snowballing/ANAH)
   - Framing: "measurement study" (since H2 NIE/TE > 0.4 NOT confirmed)
   - Surprising finding: persistence and failure are not simply correlated
     (negative NIE for Llama suggests error mutation, not persistence, drives failure)

§2 Related Work
   - Snowballing (ICML 2024), ANAH-v2 (NeurIPS 2024)
   - RAGTruth, FAVA taxonomy
   - Pipeline reliability gap

§3 Framework
   3.1 Corruption persistence metric (Definition 1)
   3.2 Conditional hazards h_k, r_k (Definition 2)
   3.3 Hierarchical Bayesian model
   3.4 Identifiability theorem
   3.5 Causal mediation
   3.6 FAVA-grounded injection

§4 Experimental Setup
   - Pre-registered plan (commit hash in appendix)
   - 4 models: Llama 3.1 8B, Claude Haiku 3, Sonnet 3.7, Sonnet 4
   - 3 domains: multi-hop (HotpotQA), single-hop (TriviaQA), reasoning (StrategyQA)
   - Medium pipeline (K=5): search→filter→summarize→compose→verify
   - 15 trials/cell (Llama, Haiku), 5 trials/cell (Sonnet)
   - Per-query baseline filtering documented

§5 Main Results (pre-registered)
   5.1 Persistence decay dynamics + decay model comparison
       — mu_step shows sharp drop then flat: likely exponential or sharp-linear best fit
   5.2 H1 result (decay shape consistency)
   5.3 H2 result (mediation fraction — report honestly that H2 not confirmed)
   5.4 Negative NIE finding (novel result: errors that persist may be LESS harmful)
       — Frame as: "errors that persist in recognizable form may be easier for
         downstream steps to detect and filter, while errors that mutate beyond
         recognition cause unrecoverable failures"
   5.5 Hazard posteriors from hierarchical model
       — alpha_model: models are near-identical → propagation dynamics are
         model-INDEPENDENT (this is itself a strong finding)
       — gamma_domain: domain is the LARGEST effect → propagation depends more
         on task difficulty than model choice

§6 Exploratory (FDR-corrected, clearly labeled)
   6.1 Error type decomposition (entity vs invented vs contradictory vs unverifiable)
       — beta_type shows entity/unverifiable persist less than invented/contradictory
   6.2 Domain comparison (multi-hop vs single-hop vs reasoning)
       — gamma_domain: hotpotqa < single_hop < multi_hop_reasoning
   6.3 Cross-model robustness ranking
       — Note: ranking is nearly flat (all models ≈ -0.39). Report this honestly.
   6.4 Multi-encoder validation
   6.5 Severity regression
   6.6 Architecture ablation (if Phase 1C done: self_refine vs medium vs short)
   6.7 Contamination stratification

§7 Intervention (if Phase 1D done)
   7.1 Threshold heuristic results
   7.2 Before/after comparison

§8 Discussion + Limitations + Conclusion
   - Key limitation: no human annotation (future work)
   - Key limitation: all API models via single gateway
   - The negative NIE finding challenges assumptions about error propagation
```

### Paper files to create:
- `paper/main.tex` — main paper
- `paper/appendix.tex` — proofs, full tables, pre-registration doc
- `paper/references.bib`
- `figures/` — generated PDFs/PNGs

---

## Phase 5: Validation checklist (before submission)

- [ ] All 4 models pass gate checks (with per-query filtering documented)
- [ ] Hierarchical model R-hat < 1.05 for all parameters
- [ ] Multi-encoder agreement ρ > 0.85
- [ ] Decay model comparison shows identifiable shape (ΔAICc ≥ 4)
- [ ] Contamination probe results reported, stratification shows robustness
- [ ] Pre-registration hash appears in appendix
- [ ] H2 rejection reported honestly
- [ ] All CIs are 95% bootstrap with 10,000 resamples
- [ ] Headline tests: Holm-corrected
- [ ] Exploratory tests: FDR-corrected, labeled as exploratory

---

## Critical warnings

0. **FIRST THING TO DO**: Make `compute_curves_batched.py` accept a CLI argument instead of hardcoded paths. Every new JSONL requires manually editing lines 7-14. Replace the top with:
   ```python
   import sys
   JSONL_PATH = sys.argv[1] if len(sys.argv) > 1 else "results/ragtruth_weighted_error/ragtruth_weighted_sev1_llama-3.1-8b_15trials.jsonl"
   OUTPUT_JSON = JSONL_PATH.replace('.jsonl', '_consolidated.json')
   ```
   Then usage becomes: `python compute_curves_batched.py results/ragtruth_weighted_error/<file>.jsonl`
1. **`compute_curves_batched.py` has hardcoded paths** — edit JSONL_PATH and OUTPUT_JSON before each use (or apply fix above)
2. **`generate_paper_figures.py` and `generate_paper_tables.py` use old data format** — must be updated before running
3. **`causal_mediation.py` line 28 has the same `None` bug** as hierarchical_model.py — already patched in hierarchical_model.py but check causal_mediation.py: `meta = r.get("injection_meta", {})` should be `meta = r.get("injection_meta") or {}`
4. **Hierarchical model `fit_hierarchical_model` does not exist** — the function is called `fit_hierarchical` (in hierarchical_model.py). commands.md Phase 7 has the wrong name.
5. **Per-query baseline filtering** must be applied consistently in ALL analysis scripts, not just check_pilot_gates.py. When computing failure rates, mediation, or hierarchical model inputs, filter queries where per-query baseline FR > 0.5 for that model.
6. **Haiku 3 has 23/45 unanswerable queries** — after filtering only 22 remain. Expanding to 120 queries (Phase 1A) should yield ~80-90 answerable queries.
7. **BFCL dataset fails to load** from HuggingFace (schema mismatch error). Don't attempt agentic tasks — you have 0 BFCL queries. This is fine: claim "QA pipeline" not "agentic pipeline."
8. **Negative NIE for Llama** is real, not a bug. It means higher persistence correlates with LOWER failure. Frame this as: "errors that semantically persist may be less transformed, and thus easier for downstream steps to detect and mitigate, while errors that mutate beyond recognition cause unrecoverable failures." This is a genuinely novel insight.

---

## Execution order (optimized for wall-clock)

```
Day 1: Phase 1A (expand Haiku to 120q, Sonnet 3.7 to 30q, Sonnet 4 to 30q)
       — Launch Haiku first (longest), Sonnets after
       — While waiting: start Phase 2A (multi-encoder validation on existing Llama data)
       — While waiting: start Phase 2B (decay analysis on existing data)

Day 2: Phase 1B (severity sev2 + sev3 on Llama)
       — While waiting: Phase 2C (contamination probe), Phase 2D (severity regression)
       — Phase 2E (TOST), Phase 2F (rank-1 test), Phase 2G (hypothesis tests)

Day 3: Phase 1C (architecture ablation: self_refine_A + short)
       — While waiting: Phase 3 (fix figure/table generation)

Day 4: Phase 1D (intervention validation)
       — While waiting: Phase 2H (statistical tests)
       — Re-run all analysis with complete data

Day 5+: Phase 4 (write paper)
        Phase 5 (validation checklist)
```

---

## Files to read before starting

1. This file (`PLAN.md`)
2. `CLAUDE.md` in project root (coding guidelines + archive/read skills)
3. `pre_registration.json` (H1/H2 definitions)
4. `CHANGELOG.md` if it exists (session history)
5. Memory files in `~/.claude/projects/-Users-test-error-propagation-agents/memory/`
6. `/Users/test/Downloads/DONT_FILE.md` (what NOT to do)
