import hashlib
import json
import os
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from config import WORKFLOW_STEPS, NUM_TRIALS, OUTPUT_DIR, INJECT_AT_VERIFY
from models import call_model
from workflow import run_workflow, TASK_TEMPLATES
from error_injection import ERROR_TYPES
from evaluation import evaluate_workflow_output
from factual_accuracy import claim_survival_score, load_ground_truth


def _derive_seed(model_name, task_query, error_step, trial_idx):
    """Deterministic seed from (model, query, trial_idx).

    P0-15 FIX: does NOT include error_step or severity. This means:
      - baseline (error_step=None)  at (model, query, trial=t)
      - injected (error_step=k)     at (model, query, trial=t)
    use the SAME per-call seed sequence through the pre-injection steps.
    Outputs will be byte-identical up to step k-1; from step k onward
    the injected run diverges only because the injector modified the
    text. This is what makes paired statistical tests (Wilcoxon
    signed-rank) legitimate: baseline(t) and injected(t) are matched on
    all nuisance variation through the first k-1 steps.

    Severity is excluded for the same reason — sev1/sev2/sev3 at the
    same (model, query, step, trial) share pre-injection state, so the
    severity sweep controls for everything but the dose.

    error_step/severity are kept in the function signature for legacy
    callers but ignored.
    """
    del error_step  # no longer part of the key
    key = f"{model_name}|{task_query}|{trial_idx}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def run_single_experiment(
    model_name: str,
    task: dict,
    error_step: int | None,
    error_type: str,
    severity: int = 1,
    judge_models: list[str] | None = None,
    save_traces: bool = True,
    ground_truth: dict | None = None,
    pos_target: str | None = None,
    tfidf_target: str | None = None,
    trial_idx: int = 0,
    use_llm_judge: bool = False,
    compound_steps: list[int] | None = None,
    max_retries: int = 1,
) -> dict:
    """Run one pipeline trial.

    `save_traces=True` (the new default) records the full input/output text
    of every step plus the injected delta. This is required for the
    factual-accuracy evaluator, the sanity-check viewer, and the step-level
    survival analysis. Set False only if you need byte-for-byte
    backward-compat output for an old analysis script.
    """
    seed = _derive_seed(model_name, task["query"], error_step, trial_idx)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # Deterministic model_fn: different seed per call within a trial
    call_counter = {"i": 0}
    def model_fn(prompt):
        call_counter["i"] += 1
        return call_model(model_name, prompt, seed=seed + call_counter["i"])

    actual_error_step = compound_steps if compound_steps else error_step
    has_injection = actual_error_step is not None
    error_fn = ERROR_TYPES.get(error_type) if has_injection else None

    error_kwargs = {"severity": severity, "return_delta": True, "rng": rng} if error_fn else {}
    if error_fn and pos_target:
        error_kwargs["pos_target"] = pos_target
    if error_fn and tfidf_target:
        error_kwargs["tfidf_target"] = tfidf_target
    # Pass query + ground_truth for answer-targeted injection (Phase 7)
    if error_fn:
        error_kwargs["query"] = task["query"]
        error_kwargs["ground_truth"] = ground_truth

    results = run_workflow(
        query=task["query"],
        model_fn=model_fn,
        error_injection_fn=error_fn,
        error_step=actual_error_step,
        error_kwargs=error_kwargs,
        max_retries=max_retries,
    )

    injected_content = None
    injection_meta = None
    injected_contents = []
    for r in results:
        if r.error_injected and r.injected_content:
            injected_contents.append(r.injected_content)
            if injected_content is None:
                injected_content = r.injected_content
                injection_meta = r.injection_meta
    if len(injected_contents) > 1:
        injected_content = " | ".join(injected_contents)

    # Issue α: distinguish "injection was attempted and produced a delta"
    # from "injection was attempted but the injector returned no-op" (e.g.
    # omission at severity 3 when the text is a single sentence triggers
    # the n <= 2 early-return in inject_omission_error). Baseline records
    # have injection_valid=None because no injection was attempted.
    injection_attempted = actual_error_step is not None
    if not injection_attempted:
        injection_valid = None  # baseline
    else:
        # attempted. Consider it valid iff the injector produced a delta.
        injection_valid = bool(injected_content)

    evaluation = evaluate_workflow_output(
        results=results,
        original_query=task["query"],
        expected_keywords=task["expected_keywords"],
        evaluator_model=(judge_models[0] if judge_models else "gpt-4o-mini"),
        injected_error=injected_content,
        ground_truth=ground_truth,
        judge_models=judge_models,
        use_llm_judge=use_llm_judge,
    )

    # per-step survival of the injected content (Phase 2.2)
    error_found_in_step = {}
    if injected_content:
        for r in results:
            score, propagated = claim_survival_score(injected_content, r.output_text)
            error_found_in_step[r.step_name] = {
                "propagated": propagated,
                "survival_score": round(score, 4),
            }
    else:
        error_found_in_step = {
            r.step_name: {"propagated": False, "survival_score": 0.0}
            for r in results
        }

    # Baseline records have NO injection at all (neither single-step nor
    # compound). Downstream analyses must distinguish these three cases:
    #   baseline:  error_step=None, compound_steps=None, is_baseline=True
    #   single:    error_step=<int>, compound_steps=None, is_baseline=False
    #   compound:  error_step=<list>, compound_steps=<list>, is_baseline=False
    # Writing the list into error_step for compound runs fixes P0-1: old
    # analysis code that only looks at error_step no longer mis-classifies
    # compound trials as baselines.
    is_baseline = (error_step is None and compound_steps is None)
    record = {
        "model": model_name,
        "task_query": task["query"],
        "error_step": compound_steps if compound_steps else error_step,
        "error_type": error_type,
        "severity": severity,
        "compound_steps": compound_steps,
        "is_baseline": is_baseline,
        "injection_valid": injection_valid,  # Issue α: True / False / None (baseline)
        "pos_target": pos_target,
        "tfidf_target": tfidf_target,
        "evaluation": evaluation,
        "injected_content": injected_content,
        "injection_meta": injection_meta,
        "error_found_in_step": error_found_in_step,
        "seed": seed,
        "max_retries": max_retries,
    }

    if save_traces:
        record["step_outputs"] = [
            {
                "step": r.step_name,
                "input_text": r.input_text,
                "output_text": r.output_text,
                "error_injected": r.error_injected,
                "pre_injection_output": r.pre_injection_output,
            }
            for r in results
        ]
    else:
        record["step_outputs"] = [
            {"step": r.step_name, "error_injected": r.error_injected} for r in results
        ]

    return record


def run_full_experiment(
    models: list[str],
    num_trials: int = NUM_TRIALS,
    error_type: str = "semantic",
    severity: int = 1,
    judge_models: list[str] | None = None,
    save_traces: bool = True,
    output_subdir: str | None = None,
    pos_target: str | None = None,
    tfidf_target: str | None = None,
    diagnostic_query: str | None = None,
    skip_baseline: bool = False,
    use_llm_judge: bool = False,
    compound_pairs: list[tuple[int, ...]] | None = None,
    max_retries: int = 1,
    max_queries: int | None = None,
):
    import threading

    subdir = output_subdir or f"{error_type}_error"
    output_dir = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(output_dir, exist_ok=True)

    ground_truth = load_ground_truth()

    all_results = []
    max_inject_step = len(WORKFLOW_STEPS) if INJECT_AT_VERIFY else len(WORKFLOW_STEPS) - 1

    tasks = TASK_TEMPLATES
    if diagnostic_query:
        tasks = [t for t in TASK_TEMPLATES if t["query"] == diagnostic_query]
        if not tasks:
            raise ValueError(f"No task matching diagnostic query: {diagnostic_query}")
    else:
        tasks = [t for t in tasks if not t.get("_placeholder")]

    # Budget control: limit number of queries
    if max_queries and len(tasks) > max_queries:
        tasks = tasks[:max_queries]
        print(f"Budget control: using {len(tasks)} queries (--queries {max_queries})")

    if compound_pairs:
        error_steps = [list(p) for p in compound_pairs]
    else:
        error_steps = list(range(max_inject_step))
        if not skip_baseline:
            error_steps = [None] + error_steps

    # Build stable JSONL filename for resume
    model_tag = "_".join(sorted(models))
    parts = [error_type, f"sev{severity}", model_tag, f"{num_trials}trials"]
    if pos_target:
        parts.append(f"pos_{pos_target}")
    if tfidf_target:
        parts.append(f"tfidf_{tfidf_target}")
    if compound_pairs:
        parts.append("compound")
    stable_name = "_".join(parts)
    jsonl_path = os.path.join(output_dir, f"{stable_name}.jsonl")

    # Load completed trials for resume
    completed = set()
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    step_key = r.get("compound_steps") or r["error_step"]
                    if step_key is None:
                        step_key = "None"
                    completed.add((r["model"], r["task_query"], str(step_key), r["trial"]))
                except Exception:
                    continue
        if completed:
            print(f"Resume: found {len(completed)} completed trials in {jsonl_path}")

    # Build list of jobs: each job is (model, task, error_step, trial, compound_steps)
    jobs = []
    for model_name in models:
        for task in tasks:
            for error_step in error_steps:
                for trial in range(num_trials):
                    if compound_pairs:
                        step_key = str(error_step)
                        compound = error_step
                        single_step = None
                    else:
                        step_key = str(error_step) if error_step is not None else "None"
                        compound = None
                        single_step = error_step
                    if (model_name, task["query"], step_key, trial) not in completed:
                        jobs.append((model_name, task, single_step, trial, compound))

    total_runs = len(jobs)
    if total_runs == 0:
        print(f"All trials already completed in {jsonl_path}")
        return jsonl_path

    # max_workers = min(10, len(jobs))
    max_workers = 10
    write_lock = threading.Lock()

    def _run_one(job):
        model_name, task, error_step, trial, compound = job
        try:
            result = run_single_experiment(
                model_name,
                task,
                error_step,
                error_type,
                severity=severity,
                judge_models=judge_models,
                save_traces=save_traces,
                ground_truth=ground_truth,
                pos_target=pos_target,
                tfidf_target=tfidf_target,
                trial_idx=trial,
                use_llm_judge=use_llm_judge,
                compound_steps=compound,
                max_retries=max_retries,
            )
            result["trial"] = trial
            return result
        except Exception as e:
            is_baseline = (error_step is None and compound is None)
            return {
                "model": model_name,
                "task_query": task["query"],
                "error_step": compound if compound else error_step,
                "error_type": error_type,
                "severity": severity,
                "compound_steps": compound,
                "is_baseline": is_baseline,
                "trial": trial,
                "error": str(e),
            }

    with tqdm(total=total_runs, desc="Running experiments") as pbar, \
         open(jsonl_path, "a") as jsonl_fh, \
         ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_run_one, job): job for job in jobs}
        for future in as_completed(futures):
            rec = future.result()
            with write_lock:
                jsonl_fh.write(json.dumps(rec) + "\n")
                jsonl_fh.flush()
            all_results.append(rec)
            pbar.update(1)

    # Consolidated JSON for back-compat tooling
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_records = []
    with open(jsonl_path) as fh:
        for line in fh:
            try:
                all_records.append(json.loads(line))
            except Exception:
                pass
    consolidated_path = os.path.join(output_dir, f"{stable_name}_{timestamp}.json")
    with open(consolidated_path, "w") as f:
        json.dump(all_records, f, indent=2)

    return consolidated_path
