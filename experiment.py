import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from config import WORKFLOW_STEPS, NUM_TRIALS, OUTPUT_DIR
from models import call_model
from workflow import run_workflow, TASK_TEMPLATES
from error_injection import ERROR_TYPES
from evaluation import evaluate_workflow_output
from factual_accuracy import claim_survival_score, load_ground_truth


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
) -> dict:
    """Run one pipeline trial.

    `save_traces=True` (the new default) records the full input/output text
    of every step plus the injected delta. This is required for the
    factual-accuracy evaluator, the sanity-check viewer, and the step-level
    survival analysis. Set False only if you need byte-for-byte
    backward-compat output for an old analysis script.
    """
    model_fn = lambda prompt: call_model(model_name, prompt)
    error_fn = ERROR_TYPES.get(error_type) if error_step is not None else None

    error_kwargs = {"severity": severity, "return_delta": True} if error_fn else {}
    if error_fn and pos_target:
        error_kwargs["pos_target"] = pos_target
    if error_fn and tfidf_target:
        error_kwargs["tfidf_target"] = tfidf_target

    results = run_workflow(
        query=task["query"],
        model_fn=model_fn,
        error_injection_fn=error_fn,
        error_step=error_step,
        error_kwargs=error_kwargs,
    )

    injected_content = None
    for r in results:
        if r.error_injected and r.injected_content:
            injected_content = r.injected_content
            break

    evaluation = evaluate_workflow_output(
        results=results,
        original_query=task["query"],
        expected_keywords=task["expected_keywords"],
        evaluator_model=(judge_models[0] if judge_models else "gpt-4o-mini"),
        injected_error=injected_content,
        ground_truth=ground_truth,
        judge_models=judge_models,
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

    record = {
        "model": model_name,
        "task_query": task["query"],
        "error_step": error_step,
        "error_type": error_type,
        "severity": severity,
        "pos_target": pos_target,
        "tfidf_target": tfidf_target,
        "evaluation": evaluation,
        "injected_content": injected_content,
        "error_found_in_step": error_found_in_step,
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
):
    subdir = output_subdir or f"{error_type}_error"
    output_dir = os.path.join(OUTPUT_DIR, subdir)
    os.makedirs(output_dir, exist_ok=True)

    ground_truth = load_ground_truth()

    all_results = []
    total_runs = len(models) * len(TASK_TEMPLATES) * (len(WORKFLOW_STEPS) + 1) * num_trials

    # Build list of jobs
    jobs = []
    for model_name in models:
        for task in TASK_TEMPLATES:
            for error_step in [None] + list(range(len(WORKFLOW_STEPS))):
                for trial in range(num_trials):
                    jobs.append((model_name, task, error_step, trial))

    max_workers = min(10, len(jobs))

    def _run_one(job):
        model_name, task, error_step, trial = job
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
            )
            result["trial"] = trial
            return result
        except Exception as e:
            return {
                "model": model_name,
                "task_query": task["query"],
                "error_step": error_step,
                "error_type": error_type,
                "severity": severity,
                "trial": trial,
                "error": str(e),
            }

    with tqdm(total=total_runs, desc="Running experiments") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_run_one, job): job for job in jobs}
            for future in as_completed(futures):
                all_results.append(future.result())
                pbar.update(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = "_".join(sorted(models))
    parts = [error_type, f"sev{severity}", model_tag, f"{num_trials}trials"]
    if pos_target:
        parts.append(f"pos_{pos_target}")
    if tfidf_target:
        parts.append(f"tfidf_{tfidf_target}")
    parts.append(timestamp)
    output_file = os.path.join(output_dir, f"{'_'.join(parts)}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    return output_file
