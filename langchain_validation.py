"""LangChain validation experiment.

Runs the same error injection through LangChain's chain architecture
to validate that the three propagation modes (persistence / reconstruction /
attenuation) are not artifacts of our custom 5-step pipeline.

Architecture:
  We build a 3-step LangChain chain that mirrors the core of our pipeline:
    1. Retrieval  (≈ our search+filter)
    2. Summarize  (≈ our summarize)
    3. Answer     (≈ our compose)

  Error injection happens between steps, using the same inject_* functions
  from error_injection.py. Evaluation uses the same factual_accuracy module.

Usage:
    # Pilot (3 queries × 3 error types × 1 severity × 5 trials)
    python langchain_validation.py --trials 5 --severity 3

    # Full validation
    python langchain_validation.py --trials 20 --severity 3

Requirements:
    pip install langchain langchain-openai langchain-community

Cost estimate:
    ~$0.003/trial (same as main pipeline). 5 trials × 3 queries × 3 etypes
    × 2 steps = 90 API calls ≈ $0.10
"""

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime

from config import OUTPUT_DIR
from error_injection import ERROR_TYPES
from evaluation import evaluate_workflow_output
from factual_accuracy import load_ground_truth, claim_survival_score
from models import call_model
from workflow import StepResult, TASK_TEMPLATES


# ---------------------------------------------------------------------------
# LangChain-style 3-step chain (no LangChain import required — we simulate
# the chain structure using the same call_model backend so the comparison
# is apples-to-apples on the LLM, differing only in prompt structure).
#
# Why not import langchain directly?
#   1. Avoids a heavy dependency for users who just want to reproduce.
#   2. The scientific question is "does prompt structure matter", not
#      "does the langchain Python package matter". Using the same
#      call_model backend isolates the variable we care about.
#
# The prompts below are modeled on LangChain's default RetrievalQA and
# StuffDocumentsChain templates.
# ---------------------------------------------------------------------------

LANGCHAIN_STEPS = ["retrieval", "summarize", "answer"]


def _langchain_retrieval(query: str, model_fn) -> str:
    """Simulates LangChain's retriever + document loader.
    Uses a single LLM call styled as a retrieval step."""
    prompt = (
        f"You are a document retrieval system. Given the query below, "
        f"return 4 relevant document excerpts. Each excerpt should be "
        f"2-3 sentences of factual information.\n\n"
        f"Query: {query}\n\n"
        f"Format: Return exactly 4 numbered excerpts."
    )
    return model_fn(prompt)


def _langchain_summarize(documents: str, query: str, model_fn) -> str:
    """Simulates LangChain's StuffDocumentsChain — combines and summarizes
    retrieved documents into a single context block."""
    prompt = (
        f"Given the following documents, combine and summarize the key "
        f"information relevant to the query.\n\n"
        f"Query: {query}\n\n"
        f"Documents:\n{documents}\n\n"
        f"Provide a concise summary of the most important facts."
    )
    return model_fn(prompt)


def _langchain_answer(context: str, query: str, model_fn) -> str:
    """Simulates LangChain's QA chain — answers the query using the
    summarized context. This mirrors the 'stuff' chain type."""
    prompt = (
        f"Use the following context to answer the question. If the context "
        f"doesn't contain enough information, say so.\n\n"
        f"Context: {context}\n\n"
        f"Question: {query}\n\n"
        f"Helpful answer:"
    )
    return model_fn(prompt)


LANGCHAIN_STEP_FUNCTIONS = {
    "retrieval": _langchain_retrieval,
    "summarize": _langchain_summarize,
    "answer": _langchain_answer,
}


def run_langchain_trial(
    model_name: str,
    task: dict,
    error_step: int | None,
    error_type: str,
    severity: int,
    trial_idx: int,
    ground_truth: dict,
) -> dict:
    """Run one LangChain-style trial with optional error injection."""
    key = f"lc|{model_name}|{task['query']}|{trial_idx}"
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    call_counter = {"i": 0}
    def model_fn(prompt):
        call_counter["i"] += 1
        return call_model(model_name, prompt, seed=seed + call_counter["i"])

    error_fn = ERROR_TYPES.get(error_type) if error_step is not None else None
    error_kwargs = {"severity": severity, "return_delta": True, "rng": rng} if error_fn else {}

    results = []
    current_input = task["query"]

    for i, step_name in enumerate(LANGCHAIN_STEPS):
        # Call the step
        if step_name == "retrieval":
            output = LANGCHAIN_STEP_FUNCTIONS[step_name](current_input, model_fn)
        else:
            output = LANGCHAIN_STEP_FUNCTIONS[step_name](current_input, task["query"], model_fn)

        # Inject error if this is the target step
        error_injected = False
        injected_content = None
        pre_injection_output = None
        injection_meta = None

        if error_fn and i == error_step:
            pre_injection_output = output
            try:
                result = error_fn(output, step_name, **error_kwargs)
            except TypeError:
                result = error_fn(output, step_name)
            if isinstance(result, tuple):
                if len(result) == 3:
                    output, injected_content, injection_meta = result
                else:
                    output, injected_content = result
            else:
                output = result
            error_injected = True

        results.append(StepResult(
            step_name=step_name,
            input_text=current_input,
            output_text=output,
            error_injected=error_injected,
            injected_content=injected_content,
            pre_injection_output=pre_injection_output,
            injection_meta=injection_meta,
        ))
        current_input = output

    # Evaluate using the LAST step as both content and verify
    # (LangChain chain has no separate verify step)
    content_text = results[-1].output_text

    # Collect injected content
    all_injected = " | ".join(
        r.injected_content for r in results
        if r.error_injected and r.injected_content
    ) or None

    # Factual accuracy on the final answer
    from factual_accuracy import evaluate_factual_accuracy
    factual = evaluate_factual_accuracy(
        pipeline_output=content_text,
        injected_error=all_injected,
        query=task["query"],
        ground_truth=ground_truth,
    )

    # Assertion preservation
    gt_entry = ground_truth.get(task["query"])
    preserved = 1.0
    if gt_entry:
        out_lower = content_text.lower()
        hits = 0
        total = len(gt_entry.get("assertions", []))
        for a in gt_entry.get("assertions", []):
            kws = [k.lower() for k in a.get("keywords", [])]
            if kws and all(kw in out_lower for kw in kws):
                hits += 1
                continue
            for alias in a.get("aliases", []):
                if alias.lower() in out_lower:
                    hits += 1
                    break
        if total > 0:
            preserved = hits / total

    # Claim survival per step
    error_found_in_step = {}
    if all_injected:
        for r in results:
            score, propagated = claim_survival_score(all_injected, r.output_text)
            error_found_in_step[r.step_name] = {
                "propagated": propagated,
                "survival_score": round(score, 4),
            }

    is_baseline = error_step is None
    injection_valid = None if is_baseline else bool(all_injected)

    return {
        "framework": "langchain",
        "model": model_name,
        "task_query": task["query"],
        "error_step": error_step,
        "error_type": error_type,
        "severity": severity,
        "is_baseline": is_baseline,
        "injection_valid": injection_valid,
        "trial": trial_idx,
        "seed": seed,
        "injected_content": all_injected,
        "injection_meta": injection_meta,
        "error_found_in_step": error_found_in_step,
        "evaluation": {
            "preserved": round(preserved, 4),
            "survival_score": round(factual.error_survival_score, 4),
            "assertions_present": factual.assertions_present,
            "assertions_total": factual.assertions_total,
            "contradictions_present": factual.contradictions_present,
            "factual_accuracy_score": round(factual.factual_accuracy_score, 4),
        },
        "step_outputs": [
            {
                "step": r.step_name,
                "output_text": r.output_text,
                "error_injected": r.error_injected,
                "pre_injection_output": r.pre_injection_output,
            }
            for r in results
        ],
    }


def main():
    parser = argparse.ArgumentParser(description="LangChain validation experiment")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--severity", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"])
    parser.add_argument("--error-types", nargs="+", default=["factual", "omission", "semantic"])
    parser.add_argument("--queries", nargs="+", default=None,
                       help="Specific queries to test (default: first 3)")
    args = parser.parse_args()

    ground_truth = load_ground_truth()

    # Select queries
    if args.queries:
        tasks = [t for t in TASK_TEMPLATES if t["query"] in args.queries]
    else:
        tasks = [t for t in TASK_TEMPLATES if not t.get("_placeholder")][:3]

    output_dir = os.path.join(OUTPUT_DIR, "langchain_validation")
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    n_steps = len(LANGCHAIN_STEPS)  # 3 steps: retrieval, summarize, answer

    total = (len(args.models) * len(tasks) * len(args.error_types)
             * (1 + n_steps) * args.trials)  # +1 for baseline
    print(f"LangChain validation: {total} trials across "
          f"{len(args.models)} models, {len(tasks)} queries, "
          f"{len(args.error_types)} error types")
    print(f"Estimated cost: ~${total * 3 * 0.00003:.2f}")

    done = 0
    for model_name in args.models:
        for task in tasks:
            for error_type in args.error_types:
                # Baseline
                for trial in range(args.trials):
                    rec = run_langchain_trial(
                        model_name, task, None, error_type,
                        args.severity, trial, ground_truth)
                    all_results.append(rec)
                    done += 1
                    print(f"\r  [{done}/{total}] ", end="", flush=True)

                # Injection at each step
                for step_idx in range(n_steps):
                    for trial in range(args.trials):
                        rec = run_langchain_trial(
                            model_name, task, step_idx, error_type,
                            args.severity, trial, ground_truth)
                        all_results.append(rec)
                        done += 1
                        print(f"\r  [{done}/{total}] ", end="", flush=True)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outpath = os.path.join(output_dir, f"langchain_sev{args.severity}_{timestamp}.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)

    # Summary
    print(f"\n\nSaved {len(all_results)} records to {outpath}")
    print(f"\nSummary by error_type and step:")

    from collections import defaultdict
    by_cell = defaultdict(list)
    for r in all_results:
        key = (r["error_type"], "BASELINE" if r["is_baseline"] else f"step={r['error_step']}")
        by_cell[key].append(r)

    print(f"{'error_type':>12} {'step':>10} {'N':>4} {'preserved':>10} {'survival':>10} {'contrad':>8}")
    print("-" * 60)
    for (etype, step), recs in sorted(by_cell.items()):
        n = len(recs)
        pres = sum(r["evaluation"]["preserved"] for r in recs) / n
        surv = sum(r["evaluation"]["survival_score"] for r in recs) / n
        contr = sum(r["evaluation"]["contradictions_present"] for r in recs) / n
        print(f"{etype:>12} {step:>10} {n:>4} {pres:>10.3f} {surv:>10.3f} {contr:>8.2f}")


if __name__ == "__main__":
    main()
