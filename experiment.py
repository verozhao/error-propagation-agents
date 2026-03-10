import json
import os
from datetime import datetime
from tqdm import tqdm
from config import WORKFLOW_STEPS, NUM_TRIALS, OUTPUT_DIR
from models import call_model
from workflow import run_workflow, TASK_TEMPLATES
from error_injection import ERROR_TYPES
from evaluation import evaluate_workflow_output


def run_single_experiment(model_name: str, task: dict, error_step: int | None, error_type: str) -> dict:
    model_fn = lambda prompt: call_model(model_name, prompt)
    error_fn = ERROR_TYPES.get(error_type) if error_step is not None else None
    
    results = run_workflow(
        query=task["query"],
        model_fn=model_fn,
        error_injection_fn=error_fn,
        error_step=error_step
    )
    
    evaluation = evaluate_workflow_output(
        results=results,
        original_query=task["query"],
        expected_keywords=task["expected_keywords"]
    )
    
    return {
        "model": model_name,
        "task_query": task["query"],
        "error_step": error_step,
        "error_type": error_type,
        "evaluation": evaluation,
        "step_outputs": [{"step": r.step_name, "error_injected": r.error_injected} for r in results],
    }


def run_full_experiment(models: list[str], num_trials: int = NUM_TRIALS, error_type: str = "semantic"):
    output_dir = os.path.join(OUTPUT_DIR, f"{error_type}_error")
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    total_runs = len(models) * len(TASK_TEMPLATES) * (len(WORKFLOW_STEPS) + 1) * num_trials
    
    with tqdm(total=total_runs, desc="Running experiments") as pbar:
        for model_name in models:
            for task in TASK_TEMPLATES:
                for error_step in [None] + list(range(len(WORKFLOW_STEPS))):
                    for trial in range(num_trials):
                        try:
                            result = run_single_experiment(model_name, task, error_step, error_type)
                            result["trial"] = trial
                            all_results.append(result)
                        except Exception as e:
                            all_results.append({
                                "model": model_name,
                                "task_query": task["query"],
                                "error_step": error_step,
                                "error_type": error_type,
                                "trial": trial,
                                "error": str(e),
                            })
                        pbar.update(1)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"experiment_{timestamp}.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    return output_file