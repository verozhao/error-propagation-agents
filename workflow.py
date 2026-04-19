import os
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

# --- 1. Load HotpotQA Dataset for Main Track Scale ---
def load_hotpotqa_tasks(num_tasks=50):
    try:
        from datasets import load_dataset
        # Load validation split of HotpotQA distractor setting
        ds = load_dataset("hotpot_qa", 'distractor', split='validation')
        tasks = []
        for i in range(min(num_tasks, len(ds))):
            item = ds[i]
            tasks.append({
                "query": item["question"],
                "expected_keywords": [item["answer"]], # The true answer acts as keyword
                "domain": "multi_hop_reasoning",
                "_placeholder": False
            })
        return tasks
    except ImportError:
        print("WARNING: 'datasets' library not found. Run pip install datasets.")
        return []

# Replace hardcoded toy queries with robust 50-query benchmark
TASK_TEMPLATES = load_hotpotqa_tasks(50)

@dataclass
class StepResult:
    step_name: str
    input_text: str
    output_text: str
    error_injected: bool
    injected_content: Optional[str] = None
    pre_injection_output: Optional[str] = None
    injection_meta: Optional[dict] = None

# --- 2. Grounded Tool Usage (Real Web Search WITH CACHE) ---
SEARCH_CACHE_FILE = "search_cache.json"

def load_search_cache():
    if os.path.exists(SEARCH_CACHE_FILE):
        with open(SEARCH_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_search_cache(cache):
    with open(SEARCH_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

# Globally load the cache
_GLOBAL_SEARCH_CACHE = load_search_cache()

def step_search(query: str, model_fn: Callable) -> str:
    print("  [TRACE] Entering Search Function...", flush=True)
    # 1. If cache hit, return the strictly consistent real search results
    # (ensures absolute variable control across 39 experimental conditions)
    if query in _GLOBAL_SEARCH_CACHE:
        print(f"  [TRACE] Cache Hit for: {query[:20]}", flush=True)
        return _GLOBAL_SEARCH_CACHE[query]
    
    # 2. If cache miss, call the real API (with retries and anti-ban delay)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                print(f"  [TRACE] Calling DuckDuckGo API (Attempt {attempt+1})...", flush=True)
                results = list(ddgs.text(query, max_results=5))
            if results:
                print("  [TRACE] Search API returned results successfully.", flush=True)
                formatted = "\n".join([f"{i+1}. {r['title']}: {r['body']}" for i, r in enumerate(results)])
                final_text = f"Real web search results for '{query}':\n{formatted}"
                
                # Write to cache and save
                _GLOBAL_SEARCH_CACHE[query] = final_text
                save_search_cache(_GLOBAL_SEARCH_CACHE)
                time.sleep(1) # Polite delay to prevent DuckDuckGo rate limiting
                return final_text
        except Exception as e:
            print(f"  [TRACE] Search API wait... {e} | attempt {attempt+1}", flush=True)
            time.sleep(5 * (attempt + 1)) # Exponential backoff
            
    # 3. Fatal error: If real search fails completely, raise an exception to halt the trial. 
    # NEVER let the LLM hallucinate fake data to preserve experimental validity!
    raise RuntimeError(f"Grounded Search Failed for query: {query}. Halting to preserve experimental validity.")

def step_filter(search_results: str, model_fn: Callable) -> str:
    prompt = f"Filter the following search results to keep only the top 3 most relevant and high-quality results:\n\n{search_results}\n\nReturn only the filtered list."
    return model_fn(prompt)

def step_summarize(filtered_results: str, model_fn: Callable) -> str:
    prompt = (
        f"Summarize the key factual information from these results. "
        f"Ensure you preserve specific entities like dates, locations, and names:\n\n{filtered_results}"
    )
    return model_fn(prompt)

def step_compose(summary: str, model_fn: Callable) -> str:
    prompt = (
        f"Answer the user's question directly using ONLY the facts provided. "
        f"DO NOT be conversational and DO NOT offer extra context. "
        f"You MUST include the specific facts, names, or locations requested. "
        f"Summary: \n\n{summary}\n\n"
        f"IMPORTANT: Your answer MUST be exactly 2-3 complete sentences. "
        f"Do not answer in a single sentence."
    )
    return model_fn(prompt)

def step_verify(recommendation: str, original_query: str, model_fn: Callable) -> str:
    prompt = (
        f"Verify if this response properly addresses the query "
        f"'{original_query}':\n\n{recommendation}\n\n"
        f"Your response must start with exactly one word: "
        f"VALID or INVALID. Then on a new line, give a one-sentence reason. "
        f"Do not use any other words before VALID/INVALID."
    )
    return model_fn(prompt)

STEP_FUNCTIONS = {
    "search": step_search,
    "filter": step_filter,
    "summarize": step_summarize,
    "compose": step_compose,
    "verify": step_verify,
}

# --- 3. Cyclic Backtracking Mitigation (The Defense Strategy) ---
def run_workflow(
    query: str,
    model_fn: Callable,
    error_injection_fn: Callable = None,
    error_step: int | list[int] | None = None,
    error_kwargs: dict | None = None,
    max_retries: int = 1,
) -> list[StepResult]:
    """Run pipeline with cyclic mitigation logic."""
    steps = ["search", "filter", "summarize", "compose", "verify"]
    error_kwargs = error_kwargs or {}

    if error_step is None:
        error_step_set = set()
    elif isinstance(error_step, int):
        error_step_set = {error_step}
    else:
        error_step_set = set(error_step)

    # Mitigation Loop
    for attempt in range(max_retries + 1):
        results = []
        # Contextual retry prompt if verification failed
        current_input = query if attempt == 0 else f"{query}\n(SYSTEM LOG: Your previous pipeline run was flagged as INVALID by the verify step. Please re-execute and correct any factual or logical errors.)"

        for i, step_name in enumerate(steps):
            # print(f"    [DEBUG] Running Task: '{query[:20]}...' | Step: {step_name}")
            if step_name == "verify":
                output = STEP_FUNCTIONS[step_name](current_input, query, model_fn)
            else:
                output = STEP_FUNCTIONS[step_name](current_input, model_fn)

            error_injected = False
            injected_content = None
            pre_injection_output = None
            injection_meta = None

            # Only inject error on the FIRST attempt to measure if the cyclic loop can mitigate it
            if attempt == 0 and error_injection_fn and i in error_step_set:
                pre_injection_output = output
                try:
                    injection_result = error_injection_fn(output, step_name, **error_kwargs)
                except TypeError:
                    injection_result = error_injection_fn(output, step_name)
                
                if isinstance(injection_result, tuple):
                    if len(injection_result) == 3:
                        output, injected_content, injection_meta = injection_result
                    else:
                        output, injected_content = injection_result
                else:
                    output = injection_result
                error_injected = True

            results.append(
                StepResult(
                    step_name=step_name,
                    input_text=current_input,
                    output_text=output,
                    error_injected=error_injected,
                    injected_content=injected_content,
                    pre_injection_output=pre_injection_output,
                    injection_meta=injection_meta,
                )
            )
            current_input = output

        # Check cyclic routing condition
        verify_text = results[-1].output_text
        first_token = ""
        if verify_text.strip():
            first_token = verify_text.strip().split(None, 1)[0].upper().rstrip(".,:;")

        is_valid = (first_token == "VALID")
        
        # If valid, or we exhausted retries, exit cycle.
        if is_valid or attempt == max_retries:
            return results

    return results