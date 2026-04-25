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
        print("WARNING: 'datasets' library not found, falling back to ground_truth.json")
        return _load_tasks_from_ground_truth(num_tasks)


def _load_tasks_from_ground_truth(num_tasks=50):
    """Fallback: build TASK_TEMPLATES from ground_truth.json when the
    HuggingFace datasets library is not installed. This guarantees the
    experiment can always run, and uses the same queries that have
    assertions/contradictions for evaluation.

    Queries are prioritized by: (1) has search cache entry, (2) number
    of assertions + contradictions. This ensures budget-constrained runs
    (small num_tasks) get the highest-signal queries first.
    """
    gt_path = os.path.join(os.path.dirname(__file__), "ground_truth.json")
    if not os.path.exists(gt_path):
        print("ERROR: ground_truth.json not found — cannot build task list.")
        return []
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)

    # Load search cache to prioritize queries with cached results
    cache_path = os.path.join(os.path.dirname(__file__), "search_cache.json")
    cached_queries = set()
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached_queries = set(json.load(f).keys())

    candidates = []
    for entry in gt.get("queries", []):
        query = entry.get("query", "")
        answer = entry.get("answer", "")
        if not query:
            continue
        # Use answer as expected_keyword; fall back to first assertion alias
        if not answer:
            for a in entry.get("assertions", []):
                for alias in a.get("aliases", []):
                    if len(alias) > 2:
                        answer = alias
                        break
                if answer:
                    break
        has_cache = query in cached_queries
        n_assert = len(entry.get("assertions", []))
        n_contra = len(entry.get("contradictions", []))
        quality = n_assert * 2 + n_contra + (100 if has_cache else 0)
        candidates.append((quality, {
            "query": query,
            "expected_keywords": [answer] if answer else [],
            "domain": entry.get("source", "ground_truth"),
            "_placeholder": False,
        }))

    # Sort by quality descending (cached queries first, then richest ground truth)
    candidates.sort(key=lambda x: x[0], reverse=True)
    tasks = [c[1] for c in candidates[:num_tasks]]
    n_cached = sum(1 for q, t in zip(candidates[:num_tasks], tasks)
                   if q[0] >= 100)
    print(f"Loaded {len(tasks)} tasks from ground_truth.json "
          f"({n_cached} with search cache)")
    return tasks

def load_triviaqa_tasks(num_tasks=60):
    """Load TriviaQA validation questions (single-hop factoid)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("trivia_qa", "rc", split="validation")
        tasks = []
        seen = set()
        for item in ds:
            if len(tasks) >= num_tasks:
                break
            q = item["question"]
            answer = item.get("answer", {}).get("value", "")
            if not answer or q in seen:
                continue
            if len(answer.split()) > 10 or len(q) > 200:
                continue
            seen.add(q)
            tasks.append({
                "query": q,
                "expected_keywords": [answer],
                "domain": "single_hop",
                "_placeholder": False,
            })
        return tasks
    except (ImportError, Exception) as e:
        print(f"WARNING: TriviaQA load failed ({e}), skipping")
        return []


def load_strategyqa_tasks(num_tasks=60):
    """Load StrategyQA questions (multi-step reasoning, yes/no)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wics/strategy-qa", split="test")
        tasks = []
        for item in ds:
            if len(tasks) >= num_tasks:
                break
            q = item.get("question", item.get("input", ""))
            answer = str(item.get("answer", item.get("target", "")))
            if not q:
                continue
            tasks.append({
                "query": q,
                "expected_keywords": [answer] if answer else [],
                "domain": "reasoning",
                "_placeholder": False,
            })
        return tasks
    except (ImportError, Exception) as e:
        print(f"WARNING: StrategyQA load failed ({e}), skipping")
        return []


def load_bfcl_tasks(num_tasks=30):
    """Load BFCL/ToolBench questions (agentic tool-use)."""
    try:
        from datasets import load_dataset
        ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="test")
        tasks = []
        for item in ds:
            if len(tasks) >= num_tasks:
                break
            q = item.get("question", "")
            if not q or len(q) > 200:
                continue
            tasks.append({
                "query": q,
                "expected_keywords": [],
                "domain": "agentic",
                "_placeholder": False,
            })
        return tasks
    except (ImportError, Exception) as e:
        print(f"WARNING: BFCL load failed ({e}), skipping")
        return []


def load_synthetic_novel_tasks():
    """Load synthetic novel queries from ground_truth.json (source=synthetic_novel)."""
    gt_path = os.path.join(os.path.dirname(__file__), "ground_truth.json")
    if not os.path.exists(gt_path):
        return []
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    tasks = []
    for entry in gt.get("queries", []):
        if entry.get("source") == "synthetic_novel":
            tasks.append({
                "query": entry.get("query", ""),
                "expected_keywords": [entry.get("answer", "")] if entry.get("answer") else [],
                "domain": "synthetic_novel",
                "_placeholder": False,
            })
    return tasks


def load_all_tasks():
    """Load combined query set: HotpotQA + TriviaQA + StrategyQA + BFCL + synthetic novel.

    Falls back to ground_truth.json if HuggingFace datasets unavailable.
    """
    hotpot = load_hotpotqa_tasks(60)
    trivia = load_triviaqa_tasks(60)
    strategy = load_strategyqa_tasks(60)
    bfcl = load_bfcl_tasks(30)
    synthetic = load_synthetic_novel_tasks()

    all_tasks = hotpot + trivia + strategy + bfcl + synthetic

    if len(all_tasks) < 30:
        print("WARNING: Few dataset queries loaded, supplementing from ground_truth.json")
        gt_tasks = _load_tasks_from_ground_truth(210 - len(all_tasks))
        existing_queries = {t["query"] for t in all_tasks}
        for t in gt_tasks:
            if t["query"] not in existing_queries:
                all_tasks.append(t)

    print(f"Loaded {len(all_tasks)} total tasks: "
          f"{len(hotpot)} HotpotQA, {len(trivia)} TriviaQA, "
          f"{len(strategy)} StrategyQA, {len(bfcl)} BFCL, {len(synthetic)} synthetic")
    return all_tasks


TASK_TEMPLATES = load_all_tasks()
assert len(TASK_TEMPLATES) >= 30, (
    f"Insufficient queries: got {len(TASK_TEMPLATES)}, need >= 30. "
    f"Check dataset availability or ground_truth.json."
)

@dataclass
class StepResult:
    step_name: str
    input_text: str
    output_text: str
    error_injected: bool
    injected_content: Optional[str] = None
    pre_injection_output: Optional[str] = None
    injection_meta: Optional[dict] = None
    retry_attempted: bool = False
    retry_recovered: bool = False

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

# --- Self-refine critique/revise prompts ---

CRITIQUE_PROMPT = """You are a strict fact-checker. Review the following draft answer and identify any factual errors, unsupported claims, or missing information.

Draft answer:
\"\"\"{draft}\"\"\"

Original query: {query}

List each issue on its own line. If the draft is accurate and complete, respond with exactly: NO ISSUES FOUND"""

REVISE_PROMPT = """Revise the following draft to fix the issues identified by the critic. Keep all correct information intact.

Draft answer:
\"\"\"{draft}\"\"\"

Critique:
\"\"\"{critique}\"\"\"

Original query: {query}

Output ONLY the revised answer, nothing else."""


def run_self_refine(draft: str, query: str, model_fn: Callable, max_iter: int = 2) -> tuple[str, list[dict]]:
    """Run critique→revise loop on a draft. Returns (final_text, refinement_log)."""
    current = draft
    log = []
    for i in range(max_iter):
        critique = model_fn(CRITIQUE_PROMPT.format(draft=current, query=query))
        if "NO ISSUES FOUND" in critique.upper():
            log.append({"iteration": i + 1, "critique": critique, "action": "accepted"})
            break
        revised = model_fn(REVISE_PROMPT.format(draft=current, critique=critique, query=query))
        log.append({"iteration": i + 1, "critique": critique, "revised": revised})
        current = revised
    return current, log


# --- 3. Cyclic Backtracking Mitigation (The Defense Strategy) ---
def run_workflow(
    query: str,
    model_fn: Callable,
    error_injection_fn: Callable = None,
    error_step: int | list[int] | None = None,
    error_kwargs: dict | None = None,
    max_retries: int = 1,
    pipeline_config=None,
    intervention_fn: Callable = None,
) -> list[StepResult]:
    """Run pipeline with cyclic mitigation logic."""
    if pipeline_config is not None:
        if isinstance(pipeline_config, list):
            steps = list(pipeline_config)
        else:
            steps = list(pipeline_config.get("steps", ["search", "filter", "summarize", "compose", "verify"]))
    else:
        steps = ["search", "filter", "summarize", "compose", "verify"]

    feedback_cfg = None
    if isinstance(pipeline_config, dict) and "feedback" in pipeline_config:
        feedback_cfg = pipeline_config["feedback"]

    error_kwargs = error_kwargs or {}

    if error_step is None:
        error_step_set = set()
    elif isinstance(error_step, int):
        error_step_set = {error_step}
    else:
        error_step_set = set(error_step)

    # Mitigation Loop
    #
    # CRITICAL: always return the FIRST attempt's results (which contain
    # the injection). Retry outcome is stored as metadata on the first
    # attempt's StepResults, NOT as a replacement.
    #
    # Before this fix, `results = []` on attempt=1 overwrote attempt=0,
    # causing 78% of injected records to lose their injection data.
    first_attempt_results = None

    for attempt in range(max_retries + 1):
        results = []
        current_input = query if attempt == 0 else f"{query}\n(SYSTEM LOG: Your previous pipeline run was flagged as INVALID by the verify step. Please re-execute and correct any factual or logical errors.)"

        for i, step_name in enumerate(steps):
            if step_name == "verify":
                output = STEP_FUNCTIONS[step_name](current_input, query, model_fn)
            else:
                output = STEP_FUNCTIONS[step_name](current_input, model_fn)

            error_injected = False
            injected_content = None
            pre_injection_output = None
            injection_meta = None

            if attempt == 0 and error_injection_fn and i in error_step_set:
                inject_mode = pipeline_config.get("inject_mode", "before_loop") if isinstance(pipeline_config, dict) else "before_loop"
                if inject_mode != "at_critique":
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

            # Self-refine loop (critique->revise) after the feedback step
            if feedback_cfg and step_name == feedback_cfg.get("after"):
                inject_mode = pipeline_config.get("inject_mode", "before_loop") if isinstance(pipeline_config, dict) else "before_loop"
                if inject_mode == "at_critique" and attempt == 0 and error_injection_fn and i in error_step_set:
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

                max_refine_iter = feedback_cfg.get("max_iter", 2)
                output, _refine_log = run_self_refine(output, query, model_fn, max_iter=max_refine_iter)

                # after_loop: inject AFTER the self-refine loop completes
                if inject_mode == "after_loop" and attempt == 0 and error_injection_fn and i in error_step_set:
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

            # Intervention: re-ask if intervention_fn flags this step
            if intervention_fn and not error_injected:
                if intervention_fn(current_input, output):
                    output = STEP_FUNCTIONS[step_name](current_input, model_fn) if step_name != "verify" else STEP_FUNCTIONS[step_name](current_input, query, model_fn)

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

        # Save first attempt (the one with injection)
        if attempt == 0:
            first_attempt_results = results

        # Check cyclic routing condition
        verify_text = results[-1].output_text
        first_token = ""
        if verify_text.strip():
            first_token = verify_text.strip().split(None, 1)[0].upper().rstrip(".,:;")

        is_valid = (first_token == "VALID")
        
        if is_valid or attempt == max_retries:
            # Always return the FIRST attempt's results (with injection data).
            # Append retry metadata so downstream analysis knows what happened.
            if first_attempt_results is not None and attempt > 0:
                # Tag first attempt results with retry outcome
                for sr in first_attempt_results:
                    sr.retry_attempted = True
                    sr.retry_recovered = is_valid
            return first_attempt_results if first_attempt_results is not None else results

    return first_attempt_results if first_attempt_results is not None else results