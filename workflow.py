from dataclasses import dataclass
from typing import Callable

TASK_TEMPLATES = [
    {
        "query": "best noise-canceling headphones 2025",
        "expected_keywords": ["sony", "bose", "apple", "airpods", "wh-1000xm5"],
    },
    {
        "query": "top programming languages 2025",
        "expected_keywords": ["python", "javascript", "rust", "typescript", "go"],
    },
    {
        "query": "healthy breakfast recipes quick",
        "expected_keywords": ["oatmeal", "eggs", "smoothie", "yogurt", "avocado"],
    },
]


@dataclass
class StepResult:
    step_name: str
    input_text: str
    output_text: str
    error_injected: bool


def step_search(query: str, model_fn: Callable) -> str:
    prompt = f"You are a search engine. Return 5 relevant results for: '{query}'. Format: numbered list with title and one-sentence description."
    return model_fn(prompt)


def step_filter(search_results: str, model_fn: Callable) -> str:
    prompt = f"Filter the following search results to keep only the top 3 most relevant and high-quality results:\n\n{search_results}\n\nReturn only the filtered list."
    return model_fn(prompt)


def step_summarize(filtered_results: str, model_fn: Callable) -> str:
    prompt = f"Summarize the key information from these results into a concise paragraph:\n\n{filtered_results}"
    return model_fn(prompt)


def step_compose(summary: str, model_fn: Callable) -> str:
    prompt = f"Based on this summary, write a helpful recommendation paragraph for a user:\n\n{summary}"
    return model_fn(prompt)


def step_verify(recommendation: str, original_query: str, model_fn: Callable) -> str:
    prompt = f"Verify if this recommendation properly addresses the query '{original_query}':\n\n{recommendation}\n\nRespond with 'VALID' or 'INVALID' followed by a brief explanation."
    return model_fn(prompt)


STEP_FUNCTIONS = {
    "search": step_search,
    "filter": step_filter,
    "summarize": step_summarize,
    "compose": step_compose,
    "verify": step_verify,
}


def run_workflow(query: str, model_fn: Callable, error_injection_fn: Callable = None, error_step: int = None) -> list[StepResult]:
    results = []
    current_input = query
    steps = ["search", "filter", "summarize", "compose", "verify"]
    
    for i, step_name in enumerate(steps):
        if step_name == "search":
            output = STEP_FUNCTIONS[step_name](current_input, model_fn)
        elif step_name == "verify":
            output = STEP_FUNCTIONS[step_name](current_input, query, model_fn)
        else:
            output = STEP_FUNCTIONS[step_name](current_input, model_fn)
        
        error_injected = False
        if error_injection_fn and error_step == i:
            output = error_injection_fn(output, step_name)
            error_injected = True
        
        results.append(StepResult(step_name, current_input, output, error_injected))
        current_input = output
    
    return results