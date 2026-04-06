from dataclasses import dataclass, field
from typing import Callable, Optional

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
    injected_content: Optional[str] = None  # delta text added by the injector
    pre_injection_output: Optional[str] = None  # output before injector ran


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


def run_workflow(
    query: str,
    model_fn: Callable,
    error_injection_fn: Callable = None,
    error_step: int = None,
    error_kwargs: dict | None = None,
) -> list[StepResult]:
    """Run the 5-step pipeline.

    error_injection_fn signature is one of:
        fn(text, step_name) -> str   (legacy — returns modified text)
        fn(text, step_name, **kw) -> (modified_text, injected_delta)

    The second form lets injectors report exactly what was added/changed
    so the trace logger and factual accuracy evaluator can use it. The
    first form is still supported for backward compatibility — when used,
    `injected_content` on the StepResult will be None.
    """
    results = []
    current_input = query
    steps = ["search", "filter", "summarize", "compose", "verify"]
    error_kwargs = error_kwargs or {}

    for i, step_name in enumerate(steps):
        if step_name == "verify":
            output = STEP_FUNCTIONS[step_name](current_input, query, model_fn)
        else:
            output = STEP_FUNCTIONS[step_name](current_input, model_fn)

        error_injected = False
        injected_content = None
        pre_injection_output = None
        if error_injection_fn and error_step == i:
            pre_injection_output = output
            try:
                injection_result = error_injection_fn(output, step_name, **error_kwargs)
            except TypeError:
                injection_result = error_injection_fn(output, step_name)
            if isinstance(injection_result, tuple):
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
            )
        )
        current_input = output

    return results
