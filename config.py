import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Pipeline configurations
PIPELINE_CONFIGS = {
    "short": ["search", "compose", "verify"],
    "medium": ["search", "filter", "summarize", "compose", "verify"],
    "self_refine_A": {
        "steps": ["search", "filter", "summarize", "compose", "verify"],
        "feedback": {"after": "compose", "type": "critique_revise", "max_iter": 2},
        "inject_mode": "before_loop",  # inject at compose BEFORE critique/revise
    },
    "self_refine_C": {
        "steps": ["search", "filter", "summarize", "compose", "verify"],
        "feedback": {"after": "compose", "type": "critique_revise", "max_iter": 2},
        "inject_mode": "at_critique",  # inject at the critique step itself
    },
}

DEFAULT_PIPELINE = "medium"
WORKFLOW_STEPS = PIPELINE_CONFIGS[DEFAULT_PIPELINE]  # backward compat

NUM_TRIALS = 15
OUTPUT_DIR = "results"

DEFAULT_OPEN_SOURCE_MODELS = ["llama-3.1-8b"]
DEFAULT_API_MODELS = ["gpt-4o-mini", "gemini-flash"]

INJECT_AT_VERIFY = False

# Judge configuration
PRIMARY_JUDGE = "gpt-4o-mini"
SECONDARY_JUDGE = "claude-sonnet-4-6"  # different provider family
STEP_SCORER = "llama-3.1-8b"  # local, free, for natural failure identification
