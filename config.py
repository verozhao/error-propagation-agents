import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load frozen experiment config (pinned model versions, parameters)
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "experiment_config.yaml")
EXPERIMENT_CONFIG = {}
try:
    import yaml
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH) as _f:
            EXPERIMENT_CONFIG = yaml.safe_load(_f) or {}
except ImportError:
    pass

PINNED_MODEL_VERSIONS = {}
if EXPERIMENT_CONFIG:
    for _section in ("pipeline", "judges"):
        for _alias, _version in EXPERIMENT_CONFIG.get("models", {}).get(_section, {}).items():
            PINNED_MODEL_VERSIONS[_alias] = _version

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
    # semantics B: post-refinement injection, not run in main sweep due to budget
    "self_refine_B": {
        "steps": ["search", "filter", "summarize", "compose", "verify"],
        "feedback": {"after": "compose", "type": "critique_revise", "max_iter": 2},
        "inject_mode": "after_loop",
    },
}

DEFAULT_PIPELINE = "medium"
WORKFLOW_STEPS = PIPELINE_CONFIGS[DEFAULT_PIPELINE]  # backward compat

NUM_TRIALS = 15
OUTPUT_DIR = "results"

DEFAULT_OPEN_SOURCE_MODELS = ["llama-3.1-8b"]
DEFAULT_API_MODELS = ["gpt-4o-mini", "gemini-flash"]

# Judge configuration
PRIMARY_JUDGE = "gpt-4o-mini"
SECONDARY_JUDGE = "claude-sonnet-4-6"  # different provider family
STEP_SCORER = "llama-3.1-8b"  # local, free, for natural failure identification
