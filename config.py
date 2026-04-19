import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

WORKFLOW_STEPS = ["search", "filter", "summarize", "compose", "verify"]
NUM_TRIALS = 15
OUTPUT_DIR = "results"

DEFAULT_OPEN_SOURCE_MODELS = ["llama-3.1-8b", "qwen-2.5-7b", "mistral-7b", "deepseek-r1-7b"]
DEFAULT_API_MODELS = ["gpt-4o-mini", "claude-haiku", "gemini-flash"]

# Verify is the terminal step; injecting there measures direct output
# corruption, not propagation. Keep False unless explicitly studying that.
INJECT_AT_VERIFY = False