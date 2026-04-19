import os
from dotenv import load_dotenv
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

USE_CMU_GATEWAY = os.getenv("USE_CMU_GATEWAY", "false").lower() == "true"
CMU_GATEWAY_URL = os.getenv("CMU_GATEWAY_URL", "https://ai-gateway.andrew.cmu.edu")
CMU_GATEWAY_API_KEY = os.getenv("CMU_GATEWAY_API_KEY", "")

OPEN_SOURCE_MODELS = {
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen-2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-2.5-72b": "Qwen/Qwen2.5-72B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-r1-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1-14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}

API_MODELS = {
    "gpt-4o": {"provider": "openai", "model": "gpt-4o"},
    "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini"},
    "claude-sonnet": {"provider": "anthropic", "model": "claude-sonnet-4-20250514"},
    "claude-haiku": {"provider": "anthropic", "model": "claude-haiku-4-20250414"},
    "claude-3-haiku": {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
    "claude-sonnet-3-5": {"provider": "anthropic", "model": "claude-3-5-sonnet"},
    "gemini-pro": {"provider": "google", "model": "gemini-1.5-pro"},
    "gemini-flash": {"provider": "google", "model": "gemini-1.5-flash"},
}

# CMU AI Gateway model ID mapping (from direct API IDs → gateway IDs)
CMU_GATEWAY_MODELS = {
    "gpt-4o": "gpt-4o-2024-08-06",
    "gpt-4o-mini": "claude-3-haiku-20240307", # as defensive programming
    "claude-sonnet-4-20250514": "claude-sonnet-4-20250514-v1:0",
    "claude-haiku-4-20250414": "claude-haiku-4-5-20251001-v1:0",
    "claude-3-haiku-20240307": "claude-3-haiku-20240307",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "gemini-1.5-pro": "gemini-1.5-pro-002",
    "gemini-1.5-flash": "gemini-1.5-flash-002",
}

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

_API_RETRY = retry(
    stop=stop_after_attempt(6),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,
)

_local_model_cache = {}
_api_clients = {}


class LocalModel:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        return response


def get_local_model(model_name: str) -> LocalModel:
    if model_name not in _local_model_cache:
        model_path = OPEN_SOURCE_MODELS[model_name]
        _local_model_cache[model_name] = LocalModel(model_path)
    return _local_model_cache[model_name]


def get_cmu_gateway_client():
    if "cmu_gateway" not in _api_clients:
        import openai
        _api_clients["cmu_gateway"] = openai.OpenAI(
            api_key=CMU_GATEWAY_API_KEY,
            base_url=CMU_GATEWAY_URL,
        )
    return _api_clients["cmu_gateway"]


def get_openai_client():
    if "openai" not in _api_clients:
        import openai
        _api_clients["openai"] = openai.OpenAI(api_key=OPENAI_API_KEY)
    return _api_clients["openai"]


def get_anthropic_client():
    if "anthropic" not in _api_clients:
        import anthropic
        _api_clients["anthropic"] = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _api_clients["anthropic"]


def get_google_model(model_name: str):
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel(model_name)


@_API_RETRY
def call_openai(model: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0, seed: int = None) -> str:
    client = get_openai_client()
    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        kwargs["seed"] = seed
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


@_API_RETRY
def call_anthropic(model: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0, seed: int = None) -> str:
    client = get_anthropic_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


@_API_RETRY
def call_google(model: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0, seed: int = None) -> str:
    gmodel = get_google_model(model)
    response = gmodel.generate_content(
        prompt,
        generation_config={"max_output_tokens": max_tokens, "temperature": temperature}
    )
    return response.text


@_API_RETRY
def call_cmu_gateway(model: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0, seed: int = None) -> str:
    client = get_cmu_gateway_client()
    gateway_model = CMU_GATEWAY_MODELS.get(model, model)
    kwargs = {
        "model": gateway_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        kwargs["seed"] = seed
    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def call_model(model_name: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0, seed: int = None) -> str:
    if model_name in OPEN_SOURCE_MODELS:
        model = get_local_model(model_name)
        if seed is not None:
            import torch
            torch.manual_seed(seed)
        return model.generate(prompt, max_tokens, temperature)

    if model_name in API_MODELS:
        cfg = API_MODELS[model_name]
        model_id = cfg["model"]

        if USE_CMU_GATEWAY:
            return call_cmu_gateway(model_id, prompt, max_tokens, temperature, seed)

        provider = cfg["provider"]
        if provider == "openai":
            return call_openai(model_id, prompt, max_tokens, temperature, seed)
        elif provider == "anthropic":
            return call_anthropic(model_id, prompt, max_tokens, temperature, seed)
        elif provider == "google":
            return call_google(model_id, prompt, max_tokens, temperature, seed)

    raise ValueError(f"Unknown model: {model_name}")


def list_available_models() -> dict:
    return {
        "open_source": list(OPEN_SOURCE_MODELS.keys()),
        "api": list(API_MODELS.keys()),
    }


def check_gpu_available() -> dict:
    if not torch.cuda.is_available():
        return {"available": False, "message": "No GPU detected"}
    
    return {
        "available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
    }