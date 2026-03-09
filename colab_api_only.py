import os
import json
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_CONFIGS = {
    "gpt-4o-mini": {"provider": "openai", "model": "gpt-4o-mini"},
    "claude-haiku": {"provider": "anthropic", "model": "claude-haiku-4-20250414"},
    "gemini-flash": {"provider": "google", "model": "gemini-1.5-flash"},
}

TASK_TEMPLATES = [
    {"query": "best noise-canceling headphones 2025", "expected_keywords": ["sony", "bose", "apple", "airpods", "wh-1000xm5"]},
    {"query": "top programming languages 2025", "expected_keywords": ["python", "javascript", "rust", "typescript", "go"]},
    {"query": "healthy breakfast recipes quick", "expected_keywords": ["oatmeal", "eggs", "smoothie", "yogurt", "avocado"]},
]

def call_api(model_name, prompt):
    cfg = API_CONFIGS[model_name]
    
    if cfg["provider"] == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        r = client.chat.completions.create(model=cfg["model"], messages=[{"role": "user", "content": prompt}], max_tokens=512)
        return r.choices[0].message.content
    
    elif cfg["provider"] == "anthropic":
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        r = client.messages.create(model=cfg["model"], max_tokens=512, messages=[{"role": "user", "content": prompt}])
        return r.content[0].text
    
    elif cfg["provider"] == "google":
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        m = genai.GenerativeModel(cfg["model"])
        return m.generate_content(prompt).text

def run_workflow(model_name, query, error_step=None):
    outputs = []
    current = query
    
    prompts = [
        lambda q: f"You are a search engine. Return 5 relevant results for: '{q}'. Format: numbered list.",
        lambda q: f"Filter to top 3 most relevant:\n\n{q}",
        lambda q: f"Summarize key information:\n\n{q}",
        lambda q: f"Write a recommendation paragraph:\n\n{q}",
        lambda q, orig: f"Verify if this addresses '{orig}':\n\n{q}\n\nRespond VALID or INVALID.",
    ]
    
    for i in range(5):
        if i == 4:
            output = call_api(model_name, prompts[i](current, query))
        else:
            output = call_api(model_name, prompts[i](current))
        
        if error_step == i:
            output = output.replace("2025", "2019").replace("best", "worst")
        
        outputs.append(output)
        current = output
    
    return outputs

def evaluate(outputs, keywords):
    final = outputs[-1]
    is_valid = "VALID" in final.upper()
    kw_score = sum(1 for k in keywords if k.lower() in outputs[-2].lower()) / len(keywords)
    return {"is_valid": is_valid, "keyword_score": kw_score, "combined": 0.5 * int(is_valid) + 0.5 * kw_score}

def run_all(model_name, num_trials=50):
    results = []
    total = len(TASK_TEMPLATES) * 6 * num_trials
    
    with tqdm(total=total, desc=model_name) as pbar:
        for task in TASK_TEMPLATES:
            for error_step in [None, 0, 1, 2, 3, 4]:
                for trial in range(num_trials):
                    try:
                        outputs = run_workflow(model_name, task["query"], error_step)
                        ev = evaluate(outputs, task["expected_keywords"])
                        results.append({"task": task["query"], "error_step": error_step, "trial": trial, "evaluation": ev})
                    except Exception as e:
                        results.append({"task": task["query"], "error_step": error_step, "trial": trial, "error": str(e)})
                    pbar.update(1)
    
    filename = f"results/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("results", exist_ok=True)
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {filename}")

if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "gemini-flash"
    trials = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    run_all(model, trials)