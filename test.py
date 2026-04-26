"""Comprehensive CMU gateway smoke test.

Probes every model in the LiteLLM dashboard. For each model:
  - Sends a 5-token "Say HI" prompt
  - Records: success, response text, latency, error message (if any)
  - Estimates per-trial cost based on dashboard pricing

Run:
  export USE_GATEWAY=true
  export GATEWAY_API_KEY=<your-key>
  export GATEWAY_URL=<your-gateway-url>
  python gateway_smoke_test.py

Total cost if all succeed: ~$0.10
"""
import os
import time
import json
from datetime import datetime

# Models to probe, sorted by ascending cost-per-trial.
# Pricing from your LiteLLM dashboard screenshot.
# $/trial assumes 10K input + 2.5K output tokens.
MODELS_TO_TEST = [
    # (gateway_id, in_price, out_price, group, notes)
    ("meta.llama3-1-8b-instruct-v1:0",        0.22, 0.22, "AWS",   "cheapest, primary workhorse"),
    ("us.meta.llama3-1-8b-instruct-v1:0",     0.22, 0.22, "AWS",   "alt routing prefix"),
    ("claude-3-haiku-20240307",               0.25, 1.25, "AWS",   "cheap Anthropic"),
    ("us.anthropic.claude-3-haiku-20240307-v1:0", 0.25, 1.25, "AWS", "fully-qualified Bedrock ID"),
    ("us.meta.llama3-2-11b-instruct-v1:0",    0.35, 0.35, "AWS",   "Llama mid"),
    ("gemini-1.5-flash-002",                  0.07, 0.30, "blank", "blank group — may not work for you"),
    ("gpt-4o-mini-2024-07-18",                0.18, 0.73, "blank", "blank group — may not work for you"),
    ("us.meta.llama3-2-90b-instruct-v1:0",    2.00, 2.00, "AWS",   "Llama large"),
    ("o1-mini-2024-09-12",                    1.21, 4.84, "Azure", "reasoning model"),
    ("gpt-4o-2024-08-06",                     2.75, 11.00, "blank", "blank group — may not work for you"),
    ("claude-3-5-sonnet-20241022",            3.00, 15.00, "AWS",   "Sonnet 3.5"),
    ("us.anthropic.claude-3-5-sonnet-20241022-v2:0", 3.00, 15.00, "AWS", "fully-qualified"),
    ("claude-3-7-sonnet-20250219-v1:0",       3.00, 15.00, "AWS",   "Sonnet 3.7"),
    ("us.anthropic.claude-3-7-sonnet-20250219-v1:0", 3.00, 15.00, "AWS", "fully-qualified"),
    ("claude-sonnet-4-20250514-v1:0",         3.00, 15.00, "AWS",   "Sonnet 4 (latest)"),
    ("us.anthropic.claude-sonnet-4-20250514-v1:0", 3.00, 15.00, "AWS", "fully-qualified"),
    ("gemini-1.5-pro-002",                    3.50, 10.50, "blank", "blank group — may not work"),
    ("gemini-2.5-flash",                      0.07, 0.30, "?",     "shown but unconfigured in screenshot"),
    ("gpt-4o-2024-05-13",                     5.50, 16.50, "blank", "blank group — may not work"),
    ("claude-opus-4-20250514-v1:0",           15.00, 75.00, "AWS",  "expensive, only test if Sonnet works"),
    ("us.anthropic.claude-opus-4-20250514-v1:0", 15.00, 75.00, "AWS", "fully-qualified"),
]

PROMPT = "Reply with exactly: HI"
MAX_TOKENS = 10


def cost_per_trial(in_price, out_price, in_tok=10_000, out_tok=2_500):
    return (in_tok * in_price + out_tok * out_price) / 1e6


def main():
    api_key = os.getenv("GATEWAY_API_KEY")
    base_url = os.getenv("GATEWAY_URL")
    
    if not api_key:
        print("ERROR: set GATEWAY_API_KEY")
        return
    
    print(f"Gateway: {base_url}")
    print(f"Testing {len(MODELS_TO_TEST)} models with prompt: {PROMPT!r}\n")
    
    try:
        import openai
    except ImportError:
        print("pip install openai")
        return
    
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    results = []
    
    for gw_id, in_p, out_p, group, notes in MODELS_TO_TEST:
        cpt = cost_per_trial(in_p, out_p)
        budget_trials = int(100 / cpt) if cpt > 0 else 0
        print(f"  {gw_id}")
        print(f"    group={group}  ${cpt:.4f}/trial  ${100} → {budget_trials:,} trials  ({notes})")
        
        t0 = time.time()
        try:
            r = client.chat.completions.create(
                model=gw_id,
                messages=[{"role": "user", "content": PROMPT}],
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
            response = r.choices[0].message.content.strip()
            latency = time.time() - t0
            print(f"    OK [{latency:.2f}s] → {response!r}\n")
            results.append({
                "gateway_id": gw_id, "group": group, "status": "ok",
                "response": response, "latency_s": round(latency, 2),
                "in_price": in_p, "out_price": out_p,
                "cost_per_trial": round(cpt, 4),
                "trials_per_100usd": budget_trials,
                "notes": notes,
            })
        except Exception as e:
            err = str(e)
            err_short = err[:200]
            latency = time.time() - t0
            print(f"    FAIL [{latency:.2f}s] → {err_short}\n")
            results.append({
                "gateway_id": gw_id, "group": group, "status": "fail",
                "error": err_short, "latency_s": round(latency, 2),
                "in_price": in_p, "out_price": out_p,
                "cost_per_trial": round(cpt, 4),
                "notes": notes,
            })
    
    # Save full results
    out_path = f"gateway_smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("=" * 70)
    print("SUMMARY — models that ACTUALLY work for you")
    print("=" * 70)
    working = [r for r in results if r["status"] == "ok"]
    working.sort(key=lambda r: r["cost_per_trial"])
    
    if not working:
        print("  No models worked. Check API key and gateway URL.")
    else:
        print(f"  {len(working)}/{len(results)} models accessible:\n")
        print(f"  {'gateway_id':<55s}  {'$/trial':>9s}  {'trials/$100':>11s}")
        print(f"  {'-'*55}  {'-'*9}  {'-'*11}")
        for r in working:
            print(f"  {r['gateway_id']:<55s}  ${r['cost_per_trial']:>7.4f}  {r['trials_per_100usd']:>11,}")
    
    print(f"\n  Full results: {out_path}")
    print()
    
    # Suggest lineup
    print("=" * 70)
    print("SUGGESTED LINEUP based on what works")
    print("=" * 70)
    if working:
        cheap = [r for r in working if r["cost_per_trial"] <= 0.005]
        mid = [r for r in working if 0.005 < r["cost_per_trial"] <= 0.02]
        expensive = [r for r in working if r["cost_per_trial"] > 0.02]
        
        print(f"  Cheap (≤$0.005/trial, primary workhorses): {len(cheap)} options")
        for r in cheap[:3]:
            print(f"    - {r['gateway_id']}")
        print(f"  Mid ($0.005-0.02/trial, secondary): {len(mid)} options")
        for r in mid[:3]:
            print(f"    - {r['gateway_id']}")
        print(f"  Expensive (>$0.02/trial, frontier subsample only): {len(expensive)} options")
        for r in expensive[:3]:
            print(f"    - {r['gateway_id']}")


if __name__ == "__main__":
    main()