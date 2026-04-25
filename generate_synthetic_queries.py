"""Generate post-2024 novel queries for contamination-free evaluation.

Uses an LLM to generate factual questions about recent events that
cannot appear in model training data. Each query is verified to have
a short, unambiguous answer.

Output: appends to ground_truth.json with source="synthetic_novel".
"""
import json
import os
from models import call_model

GENERATION_PROMPT = """Generate {n} factual questions about real events, discoveries, or developments from 2025 or later.

Requirements for EACH question:
1. The answer must be a specific entity (person, place, number, date, or organization)
2. The answer must be 1-10 words
3. The question must be unambiguous — only one correct answer
4. The question must be about a REAL, VERIFIABLE event (not hypothetical)
5. The question must require knowledge that would NOT be in training data before 2025
6. Avoid questions about AI models, elections, or sports scores (too volatile)

Format your response as a JSON array of objects:
[
  {{"question": "...", "answer": "...", "domain": "...", "date_range": "2025-Q1"}},
  ...
]

Categories to cover (roughly equal):
- Science/technology breakthroughs
- International policy/agreements
- Cultural events (awards, records)
- Business/economics (mergers, IPOs, milestones)
- Infrastructure/engineering projects

Return ONLY the JSON array, no other text."""

VERIFICATION_PROMPT = """Verify this factual question and answer pair. Is the answer correct and unambiguous?

Question: {question}
Claimed answer: {answer}

Reply with ONLY one of:
CORRECT - if the answer is accurate and unambiguous
WRONG - if the answer is incorrect
AMBIGUOUS - if multiple valid answers exist
UNVERIFIABLE - if you cannot confirm the answer"""


def generate_queries(n: int = 30, generator_model: str = "gpt-4o-mini",
                     verifier_model: str = "gemini-flash") -> list[dict]:
    """Generate and verify n synthetic novel queries."""
    batch_size = min(15, n)
    all_queries = []

    while len(all_queries) < n:
        remaining = n - len(all_queries)
        prompt = GENERATION_PROMPT.format(n=min(batch_size, remaining))

        try:
            raw = call_model(generator_model, prompt, max_tokens=2048, temperature=0.7)
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
            if clean.endswith("```"):
                clean = clean.rsplit("```", 1)[0]
            clean = clean.strip()
            if clean.startswith("json"):
                clean = clean[4:].strip()

            candidates = json.loads(clean)
        except Exception as e:
            print(f"Generation failed: {e}, retrying...")
            continue

        for candidate in candidates:
            if len(all_queries) >= n:
                break

            question = candidate.get("question", "")
            answer = candidate.get("answer", "")
            if not question or not answer or len(answer.split()) > 10:
                continue

            try:
                verification = call_model(
                    verifier_model,
                    VERIFICATION_PROMPT.format(question=question, answer=answer),
                    max_tokens=20, temperature=0.0,
                )
                if "CORRECT" not in verification.upper():
                    continue
            except Exception:
                continue

            all_queries.append({
                "question": question,
                "answer": answer,
                "domain": candidate.get("domain", "synthetic_novel"),
                "date_range": candidate.get("date_range", "2025+"),
                "source": "synthetic_novel",
                "generator": generator_model,
                "verifier": verifier_model,
            })

    return all_queries


def add_to_ground_truth(queries: list[dict], gt_path: str = "ground_truth.json"):
    """Append synthetic queries to ground_truth.json."""
    gt = {}
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            gt = json.load(f)

    if "queries" not in gt:
        gt["queries"] = []

    existing_questions = {q.get("query", q.get("question", "")) for q in gt.get("queries", [])}

    added = 0
    for q in queries:
        if q["question"] in existing_questions:
            continue
        gt["queries"].append({
            "query": q["question"],
            "answer": q["answer"],
            "source": "synthetic_novel",
            "domain": q.get("domain", "synthetic_novel"),
            "assertions": [{"keywords": [q["answer"]], "aliases": []}],
            "contradictions": [],
            "_placeholder": False,
        })
        # Add top-level entry keyed by query for persistence lookups
        gt[q["question"]] = {
            "answer": q["answer"],
            "source": "synthetic_novel",
            "assertions": [{"keywords": [q["answer"]], "aliases": []}],
            "contradictions": [],
        }
        added += 1

    with open(gt_path, "w") as f:
        json.dump(gt, f, indent=2, ensure_ascii=False)

    print(f"Added {added} synthetic novel queries to {gt_path}")
    return added


def as_task_templates(queries: list[dict]) -> list[dict]:
    """Convert generated queries to TASK_TEMPLATES format."""
    return [
        {
            "query": q["question"],
            "expected_keywords": [q["answer"]],
            "domain": "synthetic_novel",
            "_placeholder": False,
        }
        for q in queries
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic novel queries")
    parser.add_argument("--n", type=int, default=30, help="Number of queries to generate")
    parser.add_argument("--generator", type=str, default="gpt-4o-mini")
    parser.add_argument("--verifier", type=str, default="gemini-flash")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: append to ground_truth.json)")
    args = parser.parse_args()

    queries = generate_queries(n=args.n, generator_model=args.generator,
                               verifier_model=args.verifier)
    print(f"Generated {len(queries)} verified queries")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(queries, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        added = add_to_ground_truth(queries)
        print(f"Appended {added} to ground_truth.json")
