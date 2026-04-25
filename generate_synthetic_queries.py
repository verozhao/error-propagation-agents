"""Generate post-2024 novel queries for contamination-free evaluation.

Two generation modes:
1. curated (default): Load hand-verified questions from a curated JSONL file.
   These should be sourced from verifiable post-cutoff sources (Wikipedia DYK,
   arXiv abstracts post-2025-Q4, news archives) and manually checked.
2. llm_generate (legacy): Use an LLM to generate and cross-verify. This mode
   is methodologically weaker because both generator and verifier have similar
   knowledge cutoffs (~mid-2024) and cannot reliably produce/validate post-cutoff
   facts. Retained for backward compatibility but NOT recommended for the paper.

Output: appends to ground_truth.json with source="synthetic_novel".
"""
import json
import os
from models import call_model

CURATED_QUERIES_FILE = os.path.join(os.path.dirname(__file__), "synthetic_novel_curated.jsonl")

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


def load_curated_queries(path: str = CURATED_QUERIES_FILE) -> list[dict]:
    """Load hand-verified queries from a curated JSONL file.

    Expected format per line:
    {"question": "...", "answer": "...", "domain": "...", "source_url": "...", "verified_by": "human"}
    """
    if not os.path.exists(path):
        print(f"WARNING: Curated queries file not found: {path}")
        print("  To create it, source 30 questions from:")
        print("  - Wikipedia:Did_you_know/Recent_additions (last 6 months)")
        print("  - arXiv abstracts (2025-Q4 onward)")
        print("  - Major news archives (Reuters, AP)")
        print("  Format: one JSON object per line with question, answer, domain, source_url")
        return []

    queries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("question") and entry.get("answer"):
                    queries.append({
                        "question": entry["question"],
                        "answer": entry["answer"],
                        "domain": entry.get("domain", "synthetic_novel"),
                        "date_range": entry.get("date_range", "2025+"),
                        "source": "synthetic_novel",
                        "source_url": entry.get("source_url", ""),
                        "verified_by": entry.get("verified_by", "human"),
                    })
            except json.JSONDecodeError:
                continue

    print(f"Loaded {len(queries)} curated novel queries from {path}")
    return queries


def generate_queries(n: int = 30, generator_model: str = "gpt-4o-mini",
                     verifier_model: str = "gemini-flash",
                     mode: str = "curated") -> list[dict]:
    """Generate or load synthetic novel queries.

    Args:
        mode: "curated" (recommended) or "llm_generate" (legacy, weaker).
    """
    if mode == "curated":
        queries = load_curated_queries()
        if queries:
            return queries[:n]
        print("Falling back to LLM generation (curated file not found)")

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
    parser.add_argument("--mode", choices=["curated", "llm_generate"], default="curated",
                        help="curated (recommended) or llm_generate (legacy)")
    parser.add_argument("--generator", type=str, default="gpt-4o-mini")
    parser.add_argument("--verifier", type=str, default="gemini-flash")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file (default: append to ground_truth.json)")
    args = parser.parse_args()

    queries = generate_queries(n=args.n, generator_model=args.generator,
                               verifier_model=args.verifier, mode=args.mode)
    print(f"Generated {len(queries)} verified queries")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(queries, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        added = add_to_ground_truth(queries)
        print(f"Appended {added} to ground_truth.json")
