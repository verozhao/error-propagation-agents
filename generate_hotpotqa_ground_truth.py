"""Generate ground_truth.json entries for HotpotQA queries.

GAP 1 FIX: The original ground_truth.json only covers 8 legacy queries.
The 50 HotpotQA queries have expected_keywords = [answer] but no
assertions/contradictions, which forces preserved_component = 1.0 always
(degenerate metric).

This script builds assertion entries from:
  1. The known HotpotQA answer (expected_keywords)
  2. Key entities extracted from search cache results
  3. Negation-aware contradictions for yes/no questions

For multi-hop questions, assertions capture both the answer and the
intermediate entities that a correct pipeline must preserve.

Usage:
    python generate_hotpotqa_ground_truth.py          # preview
    python generate_hotpotqa_ground_truth.py --write   # update ground_truth.json
"""

import argparse
import json
import os
import re

from workflow import TASK_TEMPLATES

SEARCH_CACHE_FILE = "search_cache.json"
GROUND_TRUTH_FILE = "ground_truth.json"


def _extract_entities_from_cache(query: str, cache: dict) -> list[str]:
    """Extract capitalized multi-word entities from cached search results."""
    text = cache.get(query, "")
    if not text:
        return []
    caps = re.findall(r"(?<!\. )([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text)
    seen = set()
    entities = []
    for e in caps:
        e_lower = e.lower()
        if e_lower not in seen and len(e) > 3:
            seen.add(e_lower)
            entities.append(e)
    return entities[:10]


def _tokenize_answer(answer: str) -> list[str]:
    """Split answer into meaningful keyword tokens."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]+", answer)
    stopwords = {"the", "a", "an", "of", "in", "on", "at", "to", "for",
                 "and", "or", "is", "was", "were", "are", "be", "been",
                 "by", "with", "from", "that", "this", "it", "its"}
    return [t for t in tokens if t.lower() not in stopwords and len(t) > 1]


def generate_entry(task: dict, cache: dict) -> dict:
    """Generate a ground_truth entry for one HotpotQA task."""
    query = task["query"]
    answer = task["expected_keywords"][0] if task["expected_keywords"] else ""
    answer_lower = answer.lower().strip()

    assertions = []
    contradictions = []

    is_yes_no = answer_lower in ("yes", "no")

    if is_yes_no:
        assertions.append({
            "text": f"The answer to '{query}' is {answer_lower}",
            "keywords": [answer_lower],
            "aliases": [answer],
        })
        opposite = "no" if answer_lower == "yes" else "yes"
        contradictions.append(f"The answer is {opposite}")
    else:
        answer_keywords = _tokenize_answer(answer)
        if answer_keywords:
            assertions.append({
                "text": f"The answer includes: {answer}",
                "keywords": [k.lower() for k in answer_keywords],
                "aliases": [answer],
            })

        parts = re.split(r"[,;]", answer)
        for part in parts:
            part = part.strip()
            if not part or part == answer:
                continue
            part_kw = _tokenize_answer(part)
            if part_kw:
                assertions.append({
                    "text": f"Mentions {part}",
                    "keywords": [k.lower() for k in part_kw],
                    "aliases": [part],
                })

    entities = _extract_entities_from_cache(query, cache)
    query_lower = query.lower()
    for entity in entities:
        entity_tokens = _tokenize_answer(entity)
        if not entity_tokens:
            continue
        if entity.lower() in answer_lower:
            continue
        if entity.lower() in query_lower:
            assertions.append({
                "text": f"References {entity} (from query context)",
                "keywords": [t.lower() for t in entity_tokens],
                "aliases": [entity],
            })

    return {
        "query": query,
        "source": "hotpotqa_auto",
        "answer": answer,
        "assertions": assertions,
        "contradictions": contradictions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Write updated ground_truth.json")
    args = parser.parse_args()

    with open(SEARCH_CACHE_FILE) as f:
        cache = json.load(f)

    with open(GROUND_TRUTH_FILE) as f:
        gt = json.load(f)

    existing_queries = {e["query"] for e in gt["queries"]}

    new_entries = []
    for task in TASK_TEMPLATES:
        if task.get("_placeholder"):
            continue
        if task["query"] in existing_queries:
            continue
        entry = generate_entry(task, cache)
        if entry["assertions"]:
            new_entries.append(entry)

    print(f"Existing ground truth entries: {len(existing_queries)}")
    print(f"New entries generated: {len(new_entries)}")

    total_assertions = sum(len(e["assertions"]) for e in new_entries)
    total_contradictions = sum(len(e["contradictions"]) for e in new_entries)
    print(f"Total new assertions: {total_assertions}")
    print(f"Total new contradictions: {total_contradictions}")

    for e in new_entries[:5]:
        print(f"\n  Query: {e['query'][:70]}")
        print(f"  Answer: {e['answer']}")
        print(f"  Assertions ({len(e['assertions'])}):")
        for a in e["assertions"][:3]:
            print(f"    - {a['text'][:60]} | kw={a['keywords']}")

    if args.write:
        gt["queries"].extend(new_entries)
        gt["_meta"]["n_queries"] = len(gt["queries"])
        gt["_meta"]["status"] = (
            f"DRAFT — {len(gt['queries'])} queries "
            f"({len(existing_queries)} manual + {len(new_entries)} auto-generated from HotpotQA)"
        )
        with open(GROUND_TRUTH_FILE, "w") as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {len(gt['queries'])} entries to {GROUND_TRUTH_FILE}")
    else:
        print("\nDry run — pass --write to update ground_truth.json")


if __name__ == "__main__":
    main()
