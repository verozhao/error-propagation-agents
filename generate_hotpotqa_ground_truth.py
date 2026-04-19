"""Generate ground_truth.json entries for HotpotQA queries.

GAP 1 FIX: The original ground_truth.json only covers 8 legacy queries.
The 50 HotpotQA queries have expected_keywords = [answer] but no
assertions/contradictions, which forces preserved_component = 1.0 always
(degenerate metric).

GAP 2 FIX (Phase 7.1): Non-yes/no HotpotQA queries previously got EMPTY
contradictions lists, causing inject_factual_error to fall back to generic
filler facts ("A recent fact-check rated this claim as misleading") that
don't actually attack the answer. This made factual injection ~toothless
for 41/58 queries, artificially deflating factual error failure rates.

Now ALL queries get answer-targeted contradictions built from:
  1. Direct negation of the answer entity
  2. Plausible alternative entities (category-aware swaps)
  3. Attribute contradictions derived from assertion keywords
  4. Temporal/quantitative distortions for numeric answers

Usage:
    python generate_hotpotqa_ground_truth.py          # preview new entries
    python generate_hotpotqa_ground_truth.py --write   # update ground_truth.json
    python generate_hotpotqa_ground_truth.py --patch   # backfill contradictions for existing entries
    python generate_hotpotqa_ground_truth.py --patch --write  # backfill AND save
"""

import argparse
import json
import os
import re
import random

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


# ---------------------------------------------------------------------------
# Phase 7.1: Contradiction generation for non-yes/no questions
# ---------------------------------------------------------------------------

_PERSON_ALTS = [
    "James Mitchell", "Robert Chen", "Sarah Williams", "Maria Gonzalez",
    "Thomas Burke", "Heinrich Mueller", "Yuki Tanaka", "Pierre Dubois",
    "Catherine Bell", "Alexander Wright", "Olga Petrov", "Samuel Thornton",
]
_PLACE_ALTS = [
    "Boston", "Philadelphia", "San Francisco", "Chicago", "Denver",
    "Seattle", "Atlanta", "Portland", "Detroit", "Baltimore",
    "London", "Paris", "Berlin", "Tokyo", "Sydney",
]
_ORG_ALTS = [
    "Columbia Records", "Universal Studios", "Warner Bros", "Paramount",
    "SM Entertainment", "JYP Entertainment", "Sony Music", "Atlantic Records",
    "Oxford University Press", "Cambridge University", "MIT Press",
]
_ROLE_ALTS = [
    "Secretary of State", "Ambassador to France", "Attorney General",
    "Deputy Director", "Chief of Staff", "Press Secretary",
    "National Security Advisor", "Surgeon General", "Under Secretary",
]

_NEGATION_TEMPLATES = [
    "{answer} is a widely debunked misconception",
    "{answer} has been confirmed as incorrect by primary sources",
    "the correct answer is not {answer}",
    "historians have conclusively disproven the claim about {answer}",
    "recent scholarship shows {answer} was misattributed",
]
_ATTRIBUTE_TEMPLATES = [
    "{kw} was never associated with this topic",
    "{kw} is incorrect in this context",
    "the reference to {kw} has been retracted",
    "contrary to popular belief, {kw} plays no role here",
    "verified records show no connection to {kw}",
    "{kw} is a common error repeated in secondary sources",
]
_SWAP_TEMPLATES = [
    "the actual answer is {alt}, not {answer}",
    "it was {alt}, not {answer}, according to corrected records",
    "{alt} is the verified correct answer; {answer} is erroneous",
]


def _detect_answer_category(answer: str, query: str) -> str:
    """Heuristic category detection for answer entity."""
    a = answer.lower().strip()
    q = query.lower()

    if a in ("yes", "no"):
        return "boolean"
    if re.fullmatch(r"\d{4}", a) or re.fullmatch(r"[\d,.]+ ?\w*", a):
        return "numeric"
    if any(w in q for w in ["position", "role", "title", "office"]):
        return "role"
    if any(w in q for w in ["city", "country", "where", "located", "based in"]):
        return "place"
    if any(w in a for w in ["entertainment", "records", "studios", "university",
                             "inc", "corp", "company", "group", "institute"]):
        return "org"
    tokens = _tokenize_answer(answer)
    if len(tokens) >= 2 and all(t[0].isupper() for t in answer.split() if t):
        return "person"
    return "entity"


def generate_contradictions(answer: str, query: str, assertions: list,
                            cache_entities: list, rng: random.Random) -> list[str]:
    """Generate answer-targeted contradictions for any query type.

    Returns 8-15 contradiction strings that directly attack the correct
    answer, suitable for insertion by inject_factual_error().
    """
    contradictions = []
    answer_lower = answer.lower().strip()
    category = _detect_answer_category(answer, query)

    # --- 1. Boolean ---
    if category == "boolean":
        opposite = "no" if answer_lower == "yes" else "yes"
        contradictions.append(f"The answer is {opposite}")
        contradictions.append(f"This is definitively {opposite}, not {answer_lower}")
        contradictions.append(f"Verified sources confirm the answer is {opposite}")
        return contradictions

    # --- 2. Direct negation of full answer ---
    for tmpl in rng.sample(_NEGATION_TEMPLATES, k=min(3, len(_NEGATION_TEMPLATES))):
        contradictions.append(tmpl.format(answer=answer))

    # --- 3. Keyword-level contradictions from assertions ---
    for a in assertions:
        for alias in a.get("aliases", []):
            if len(alias) > 3:
                tmpl = rng.choice(_ATTRIBUTE_TEMPLATES)
                contradictions.append(tmpl.format(kw=alias))
        for kw in a.get("keywords", []):
            if len(kw) > 3 and kw.lower() != answer_lower:
                tmpl = rng.choice(_ATTRIBUTE_TEMPLATES)
                contradictions.append(tmpl.format(kw=kw))

    # --- 4. Category-aware entity swap ---
    alt_pool = {
        "person": _PERSON_ALTS,
        "place": _PLACE_ALTS,
        "org": _ORG_ALTS,
        "role": _ROLE_ALTS,
        "entity": _PERSON_ALTS + _PLACE_ALTS,
    }.get(category, _PERSON_ALTS)

    alt_pool = [a for a in alt_pool if a.lower() not in answer_lower]
    if alt_pool:
        for alt in rng.sample(alt_pool, k=min(3, len(alt_pool))):
            tmpl = rng.choice(_SWAP_TEMPLATES)
            contradictions.append(tmpl.format(alt=alt, answer=answer))

    # --- 5. Numeric-specific distortions ---
    if category == "numeric":
        year_match = re.search(r"(\d{4})", answer)
        num_match = re.search(r"([\d,]+)", answer)
        if year_match:
            yr = int(year_match.group(1))
            for delta in [-7, -13, 8, 15]:
                contradictions.append(f"the correct year is {yr + delta}, not {yr}")
        elif num_match:
            raw = num_match.group(1).replace(",", "")
            if raw.isdigit():
                n = int(raw)
                contradictions.append(f"the actual figure is {n * 2:,}, not {n:,}")
                contradictions.append(f"verified records show {max(1, n // 3):,}, not {n:,}")

    # --- 6. Cache-entity distractors ---
    for ent in cache_entities[:3]:
        if ent.lower() not in answer_lower and ent.lower() not in query.lower():
            contradictions.append(
                f"this is commonly confused with {ent}, which is the actual answer")

    # Deduplicate and cap at 15
    seen = set()
    unique = []
    for c in contradictions:
        c_norm = c.lower().strip()
        if c_norm not in seen:
            seen.add(c_norm)
            unique.append(c)
    rng.shuffle(unique)
    return unique[:15]


def generate_entry(task: dict, cache: dict) -> dict:
    """Generate a ground_truth entry for one HotpotQA task."""
    query = task["query"]
    answer = task["expected_keywords"][0] if task["expected_keywords"] else ""

    assertions = []
    is_yes_no = answer.lower().strip() in ("yes", "no")

    if is_yes_no:
        assertions.append({
            "text": f"The answer to '{query}' is {answer.lower()}",
            "keywords": [answer.lower()],
            "aliases": [answer],
        })
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
    for entity in entities:
        entity_tokens = _tokenize_answer(entity)
        if not entity_tokens:
            continue
        if entity.lower() in answer.lower():
            continue
        if entity.lower() in query.lower():
            assertions.append({
                "text": f"References {entity} (from query context)",
                "keywords": [t.lower() for t in entity_tokens],
                "aliases": [entity],
            })

    # Phase 7.1: generate contradictions for ALL query types
    rng = random.Random(hash(query))
    contradictions = generate_contradictions(
        answer, query, assertions, entities, rng)

    return {
        "query": query,
        "source": "hotpotqa_auto",
        "answer": answer,
        "assertions": assertions,
        "contradictions": contradictions,
    }


def patch_existing_entries(gt: dict, cache: dict) -> int:
    """Backfill contradictions for existing entries that have empty lists.

    Returns the number of entries patched.
    """
    patched = 0
    for entry in gt["queries"]:
        existing = entry.get("contradictions", [])
        if len(existing) >= 8:
            continue  # already has enough contradictions (sev3 needs k=8)

        query = entry["query"]
        answer = entry.get("answer", "")

        # For entries without "answer" field, infer from first assertion alias
        if not answer:
            for a in entry.get("assertions", []):
                for alias in a.get("aliases", []):
                    if len(alias) > 3:
                        answer = alias
                        break
                if answer:
                    break
        if not answer:
            continue

        entities = _extract_entities_from_cache(query, cache)
        rng = random.Random(hash(query))
        contradictions = generate_contradictions(
            answer, query, entry.get("assertions", []), entities, rng)

        if contradictions:
            entry["contradictions"] = contradictions
            patched += 1

    return patched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="Write updated ground_truth.json")
    parser.add_argument("--patch", action="store_true",
                        help="Backfill contradictions for existing entries with empty lists")
    args = parser.parse_args()

    cache = {}
    if os.path.exists(SEARCH_CACHE_FILE):
        with open(SEARCH_CACHE_FILE) as f:
            cache = json.load(f)

    with open(GROUND_TRUTH_FILE) as f:
        gt = json.load(f)

    existing_queries = {e["query"] for e in gt["queries"]}

    # --- Patch mode ---
    if args.patch:
        before = sum(1 for e in gt["queries"] if e.get("contradictions"))
        patched = patch_existing_entries(gt, cache)
        after = sum(1 for e in gt["queries"] if e.get("contradictions"))
        print(f"Contradictions coverage: {before}/{len(gt['queries'])} -> {after}/{len(gt['queries'])}")
        print(f"Patched {patched} entries")

        for entry in gt["queries"]:
            if entry.get("source") == "hotpotqa_auto" and entry.get("contradictions"):
                print(f"\n  Example: {entry['query'][:60]}")
                print(f"  Answer: {entry.get('answer', 'N/A')}")
                for c in entry["contradictions"][:4]:
                    print(f"    -> {c}")
                break

        if args.write:
            with open(GROUND_TRUTH_FILE, "w") as f:
                json.dump(gt, f, indent=2, ensure_ascii=False)
            print(f"\nSaved to {GROUND_TRUTH_FILE}")
        else:
            print("\nDry run -- pass --patch --write to save")
        return

    # --- Normal mode: generate new entries ---
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
        print(f"  Contradictions ({len(e['contradictions'])}):")
        for c in e["contradictions"][:4]:
            print(f"    -> {c}")

    if args.write:
        gt["queries"].extend(new_entries)
        gt["_meta"]["n_queries"] = len(gt["queries"])
        gt["_meta"]["status"] = (
            f"DRAFT -- {len(gt['queries'])} queries "
            f"({len(existing_queries)} manual + {len(new_entries)} auto-generated from HotpotQA)"
        )
        with open(GROUND_TRUTH_FILE, "w") as f:
            json.dump(gt, f, indent=2, ensure_ascii=False)
        print(f"\nWrote {len(gt['queries'])} entries to {GROUND_TRUTH_FILE}")
    else:
        print("\nDry run -- pass --write to update ground_truth.json")


if __name__ == "__main__":
    main()
