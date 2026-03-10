import random
import re

ERROR_SUBSTITUTIONS = {
    "2025": "2019",
    "2024": "2018",
    "best": "worst",
    "top": "outdated",
    "recommended": "not recommended",
    "high-quality": "low-quality",
    "popular": "unpopular",
    "effective": "ineffective",
}


def inject_semantic_error(text: str, step_name: str) -> str:
    modified = text
    substitutions_made = 0
    
    for original, replacement in ERROR_SUBSTITUTIONS.items():
        if original.lower() in modified.lower() and substitutions_made < 2:
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            modified = pattern.sub(replacement, modified, count=1)
            substitutions_made += 1
    
    if substitutions_made == 0:
        sentences = modified.split(". ")
        if len(sentences) > 1:
            idx = random.randint(0, len(sentences) - 1)
            sentences[idx] = "This information may be outdated or incorrect."
            modified = ". ".join(sentences)
    
    return modified


def inject_factual_error(text: str, step_name: str) -> str:
    fake_facts = [
        "According to recent studies, this has been discontinued.",
        "Note: This product was recalled in 2023.",
        "Warning: Multiple reports indicate quality issues.",
    ]
    sentences = text.split(". ")
    insert_pos = len(sentences) // 2
    sentences.insert(insert_pos, random.choice(fake_facts))
    return ". ".join(sentences)


def inject_omission_error(text: str, step_name: str) -> str:
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences) > 2:
        remove_idx = random.randint(1, len(sentences) - 1)  # keep first sentence
        sentences.pop(remove_idx)
    return ". ".join(sentences)


ERROR_TYPES = {
    "semantic": inject_semantic_error,
    "factual": inject_factual_error,
    "omission": inject_omission_error,
}