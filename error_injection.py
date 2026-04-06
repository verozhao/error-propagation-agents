"""Error injection functions.

Each injector accepts (text, step_name, severity=1, return_delta=False) and
returns either the modified text (legacy) or a tuple (modified_text,
injected_delta) when `return_delta=True`. The delta is a short string that
captures *what* was inserted, removed, or substituted, so downstream
components (trace logger, factual accuracy evaluator) can reason about it.

Severity scales:
    factual:  1 = 1 false claim, 2 = 2 false claims, 3 = 3 false claims
    omission: 1 = remove 1 sentence (legacy), 2 = remove ~30%, 3 = remove ~50%
    semantic: 1 = up to 2 substitutions (legacy), 2 = up to 5, 3 = unlimited
"""

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
    "leading": "lagging",
    "improved": "degraded",
    "reliable": "unreliable",
    "premium": "budget",
}


FAKE_FACTS = [
    "According to recent studies, this has been discontinued.",
    "Note: This product was recalled in 2023.",
    "Warning: Multiple reports indicate quality issues.",
    "A 2024 lawsuit found this option to be unsafe.",
    "This option has been banned in several regions.",
    "Independent reviewers have flagged this as fraudulent.",
    "Recent benchmarks show this performs worst in its category.",
    "Manufacturer announced an end-of-life notice for this product.",
]


def _maybe_return(modified: str, delta: str, return_delta: bool):
    if return_delta:
        return modified, delta
    return modified


def inject_semantic_error(text: str, step_name: str, severity: int = 1, return_delta: bool = False):
    if severity <= 1:
        max_subs = 2
    elif severity == 2:
        max_subs = 5
    else:
        max_subs = 10**9  # unlimited

    modified = text
    swaps: list[tuple[str, str]] = []

    for original, replacement in ERROR_SUBSTITUTIONS.items():
        if len(swaps) >= max_subs:
            break
        if original.lower() in modified.lower():
            pattern = re.compile(re.escape(original), re.IGNORECASE)
            new_modified, n = pattern.subn(replacement, modified, count=(1 if severity <= 1 else 0))
            if n > 0:
                modified = new_modified
                swaps.extend([(original, replacement)] * n)
                if len(swaps) >= max_subs:
                    # trim — re-do with exact count
                    pass

    if not swaps:
        sentences = modified.split(". ")
        if len(sentences) > 1:
            idx = random.randint(0, len(sentences) - 1)
            insertion = "This information may be outdated or incorrect."
            sentences[idx] = insertion
            modified = ". ".join(sentences)
            delta = insertion
        else:
            delta = ""
    else:
        delta = "; ".join(f"{o}->{r}" for o, r in swaps)

    return _maybe_return(modified, delta, return_delta)


def inject_factual_error(text: str, step_name: str, severity: int = 1, return_delta: bool = False):
    n_claims = max(1, min(3, severity))
    sentences = text.split(". ")
    chosen = random.sample(FAKE_FACTS, k=min(n_claims, len(FAKE_FACTS)))

    # spread insertions across the text
    if len(sentences) <= 1:
        modified = text + " " + " ".join(chosen)
    else:
        positions = sorted(
            {max(1, (len(sentences) * (i + 1)) // (n_claims + 1)) for i in range(n_claims)}
        )
        # insert from the end to keep indices valid
        for pos, claim in zip(reversed(positions), reversed(chosen)):
            sentences.insert(pos, claim)
        modified = ". ".join(sentences)

    delta = " ".join(chosen)
    return _maybe_return(modified, delta, return_delta)


def inject_omission_error(text: str, step_name: str, severity: int = 1, return_delta: bool = False):
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences) <= 2:
        return _maybe_return(text, "", return_delta)

    if severity <= 1:
        n_remove = 1
    elif severity == 2:
        n_remove = max(1, int(round(len(sentences) * 0.30)))
    else:
        n_remove = max(1, int(round(len(sentences) * 0.50)))
    n_remove = min(n_remove, len(sentences) - 1)  # keep at least one sentence

    # always keep the first sentence; sample from the rest
    candidates = list(range(1, len(sentences)))
    remove_idxs = sorted(random.sample(candidates, k=min(n_remove, len(candidates))))
    removed_sentences = [sentences[i] for i in remove_idxs]
    kept = [s for i, s in enumerate(sentences) if i not in set(remove_idxs)]
    modified = ". ".join(kept)
    delta = "REMOVED: " + " | ".join(removed_sentences)
    return _maybe_return(modified, delta, return_delta)


ERROR_TYPES = {
    "semantic": inject_semantic_error,
    "factual": inject_factual_error,
    "omission": inject_omission_error,
}
