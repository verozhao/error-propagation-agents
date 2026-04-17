"""Error injection functions — unified monotonic severity parameterization.

Severity encodes a *physical* quantity, logged alongside the integer:

| Error type | Mechanism                           | Sev1 | Sev2 | Sev3 |
|------------|-------------------------------------|------|------|------|
| factual    | Insert K fake facts (never replace) | K=1  | K=2  | K=8  |
| omission   | Remove fraction rho of sentences    | 0.10 | 0.25 | 0.75 |
| semantic   | Apply S polarity substitutions      | S=1  | S=2  | S=8  |

POS-targeted injection (pos_target parameter):
    "noun" = only corrupt nouns/proper nouns
    "verb" = only corrupt verbs
    "adj"  = only corrupt adjectives
    None   = default (any word, original behavior)

TF-IDF-targeted injection (tfidf_target parameter):
    "high" = corrupt highest TF-IDF word first
    "low"  = corrupt lowest TF-IDF word first
    None   = default (original behavior)
"""

import random
import re
import math
from collections import Counter

# --- Monotonic severity tables ---
FACTUAL_INSERT_COUNT = {1: 1, 2: 2, 3: 8}
OMISSION_FRACTION    = {1: 0.10, 2: 0.25, 3: 0.75}
SEMANTIC_SUB_COUNT   = {1: 1, 2: 2, 3: 8}

# --- Sentence splitter (replaces every text.split(". ")) ---
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _split_sents(text):
    return [s.strip() for s in _SENT_SPLIT.split((text or "").strip()) if s.strip()]


# --- POS-targeted substitution tables ---
_NOUN_SWAPS = {
    "sony": "Zenith", "bose": "RadioShack", "apple": "Blackberry",
    "python": "COBOL", "javascript": "ActionScript", "typescript": "CoffeeScript",
    "rust": "Pascal", "java": "Fortran", "go": "Ada",
    "oatmeal": "candy", "eggs": "soda", "smoothie": "milkshake",
    "yogurt": "ice cream", "avocado": "lard",
    "headphones": "speakers", "battery": "antenna", "recipe": "procedure",
    "performance": "latency", "quality": "deficiency",
}
_VERB_SWAPS = {
    "recommend": "avoid", "improve": "worsen", "use": "abandon",
    "buy": "return", "compare": "ignore", "test": "skip",
    "review": "dismiss", "support": "undermine", "enhance": "degrade",
    "optimize": "bloat", "accelerate": "stall", "boost": "diminish",
    "prefer": "reject", "adopt": "discard", "cook": "burn",
}
_ADJ_SWAPS = {
    "best": "worst", "top": "bottom", "good": "bad", "great": "terrible",
    "popular": "unpopular", "reliable": "unreliable", "quick": "slow",
    "healthy": "unhealthy", "fast": "sluggish", "cheap": "overpriced",
    "premium": "budget", "excellent": "awful", "innovative": "obsolete",
    "advanced": "primitive", "powerful": "weak", "effective": "ineffective",
    "leading": "lagging", "recommended": "not recommended",
}
_POS_SWAP_TABLES = {"noun": _NOUN_SWAPS, "verb": _VERB_SWAPS, "adj": _ADJ_SWAPS}


def _tfidf_rank_words(text: str) -> list[tuple[str, float]]:
    """Rank words in text by TF (proxy for TF-IDF within a single document)."""
    words = re.findall(r'[a-zA-Z]{3,}', text)
    if not words:
        return []
    tf = Counter(w.lower() for w in words)
    total = sum(tf.values())
    scored = []
    for word, count in tf.items():
        tf_score = count / total
        idf_approx = math.log(total / count) + 1
        scored.append((word, tf_score * idf_approx))
    return sorted(scored, key=lambda x: x[1], reverse=True)


def _apply_pos_targeted_swap(text: str, pos_target: str, max_subs: int) -> tuple[str, str]:
    """Swap words matching a specific POS category."""
    table = _POS_SWAP_TABLES.get(pos_target, {})
    modified = text
    swaps = []
    for original, replacement in table.items():
        if len(swaps) >= max_subs:
            break
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        new_modified, n = pattern.subn(replacement, modified, count=1)
        if n > 0:
            modified = new_modified
            swaps.append(f"{original}->{replacement}")
    delta = f"POS[{pos_target}]: " + "; ".join(swaps) if swaps else ""
    return modified, delta


def _apply_tfidf_targeted_swap(text: str, tfidf_target: str, swap_table: dict) -> tuple[str, str]:
    """Swap the highest or lowest TF-IDF word that has a substitution available."""
    ranked = _tfidf_rank_words(text)
    if tfidf_target == "low":
        ranked = list(reversed(ranked))

    modified = text
    swap_table_lower = {k.lower(): v for k, v in swap_table.items()}
    for word, score in ranked:
        if word in swap_table_lower:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            modified = pattern.sub(swap_table_lower[word], modified, count=1)
            delta = f"TFIDF[{tfidf_target}]: {word}(score={score:.3f})->{swap_table_lower[word]}"
            return modified, delta
    return text, ""


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
    "excellent": "terrible",
    "superior": "inferior",
    "innovative": "obsolete",
    "advanced": "primitive",
    "powerful": "weak",
    "fastest": "slowest",
    "highly rated": "poorly rated",
    "strongly": "weakly",
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


def _maybe_return(modified, delta, return_delta, meta=None):
    if return_delta:
        if meta is not None:
            return (modified, delta, meta)
        return (modified, delta)
    return modified


def inject_semantic_error(text: str, step_name: str, severity: int = 1, return_delta: bool = False,
                          pos_target: str | None = None, tfidf_target: str | None = None,
                          rng=None):
    rng = rng or random
    max_subs = SEMANTIC_SUB_COUNT.get(int(severity), 1)

    modified = text
    delta = ""

    # POS-targeted mode
    if pos_target and pos_target in _POS_SWAP_TABLES:
        modified, delta = _apply_pos_targeted_swap(text, pos_target, max_subs)
    # TF-IDF-targeted mode
    elif tfidf_target in ("high", "low"):
        modified, delta = _apply_tfidf_targeted_swap(text, tfidf_target, ERROR_SUBSTITUTIONS)
    else:
        # Default: iterate dictionary, each match counts toward max_subs, ONE occurrence per match
        swaps = []
        items = list(ERROR_SUBSTITUTIONS.items())
        rng.shuffle(items)  # avoid dictionary-order bias
        for original, replacement in items:
            if len(swaps) >= max_subs:
                break
            pat = re.compile(re.escape(original), re.IGNORECASE)
            new_modified, n = pat.subn(replacement, modified, count=1)
            if n > 0:
                modified = new_modified
                swaps.append((original, replacement))
        if swaps:
            delta = "; ".join(f"{o}->{r}" for o, r in swaps)

    if not delta:
        # Fallback: sentinel substitution (not a framing prepend!)
        sents = _split_sents(modified)
        if len(sents) > 1:
            idx = rng.randrange(len(sents))
            sents[idx] = "This information may be outdated or incorrect."
            modified = ". ".join(sents)
            delta = "SENTINEL"

    meta = {"n_subs": len(delta.split(";")) if ";" in delta else (1 if delta else 0),
            "severity_physical": len(delta.split(";")) if ";" in delta else (1 if delta and delta != "SENTINEL" else 0)}
    return _maybe_return(modified, delta, return_delta, meta=meta)


def inject_factual_error(text: str, step_name: str, severity: int = 1, return_delta: bool = False,
                         rng=None):
    rng = rng or random
    k = FACTUAL_INSERT_COUNT.get(int(severity), 1)
    sentences = _split_sents(text)
    if not sentences:
        return _maybe_return(text, "", return_delta, meta={"n_inserts": 0, "severity_physical": 0})

    chosen = rng.sample(FAKE_FACTS, k=min(k, len(FAKE_FACTS)))
    # insert at evenly-spaced positions, back-to-front so indices stay valid
    n = len(sentences)
    positions = sorted({max(1, round((i + 1) * n / (k + 1))) for i in range(k)}, reverse=True)
    for pos, fact in zip(positions, chosen):
        sentences.insert(pos, fact)
    modified = ". ".join(sentences)
    delta = f"INSERTED {len(chosen)}: " + " | ".join(chosen)
    meta = {"n_inserts": len(chosen), "severity_physical": len(chosen)}
    return _maybe_return(modified, delta, return_delta, meta=meta)


def inject_omission_error(text: str, step_name: str, severity: int = 1, return_delta: bool = False,
                          rng=None):
    rng = rng or random
    rho = OMISSION_FRACTION.get(int(severity), 0.10)
    sentences = _split_sents(text)
    n = len(sentences)
    if n <= 2:
        return _maybe_return(text, "", return_delta,
                             meta={"n_removed": 0, "n_total": n, "severity_physical": 0.0})
    n_remove = int(round(rho * n))
    n_remove = min(n_remove, n - 2)
    if n_remove < 1:
        return _maybe_return(text, "", return_delta,
                             meta={"n_removed": 0, "n_total": n, "severity_physical": 0.0})
    eligible = list(range(1, n))
    remove_idxs = set(rng.sample(eligible, k=n_remove))
    kept = [s for i, s in enumerate(sentences) if i not in remove_idxs]
    removed = [sentences[i] for i in sorted(remove_idxs)]
    modified = ". ".join(kept)
    delta = f"REMOVED {n_remove}/{n}: " + " | ".join(removed)
    meta = {"n_removed": n_remove, "n_total": n, "severity_physical": n_remove / n}
    return _maybe_return(modified, delta, return_delta, meta=meta)


ERROR_TYPES = {
    "semantic": inject_semantic_error,
    "factual": inject_factual_error,
    "omission": inject_omission_error,
}
