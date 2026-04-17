"""Error injection functions.

Each injector accepts (text, step_name, severity=1, return_delta=False) and
returns either the modified text (legacy) or a tuple (modified_text,
injected_delta) when `return_delta=True`.

Severity scales:
    factual:  1 = insert 1 false claim (legacy), 2 = replace 1-2 sentences,
              3 = replace ~30% of sentences, 4 = replace ~50%
    omission: 1 = remove ~20%, 2 = remove ~40%, 3 = remove ~60%
    semantic: 1 = up to 2 substitutions (legacy), 2 = up to 5, 3 = unlimited,
              4 = unlimited + misleading framing sentence

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
    """Rank words in text by TF (proxy for TF-IDF within a single document).
    Returns list of (word, score) sorted by score descending."""
    words = re.findall(r'[a-zA-Z]{3,}', text)
    if not words:
        return []
    tf = Counter(w.lower() for w in words)
    total = sum(tf.values())
    # IDF approximation: rarer words in the text get higher weight
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

# Stronger claims for severity >= 2: replace existing sentences instead of inserting
FACTUAL_REPLACEMENTS = [
    "However, this option was officially discontinued in late 2023 due to widespread hardware failures and has no manufacturer support.",
    "Independent testing by Consumer Reports found this option scored last in its category, with a recommendation to avoid purchase.",
    "A major class-action lawsuit in 2024 revealed that the advertised specifications were fabricated, and a full product recall was issued.",
    "Recent third-party audits discovered that this product contains components banned under EU safety regulations since 2022.",
    "Multiple verified reports confirm that the latest firmware update has permanently bricked units, with no fix available.",
    "A comprehensive 2024 investigation found that performance claims were inflated by 300% compared to actual measured results.",
    "The manufacturer filed for bankruptcy in Q3 2024, ending all warranty coverage and future product development.",
    "Security researchers disclosed critical vulnerabilities that expose all user data, with the manufacturer refusing to patch.",
]


def _maybe_return(modified: str, delta: str, return_delta: bool):
    if return_delta:
        return modified, delta
    return modified


def inject_semantic_error(text: str, step_name: str, severity: int = 1, return_delta: bool = False,
                          pos_target: str | None = None, tfidf_target: str | None = None):
    if severity <= 1:
        max_subs = 2
    elif severity == 2:
        max_subs = 5
    else:
        max_subs = 10**9  # unlimited

    modified = text
    delta = ""

    # POS-targeted mode: only swap words of the specified part of speech
    if pos_target and pos_target in _POS_SWAP_TABLES:
        modified, delta = _apply_pos_targeted_swap(text, pos_target, max_subs)
    # TF-IDF-targeted mode: swap highest or lowest importance word
    elif tfidf_target in ("high", "low"):
        modified, delta = _apply_tfidf_targeted_swap(text, tfidf_target, ERROR_SUBSTITUTIONS)
    else:
        # Default behavior
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
        if swaps:
            delta = "; ".join(f"{o}->{r}" for o, r in swaps)

    if not delta:
        sentences = modified.split(". ")
        if len(sentences) > 1:
            idx = random.randint(0, len(sentences) - 1)
            insertion = "This information may be outdated or incorrect."
            sentences[idx] = insertion
            modified = ". ".join(sentences)
            delta = insertion

    # severity 4: also prepend a misleading framing sentence
    if severity >= 4:
        framing = random.choice([
            "Note: The following information is based on outdated 2019 data and may no longer be accurate.",
            "IMPORTANT: Recent investigations have found significant factual errors in the sources below.",
            "Warning: This content has been flagged for containing misleading claims by multiple fact-checkers.",
        ])
        modified = framing + " " + modified
        delta = (delta + " | FRAMING: " + framing) if delta else framing

    return _maybe_return(modified, delta, return_delta)


def inject_factual_error(text: str, step_name: str, severity: int = 1, return_delta: bool = False):
    sentences = text.split(". ")

    if severity <= 1:
        # Legacy behavior: insert 1 fake fact at midpoint
        chosen = random.sample(FAKE_FACTS, k=1)
        if len(sentences) <= 1:
            modified = text + " " + chosen[0]
        else:
            pos = len(sentences) // 2
            sentences.insert(pos, chosen[0])
            modified = ". ".join(sentences)
        return _maybe_return(modified, chosen[0], return_delta)

    # severity >= 2: replace existing sentences with plausible false claims
    if severity == 2:
        n_replace = min(2, max(1, len(sentences) - 1))
    elif severity == 3:
        n_replace = max(3, int(round(len(sentences) * 0.3)))
    else:
        n_replace = max(4, int(round(len(sentences) * 0.5)))
    n_replace = min(n_replace, len(sentences) - 1, len(FACTUAL_REPLACEMENTS))

    if len(sentences) <= 2:
        chosen = random.sample(FACTUAL_REPLACEMENTS, k=1)
        modified = text + " " + chosen[0]
        delta = "REPLACED: " + chosen[0]
    else:
        candidates = list(range(1, len(sentences)))  # keep first sentence
        replace_idxs = sorted(random.sample(candidates, k=min(n_replace, len(candidates))))
        replacements = random.sample(FACTUAL_REPLACEMENTS, k=len(replace_idxs))
        delta_parts = []
        for idx, repl in zip(replace_idxs, replacements):
            delta_parts.append(f"[{sentences[idx][:50]}...] -> [{repl[:50]}...]")
            sentences[idx] = repl
        modified = ". ".join(sentences)
        delta = "REPLACED: " + " | ".join(delta_parts)

    return _maybe_return(modified, delta, return_delta)


def inject_omission_error(text: str, step_name: str, severity: int = 1, return_delta: bool = False):
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences) <= 2:
        return _maybe_return(text, "", return_delta)

    # Proportional removal instead of fixed count
    if severity <= 1:
        n_remove = max(1, int(round(len(sentences) * 0.20)))
    elif severity == 2:
        n_remove = max(2, int(round(len(sentences) * 0.40)))
    elif severity == 3:
        n_remove = max(3, int(round(len(sentences) * 0.60)))
    else:
        n_remove = max(4, int(round(len(sentences) * 0.80)))
    n_remove = min(n_remove, len(sentences) - 1)

    candidates = list(range(1, len(sentences)))
    remove_idxs = sorted(random.sample(candidates, k=min(n_remove, len(candidates))))
    removed_sentences = [sentences[i] for i in remove_idxs]
    kept = [s for i, s in enumerate(sentences) if i not in set(remove_idxs)]
    modified = ". ".join(kept)
    delta = f"REMOVED ({len(removed_sentences)}/{len(sentences)}): " + " | ".join(removed_sentences)
    return _maybe_return(modified, delta, return_delta)


ERROR_TYPES = {
    "semantic": inject_semantic_error,
    "factual": inject_factual_error,
    "omission": inject_omission_error,
}
