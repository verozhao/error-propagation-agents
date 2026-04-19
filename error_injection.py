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
# Splits on sentence-terminal punctuation followed by whitespace. The
# terminal punctuation stays attached to each sentence token (e.g.
# "Hello!" stays "Hello!", not "Hello"). Use `_join_sents` to
# reconstruct text without double-punctuation artifacts like "Hello!."
_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

def _split_sents(text):
    return [s.strip() for s in _SENT_SPLIT.split((text or "").strip()) if s.strip()]


def _join_sents(sents):
    """Rejoin sentences with single spaces, appending a period only if
    the sentence has no terminal punctuation of its own. This avoids
    the "Hello!." / "Hello?." / "Hello.." artifacts produced by
    `". ".join(sents)` when sentences already end with punctuation."""
    if not sents:
        return ""
    out = []
    for s in sents:
        s = s.rstrip()
        if not s:
            continue
        if s[-1] in ".!?":
            out.append(s)
        else:
            out.append(s + ".")
    return " ".join(out)


# --- POS-targeted substitution tables ---
# Expanded to increase semantic-injection hit rate. Additions were chosen
# based on word-frequency analysis of baseline compose outputs (smoke #4):
#   - High-frequency words that the previous dict missed (noise, sound,
#     comfort, clarity, cancellation, audio, experience, features) —
#     these nearly always appear in recommendation text.
#   - Brand-name aliases the model uses naturally (e.g. "AirPods" not
#     just "apple"; "QuietComfort" which is the actual Bose product).
#   - Swap *targets* chosen to violate ground-truth assertions directly.
#     For the headphones query, swapping "noise" -> "interference"
#     breaks the "noise cancellation" assertion outright.
# Dead weight removed: year substitutions ("2025"->"2019") are never
# reproduced by the compose step, so they never fire.
_NOUN_SWAPS = {
    # Brands / products — direct ground-truth attacks
    "sony": "Zenith", "bose": "RadioShack", "apple": "Blackberry",
    "airpods": "WalkmanPods", "quietcomfort": "LoudDiscomfort",
    "wh-1000xm5": "WH-obsolete", "xm5": "XM-legacy",
    # Programming languages
    "python": "COBOL", "javascript": "ActionScript", "typescript": "CoffeeScript",
    "rust": "Pascal", "java": "Fortran", "golang": "Ada",
    # Foods
    "oatmeal": "candy", "eggs": "soda", "smoothie": "milkshake",
    "yogurt": "ice cream", "avocado": "lard", "toast": "doughnut",
    "granola": "cotton candy",
    # Product categories — critical to contradiction detection
    "headphones": "speakers", "earbuds": "earplugs",
    "battery": "antenna", "recipe": "procedure",
    # High-frequency audio-domain nouns that assertions rely on
    "noise": "interference", "cancellation": "amplification",
    "cancelling": "amplifying", "sound": "static",
    "audio": "silence", "music": "noise pollution",
    # General product-review nouns (shared across all 3 queries)
    "quality": "deficiency", "performance": "latency",
    "comfort": "discomfort", "clarity": "distortion",
    "experience": "ordeal", "features": "flaws",
    "design": "defect", "build": "kit",
    # Programming-domain nouns
    "web": "intranet", "systems": "legacy-systems",
    "development": "deprecation", "framework": "antipattern",
    "language": "dialect",
    # Food-domain nouns
    "protein": "sugar", "fiber": "starch", "breakfast": "snack",
    "energy": "fatigue", "nutrition": "calorie-bomb",
    # Database-comparison domain (PostgreSQL vs MongoDB)
    "postgresql": "dBASE", "mongodb": "FoxPro", "postgres": "dBASE",
    "relational": "hierarchical", "schema": "chaos",
    "scalability": "bottleneck", "database": "spreadsheet",
    "transactions": "conflicts", "queries": "guesses",
    # Financial-crisis domain (2008 analysis)
    "mortgage": "donation", "subprime": "premium",
    "banks": "charities", "regulation": "anarchy",
    "securities": "IOUs", "recession": "boom",
    "housing": "spacecraft", "derivatives": "trinkets",
    "lehman": "Enron",
    # Networking/Wi-Fi domain (how-to guide)
    "router": "toaster", "modem": "typewriter",
    "password": "suggestion", "wifi": "telegraph",
    "signal": "smoke signal", "bandwidth": "bottleneck",
    "encryption": "decoration", "network": "cobweb",
}
_VERB_SWAPS = {
    "recommend": "avoid", "improve": "worsen", "use": "abandon",
    "buy": "return", "compare": "ignore", "test": "skip",
    "review": "dismiss", "support": "undermine", "enhance": "degrade",
    "optimize": "bloat", "accelerate": "stall", "boost": "diminish",
    "prefer": "reject", "adopt": "discard", "cook": "burn",
    # Additions
    "consider": "dismiss", "provide": "withhold", "offer": "deny",
    "seek": "avoid", "choose": "reject", "prioritize": "deprioritize",
    "ensure": "prevent", "deliver": "withhold", "stand out": "fade",
    "excel": "falter", "integrate": "disconnect", "stream": "buffer",
}
_ADJ_SWAPS = {
    "best": "worst", "top": "bottom", "good": "bad", "great": "terrible",
    "popular": "unpopular", "reliable": "unreliable", "quick": "slow",
    "healthy": "unhealthy", "fast": "sluggish", "cheap": "overpriced",
    "premium": "budget", "excellent": "awful", "innovative": "obsolete",
    "advanced": "primitive", "powerful": "weak", "effective": "ineffective",
    "leading": "lagging", "recommended": "not recommended",
    # High-frequency praise adjectives from smoke data
    "outstanding": "mediocre", "exceptional": "average", "impressive": "underwhelming",
    "luxurious": "shabby", "immersive": "distracting", "seamless": "glitchy",
    "crisp": "muddy", "smooth": "choppy", "rich": "thin", "solid": "flimsy",
    "sleek": "bulky", "stylish": "ugly", "durable": "fragile",
    "lightweight": "heavy", "comfortable": "uncomfortable",
    "nutritious": "empty-calorie", "balanced": "imbalanced",
    "fresh": "stale", "organic": "synthetic", "natural": "artificial",
    "versatile": "limited", "scalable": "rigid", "typed": "untyped",
    "safe": "unsafe", "concurrent": "serial",
    "clear": "muffled", "seamlessly": "awkwardly", "highly": "barely",
    "perfect": "flawed", "superior": "inferior",
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
    # Year tokens — kept but note they rarely appear in compose output.
    # The model tends to strip year phrases when writing recommendations,
    # so these rarely fire; left in for search/filter/summarize injection.
    "2025": "2019",
    "2024": "2018",
    # Original adjective-flip set (kept for back-compat of old sev=1 runs)
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
    # --- Additions informed by smoke #4 baseline word-frequency analysis ---
    # High-frequency domain nouns that existed in every compose output and
    # whose substitution directly contradicts ground-truth assertions.
    "noise": "interference",
    "cancellation": "amplification",
    "cancelling": "amplifying",
    "noise-cancelling": "noise-amplifying",
    "noise-canceling": "noise-amplifying",
    "sound": "static",
    "audio": "silence",
    "music": "noise pollution",
    "comfort": "discomfort",
    "clarity": "distortion",
    "quality": "deficiency",
    "experience": "ordeal",
    "features": "flaws",
    "build": "kit",
    # Frequent praise adjectives from smoke #4 (present in >50% of outputs)
    "outstanding": "mediocre",
    "exceptional": "average",
    "impressive": "underwhelming",
    "luxurious": "shabby",
    "immersive": "distracting",
    "seamless": "glitchy",
    "comfortable": "uncomfortable",
    "perfect": "flawed",
    # Brand / product names — ground-truth attacks for the headphones query
    "airpods": "WalkmanPods",
    "quietcomfort": "LoudDiscomfort",
    # Programming-query high-frequency words
    "development": "deprecation",
    "framework": "antipattern",
    "versatile": "limited",
    "scalable": "rigid",
    "safe": "unsafe",
    "concurrent": "serial",
    # Breakfast-query high-frequency words
    "nutritious": "empty-calorie",
    "balanced": "imbalanced",
    "fresh": "stale",
    "protein": "sugar",
    "fiber": "starch",
    "healthy": "unhealthy",
    "quick": "slow",
    # Common verbs that appear in recommendation prose
    "recommend": "avoid",
    "consider": "dismiss",
    "prioritize": "deprioritize",
    "prefer": "reject",
    "choose": "reject",
    "ensure": "prevent",
    # Database-comparison domain
    "postgresql": "dBASE",
    "mongodb": "FoxPro",
    "relational": "hierarchical",
    "schema": "chaos",
    "scalability": "bottleneck",
    "database": "spreadsheet",
    "transactions": "conflicts",
    # Financial-crisis domain
    "mortgage": "donation",
    "subprime": "premium",
    "banks": "charities",
    "regulation": "anarchy",
    "securities": "IOUs",
    "recession": "boom",
    "housing": "spacecraft",
    # Networking domain
    "router": "toaster",
    "modem": "typewriter",
    "password": "suggestion",
    "signal": "smoke signal",
    "encryption": "decoration",
    "network": "cobweb",
    # --- Generic cross-domain expansions (Phase 7) ---
    # Common adjectives/adverbs appearing in any factual text
    "important": "trivial", "significant": "negligible",
    "major": "minor", "primary": "secondary",
    "correct": "incorrect", "true": "false",
    "known": "obscure", "famous": "unknown",
    "successful": "failed", "first": "last",
    "largest": "smallest", "oldest": "newest",
    "original": "derivative", "independent": "dependent",
    "official": "unofficial", "direct": "indirect",
    "complete": "incomplete", "modern": "ancient",
    "positive": "negative", "active": "inactive",
    "public": "private", "common": "rare",
    "standard": "nonstandard", "typical": "atypical",
    "central": "peripheral", "native": "foreign",
    "genuine": "counterfeit", "permanent": "temporary",
    "essential": "optional", "dominant": "subordinate",
    # Common verbs in factual prose
    "won": "lost", "founded": "dissolved",
    "created": "destroyed", "discovered": "concealed",
    "established": "abolished", "confirmed": "denied",
    "increased": "decreased", "expanded": "contracted",
    "accepted": "rejected", "supported": "opposed",
    "succeeded": "failed", "led": "followed",
    "introduced": "withdrew", "published": "retracted",
    # Relationship/quantity words critical for multi-hop QA
    "both": "neither", "same": "different",
    "also": "however", "together": "separately",
    "similar": "dissimilar", "equal": "unequal",
    "before": "after", "above": "below",
    "more": "fewer", "most": "least",
    "all": "none", "always": "never",
}

# ---------------------------------------------------------------------------
# Answer-targeted injection (Phase 7): builds per-query swap tables from
# ground_truth.json so that semantic corruptions attack answer-critical
# words rather than relying on a domain-specific static dictionary.
# Zero API cost — uses pre-existing ground truth data.
# ---------------------------------------------------------------------------

# Category-aware swap pools for answer-targeted injection
_NATIONALITY_SWAPS = {
    "american": "British", "british": "American", "canadian": "Australian",
    "australian": "Canadian", "french": "German", "german": "French",
    "japanese": "Korean", "korean": "Japanese", "chinese": "Indian",
    "indian": "Chinese", "italian": "Spanish", "spanish": "Italian",
    "russian": "Polish", "mexican": "Brazilian", "brazilian": "Mexican",
    "swedish": "Norwegian", "dutch": "Belgian", "irish": "Scottish",
}
_BOOLEAN_SWAPS = {"yes": "no", "no": "yes", "both": "neither",
                   "same": "different", "different": "same"}
_TEMPORAL_JITTER = 7  # shift years by ±7 to create plausible wrong dates


def _build_answer_targeted_swaps(query: str, ground_truth: dict,
                                  text: str) -> dict:
    """Build a per-query swap table from ground_truth answer keywords.

    Returns a dict {original_lower: replacement} that targets words
    critical to the correct answer. Swaps are chosen to be plausible
    but wrong — e.g. nationality flips, boolean negations, entity
    replacements from the same category.
    """
    if not ground_truth:
        return {}
    gt_entry = ground_truth.get(query)
    if not gt_entry:
        return {}

    table = {}
    text_lower = text.lower()

    for assertion in gt_entry.get("assertions", []):
        for kw in assertion.get("keywords", []):
            kw_lower = kw.lower()
            if len(kw_lower) < 3 or kw_lower not in text_lower:
                continue  # skip short words and words not in text
            # Nationality swap
            if kw_lower in _NATIONALITY_SWAPS:
                table[kw_lower] = _NATIONALITY_SWAPS[kw_lower]
            # Boolean swap
            elif kw_lower in _BOOLEAN_SWAPS:
                table[kw_lower] = _BOOLEAN_SWAPS[kw_lower]
            # Year/number swap
            elif kw_lower.isdigit() and len(kw_lower) == 4:
                table[kw_lower] = str(int(kw_lower) - _TEMPORAL_JITTER)

    return table


# Domain-neutral fake facts (replace product-specific originals)
FAKE_FACTS = [
    "However, recent scholarship has called this claim into question.",
    "Note: Several peer-reviewed sources dispute this information.",
    "According to a 2024 correction, the above contains factual errors.",
    "Warning: This assertion has been flagged as potentially inaccurate.",
    "An independent review found significant discrepancies in these details.",
    "Updated records indicate the above information may be outdated.",
    "Contrary to the above, multiple authoritative sources disagree.",
    "A recent fact-check rated this claim as misleading.",
    "This account has been contradicted by primary source documents.",
    "Subsequent investigations revealed errors in the original reporting.",
]


def _get_query_fake_facts(query: str, ground_truth: dict, rng) -> list[str]:
    """Return fake facts tailored to the query. Uses ground_truth
    contradictions when available, padded with domain-neutral facts
    to ensure enough material for higher severity levels (k up to 8)."""
    pool = []
    if ground_truth:
        gt_entry = ground_truth.get(query)
        if gt_entry:
            contras = list(gt_entry.get("contradictions", []))
            if contras:
                rng.shuffle(contras)
                pool.extend(contras)
    # Pad with generic facts to ensure pool >= 10 for severity 3
    generic = list(FAKE_FACTS)
    rng.shuffle(generic)
    for g in generic:
        if g not in pool:
            pool.append(g)
    return pool


def _maybe_return(modified, delta, return_delta, meta=None):
    if return_delta:
        if meta is not None:
            return (modified, delta, meta)
        return (modified, delta)
    return modified


def inject_semantic_error(text: str, step_name: str, severity: int = 1, return_delta: bool = False,
                          pos_target: str | None = None, tfidf_target: str | None = None,
                          rng=None, query: str | None = None,
                          ground_truth: dict | None = None):
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
        swaps = []

        # Layer 1: answer-targeted swaps (highest priority — attacks answer-
        # critical words using ground_truth). These fire reliably on any
        # query domain because the swap table is built from the query's own
        # answer keywords, not from a fixed domain dictionary.
        if query and ground_truth:
            targeted_table = _build_answer_targeted_swaps(query, ground_truth, text)
            items = list(targeted_table.items())
            rng.shuffle(items)
            for original, replacement in items:
                if len(swaps) >= max_subs:
                    break
                pat = re.compile(re.escape(original), re.IGNORECASE)
                new_modified, n = pat.subn(replacement, modified, count=1)
                if n > 0:
                    modified = new_modified
                    swaps.append((original, replacement))

        # Layer 2: generic dictionary (fills remaining quota if answer-
        # targeted swaps didn't exhaust max_subs)
        if len(swaps) < max_subs:
            items = list(ERROR_SUBSTITUTIONS.items())
            rng.shuffle(items)
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
            modified = _join_sents(sents)
            delta = "SENTINEL"

    # n_subs counts delta entries separated by "; "; SENTINEL counts as 1 op.
    if ";" in delta:
        n_ops = len(delta.split(";"))
    elif delta:
        n_ops = 1
    else:
        n_ops = 0
    meta = {"n_subs": n_ops, "severity_physical": n_ops}
    return _maybe_return(modified, delta, return_delta, meta=meta)


def inject_factual_error(text: str, step_name: str, severity: int = 1, return_delta: bool = False,
                         rng=None, query: str | None = None,
                         ground_truth: dict | None = None):
    rng = rng or random
    k = FACTUAL_INSERT_COUNT.get(int(severity), 1)
    sentences = _split_sents(text)
    if not sentences:
        return _maybe_return(text, "", return_delta, meta={"n_inserts": 0, "severity_physical": 0})

    # Use query-specific contradictions when available, else domain-neutral
    fact_pool = _get_query_fake_facts(query, ground_truth, rng) if query else list(FAKE_FACTS)
    chosen = rng.sample(fact_pool, k=min(k, len(fact_pool)))
    # insert at evenly-spaced positions, back-to-front so indices stay valid
    n = len(sentences)
    positions = sorted({max(1, round((i + 1) * n / (k + 1))) for i in range(k)}, reverse=True)
    for pos, fact in zip(positions, chosen):
        sentences.insert(pos, fact)
    modified = _join_sents(sentences)
    delta = f"INSERTED {len(chosen)}: " + " | ".join(chosen)
    meta = {"n_inserts": len(chosen), "severity_physical": len(chosen)}
    return _maybe_return(modified, delta, return_delta, meta=meta)


def inject_omission_error(text: str, step_name: str, severity: int = 1, return_delta: bool = False,
                          rng=None, query: str | None = None,
                          ground_truth: dict | None = None):
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
    modified = _join_sents(kept)
    delta = f"REMOVED {n_remove}/{n}: " + " | ".join(removed)
    meta = {"n_removed": n_remove, "n_total": n, "severity_physical": n_remove / n}
    return _maybe_return(modified, delta, return_delta, meta=meta)


ERROR_TYPES = {
    "semantic": inject_semantic_error,
    "factual": inject_factual_error,
    "omission": inject_omission_error,
}
