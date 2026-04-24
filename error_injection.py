"""Error injection functions — FAVA-grounded taxonomy from RAGTruth empirical data.

Four error operations validated on 10,725 RAGTruth spans (ACL 2024):

| Error type     | RAGTruth %  | Mechanism                                         |
|----------------|-------------|----------------------------------------------------|
| entity         | 51.0%       | Substitute a key entity with a wrong same-type one |
| invented       | 28.0%       | Insert a fabricated but plausible claim            |
| unverifiable   | 13.8%       | Add a specific but unsourced/unverifiable claim    |
| contradictory  |  6.9%       | Replace a factual sentence with its negation       |

Each function supports two modes:
  - Rule-based (default): fast, deterministic, zero extra cost
  - LLM-based: pass `injection_model_fn` for natural-sounding errors

Severity controls injection intensity (number of operations per call).
Continuous severity (embedding cosine distance) is computed post-hoc.

Reference:
  - RAGTruth: Niu et al., ACL 2024  (label distribution, span statistics)
  - FAVA:     Mishra et al., COLM 2024 (six-type taxonomy, reduced to four)
"""

import json as _json
import math
import os as _os
import random
import re
from collections import Counter

# ============================================================
# Sentence utilities
# ============================================================

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')


def _split_sents(text):
    return [s.strip() for s in _SENT_SPLIT.split((text or "").strip()) if s.strip()]


def _join_sents(sents):
    """Rejoin sentences, appending a period only when needed."""
    out = []
    for s in sents:
        s = s.rstrip()
        if not s:
            continue
        out.append(s if s[-1] in ".!?" else s + ".")
    return " ".join(out)


def _maybe_return(modified, delta, return_delta, meta=None):
    if return_delta:
        return (modified, delta, meta) if meta is not None else (modified, delta)
    return modified


# ============================================================
# Taxonomy weights (loaded from JSON or hardcoded defaults)
# ============================================================

_TAXONOMY_PATH = _os.path.join(_os.path.dirname(__file__), "error_operation_taxonomy.json")
_INJECTION_WEIGHTS = {
    "search":    {"entity": 0.42, "invented": 0.44, "unverifiable": 0.08, "contradictory": 0.06},
    "filter":    {"entity": 0.42, "invented": 0.44, "unverifiable": 0.08, "contradictory": 0.06},
    "summarize": {"entity": 0.39, "invented": 0.40, "unverifiable": 0.11, "contradictory": 0.10},
    "compose":   {"entity": 0.58, "invented": 0.20, "unverifiable": 0.16, "contradictory": 0.06},
    "verify":    {"entity": 0.51, "invented": 0.28, "unverifiable": 0.14, "contradictory": 0.07},
}
_VALID_TYPES = {"entity", "invented", "unverifiable", "contradictory"}
if _os.path.exists(_TAXONOMY_PATH):
    try:
        with open(_TAXONOMY_PATH) as _f:
            _tax = _json.load(_f)
        if "injection_weights_by_step" in _tax:
            for step, weights in _tax["injection_weights_by_step"].items():
                # Filter to valid types only, renormalize
                filtered = {k: v for k, v in weights.items() if k in _VALID_TYPES}
                if filtered:
                    total = sum(filtered.values())
                    _INJECTION_WEIGHTS[step] = {k: v / total for k, v in filtered.items()}
    except Exception:
        pass


# ============================================================
# Answer-targeted swap generation (for entity injection)
# ============================================================

_NATIONALITY_SWAPS = {
    "american": "British", "british": "American", "canadian": "Australian",
    "australian": "Canadian", "french": "German", "german": "French",
    "japanese": "Korean", "korean": "Japanese", "chinese": "Indian",
    "indian": "Chinese", "italian": "Spanish", "spanish": "Italian",
    "russian": "Polish", "mexican": "Brazilian", "brazilian": "Mexican",
    "swedish": "Norwegian", "dutch": "Belgian", "irish": "Scottish",
}
_BOOLEAN_SWAPS = {
    "yes": "no", "no": "yes", "both": "neither",
    "same": "different", "different": "same",
}
_TEMPORAL_JITTER = 7


def _build_answer_targeted_swaps(query: str, ground_truth: dict, text: str) -> dict:
    """Build a per-query swap table from ground_truth answer keywords."""
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
                continue
            if kw_lower in _NATIONALITY_SWAPS:
                table[kw_lower] = _NATIONALITY_SWAPS[kw_lower]
            elif kw_lower in _BOOLEAN_SWAPS:
                table[kw_lower] = _BOOLEAN_SWAPS[kw_lower]
            elif kw_lower.isdigit() and len(kw_lower) == 4:
                table[kw_lower] = str(int(kw_lower) - _TEMPORAL_JITTER)
    return table


# Generic entity swap table — domain-neutral, covers common entity categories.
_ENTITY_SWAPS = {
    # Proper nouns / brands (cross-domain)
    "sony": "Panasonic", "bose": "JBL", "apple": "Samsung",
    "google": "Yahoo", "microsoft": "Oracle", "amazon": "Alibaba",
    "tesla": "Rivian", "nvidia": "AMD", "intel": "Qualcomm",
    # Programming languages
    "python": "COBOL", "javascript": "ActionScript", "typescript": "CoffeeScript",
    "rust": "Pascal", "java": "Fortran", "golang": "Ada",
    # Databases
    "postgresql": "dBASE", "mongodb": "FoxPro", "mysql": "Access",
    "redis": "Memcached", "sqlite": "FoxPro",
    # Quantities — adjective/comparative swaps that change entity attributes
    "first": "last", "largest": "smallest", "oldest": "newest",
    "highest": "lowest", "fastest": "slowest", "best": "worst",
    "most": "least", "major": "minor", "primary": "secondary",
    # Temporal
    "2025": "2019", "2024": "2018", "2023": "2017",
    # Boolean / relational
    "true": "false", "correct": "incorrect", "positive": "negative",
    "before": "after", "above": "below", "more": "fewer",
}


# ============================================================
# LLM prompt templates
# ============================================================

ENTITY_INJECTION_PROMPT = """You are an error injection tool. Given the following text, introduce exactly one entity substitution error.

Rules:
1. Find one key entity (person name, place, number, date, or organization)
2. Replace it with a WRONG entity of the same type (e.g., wrong person name, wrong number)
3. The replacement must look plausible — same category, similar format
4. Change NOTHING else — grammar, structure, and all other facts stay identical
5. Output ONLY a JSON object with exactly these fields:

{{"modified_text": "...", "original_span": "...", "replacement_span": "...", "error_type": "entity"}}

Text to modify:
{text}"""

INVENTED_INJECTION_PROMPT = """You are an error injection tool. Given the following text, insert one fabricated detail that does NOT exist in the original.

Rules:
1. Add one specific but entirely made-up detail (a fake statistic, a non-existent person, a fabricated event)
2. The fabricated detail must sound plausible and blend naturally with the surrounding text
3. Insert it as a new clause or sentence — do NOT replace existing content
4. Keep everything else unchanged
5. Output ONLY a JSON object with exactly these fields:

{{"modified_text": "...", "inserted_text": "...", "insertion_position": "after sentence N", "error_type": "invented"}}

Text to modify:
{text}"""

UNVERIFIABLE_INJECTION_PROMPT = """You are an error injection tool. Given the following text, add one unverifiable claim.

Rules:
1. Insert a specific, factual-sounding statement that CANNOT be verified from the source text
2. Examples: an unsourced percentage, an unattributed quote, a private detail, a vague temporal claim
3. It should NOT directly contradict the text — just add information that has no source
4. Keep the original text intact; insert the new claim naturally
5. Output ONLY a JSON object with exactly these fields:

{{"modified_text": "...", "inserted_text": "...", "error_type": "unverifiable"}}

Text to modify:
{text}"""

CONTRADICTORY_INJECTION_PROMPT = """You are an error injection tool. Given the following text, replace one factual sentence with its contradiction.

Rules:
1. Pick one sentence that states a clear fact
2. Replace it with a sentence that says the OPPOSITE (negate the core claim)
3. The contradictory sentence must be grammatically correct and self-consistent
4. Change ONLY that one sentence — all other text stays identical
5. Output ONLY a JSON object with exactly these fields:

{{"modified_text": "...", "original_sentence": "...", "contradictory_sentence": "...", "error_type": "contradictory"}}

Text to modify:
{text}"""


def _parse_llm_injection(response: str, original_text: str):
    """Try to parse LLM JSON response; return (modified, delta, meta) or None."""
    try:
        clean = response.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean.rsplit("```", 1)[0]
        clean = clean.strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()

        data = _json.loads(clean)
        modified = data.get("modified_text", "")
        if not modified or modified == original_text:
            return None

        error_type = data.get("error_type", "unknown")
        parts = []
        for key in ["original_span", "replacement_span", "inserted_text",
                     "original_sentence", "contradictory_sentence", "insertion_position"]:
            if key in data:
                parts.append(f"{key}={data[key]}")
        delta = f"LLM[{error_type}]: " + "; ".join(parts) if parts else f"LLM[{error_type}]"
        meta = {"error_type": error_type, "injection_method": "llm",
                **{k: v for k, v in data.items() if k != "modified_text"}}
        return modified, delta, meta
    except Exception:
        return None


# ============================================================
# Injection functions
# ============================================================

def inject_entity_error(text: str, step_name: str, severity: int = 1,
                        return_delta: bool = False, rng=None,
                        query: str | None = None,
                        ground_truth: dict | None = None,
                        injection_model_fn=None, **kwargs):
    """Entity substitution: replace a key entity with a wrong same-type entity.

    RAGTruth: 51.0% of all hallucination spans. Avg span: 7.4 words.
    Cross-tab: 86% from Evident Conflict, 84% from Subtle Conflict.
    """
    rng = rng or random
    n_subs = min(severity, 3)

    # --- LLM mode ---
    if injection_model_fn is not None:
        try:
            resp = injection_model_fn(ENTITY_INJECTION_PROMPT.format(text=text))
            parsed = _parse_llm_injection(resp, text)
            if parsed:
                return _maybe_return(*parsed, return_delta)
        except Exception:
            pass

    # --- Rule-based mode ---
    modified = text
    swaps = []

    # Layer 1: ground-truth answer keyword attacks
    if query and ground_truth:
        targeted = _build_answer_targeted_swaps(query, ground_truth, text)
        items = list(targeted.items())
        rng.shuffle(items)
        for orig, repl in items:
            if len(swaps) >= n_subs:
                break
            pat = re.compile(re.escape(orig), re.IGNORECASE)
            new_text, n = pat.subn(repl, modified, count=1)
            if n > 0:
                modified = new_text
                swaps.append((orig, repl))

    # Layer 2: generic entity swap table
    if len(swaps) < n_subs:
        items = list(_ENTITY_SWAPS.items())
        rng.shuffle(items)
        for orig, repl in items:
            if len(swaps) >= n_subs:
                break
            pat = re.compile(re.escape(orig), re.IGNORECASE)
            new_text, n = pat.subn(repl, modified, count=1)
            if n > 0:
                modified = new_text
                swaps.append((orig, repl))

    delta = "ENTITY: " + "; ".join(f"{o}->{r}" for o, r in swaps) if swaps else ""

    # Fallback: swap a number if any exists
    if not delta:
        numbers = re.findall(r'\b\d{2,}\b', modified)
        if numbers:
            target = rng.choice(numbers)
            wrong = str(int(target) + rng.choice([-7, -3, 3, 7, 13]))
            modified = modified.replace(target, wrong, 1)
            delta = f"ENTITY: {target}->{wrong}"

    n_ops = len(swaps) if swaps else (1 if delta else 0)
    meta = {"error_type": "entity", "n_subs": n_ops,
            "severity_physical": n_ops, "injection_method": "rule"}
    return _maybe_return(modified, delta, return_delta, meta=meta)


# --- Invented detail pools ---
_INVENTED_CLAIMS = [
    "According to a 2023 internal review, this has been independently verified by three separate institutions.",
    "Studies conducted at the University of Cambridge in 2022 confirmed these findings with 94.7% confidence.",
    "The methodology was originally developed by Dr. Heinz Mueller at ETH Zurich in 2019.",
    "An independent audit by Deloitte found similar results across 47 countries.",
    "This approach was first proposed in a landmark 2018 paper published in Nature.",
    "Internal documents obtained through FOIA requests corroborate this analysis.",
    "A longitudinal study tracking 12,500 participants over 8 years supports this conclusion.",
    "The framework has since been adopted by over 200 organizations in the Fortune 500.",
    "Peer review by the International Standards Organization rated this methodology as 'gold standard'.",
    "Cross-validation with the European Research Council dataset shows 97.2% agreement.",
    "This was subsequently confirmed by a meta-analysis of 34 independent studies.",
    "A report by the World Economic Forum cited this as one of the top emerging trends of 2024.",
    "Research from MIT's Computer Science and Artificial Intelligence Laboratory reached identical conclusions.",
    "The United Nations Development Programme published corroborating data in their 2023 annual report.",
    "A randomized controlled trial with 8,400 subjects published in The Lancet supports this finding.",
]


def inject_invented_error(text: str, step_name: str, severity: int = 1,
                          return_delta: bool = False, rng=None,
                          query: str | None = None,
                          ground_truth: dict | None = None,
                          injection_model_fn=None, **kwargs):
    """Invented detail: insert a fabricated but plausible claim.

    RAGTruth: 28.0% of all spans. Avg span: 13.8 words.
    Cross-tab: 64% from Evident Baseless Info.
    """
    rng = rng or random
    k = min(severity, 3)

    if injection_model_fn is not None:
        try:
            resp = injection_model_fn(INVENTED_INJECTION_PROMPT.format(text=text))
            parsed = _parse_llm_injection(resp, text)
            if parsed:
                return _maybe_return(*parsed, return_delta)
        except Exception:
            pass

    sentences = _split_sents(text)
    if not sentences:
        return _maybe_return(text, "", return_delta,
                             meta={"error_type": "invented", "n_inserts": 0,
                                   "severity_physical": 0, "injection_method": "rule"})

    chosen = rng.sample(_INVENTED_CLAIMS, k=min(k, len(_INVENTED_CLAIMS)))
    n = len(sentences)
    positions = sorted({max(1, round((i + 1) * n / (k + 1))) for i in range(k)}, reverse=True)
    for pos, claim in zip(positions, chosen):
        sentences.insert(pos, claim)

    modified = _join_sents(sentences)
    delta = f"INVENTED {len(chosen)}: " + " | ".join(chosen)
    meta = {"error_type": "invented", "n_inserts": len(chosen),
            "severity_physical": len(chosen), "injection_method": "rule"}
    return _maybe_return(modified, delta, return_delta, meta=meta)


# --- Unverifiable fragment pools ---
_UNVERIFIABLE_FRAGMENTS = [
    "reportedly",
    "according to some estimates",
    "sources suggest that",
    "approximately 73% of cases show that",
    "in most documented instances",
    "as widely reported",
    "typically around 85% of the time",
    "based on preliminary data",
    "historically speaking",
    "by some accounts nearly 90% confirm that",
    "as noted by several analysts",
    "with an estimated 67% success rate",
    "industry experts generally agree that",
    "based on internal benchmarks",
]


def inject_unverifiable_error(text: str, step_name: str, severity: int = 1,
                               return_delta: bool = False, rng=None,
                               query: str | None = None,
                               ground_truth: dict | None = None,
                               injection_model_fn=None, **kwargs):
    """Unverifiable claim: add a specific but unverifiable statement.

    RAGTruth: 13.8% of all spans. Avg span: 4.4 words (shortest type).
    Cross-tab: 86% from Subtle Baseless Info.
    """
    rng = rng or random

    if injection_model_fn is not None:
        try:
            resp = injection_model_fn(UNVERIFIABLE_INJECTION_PROMPT.format(text=text))
            parsed = _parse_llm_injection(resp, text)
            if parsed:
                return _maybe_return(*parsed, return_delta)
        except Exception:
            pass

    sentences = _split_sents(text)
    if not sentences:
        return _maybe_return(text, "", return_delta,
                             meta={"error_type": "unverifiable", "n_inserts": 0,
                                   "severity_physical": 0, "injection_method": "rule"})

    k = max(1, min(severity, len(sentences) - 1, len(_UNVERIFIABLE_FRAGMENTS)))
    eligible = list(range(1, len(sentences))) or [0]
    targets = rng.sample(eligible, k=min(k, len(eligible)))
    fragments_used = rng.sample(_UNVERIFIABLE_FRAGMENTS, k=min(k, len(_UNVERIFIABLE_FRAGMENTS)))

    for idx, frag in zip(sorted(targets), fragments_used):
        sent = sentences[idx]
        if sent and sent[0].isupper():
            sentences[idx] = frag.capitalize() + ", " + sent[0].lower() + sent[1:]
        else:
            sentences[idx] = frag + ", " + sent

    modified = _join_sents(sentences)
    delta = f"UNVERIFIABLE {len(fragments_used)}: " + " | ".join(fragments_used)
    meta = {"error_type": "unverifiable", "n_inserts": len(fragments_used),
            "severity_physical": len(fragments_used), "injection_method": "rule"}
    return _maybe_return(modified, delta, return_delta, meta=meta)


def inject_contradictory_error(text: str, step_name: str, severity: int = 1,
                                return_delta: bool = False, rng=None,
                                query: str | None = None,
                                ground_truth: dict | None = None,
                                injection_model_fn=None, **kwargs):
    """Contradictory sentence: replace a fact with its negation.

    RAGTruth: 6.9% of all spans. Avg span: 16.1 words (sentence-level).
    Cross-tab: 13% from Evident Conflict, 15% from Subtle Conflict.
    """
    rng = rng or random

    if injection_model_fn is not None:
        try:
            resp = injection_model_fn(CONTRADICTORY_INJECTION_PROMPT.format(text=text))
            parsed = _parse_llm_injection(resp, text)
            if parsed:
                return _maybe_return(*parsed, return_delta)
        except Exception:
            pass

    # Prefer ground_truth contradictions
    if query and ground_truth:
        gt_entry = ground_truth.get(query)
        if gt_entry:
            contras = list(gt_entry.get("contradictions", []))
            if contras:
                rng.shuffle(contras)
                sentences = _split_sents(text)
                if len(sentences) > 1:
                    k = min(severity, len(contras), len(sentences) - 1)
                    targets = rng.sample(range(1, len(sentences)), k=k)
                    replacements = []
                    for idx, contra in zip(sorted(targets), contras[:k]):
                        replacements.append((sentences[idx], contra))
                        sentences[idx] = contra
                    modified = _join_sents(sentences)
                    delta = "CONTRADICTORY: " + " | ".join(
                        f"[{orig[:30]}...]->[{repl[:30]}...]" for orig, repl in replacements)
                    meta = {"error_type": "contradictory", "n_replaced": len(replacements),
                            "severity_physical": len(replacements), "injection_method": "rule_gt"}
                    return _maybe_return(modified, delta, return_delta, meta=meta)

    # Fallback: negate a sentence
    sentences = _split_sents(text)
    if len(sentences) <= 1:
        return _maybe_return(text, "", return_delta,
                             meta={"error_type": "contradictory", "n_replaced": 0,
                                   "severity_physical": 0, "injection_method": "rule"})

    idx = rng.randrange(1, len(sentences))
    original_sent = sentences[idx]
    negated = original_sent
    negation_applied = False

    for pos, neg in [(" is ", " is not "), (" was ", " was not "),
                     (" are ", " are not "), (" were ", " were not "),
                     (" has ", " has not "), (" had ", " had not "),
                     (" can ", " cannot "), (" will ", " will not "),
                     (" should ", " should not "), (" does ", " does not "),
                     (" do ", " do not "), (" did ", " did not ")]:
        if pos in negated:
            negated = negated.replace(pos, neg, 1)
            negation_applied = True
            break
        if neg in negated:
            negated = negated.replace(neg, pos, 1)
            negation_applied = True
            break

    if not negation_applied:
        negated = ("Contrary to common belief, it is not the case that "
                   + original_sent[0].lower() + original_sent[1:])

    sentences[idx] = negated
    modified = _join_sents(sentences)
    delta = f"CONTRADICTORY: [{original_sent[:40]}...]->[{negated[:40]}...]"
    meta = {"error_type": "contradictory", "n_replaced": 1,
            "severity_physical": 1, "injection_method": "rule"}
    return _maybe_return(modified, delta, return_delta, meta=meta)


# ============================================================
# RAGTruth-weighted sampler
# ============================================================

_FAVA_DISPATCH = {
    "entity": inject_entity_error,
    "invented": inject_invented_error,
    "unverifiable": inject_unverifiable_error,
    "contradictory": inject_contradictory_error,
}


def inject_ragtruth_weighted(text: str, step_name: str, severity: int = 1,
                              return_delta: bool = False, rng=None,
                              query: str | None = None,
                              ground_truth: dict | None = None,
                              injection_model_fn=None, **kwargs):
    """Sample an error type using RAGTruth empirical weights for this step.

    Recommended for the main experimental sweep.
    """
    rng = rng or random
    weights = _INJECTION_WEIGHTS.get(step_name, _INJECTION_WEIGHTS["verify"])

    types = list(weights.keys())
    probs = [weights[t] for t in types]
    chosen_type = rng.choices(types, weights=probs, k=1)[0]

    return _FAVA_DISPATCH[chosen_type](
        text, step_name, severity=severity, return_delta=return_delta,
        rng=rng, query=query, ground_truth=ground_truth,
        injection_model_fn=injection_model_fn, **kwargs)


# ============================================================
# Registry
# ============================================================

ERROR_TYPES = {
    "entity": inject_entity_error,
    "invented": inject_invented_error,
    "unverifiable": inject_unverifiable_error,
    "contradictory": inject_contradictory_error,
    "ragtruth_weighted": inject_ragtruth_weighted,
}
