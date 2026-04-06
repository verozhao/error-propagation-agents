"""Factual accuracy evaluation for pipeline outputs.

Implements Phase 1.1 of the evaluation overhaul: rather than relying on
self-judging LLMs and keyword matching, we measure (a) whether an injected
false claim survives into the final output, and (b) whether known
ground-truth facts about the query domain are preserved.

The injected-claim survival score uses two cheap signals stacked:

  1. token-overlap (Jaccard on content lemmas) — robust, no API calls
  2. n-gram containment of distinctive trigrams from the injected claim

We deliberately avoid embedding-model calls in the default path so this
function can be applied retroactively to large result sets without API
spend. An optional `embedding_fn` hook is provided for callers that want
true semantic similarity (see `make_openai_embedding_fn`).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

_GT_CACHE: dict[str, dict] = {}

_STOPWORDS = frozenset("""
a an and are as at be by for from has have he her hers him his i in is it its
me my of on or our she that the their them they this to was we were will with
you your these those there here also but not no yes do does did been being
""".split())

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _content_tokens(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in _STOPWORDS and len(t) > 1}


def _trigrams(tokens: list[str]) -> set[tuple[str, str, str]]:
    return {tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)}


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in parts if p]


def load_ground_truth(path: str | None = None) -> dict:
    """Load ground_truth.json keyed by query string."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "ground_truth.json")
    if path in _GT_CACHE:
        return _GT_CACHE[path]
    with open(path) as f:
        raw = json.load(f)
    by_query = {entry["query"]: entry for entry in raw["queries"]}
    _GT_CACHE[path] = by_query
    return by_query


def claim_survival_score(
    injected_claim: str,
    output_text: str,
    embedding_fn: Optional[Callable[[list[str]], list[list[float]]]] = None,
) -> tuple[float, bool]:
    """Return (continuous_score in [0,1], propagated_bool).

    The continuous score is the max of:
      - Jaccard overlap of content tokens between the claim and the
        sentence in `output_text` that overlaps with it the most
      - fraction of distinctive trigrams from the claim that appear
        anywhere in `output_text`
      - (optional) max cosine similarity from the embedding hook

    `propagated_bool` is True if the continuous score is >= 0.5 OR if any
    distinctive trigram appears verbatim in the output.
    """
    claim = (injected_claim or "").strip()
    if not claim or not output_text:
        return 0.0, False

    claim_tokens = _content_tokens(claim)
    if not claim_tokens:
        return 0.0, False

    output_tokens_seq = _tokens(output_text)
    output_token_set = set(output_tokens_seq)

    # 1. trigram containment
    claim_seq = _tokens(claim)
    claim_tris = _trigrams(claim_seq)
    output_tris = _trigrams(output_tokens_seq)
    if claim_tris:
        tri_score = len(claim_tris & output_tris) / len(claim_tris)
    else:
        tri_score = 0.0
    verbatim_hit = bool(claim_tris & output_tris)

    # 2. best per-sentence Jaccard
    best_jacc = 0.0
    for sent in _split_sentences(output_text):
        sent_tokens = _content_tokens(sent)
        if not sent_tokens:
            continue
        inter = len(claim_tokens & sent_tokens)
        union = len(claim_tokens | sent_tokens)
        if union:
            best_jacc = max(best_jacc, inter / union)

    # 3. token recall (claim tokens that appear anywhere)
    recall = len(claim_tokens & output_token_set) / len(claim_tokens)

    score = max(tri_score, best_jacc, 0.6 * recall)

    # 4. optional embedding cosine
    if embedding_fn is not None:
        sents = _split_sentences(output_text) or [output_text]
        try:
            embs = embedding_fn([claim] + sents)
            claim_vec = embs[0]
            best_cos = max(_cosine(claim_vec, s) for s in embs[1:])
            score = max(score, best_cos)
        except Exception:
            pass

    score = max(0.0, min(1.0, score))
    propagated = verbatim_hit or score >= 0.5
    return score, propagated


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class FactualAccuracyResult:
    factual_accuracy_score: float
    error_propagated: bool
    error_survival_score: float
    assertions_present: int
    assertions_total: int
    contradictions_present: int

    def to_dict(self) -> dict:
        return {
            "factual_accuracy_score": self.factual_accuracy_score,
            "error_propagated": self.error_propagated,
            "error_survival_score": self.error_survival_score,
            "assertions_present": self.assertions_present,
            "assertions_total": self.assertions_total,
            "contradictions_present": self.contradictions_present,
        }


def evaluate_factual_accuracy(
    pipeline_output: str,
    injected_error: Optional[str] = None,
    query: Optional[str] = None,
    ground_truth: Optional[dict] = None,
    embedding_fn: Optional[Callable] = None,
) -> FactualAccuracyResult:
    """Score the factual accuracy of `pipeline_output`.

    Args:
        pipeline_output: the final text from the pipeline.
        injected_error: the exact text of an injected false claim, if any.
        query: the original task query (used to look up ground truth).
        ground_truth: pre-loaded ground truth dict; if None, loads default.
        embedding_fn: optional callable(list[str]) -> list[list[float]].
    """
    if ground_truth is None:
        ground_truth = load_ground_truth()

    out_lower = (pipeline_output or "").lower()
    out_tokens = set(_tokens(pipeline_output))

    survival_score = 0.0
    propagated = False
    if injected_error:
        survival_score, propagated = claim_survival_score(
            injected_error, pipeline_output, embedding_fn=embedding_fn
        )

    gt_entry = ground_truth.get(query) if query else None
    assertions_present = 0
    assertions_total = 0
    contradictions_present = 0

    if gt_entry:
        for assertion in gt_entry.get("assertions", []):
            assertions_total += 1
            kws = [k.lower() for k in assertion.get("keywords", [])]
            if kws and all(kw in out_lower for kw in kws):
                assertions_present += 1
                continue
            # alias fallback: any alias appears as substring
            for alias in assertion.get("aliases", []):
                if alias.lower() in out_lower:
                    assertions_present += 1
                    break

        for contradiction in gt_entry.get("contradictions", []):
            # contradiction detection must respect negation words, so we
            # check trigram containment on the *full* token sequence (not
            # the stopword-filtered set, which strips "not", "no", etc.).
            c_seq = _tokens(contradiction)
            if len(c_seq) < 3:
                if contradiction.lower() in out_lower:
                    contradictions_present += 1
                continue
            c_tris = _trigrams(c_seq)
            out_seq = _tokens(pipeline_output)
            out_tris_full = _trigrams(out_seq)
            if c_tris and len(c_tris & out_tris_full) / len(c_tris) >= 0.5:
                contradictions_present += 1

    if assertions_total > 0:
        preserved = assertions_present / assertions_total
    else:
        preserved = 1.0

    contradiction_penalty = 0.0
    if assertions_total > 0 and contradictions_present > 0:
        contradiction_penalty = min(0.5, 0.15 * contradictions_present)

    factual = preserved * (1.0 - survival_score) - contradiction_penalty
    factual = max(0.0, min(1.0, factual))

    return FactualAccuracyResult(
        factual_accuracy_score=factual,
        error_propagated=propagated,
        error_survival_score=survival_score,
        assertions_present=assertions_present,
        assertions_total=assertions_total,
        contradictions_present=contradictions_present,
    )


def make_openai_embedding_fn(model: str = "text-embedding-3-small"):
    """Optional embedding hook. Caller pays per call — not used by default."""

    def _fn(texts: list[str]) -> list[list[float]]:
        from models import get_openai_client

        client = get_openai_client()
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    return _fn
