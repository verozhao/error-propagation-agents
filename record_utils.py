"""Shared helpers for interpreting experiment records.

Introduced when P0-1 (is_baseline) and Issue α (injection_valid) added
two semantically-distinct flags. Use these helpers from every analysis
script so "baseline" and "usable injection" always mean the same thing.

Three record categories:
  - BASELINE:           is_baseline=True
  - INJECTED_VALID:     is_baseline=False and injection_valid=True
                        (injection was attempted AND produced a delta)
  - INJECTED_INVALID:   is_baseline=False and injection_valid=False
                        (injection attempted but injector no-opped, e.g.
                         omission on a single-sentence text; safe subset
                         of records that analyses should usually drop)

Legacy records (from before either flag existed) are inferred
conservatively: we consider them baseline iff both error_step and
compound_steps are None. Legacy injected records are treated as
INJECTED_VALID if they have non-empty injected_content, else INJECTED_INVALID.
"""
from __future__ import annotations


def is_baseline(record: dict) -> bool:
    """Canonical baseline test. Uses explicit flag when available."""
    flag = record.get("is_baseline")
    if flag is not None:
        return bool(flag)
    # legacy fallback
    return (record.get("error_step") is None
            and record.get("compound_steps") is None)


def injection_is_valid(record: dict) -> bool | None:
    """Return True/False for injected records; None for baselines.

    Issue α: a non-baseline record whose injector produced no delta
    (empty injected_content) should not be counted as evidence of
    propagation or degradation. Most analyses should drop these.
    """
    if is_baseline(record):
        return None
    # explicit flag set by experiment.py after the Issue α fix
    flag = record.get("injection_valid")
    if flag is not None:
        return bool(flag)
    # legacy fallback: treat non-empty injected_content as valid
    return bool(record.get("injected_content"))


def usable_for_injection_analysis(record: dict) -> bool:
    """True if the record is a non-baseline injection that actually
    perturbed the pipeline. Excludes baselines AND failed-injection
    no-ops. This is the predicate most per-step / regression analyses
    should use.
    """
    if is_baseline(record):
        return False
    return injection_is_valid(record) is True


def usable_for_baseline_pool(record: dict) -> bool:
    """True if the record should contribute to the baseline pool
    (for failure-rate normalisation, TF-IDF similarity, etc.)."""
    return is_baseline(record)
