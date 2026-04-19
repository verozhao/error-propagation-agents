#!/usr/bin/env python3
"""
Phase 7 smoke test — run BEFORE full sweep to verify changes work with real LLMs.

Usage:
    # Quick (1 query, 1 model, ~$0.05):
    python smoke_test_phase7.py

    # With specific model:
    python smoke_test_phase7.py --model gpt-4o-mini
"""
import argparse
import json
import random
import sys

from error_injection import (
    ERROR_TYPES, ERROR_SUBSTITUTIONS, FAKE_FACTS,
    inject_semantic_error, inject_factual_error, inject_omission_error,
    _build_answer_targeted_swaps, _get_query_fake_facts,
)
from factual_accuracy import load_ground_truth, claim_survival_score, evaluate_factual_accuracy
from workflow import run_workflow

PASS = FAIL = 0
def check(name, ok, detail=""):
    global PASS, FAIL
    if ok: PASS += 1; print(f"  ✓ {name}")
    else:  FAIL += 1; print(f"  ✗ {name}  ← {detail}")


def run_offline_checks():
    """Checks that need no API calls."""
    print("=" * 60)
    print("OFFLINE CHECKS (no API)")
    print("=" * 60)

    ground_truth = load_ground_truth()

    # Dictionary expansion
    check("ERROR_SUBSTITUTIONS ≥ 130", len(ERROR_SUBSTITUTIONS) >= 130,
          f"got {len(ERROR_SUBSTITUTIONS)}")
    check("FAKE_FACTS domain-neutral", "product" not in FAKE_FACTS[0].lower())
    check("Generic words present", "both" in ERROR_SUBSTITUTIONS and "important" in ERROR_SUBSTITUTIONS)

    # Answer-targeted swaps
    q = "Were Scott Derrickson and Ed Wood of the same nationality?"
    text = "Yes, both are American filmmakers from the same country."
    table = _build_answer_targeted_swaps(q, ground_truth, text)
    check("Targeted table built", len(table) > 0, f"table={table}")

    # Injection on HotpotQA content
    rng = random.Random(42)
    _, delta, meta = inject_semantic_error(
        text, 'filter', severity=2, return_delta=True, rng=rng,
        query=q, ground_truth=ground_truth)
    check(f"Semantic sev2 fires (n_subs={meta['n_subs']})", meta["n_subs"] > 0)

    # Compose dead zone fix
    compose_3s = ("Scott Derrickson and Ed Wood are both American filmmakers. "
                  "They share the same nationality. "
                  "Derrickson is known for horror films.")
    rng2 = random.Random(42)
    _, delta_c, meta_c = inject_semantic_error(
        compose_3s, 'compose', severity=2, return_delta=True, rng=rng2,
        query=q, ground_truth=ground_truth)
    check(f"Semantic fires on 3-sent compose (n_subs={meta_c['n_subs']})", meta_c["n_subs"] > 0)

    rng3 = random.Random(42)
    _, _, meta_o = inject_omission_error(compose_3s, 'compose', severity=3, return_delta=True, rng=rng3)
    check(f"Omission fires on 3-sent compose (n_removed={meta_o['n_removed']})", meta_o["n_removed"] > 0)

    # Severity monotonicity
    prev = 0
    for sev in [1, 2, 3]:
        rng_m = random.Random(42)
        _, _, m = inject_semantic_error(text, 'filter', severity=sev, return_delta=True,
                                        rng=rng_m, query=q, ground_truth=ground_truth)
        check(f"Monotonic sev {sev} (n_subs={m['n_subs']} ≥ {prev})", m["n_subs"] >= prev)
        prev = m["n_subs"]

    # Factual injection uses contradictions
    rng_f = random.Random(42)
    pool = _get_query_fake_facts(q, ground_truth, rng_f)
    check("Fact pool has contradiction", "The answer is no" in pool)
    check("Fact pool padded ≥ 10", len(pool) >= 10)

    print(f"\nOffline: {PASS} passed, {FAIL} failed\n")


def run_live_check(model_name="claude-3-haiku"):
    """Single-query pipeline test with real LLM (~$0.05)."""
    from models import call_model

    print("=" * 60)
    print(f"LIVE CHECK — {model_name} (1 query, ~$0.05)")
    print("=" * 60)

    ground_truth = load_ground_truth()
    query = "Were Scott Derrickson and Ed Wood of the same nationality?"

    seed = 12345
    call_i = {"i": 0}
    def model_fn(prompt):
        call_i["i"] += 1
        return call_model(model_name, prompt, seed=seed + call_i["i"])

    # Baseline
    print("\n--- Baseline ---")
    call_i["i"] = 0
    results_base = run_workflow(query=query, model_fn=model_fn)
    compose_base = results_base[-2].output_text
    verify_base = results_base[-1].output_text
    fa_base = evaluate_factual_accuracy(compose_base, query=query, ground_truth=ground_truth)

    print(f"  Compose ({len(compose_base)} chars): {compose_base[:100]}")
    print(f"  Verify: {verify_base[:50]}")
    print(f"  FA score: {fa_base.factual_accuracy_score:.3f}")
    print(f"  Assertions: {fa_base.assertions_present}/{fa_base.assertions_total}")
    check("Baseline FA > 0.5", fa_base.factual_accuracy_score > 0.5)
    check("Compose ≥ 2 sentences", compose_base.count('.') >= 2,
          f"got {compose_base.count('.')} periods")

    # Injected runs
    for etype_name in ["semantic", "factual", "omission"]:
        etype_fn = ERROR_TYPES[etype_name]
        print(f"\n--- {etype_name} @ filter (sev=2) ---")
        rng = random.Random(42)
        call_i["i"] = 0
        error_kwargs = {"severity": 2, "return_delta": True, "rng": rng,
                        "query": query, "ground_truth": ground_truth}
        results_inj = run_workflow(
            query=query, model_fn=model_fn,
            error_injection_fn=etype_fn, error_step=1,
            error_kwargs=error_kwargs)

        inj_steps = [r for r in results_inj if r.error_injected]
        injected = inj_steps[0].injected_content if inj_steps else ""
        compose_inj = results_inj[-2].output_text

        survival, propagated = claim_survival_score(injected, compose_inj)
        fa_inj = evaluate_factual_accuracy(
            compose_inj, injected_error=injected,
            query=query, ground_truth=ground_truth)

        delta = fa_base.factual_accuracy_score - fa_inj.factual_accuracy_score
        print(f"  Injected: {injected[:80]}")
        print(f"  Compose:  {compose_inj[:80]}")
        print(f"  Survival: {survival:.3f}, propagated: {propagated}")
        print(f"  FA: {fa_inj.factual_accuracy_score:.3f} (Δ={delta:+.3f})")
        check(f"{etype_name} injection produced delta OR survival",
              delta > 0.01 or propagated or survival > 0.1,
              f"delta={delta:.3f}, survival={survival:.3f}")

    print(f"\nLive: {PASS} passed, {FAIL} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-3-haiku")
    parser.add_argument("--offline-only", action="store_true",
                        help="Skip live LLM check (no API cost)")
    args = parser.parse_args()

    run_offline_checks()
    if not args.offline_only:
        run_live_check(args.model)

    print(f"\n{'=' * 60}")
    print(f"TOTAL: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("🟢 All checks passed — ready for full sweep.")
    else:
        print(f"🔴 {FAIL} check(s) failed.")
    print("=" * 60)
    sys.exit(FAIL)
