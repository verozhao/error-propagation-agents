"""Pre-flight test suite: verify all code fixes before running the full sweep.

Run this BEFORE ./run_severity_sweeps.sh to confirm everything is wired correctly.

Usage:
    python test_fixes.py           # all tests (no API calls)
    python test_fixes.py --live    # include 1-trial smoke test (costs ~$0.01)
"""

import argparse
import json
import os
import random
import sys

PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


def test_task_loading():
    """Test 1: TASK_TEMPLATES loads from ground_truth.json fallback."""
    print("\n=== TEST 1: Task loading ===")
    from workflow import TASK_TEMPLATES
    check("TASK_TEMPLATES non-empty", len(TASK_TEMPLATES) > 0,
          f"got {len(TASK_TEMPLATES)}")
    check("All have expected_keywords",
          all(t.get("expected_keywords") for t in TASK_TEMPLATES),
          "some queries missing expected_keywords")
    check("No placeholders in top 15",
          not any(t.get("_placeholder") for t in TASK_TEMPLATES[:15]))
    return TASK_TEMPLATES


def test_contradictions():
    """Test 2: All queries have sufficient contradictions."""
    print("\n=== TEST 2: Contradictions coverage ===")
    with open("ground_truth.json") as f:
        gt = json.load(f)

    total = len(gt["queries"])
    has_c = sum(1 for q in gt["queries"] if q.get("contradictions"))
    check(f"All queries have contradictions ({has_c}/{total})", has_c == total)

    min_c = min(len(q.get("contradictions", [])) for q in gt["queries"])
    check(f"Min contradictions >= 3 (got {min_c})", min_c >= 3,
          "boolean queries should have >= 3")

    # Check that HotpotQA queries don't just have generic filler
    from error_injection import FAKE_FACTS
    generic = set(FAKE_FACTS)
    hotpot_specific = 0
    for q in gt["queries"]:
        if q.get("source") == "hotpotqa_auto":
            contras = q.get("contradictions", [])
            if contras and contras[0] not in generic:
                hotpot_specific += 1
    n_hotpot = sum(1 for q in gt["queries"] if q.get("source") == "hotpotqa_auto")
    if n_hotpot > 0:
        check(f"HotpotQA queries use targeted contradictions ({hotpot_specific}/{n_hotpot})",
              hotpot_specific > n_hotpot * 0.8)
    return gt


def test_cache_coverage(tasks):
    """Test 3: Top-15 queries have search cache entries."""
    print("\n=== TEST 3: Search cache coverage ===")
    with open("search_cache.json") as f:
        cache = json.load(f)

    top15 = tasks[:15]
    cached = sum(1 for t in top15 if t["query"] in cache)
    check(f"Top-15 budget queries cached ({cached}/15)", cached >= 12,
          "uncached queries will need live DuckDuckGo")

    # Also check total
    all_cached = sum(1 for t in tasks if t["query"] in cache)
    check(f"Total cached ({all_cached}/{len(tasks)})", all_cached >= len(tasks) * 0.7)


def test_factual_injection(gt):
    """Test 4: inject_factual_error uses query-specific contradictions."""
    print("\n=== TEST 4: Factual injection quality ===")
    from error_injection import inject_factual_error

    gt_map = {q["query"]: q for q in gt["queries"]}
    rng = random.Random(42)

    # Pick a non-boolean HotpotQA query
    test_queries = [q for q in gt["queries"]
                    if q.get("source") == "hotpotqa_auto"
                    and q.get("answer", "").lower() not in ("yes", "no")]

    if not test_queries:
        check("Has testable HotpotQA queries", False, "no non-boolean HotpotQA queries found")
        return

    tq = test_queries[0]
    text = f"The answer is {tq['answer']}. This is a factual statement about the topic."
    result, delta, meta = inject_factual_error(
        text, "compose", severity=3, return_delta=True,
        rng=rng, query=tq["query"], ground_truth=gt_map)

    check("Injection produced delta", bool(delta), "empty delta")
    check("Meta has n_inserts", meta.get("n_inserts", 0) > 0,
          f"n_inserts={meta.get('n_inserts')}")

    # Verify injected content is NOT all generic
    from error_injection import FAKE_FACTS
    generic = set(FAKE_FACTS)
    inserted_facts = delta.replace("INSERTED ", "").split(" | ") if "INSERTED" in delta else []
    non_generic = [f for f in inserted_facts if f.strip() not in generic]
    check(f"Uses query-specific fakes ({len(non_generic)}/{len(inserted_facts)})",
          len(non_generic) > 0, "all inserted facts are generic filler")


def test_semantic_injection(gt):
    """Test 5: inject_semantic_error with answer-targeted swaps."""
    print("\n=== TEST 5: Semantic injection quality ===")
    from error_injection import inject_semantic_error

    gt_map = {q["query"]: q for q in gt["queries"]}
    rng = random.Random(42)

    tq = [q for q in gt["queries"]
          if q.get("answer") and len(q.get("answer", "")) > 3
          and q.get("answer", "").lower() not in ("yes", "no")]
    if not tq:
        check("Has testable queries", False)
        return
    tq = tq[0]

    text = f"The answer is {tq['answer']}. {tq['answer']} is relevant to this question."
    result, delta, meta = inject_semantic_error(
        text, "compose", severity=2, return_delta=True,
        rng=rng, query=tq["query"], ground_truth=gt_map)

    check("Semantic injection produced delta", bool(delta), "empty delta")
    check("Text was modified", result != text, "output == input")


def test_seed_determinism():
    """Test 6: Seed doesn't change with error_step (paired test validity)."""
    print("\n=== TEST 6: Seed determinism ===")
    from experiment import _derive_seed

    s1 = _derive_seed("model", "query", None, 0)
    s2 = _derive_seed("model", "query", 2, 0)
    s3 = _derive_seed("model", "query", 3, 0)
    check("Seed ignores error_step (baseline vs step2)", s1 == s2,
          f"{s1} != {s2}")
    check("Seed ignores error_step (step2 vs step3)", s2 == s3,
          f"{s2} != {s3}")

    s4 = _derive_seed("model", "query", 2, 1)
    check("Seed changes with trial_idx", s1 != s4,
          f"trial 0 and 1 got same seed {s1}")


def test_experiment_max_queries():
    """Test 7: --queries flag limits task count."""
    print("\n=== TEST 7: Budget control (max_queries) ===")
    from workflow import TASK_TEMPLATES

    # Simulate what run_full_experiment does
    tasks = [t for t in TASK_TEMPLATES if not t.get("_placeholder")]
    original = len(tasks)
    tasks = tasks[:15]
    check(f"max_queries=15 slices {original} -> {len(tasks)}", len(tasks) == 15)


def test_live_smoke(models):
    """Test 8: Single-trial end-to-end (costs ~$0.01)."""
    print("\n=== TEST 8: Live smoke test (1 trial, 1 query) ===")
    from experiment import run_single_experiment
    from workflow import TASK_TEMPLATES
    from factual_accuracy import load_ground_truth

    gt = load_ground_truth()
    task = TASK_TEMPLATES[0]

    for model in models:
        print(f"  Running 1 trial with {model}...")
        try:
            result = run_single_experiment(
                model_name=model,
                task=task,
                error_step=1,
                error_type="factual",
                severity=2,
                ground_truth=gt,
                trial_idx=0,
                use_llm_judge=False,
            )
            ev = result["evaluation"]
            check(f"{model}: got combined_score={ev['combined_score']:.3f}",
                  0 <= ev["combined_score"] <= 1)
            check(f"{model}: injection_valid={result['injection_valid']}",
                  result["injection_valid"] is True,
                  "injection may have no-op'd")

            # Check factual injection used contradictions
            injected = result.get("injected_content", "")
            from error_injection import FAKE_FACTS
            is_generic = injected in FAKE_FACTS
            check(f"{model}: injected content is query-specific",
                  not is_generic and bool(injected),
                  f"got: {injected[:60]}")
        except Exception as e:
            check(f"{model}: smoke test", False, str(e)[:80])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true",
                        help="Run 1-trial smoke test with API calls (~$0.01)")
    parser.add_argument("--models", nargs="+", default=["claude-3-haiku"],
                        help="Models for live smoke test")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    tasks = test_task_loading()
    gt = test_contradictions()
    test_cache_coverage(tasks)
    test_factual_injection(gt)
    test_semantic_injection(gt)
    test_seed_determinism()
    test_experiment_max_queries()

    if args.live:
        test_live_smoke(args.models)

    print("\n" + "=" * 40)
    print(f"PASSED: {PASS}  FAILED: {FAIL}")
    if FAIL > 0:
        print("⚠️  Fix failures before running sweep!")
        sys.exit(1)
    else:
        print("✓ All tests passed — ready to run sweep")
        print("\n  ./run_severity_sweeps.sh --full")


if __name__ == "__main__":
    main()
