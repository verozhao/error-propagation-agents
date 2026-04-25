import argparse
from models import list_available_models, check_gpu_available
from experiment import run_full_experiment
from analysis import generate_report


def main():
    parser = argparse.ArgumentParser(description="Error Propagation Experiment")
    parser.add_argument("--mode", choices=["run", "analyze", "check"], required=True)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--use-api", action="store_true", help="Use API models instead of local")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--error-type", default="ragtruth_weighted",
                        choices=["entity", "invented", "unverifiable", "contradictory", "ragtruth_weighted"],
                        help="Error type grounded in FAVA taxonomy (default: ragtruth_weighted)")
    parser.add_argument("--severity", type=int, default=1, choices=[1, 2, 3],
                        help="Injection intensity: number of error operations per step (1-3)")
    parser.add_argument("--injection-model", type=str, default=None,
                        help="Model to use for LLM-based error generation (e.g., llama-3.1-8b). "
                             "If not set, uses fast rule-based injection.")
    parser.add_argument("--results-file", type=str, default=None)
    parser.add_argument("--diagnostic-query", type=str, default=None,
                        help="Run only this single query (for diagnostic mini-sweep)")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline (error_step=None) runs; reuse sev=1 baselines")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Override judge model for evaluation")
    parser.add_argument("--use-llm-judge", action="store_true",
                        help="Enable LLM judge (adds cost). Default off — uses algorithmic metric.")
    parser.add_argument("--compound-steps", type=str, default=None,
                        help="Compound injection: comma-separated pair(s) like '0,3' or '0,3;1,3'")
    parser.add_argument("--no-retry", action="store_true",
                        help="Disable verify-triggered retry (ablation)")
    parser.add_argument("--queries", type=int, default=None,
                        help="Max number of queries to use (default: all)")
    parser.add_argument("--pipeline", default="medium",
                        choices=["short", "medium", "self_refine_A", "self_refine_B", "self_refine_C"],
                        help="Pipeline configuration (default: medium)")
    parser.add_argument("--intervention", default="none",
                        choices=["none", "threshold", "learned", "optimal"],
                        help="Intervention strategy (default: none)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only baseline (no injection) for natural failure analysis")
    args = parser.parse_args()
    
    if args.mode == "check":
        print("Available models:", list_available_models())
        print("GPU status:", check_gpu_available())
        return
    
    if args.mode == "run":
        if args.models is None:
            from config import DEFAULT_OPEN_SOURCE_MODELS, DEFAULT_API_MODELS
            args.models = DEFAULT_API_MODELS if args.use_api else DEFAULT_OPEN_SOURCE_MODELS
        
        compound_pairs = None
        if args.compound_steps:
            compound_pairs = [
                tuple(int(x) for x in pair.split(","))
                for pair in args.compound_steps.split(";")
            ]

        output_file = run_full_experiment(
            models=args.models,
            num_trials=args.trials,
            error_type=args.error_type,
            severity=args.severity,
            diagnostic_query=args.diagnostic_query,
            skip_baseline=args.skip_baseline,
            judge_models=[args.judge_model] if args.judge_model else None,
            use_llm_judge=args.use_llm_judge,
            compound_pairs=compound_pairs,
            max_retries=0 if args.no_retry else 1,
            max_queries=args.queries,
            injection_model=args.injection_model,
            pipeline=args.pipeline,
            baseline_only=args.baseline_only,
            intervention=args.intervention,
        )
        print(f"Results saved to: {output_file}")
    
    elif args.mode == "analyze":
        generate_report(args.results_file, error_type=args.error_type)


if __name__ == "__main__":
    main()