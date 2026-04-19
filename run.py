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
    parser.add_argument("--error-type", default="semantic", choices=["semantic", "factual", "omission"])
    parser.add_argument("--severity", type=int, default=1, choices=[1, 2, 3],
                        help="Error injection severity level (1-3, default 1)")
    parser.add_argument("--pos-target", default=None, choices=["noun", "verb", "adj"],
                        help="POS-targeted injection: only corrupt this word class")
    parser.add_argument("--tfidf-target", default=None, choices=["high", "low"],
                        help="TF-IDF-targeted injection: corrupt highest or lowest importance word")
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
            pos_target=args.pos_target,
            tfidf_target=args.tfidf_target,
            diagnostic_query=args.diagnostic_query,
            skip_baseline=args.skip_baseline,
            judge_models=[args.judge_model] if args.judge_model else None,
            use_llm_judge=args.use_llm_judge,
            compound_pairs=compound_pairs,
            max_retries=0 if args.no_retry else 1,
        )
        print(f"Results saved to: {output_file}")
    
    elif args.mode == "analyze":
        generate_report(args.results_file, error_type=args.error_type)


if __name__ == "__main__":
    main()