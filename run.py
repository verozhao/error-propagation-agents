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
    parser.add_argument("--severity", type=int, default=1, choices=[1, 2, 3, 4],
                        help="Error injection severity level (1-4, default 1)")
    parser.add_argument("--results-file", type=str, default=None)
    args = parser.parse_args()
    
    if args.mode == "check":
        print("Available models:", list_available_models())
        print("GPU status:", check_gpu_available())
        return
    
    if args.mode == "run":
        if args.models is None:
            from config import DEFAULT_OPEN_SOURCE_MODELS, DEFAULT_API_MODELS
            args.models = DEFAULT_API_MODELS if args.use_api else DEFAULT_OPEN_SOURCE_MODELS
        
        output_file = run_full_experiment(
            models=args.models,
            num_trials=args.trials,
            error_type=args.error_type,
            severity=args.severity
        )
        print(f"Results saved to: {output_file}")
    
    elif args.mode == "analyze":
        generate_report(args.results_file, error_type=args.error_type)


if __name__ == "__main__":
    main()