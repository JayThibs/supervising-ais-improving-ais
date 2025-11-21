import argparse
from interpret_case_studies import analyze_case_study_exps
from interpret_ablations import analyze_ablation_exps
from results_paths_dicts import get_paths_to_results


def main():
    parser = argparse.ArgumentParser()
    # Progressive summarization options
    parser.add_argument("--api_provider", type=str, choices=["openai", "openrouter", "anthropic", "gemini"], help="LLM provider for summarization", default="openrouter")
    parser.add_argument("--model_str", type=str, help="Model name for summarization", default="openai/gpt-5-2025-08-07")
    parser.add_argument("--api_key_path", type=str, default="../../../../../data/api_keys/openrouter_key.txt", help="Path to API key file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging of API calls and model interactions during summarization.")

    parser.add_argument("--summary_max_tokens", type=int, default=30000)
    parser.add_argument("--summary_temperature", type=float, default=1.0)
    parser.add_argument("--summary_max_thinking_tokens", type=int, default=20000)
    parser.add_argument("--summary_save_dir", type=str, default="summaries_3")

    # Preset experimental configuration options
    parser.add_argument("--experiment_name", type=str, choices=["ablations_gemini_diversified", "case_studies_gemini_diversified", "case_studies_qwen_diversified", "case_studies_qwen_un_diverse"], help="Name of the experiment to analyze", default="ablations_gemini_diversified")


    parser.add_argument("--skip_plots", action="store_true")

    # Hypothesis selection options
    parser.add_argument("--do_summaries", action="store_true")
    parser.add_argument("--summary_filter", type=str, default="permutation", choices=["none", "permutation", "crossval"], 
                        help="Filter hypotheses by BH-significant p-values of the chosen type")
    parser.add_argument("--summary_bh_alpha", type=float, default=0.05)
    parser.add_argument("--summary_min_accuracy", type=float, default=None)
    parser.add_argument("--summary_top_k", type=int, default=None)

    # Analysis description options
    parser.add_argument("--alpha", type=float, default=0.05)


    args = parser.parse_args()

    summary_opts = None
    if args.do_summaries:
        if not args.api_provider or not args.model_str or not args.api_key_path:
            raise ValueError("--summaries requires --api_provider, --model_str, and --api_key_path.")
        summary_opts = {
            "enabled": True,
            "api_provider": args.api_provider,
            "model_str": args.model_str,
            "api_key_path": args.api_key_path,
            "save_dir": args.summary_save_dir,
            "max_tokens": args.summary_max_tokens,
            "temperature": args.summary_temperature,
            "max_thinking_tokens": args.summary_max_thinking_tokens,
            "filter_type": args.summary_filter,
            "bh_alpha": args.summary_bh_alpha,
            "min_accuracy": args.summary_min_accuracy,
            "top_k": args.summary_top_k,
            "verbose": args.verbose,
        }

    paths_to_results = get_paths_to_results(args.experiment_name)

    if args.experiment_name in ["case_studies_gemini_diversified", "case_studies_qwen_diversified", "case_studies_qwen_un_diverse"]:
        analyze_case_study_exps(
            paths_to_results=paths_to_results,
            out_dir_tables="tables_3/case_studies",
            out_dir_figs="figs_3/case_studies",
            out_dir_summaries="summaries_3/case_studies",
            alpha=args.alpha,
            make_plots=not args.skip_plots,
            do_progressive_summaries=args.do_summaries,
            summary_opts=summary_opts if args.do_summaries else None,
        )
    elif args.experiment_name in ["ablations_gemini_diversified"]:
        analyze_ablation_exps(
            paths_to_results=paths_to_results,
            out_dir_tables="tables_3/ablation",
            out_dir_figs="figs_3/ablation",
            alpha=args.alpha,
            make_plots=not args.skip_plots,
            api_provider=args.api_provider,
            model_str=args.model_str,
            api_key_path=args.api_key_path,
            max_tokens=args.summary_max_tokens,
            temperature=args.summary_temperature,
            max_thinking_tokens=args.summary_max_thinking_tokens,
            verbose=args.verbose,
        )
    else:
        raise ValueError("No log file or exp config file provided, and no preset experimental configuration used.")

if __name__ == "__main__":
    main()
