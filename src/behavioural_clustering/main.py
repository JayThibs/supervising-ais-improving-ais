import argparse
import os
import sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

print("Loading evaluation pipeline...")
from behavioural_clustering.evaluation.evaluator_pipeline import EvaluatorPipeline

print("Loading run configuration manager...")
from behavioural_clustering.config.run_configuration_manager import RunConfigurationManager

print("Loading LLM clustering algorithms...")
from behavioural_clustering.utils.llm_clustering import update_clustering_factory

def get_args():
    parser = argparse.ArgumentParser(
        description="Language Model Unsupervised Behavioural Evaluator"
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Specify the name of the run configuration to use.",
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        default=[],
        help="Specify sections to skip (space-separated).",
    )
    parser.add_argument(
        "--run-only",
        type=str,
        default=None,
        help="Specify a single section to run.",
    )
    parser.add_argument(
        "--list-sections",
        action="store_true",
        help="List available sections and exit.",
    )
    parser.add_argument(
        "--clustering-algorithm",
        type=str,
        default=None,
        help="Specify the clustering algorithm to use (e.g., KMeans, SpectralClustering, k-LLMmeans, SPILL).",
    )
    # NEW ARG for iterative runs
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="Run an iterative evaluation pipeline."
    )
    return parser.parse_args()

def main(args):
    run_config_manager = RunConfigurationManager()

    print("Updating clustering factory with LLM-based methods...")
    update_clustering_factory()

    if args.list_sections:
        run_config_manager.print_available_sections()
        return

    if args.run:
        selected_run = args.run
    else:
        available_runs = run_config_manager.list_configurations()
        if not available_runs:
            raise ValueError("No run configurations available.")
        print("Available run configurations:")
        for i, run in enumerate(available_runs):
            print(f"{i + 1}. {run}")
        choice = input("Enter the number of the run configuration to use (or press Enter to exit): ")
        if not choice:
            print("Exiting...")
            return
        selected_run = available_runs[int(choice) - 1]

    run_settings = run_config_manager.get_configuration(selected_run)
    if run_settings:
        print(f"Using run settings: {run_settings.name}")

        # Update run_sections based on command-line arguments
        if args.skip:
            run_settings.run_sections = [section for section in run_settings.run_sections if section not in args.skip]
            run_settings.update_run_sections()  # Validate the updated run_sections
        if args.run_only:
            run_settings.update_run_sections(args.run_only)
            
        if args.clustering_algorithm:
            if args.clustering_algorithm in run_settings.clustering_settings.all_clustering_algorithms:
                print(f"Using clustering algorithm: {args.clustering_algorithm}")
                run_settings.clustering_settings.main_clustering_algorithm = args.clustering_algorithm
            else:
                print(f"Warning: Clustering algorithm '{args.clustering_algorithm}' not found in available algorithms.")
                print(f"Available algorithms: {run_settings.clustering_settings.all_clustering_algorithms}")
                print(f"Using default: {run_settings.clustering_settings.main_clustering_algorithm}")

        # Display which sections will run
        print("Sections that will run:")
        for section in run_settings.run_sections:
            print(f"- {section}")

        print("Loading evaluator pipeline...")
        evaluator = EvaluatorPipeline(run_settings)

        print("Running evaluation...")
        if args.iterative:
            # NEW: Run iterative evaluation approach
            evaluator.run_iterative_evaluation()
        else:
            evaluator.run_evaluations()
    else:
        raise ValueError(f"Run settings not found for {selected_run}")

if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")
