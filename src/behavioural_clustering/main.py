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
    return parser.parse_args()


def main(args):
    run_config_manager = RunConfigurationManager()

    if args.list_sections:
        run_config_manager.print_available_sections()
        return

    if args.run:
        selected_run = args.run
    else:
        available_runs = run_config_manager.list_configurations()
        if not available_runs:
            raise ValueError("No run configurations available.")
        selected_run = available_runs[0]  # Default to the first available configuration

    run_settings = run_config_manager.get_configuration(selected_run)
    if run_settings:
        print(f"Using run settings: {run_settings.name}")

        # Update skip_sections and run_only based on command-line arguments
        run_settings.skip_sections.extend(args.skip)
        if args.run_only:
            run_settings.run_only = args.run_only

        print("Loading evaluator pipeline...")
        evaluator = EvaluatorPipeline(run_settings)
        print("Loading and preprocessing data...")
        evaluator.setup_evaluations()
        print("Running evaluation...")
        evaluator.run_evaluations()
    else:
        raise ValueError(f"Run settings not found for {selected_run}")


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")