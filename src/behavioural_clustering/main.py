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

        # Display which sections will run
        print("Sections that will run:")
        for section in run_settings.run_sections:
            print(f"- {section}")

        print("Loading evaluator pipeline...")
        evaluator = EvaluatorPipeline(run_settings)
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