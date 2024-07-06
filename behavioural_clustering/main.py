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
    return parser.parse_args()


def main(args):
    run_config_manager = RunConfigurationManager()
    if args.run:
        # python behavioural_clustering.main --run quick_full_test
        selected_run = args.run
    else:
        # python behavioural_clustering.main (can edit in run_configuration_manager.py > selected_run)
        selected_run = run_config_manager.selected_run
    run_settings = run_config_manager.get_configuration(selected_run)
    if run_settings:
        print(f"Using run settings: {run_settings.name}")
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
