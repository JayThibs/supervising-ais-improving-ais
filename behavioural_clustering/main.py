import argparse

print("Loading evaluation pipeline...")
from evaluator_pipeline import EvaluatorPipeline

print("loaded query_model_on_statements")
from config.run_configuration_manager import RunConfigurationManager

# print("Loading data preparation...")
# from data_preparation import DataPreparation

# print("Loading model evaluation...")
# from model_evaluation import ModelEvaluation

# print("query_model_on_statements...")
# from utils import query_model_on_statements


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
        # python behavioural_clustering/main.py --run quick_full_test
        selected_run = args.run
    else:
        # python behavioural_clustering/main.py (can edit in run_configuration_manager.py > selected_run)
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