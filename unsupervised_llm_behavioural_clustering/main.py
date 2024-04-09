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


def main():
    run_config_manager = RunConfigurationManager()
    selected_run = input(
        f"Enter the name of the run configuration you want to use (default: {run_config_manager.default_configuration}): "
    )
    run_settings = run_config_manager.get_configuration(
        selected_run or run_config_manager.default_configuration
    )
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
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
