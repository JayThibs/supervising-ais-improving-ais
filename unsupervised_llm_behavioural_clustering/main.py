from evaluator_pipeline import EvaluatorPipeline
from data_preparation import DataPreparation
from model_evaluation import ModelEvaluation
from utils import query_model
import argparse
import numpy as np
from sklearn.manifold import TSNE


def get_args():
    parser = argparse.ArgumentParser(
        description="Language Model Unsupervised Behavioural Evaluator"
    )

    parser.add_argument(
        "--file_paths",
        nargs="+",
        type=str,
        required=True,
        help="List of file paths for loading evaluation data.",
    )
    parser.add_argument(
        "--texts_subset",
        type=int,
        default=10,
        help="Number of texts to consider as a subset for evaluation. 10 works as a test.",
    )
    parser.add_argument(
        "--run_tests",
        action="store_true",
        help="Run tests on short texts.",
    )

    parser.add_argument(
        "--llm", type=str, required=True, help="Language Model to use for evaluation."
    )
    parser.add_argument(
        "--prompt", type=str, required=True, help="Prompt to use for model evaluation."
    )

    parser.add_argument(
        "--n_clusters", type=int, default=200, help="Number of clusters for KMeans."
    )

    parser.add_argument(
        "--plot_dim", type=str, default="16,16", help="Dimensions of the plot."
    )

    parser.add_argument(
        "--save_path", type=str, default="data/plots", help="Path for saving plots."
    )

    return parser.parse_args()


# # Include function arguments for variables that are not defined within main()
def main(args):
    pipeline = EvaluatorPipeline()
    pipeline.setup()

    if args.run_tests:
        data_preparation = DataPreparation()
        model_evaluation = ModelEvaluation()
        model_evaluation.run_short_text_tests()
    else:
        evaluator = EvaluatorPipeline(args)
        evaluator.setup()
        evaluator.run_evaluation()


if __name__ == "__main__":
    try:
        args = get_args()
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")
