print("Loading evaluation pipeline...")
from evaluator_pipeline import EvaluatorPipeline

print("Loading data preparation...")
from data_preparation import DataPreparation

print("Loading model evaluation...")
from model_evaluation import ModelEvaluation

print("query_model_on_statements...")
from utils import query_model_on_statements

print("loaded query_model_on_statements")
import argparse
import numpy as np
from sklearn.manifold import TSNE


def get_args():
    parser = argparse.ArgumentParser(
        description="Language Model Unsupervised Behavioural Evaluator"
    )

    parser.add_argument(
        "--texts_subset",
        type=int,
        default=10,
        help="Number of texts to consider as a subset for evaluation. 10 works as a test.",
    )

    parser.add_argument(
        "--hide-plots",
        action="store_true",
        help="Hide the plots while still saving them.",
    )

    parser.add_argument(
        "--model-family",
        type=str,
        required=True,
        help="Language Model family to use. Options: 'openai', 'anthropic', 'local'.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Language Model to use for evaluation. Options: 'gpt-3.5-turbo', 'gpt-4', etc.",
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=200,
        help="Number of clusters for KMeans.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode on a small data subset.",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        default=5000,
        help="Number of points to use for clustering.",
    )

    return parser.parse_args()


def main(args):
    print("Loading evaluator pipeline...")
    evaluator = EvaluatorPipeline(args)
    print("Loading and preprocessing data...")
    evaluator.setup()

    if args.test_mode:
        print("Running short text tests...")
        evaluator.run_short_text_tests()
    else:
        print("Running evaluation...")
        evaluator.run_evaluation()


if __name__ == "__main__":
    try:
        """
        Run using:

        python unsupervised_llm_behavioural_clustering/main.py --model-family="openai" --model="gpt-3.5-turbo" --test-mode
        """
        args = get_args()
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")
