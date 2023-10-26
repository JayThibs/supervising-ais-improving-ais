from evaluator_pipeline import EvaluatorPipeline
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
        help="Number of texts to consider as a subset for evaluation.",
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
# def main(args):
#     data_prep = DataPreparation()
#     model_eval = ModelEvaluation()
#     viz = Visualization()

#     # Load API key
#     api_key = data_prep.load_api_key("OPENAI_API_KEY")

#     # Clone repository
#     data_prep.clone_repo("https://github.com/anthropics/evals.git", "data/evals")

#     ### TODO: Add more evaluation data from other sources

#     # Load evaluation data
#     all_texts = data_prep.load_evaluation_data(args.file_paths)

#     # Prepare the data subset and other variables (placeholders for demonstration)
#     texts_subset = all_texts[:10]
#     llm = "some_model"
#     prompt = "some_prompt"

#     # Evaluate the model
#     generation_results = model_eval.generate_responses(texts_subset, llm, prompt)

#     # Embed the responses
#     joint_embeddings_all_llms = model_eval.embed_responses(generation_results)

#     # Perform clustering
#     combined_embeddings = joint_embeddings_all_llms  # Placeholder assignment
#     clustering = model_eval.perform_clustering(combined_embeddings)

#     # Analyze the clusters
#     rows = model_eval.analyze_clusters(joint_embeddings_all_llms, clustering)

#     # Prepare variables for visualization (placeholders for demonstration)
#     combined_embeddings = np.array([e[3] for e in joint_embeddings_all_llms])
#     iterations = 2000
#     p = 50
#     dim_reduce_tsne = TSNE(
#         perplexity=p,
#         n_iter=iterations,
#         angle=0.8,
#         init="pca",
#         early_exaggeration=22,
#         learning_rate="auto",
#         random_state=42,
#     ).fit_transform(X=combined_embeddings)

#     approvals_statements_and_embeddings = "some_data"

#     colors = ["red", "black", "green", "blue"]
#     shapes = ["o", "o", "*", "+"]
#     labels = ["Unaware", "Other AI", "Aware", "Other human"]
#     sizes = [5, 30, 200, 300]
#     order = [2, 1, 3, 0]

#     # Visualize the results
#     plot_dimension_reduction(dim_reduce_tsne, joint_embeddings_all_llms)
#     visualize_hierarchical_clustering(
#         clustering, approvals_statements_and_embeddings, rows, colors
#     )


if __name__ == "__main__":
    try:
        args = get_args()
        evaluator = EvaluatorPipeline(args)
        evaluator.setup()
        evaluator.run_evaluation()
    except Exception as e:
        print(f"An error occurred: {e}")
