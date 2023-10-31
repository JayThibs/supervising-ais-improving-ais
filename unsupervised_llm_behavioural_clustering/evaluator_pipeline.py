import os
import argparse
import numpy as np
import pickle
from data_preparation import DataPreparation
from model_evaluation import ModelEvaluation
from visualization import Visualization


class EvaluatorPipeline:
    def __init__(self, args):
        self.model_eval = ModelEvaluation()
        # self.viz = Visualization()
        self.args = args
        self.data_dir = f"{os.getcwd()}/data"
        print(self.data_dir)
        self.evals_dir = f"{self.data_dir}/evals"
        print(self.evals_dir)

    def setup(self):
        # Create a new data/evals directory if it doesn't exist
        if not os.path.exists(self.evals_dir):
            os.makedirs(self.evals_dir)

        self.api_key = DataPreparation.load_api_key(
            self, "OPENAI_API_KEY"
        )  # TODO: Make more general to include anthropic and local models

        # Clone the anthropic evals repo inside the data/evals directory
        if not os.path.exists(
            os.path.join(self.evals_dir, "anthropic-model-written-evals")
        ):
            DataPreparation.clone_repo(
                self,
                "https://github.com/anthropics/evals.git",
                "anthropic-model-written-evals",
            )
        self.data_prep = DataPreparation()
        self.all_texts = self.data_prep.load_evaluation_data()

    def save_results(self, results, file_name):
        with open(f"{self.data_dir}/{file_name}", "wb") as f:
            pickle.dump(results, f)

    def load_results(self, file_name):
        with open(f"{self.data_dir}/{file_name}", "rb") as f:
            return pickle.load(f)

    def run_short_text_tests(self):
        self.model_eval.load_and_preprocess_data(n_points=self.args.n_points)
        self.model_eval.run_short_text_tests()

    def run_evaluation(self):
        # Generate responses and embed them
        generation_results = self.model_eval.generate_responses(
            self.all_texts[: self.args.texts_subset], self.args.llm, self.args.prompt
        )
        joint_embeddings_all_llms = self.model_eval.embed_responses(generation_results)

        # Perform clustering and store the results
        self.model_eval.run_clustering(joint_embeddings_all_llms)

        # Choose which clustering result to analyze further
        chosen_clustering = self.model_eval.clustering_results["Spectral"]

        # Analyze the clusters and get a summary table
        rows = self.model_eval.analyze_clusters(chosen_clustering)

        # Save and display the results
        self.model_eval.save_and_display_results(chosen_clustering, rows)

        # Perform dimensionality reduction
        dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
            joint_embeddings_all_llms, iterations=2000, p=50
        )

        # Visualizations
        self.viz.plot_dimension_reduction(dim_reduce_tsne)
        self.viz.plot_embedding_responses(joint_embeddings_all_llms)
        self.run_approvals_based_evalusation_and_plotting()
        self.viz.visualize_hierarchical_clustering(chosen_clustering, rows)

    def run_approvals_based_evalusation_and_plotting(self):
        approvals_statements_and_embeddings = self.model_eval.generate_approval_data()
        dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
            approvals_statements_and_embeddings
        )

        self.viz.plot_approvals(dim_reduce_tsne, approvals_statements_and_embeddings)
