import os
import argparse
import numpy as np
import pickle
import datetime
import pdb
from matplotlib import pyplot as plt
from data_preparation import DataPreparation
from model_evaluation import ModelEvaluation
from visualization import Visualization


class EvaluatorPipeline:
    def __init__(self, args):
        self.args = args
        self.data_dir = f"{os.getcwd()}/data"
        self.evals_dir = f"{self.data_dir}/evals"
        self.results_dir = f"{self.data_dir}/results/pickle_files"
        self.viz_dir = f"{self.data_dir}/results/plots"
        self.model_eval = ModelEvaluation(args)
        self.viz = Visualization(save_path=self.viz_dir)

        # Define plotting aesthetics
        self.colors = ["red", "black", "green", "blue"]
        self.shapes = ["o", "o", "*", "+"]
        self.labels = ["Google Chat", "Bing Chat", "Bing Chat Emoji", "Bing Chat Janus"]
        self.sizes = [5, 30, 200, 300]

        print(self.data_dir)
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
        self.all_texts = self.data_prep.load_evaluation_data(self.data_prep.file_paths)

    def save_results(self, results, file_name):
        results_dir = f"{self.data_dir}/results/pickle_files"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        with open(f"{results_dir}/{file_name}", "wb") as f:
            pickle.dump(results, f)

    def load_results(self, file_name):
        results_dir = f"{self.data_dir}/results/pickle_files"
        with open(f"{results_dir}/{file_name}", "rb") as f:
            return pickle.load(f)

    def generate_plot_filename(self, model_family, model, plot_type):
        # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{self.viz_dir}/{model_family}-{model}-{plot_type}.png"
        return filename

    def run_short_text_tests(self):
        text_subset = self.model_eval.load_and_preprocess_data(
            self.data_prep, n_points=3
        )
        print(text_subset)
        generation_results, file_name = self.model_eval.run_short_text_tests(
            text_subset
        )

        print("Embedding responses...")
        # pdb.set_trace()
        print(generation_results)
        print(self.model_eval)
        print(self.args.model)
        combined_embeddings = self.model_eval.embed_responses(
            generation_results=generation_results, llms=[self.args.model]
        )
        embeddings = np.array(combined_embeddings, dtype=np.float64)
        if not np.isfinite(embeddings).all():
            print("Embeddings contain non-finite values.")

        # Perform clustering and store the results
        print("Running clustering...")
        print(embeddings)
        self.model_eval.run_clustering(embeddings)

        # Choose which clustering result to analyze further
        print("Analyzing clusters...")
        chosen_clustering = self.model_eval.clustering_results["KMeans"]
        labels = chosen_clustering.labels_
        plt.hist(labels, bins=3)

        # Perform dimensionality reduction
        print("Performing dimensionality reduction...")
        dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
            embeddings, iterations=300, perplexity=2
        )
        print("Type of dim_reduce_tsne:", type(dim_reduce_tsne))
        if not np.isfinite(dim_reduce_tsne).all():
            print("dim_reduce_tsne contains non-finite values.")
        if np.isnan(dim_reduce_tsne).any() or np.isinf(dim_reduce_tsne).any():
            print("dim_reduce_tsne contains NaN or inf values.")
        print("dim_reduce_tsne:", dim_reduce_tsne.dtype)

        # Visualizations
        tsne_filename = self.generate_plot_filename(
            self.args.model_family, self.args.model, "tsne_embedding_responses"
        )
        approval_filename = self.generate_plot_filename(
            self.args.model_family, self.args.model, "approval_responses"
        )
        pdb.set_trace()
        self.viz.plot_embedding_responses(
            dim_reduce_tsne, labels, [self.args.model], tsne_filename
        )
        plt.hist(labels, bins=50)
        plt.show()

        # Analyze the clusters and get a summary table
        print("Compiling cluster table...")
        print(chosen_clustering)

        rows = self.model_eval.analyze_clusters(chosen_clustering, embeddings)

        # Save and display the results
        print("Saving and displaying results...")
        self.model_eval.save_and_display_results(chosen_clustering, rows)

        self.run_approvals_based_evalusation_and_plotting(approval_filename)
        self.viz.visualize_hierarchical_clustering(chosen_clustering, rows)

    def run_evaluation(self):
        # Generate responses and embed them
        generation_results = self.model_eval.generate_responses(
            self.all_texts[: self.args.texts_subset], self.args.llm, self.args.prompt
        )
        combined_embeddings = self.model_eval.embed_responses(generation_results)

        # Perform clustering and store the results
        self.model_eval.run_clustering(combined_embeddings)

        # Choose which clustering result to analyze further
        chosen_clustering = self.model_eval.clustering_results["Spectral"]
        labels = chosen_clustering.labels_

        # Analyze the clusters and get a summary table
        rows = self.model_eval.analyze_clusters(chosen_clustering)

        # Save and display the results
        self.model_eval.save_and_display_results(chosen_clustering, rows)

        # Perform dimensionality reduction
        dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
            combined_embeddings, iterations=2000, p=50
        )

        # Visualizations
        tsne_filename = self.generate_plot_filename(
            self.args.model_family, self.args.model, "tsne_embedding_responses"
        )
        approval_filename = self.generate_plot_filename(
            self.args.model_family, self.args.model, "approval_responses"
        )
        self.viz.plot_embedding_responses(
            dim_reduce_tsne, labels, [self.args.model], tsne_filename
        )
        self.run_approvals_based_evalusation_and_plotting(approval_filename)
        self.viz.visualize_hierarchical_clustering(chosen_clustering, rows)

    def run_approvals_based_evaluation_and_plotting(self, approval_filename):
        # Generate the data for approvals
        approvals_statements_and_embeddings = (
            self.model_eval.generate_approval_responses(
                self.all_texts[: self.args.texts_subset],
                self.args.model_family,
                self.args.model,
                self.args.prompt_template,
                self.args.role_description,
            )
        )

        # Perform dimensionality reduction on the embeddings part of the approvals data
        # assuming that the embeddings are the second element in the tuple
        embeddings = np.array(
            [approval[1] for approval in approvals_statements_and_embeddings]
        )
        dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
            embeddings, iterations=2000, p=50
        )

        # Extract the condition (approval or disapproval) from the approvals data
        # assuming that the condition is the first element in the tuple
        conditions = np.array(
            [approval[0] for approval in approvals_statements_and_embeddings]
        )

        # Approvals
        self.viz.plot_approvals(
            dim_reduce_tsne,
            conditions,
            approval_filename,
            condition=1,
            colors=self.colors,
            shapes=self.shapes,
            labels=self.labels,
            sizes=self.sizes,
            title="Embeddings of approvals for different chat modes",
        )

        # Disapprovals
        self.viz.plot_approvals(
            dim_reduce_tsne,
            conditions,
            approval_filename,
            condition=0,
            colors=self.colors,
            shapes=self.shapes,
            labels=self.labels,
            sizes=self.sizes,
            title="Embeddings of disapprovals for different chat modes",
        )

        # No response for different chat modes
        self.viz.plot_approvals(
            dim_reduce_tsne,
            conditions,
            approval_filename,
            condition=-1,
            colors=self.colors,
            shapes=self.shapes,
            labels=self.labels,
            sizes=self.sizes,
            title="Embeddings of no response for different chat modes",
        )
