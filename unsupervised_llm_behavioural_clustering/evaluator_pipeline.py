import os
import json
import numpy as np
import pickle
from datetime import datetime
import pdb
from matplotlib import pyplot as plt
from data_preparation import DataPreparation
from model_evaluation import ModelEvaluation
from visualization import Visualization
from clustering import Clustering
from utils import embed_texts


class EvaluatorPipeline:
    def __init__(self, args):
        self.args = args
        self.test_mode = args.test_mode
        self.data_dir = f"{os.getcwd()}/data"
        self.evals_dir = f"{self.data_dir}/evals"
        self.results_dir = f"{self.data_dir}/results"
        self.pickle_dir = f"{self.results_dir}/pickle_files"
        self.viz_dir = f"{self.results_dir}/plots"
        self.model_eval = ModelEvaluation(args)
        self.viz = Visualization(save_path=self.viz_dir)

        if self.args.new_generation:
            self.saved_query_results = None
            if "test_query_results.pickle" in os.listdir(self.pickle_dir):
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                os.rename(
                    f"{self.pickle_dir}/test_query_results.pickle",
                    f"{self.pickle_dir}/test_query_results_{timestamp}.pickle",
                )
        else:
            if "test_query_results.pickle" in os.listdir(self.pickle_dir):
                self.saved_query_results = self.load_results(
                    "test_query_results.pickle", "pickle_files"
                )
            else:
                self.query_results = None

        # Load approval prompts from same directory as this file
        with open(f"{self.data_dir}/prompts/approval_prompts.json", "r") as file:
            # { "google_chat_desc": [ "prompt"], "bing_chat_desc": [...], ...}
            self.approval_prompts = json.load(file)
            self.approval_prompts = [
                prompt for prompt in self.approval_prompts.values()
            ]
        self.approval_question_prompt_template = self.args.approval_prompt_template
        self.use_saved_approvals = self.args.use_saved_approvals

    def setup(self):
        # Create a new data/evals directory if it doesn't exist
        dirs = [self.data_dir, self.evals_dir, self.results_dir, self.viz_dir]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

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

    def save_results(self, results, file_name, sub_dir):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        with open(f"{self.results_dir}/{sub_dir}/{file_name}", "wb") as f:
            pickle.dump(results, f)

    def load_results(self, file_name, sub_dir):
        with open(f"{self.results_dir}/{sub_dir}/{file_name}", "rb") as f:
            return pickle.load(f)

    def generate_plot_filename(self, model_family, model, plot_type):
        # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{self.viz_dir}/{model_family}-{model}-{plot_type}.png"
        return filename

    def run_short_text_tests(self):
        n_statements = 100
        self.text_subset = self.model_eval.load_and_preprocess_data(
            self.data_prep, n_statements=n_statements
        ).tolist()
        print(self.text_subset)
        if self.saved_query_results is None:
            query_results, file_name = self.model_eval.run_short_text_tests(
                self.text_subset, n_statements=n_statements
            )
            print("Saving generated results...")
            self.save_results(
                query_results, "test_query_results.pickle", "pickle_files"
            )  # last saved
            self.save_results(
                query_results, file_name, "pickle_files"
            )  # full file name
            print(f"{file_name} saved.")
        else:
            query_results = self.saved_query_results

        print("Embedding responses...")
        # pdb.set_trace()
        print(query_results)
        print(self.model_eval)
        print(self.args.model)
        joint_embeddings_all_llms = self.model_eval.embed_responses(
            query_results=query_results, llms=[self.args.model]
        )
        combined_embeddings = np.array([e[3] for e in joint_embeddings_all_llms])
        print(f"combined_embeddings: {combined_embeddings}")
        print(f"combined_embeddings.shape: {combined_embeddings.shape}")
        print(f"combined_embeddings[0].shape: {combined_embeddings[0].shape}")
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
        print(f"labels: {labels}")
        print(f"self.args.model: {self.args.model}")
        self.viz.plot_embedding_responses(
            dim_reduce_tsne, labels, [self.args.model], tsne_filename
        )
        # plt.hist(labels, bins=50)
        # plt.show()
        # plt.close()

        # Analyze the clusters and get a summary table
        print("Compiling cluster table...")
        print(chosen_clustering)

        rows = self.model_eval.analyze_clusters(
            chosen_clustering,
            joint_embeddings_all_llms,
            query_results=query_results,
        )

        # Save and display the results
        print("Saving and displaying results...")
        pdb.set_trace()
        self.model_eval.save_and_display_results(chosen_clustering, rows)

        self.run_approvals_based_evaluation_and_plotting(approval_filename)
        self.clustering_obj = Clustering(self.args)
        statement_clustering = self.clustering_obj.cluster_statement_embeddings(
            self.statement_embeddings
        )
        self.clustering_obj.cluster_approval_stats(
            self.approvals_statements_and_embeddings, statement_clustering, labels
        )
        print("Calculating hierarchical cluster data...")
        (
            Z,
            leaf_labels,
            original_cluster_sizes,
            merged_cluster_sizes,
        ) = self.clustering_obj.calculate_hierarchical_cluster_data(
            statement_clustering, self.approvals_statements_and_embeddings, rows
        )
        print("Visualizing hierarchical cluster...")
        self.viz.visualize_hierarchical_cluster(
            Z, leaf_labels, original_cluster_sizes, merged_cluster_sizes, labels
        )

        be_nice_conditions = pickle.load(
            f"{os.getcwd()}/data/results/pickle_files/conditions.pkl", "rb"
        )
        Anthropic_5000_random_statements_embs = pickle.load(
            open("Anthropic_5000_random_statements_embs.pkl", "rb")
        )

        data_include_statements_and_embeddings_4_prompts = [
            [[], s[0], s[2]] for s in Anthropic_5000_random_statements_embs
        ]

        for condition in be_nice_conditions:
            for i, record in enumerate(condition):
                if record[1] == "no":
                    data_include_statements_and_embeddings_4_prompts[i][0].append(0)
                elif record[1] == "yes":
                    data_include_statements_and_embeddings_4_prompts[i][0].append(1)
                else:
                    data_include_statements_and_embeddings_4_prompts[i][0].append(-1)

        statement_embeddings = np.array(
            [e[2] for e in data_include_statements_and_embeddings_4_prompts]
        )

        # TODO: What is the difference between this statement_clustering and the one above?
        statement_clustering = self.clustering_obj.cluster_statement_embeddings(
            self.statement_embeddings
        )
        self.viz.plot_approvals(
            dim_reduce_tsne,
            data_include_statements_and_embeddings_4_prompts,  # TODO: ???
            1,
            plot_type="awareness",
            filename="embedding_of_approvals_diff_chats.png",
            title=f"Embeddings of approvals for different chat modes",
        )

    def run_evaluation(self):
        # Generate responses and embed them
        query_results = self.model_eval.generate_responses(
            self.all_texts[: self.args.texts_subset], self.args.llm, self.args.prompt
        )
        combined_embeddings = self.model_eval.embed_responses(query_results)

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

        if self.use_saved_approvals:
            self.approvals_statements_and_embeddings = pickle.load(
                open(
                    f"{os.getcwd()}/data/results/pickle_files/approvals_statements_and_embeddings_G_B_BE.pkl",
                    "rb",
                )
            )

        else:
            print("Embedding statements...")
            statement_embeddings = embed_texts(
                self.text_subset, model="text-embedding-ada-002"
            )
            print("Generating approvals...")
            print(self.text_subset)
            print(type(self.text_subset))
            all_condition_approvals = [
                self.model_eval.get_model_approvals(
                    statements=self.text_subset,
                    prompt_template=self.approval_question_prompt_template,
                    model_family=self.args.model_family,
                    model=self.args.model,
                    system_message=role_description,
                )
                for role_description in self.approval_prompts
            ]

            self.approvals_statements_and_embeddings = []
            for i in range(len(self.text_subset)):
                record_approvals = [
                    condition_approvals[i]
                    for condition_approvals in all_condition_approvals
                ]
                record = [
                    record_approvals,
                    self.text_subset[i],
                    statement_embeddings[i],
                ]
                self.approvals_statements_and_embeddings.append(record)
                pickle.dump(
                    self.approvals_statements_and_embeddings,
                    open(
                        f"{os.getcwd()}/data/results/pickle_files/approvals_statements_and_embeddings_G_B_BE.pkl",
                        "wb",
                    ),
                )

        # Perform dimensionality reduction on the embeddings part of the approvals data
        # assuming that the embeddings are the second element in the tuple
        print("Performing dimensionality reduction...")
        pdb.set_trace()
        print(self.approvals_statements_and_embeddings)
        self.statement_embeddings = np.array(
            [approval[2] for approval in self.approvals_statements_and_embeddings]
        )
        dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
            self.statement_embeddings, iterations=2000, p=50
        )

        # Extract the condition (approval or disapproval) from the approvals data
        # assuming that the condition is the first element in the tuple
        conditions = np.array(
            [approval[0] for approval in self.approvals_statements_and_embeddings]
        )

        pickle.dump(
            conditions,
            open(f"{os.getcwd()}/data/results/pickle_files/conditions.pkl", "wb"),
        )

        # Refactored approvals plotting
        for condition in [1, 0, -1]:
            condition_title = {
                1: "approvals",
                0: "disapprovals",
                -1: "no response",
            }[condition]
            self.viz.plot_approvals(
                dim_reduce_tsne,
                self.approvals_statements_and_embeddings,
                condition,
                plot_type="approvals",
                filename=approval_filename,
                title=f"Embeddings of {condition_title} for different chat modes",
            )
