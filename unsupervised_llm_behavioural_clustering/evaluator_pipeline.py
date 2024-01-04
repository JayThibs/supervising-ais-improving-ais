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
from utils import embed_texts, load_pkl_or_not


class EvaluatorPipeline:
    def __init__(self, args):
        self.args = args
        # Set up directories
        self.test_mode = args.test_mode
        self.data_dir = f"{os.getcwd()}/data"
        self.evals_dir = f"{self.data_dir}/evals"
        self.results_dir = f"{self.data_dir}/results"
        self.pickle_dir = f"{self.results_dir}/pickle_files"
        self.viz_dir = f"{self.results_dir}/plots"

        # model information
        self.llms = []
        print(f"self.args.model: {self.args.model}")
        print(f"self.args.model type: {type(self.args.model)}")
        for model in self.args.model:
            if "gpt-" in model and (
                not any(char.isdigit() for char in model.split("-")[-1][0])
                or "turbo" in model
            ):
                model_family = "openai"
            elif "claude" in model:
                model_family = "anthropic"
            else:
                model_family = "local"
            self.llms.append((model_family, model))

        # Load approval prompts from same directory as this file
        with open(f"{self.data_dir}/prompts/approval_prompts.json", "r") as file:
            # { "google_chat_desc": [ "prompt"], "bing_chat_desc": [...], ...}
            self.approval_prompts = json.load(file)
            self.personas = list(self.approval_prompts.keys())
            self.approval_prompts = [
                prompt for prompt in self.approval_prompts.values()
            ]
        self.approval_question_prompt_template = self.args.approval_prompt_template
        self.set_reuse_flags()
        self.hide_plots = self.process_hide_plots(self.args.hide_plots)
        self.plot_statement_clustering = False

        # Set up objects
        self.model_eval = ModelEvaluation(args, self.llms)
        self.viz = Visualization(save_path=self.viz_dir, personas=self.personas)

        if self.args.new_generation:
            self.saved_query_results = None
            if "all_query_results.pickle" in os.listdir(self.pickle_dir):
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                os.rename(
                    f"{self.pickle_dir}/all_query_results.pickle",
                    f"{self.pickle_dir}/all_query_results_{timestamp}.pickle",
                )
        else:
            if "all_query_results.pickle" in os.listdir(self.pickle_dir):
                self.saved_query_results = self.load_results(
                    "all_query_results.pickle", "pickle_files"
                )
            else:
                self.query_results = None

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

    def process_reuse_data(self, reuse_data, all_data_types):
        reuse_types = set()

        if "all" in reuse_data:
            reuse_types = all_data_types.copy()
            for item in reuse_data:
                if item.startswith("!"):
                    exclude_type = item[1:]
                    reuse_types.discard(exclude_type)
        else:
            for item in reuse_data:
                if item in all_data_types:
                    reuse_types.add(item)

        return reuse_types

    def set_reuse_flags(self):
        data_types = [
            "embedding_clustering",
            "joint_embeddings",
            "tsne",
            "approvals",
            "hierarchical",
            "awareness",
            "cluster_rows",
            "conditions",
        ]

        reuse_data_types = self.process_reuse_data(self.args.reuse_data, data_types)
        print("Data types to reuse:", reuse_data_types)

        for data_type in data_types:
            setattr(self, f"reuse_{data_type}", data_type in reuse_data_types)

    def process_hide_plots(self, hide_plots):
        all_plot_types = {"tsne", "approval", "awareness", "hierarchical"}
        hide_types = set()

        if "all" in hide_plots:
            hide_types = all_plot_types.copy()
            for plot_type in hide_plots:
                if plot_type.startswith("!"):
                    include_type = plot_type[1:]
                    hide_types.discard(include_type)
        else:
            for plot_type in hide_plots:
                if plot_type in all_plot_types:
                    hide_types.add(plot_type)

        return hide_types

    def save_results(self, results, file_name, sub_dir):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        with open(f"{self.results_dir}/{sub_dir}/{file_name}", "wb") as f:
            pickle.dump(results, f)

    def load_results(self, file_name, sub_dir):
        with open(f"{self.results_dir}/{sub_dir}/{file_name}", "rb") as f:
            return pickle.load(f)

    def generate_plot_filename(self, model_names: list, plot_type):
        # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{self.viz_dir}/"
        for model in model_names:
            filename += f"{model}-"
        filename += f"{plot_type}.png"
        return filename

    def run_short_text_tests(self):
        if self.test_mode:
            n_statements = 300
            self.perplexity = 3

        else:
            n_statements = self.args.n_statements
            self.perplexity = 50
        self.text_subset = self.model_eval.load_and_preprocess_data(
            self.data_prep, n_statements=n_statements
        )

        if self.saved_query_results is None:
            all_query_results = []
            for llm in self.llms:
                print(f"Generating responses for {llm}...")
                query_results, file_name = self.model_eval.run_short_text_tests(
                    self.text_subset, n_statements, llm
                )
                all_query_results.append(query_results)
                self.save_results(query_results, file_name, "pickle_files")
                print(f"{file_name} saved.")
            self.save_results(
                all_query_results, "all_query_results.pickle", "pickle_files"
            )  # last saved
            print(f"{file_name} saved.")
        else:
            all_query_results = [self.saved_query_results]

        all_model_info = []
        for result in all_query_results:
            all_model_info.append(result["model_info"])
            # {"model_family": ..., "model": ..., "system_message": ...}

        print("Embedding responses...")
        # print(query_results)
        # if not exists and no reload joint embeddings flag, create joint_embeddings_all_llms, else load it
        file_loaded, joint_embeddings_all_llms = load_pkl_or_not(
            "joint_embeddings_all_llms.pkl",
            self.pickle_dir,
            self.reuse_joint_embeddings,
        )
        if not file_loaded:
            # File doesn't exist or needs to be updated, generate new content
            joint_embeddings_all_llms = self.model_eval.embed_responses(
                query_results=all_query_results, llms=self.llms
            )
            # Save the new content
            with open(f"{self.pickle_dir}/joint_embeddings_all_llms.pkl", "wb") as f:
                pickle.dump(joint_embeddings_all_llms, f)

        combined_embeddings = np.array(
            [e[3] for e in joint_embeddings_all_llms]
        )  # grab the embeddings of the inputs + responses
        embeddings = np.array(combined_embeddings, dtype=np.float64)
        if not np.isfinite(embeddings).all():
            print("Embeddings contain non-finite values.")

        # Perform clustering and store the results
        print("Running clustering...")
        self.model_eval.run_clustering(embeddings)

        # Choose which clustering result to analyze further
        print("Analyzing clusters...")
        file_loaded, chosen_clustering = load_pkl_or_not(
            "chosen_clustering.pkl", self.pickle_dir, self.reuse_embedding_clustering
        )
        if not file_loaded:
            # File doesn't exist or needs to be updated, generate new content
            chosen_clustering = self.model_eval.clustering_results["KMeans"]
            # Save the new content
            with open(f"{self.pickle_dir}/chosen_clustering.pkl", "wb") as f:
                pickle.dump(chosen_clustering, f)
        labels = chosen_clustering.labels_
        plt.hist(labels, bins=3)

        # Perform dimensionality reduction
        print("Performing dimensionality reduction...")
        dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
            embeddings, iterations=300, perplexity=self.perplexity
        )
        print("Type of dim_reduce_tsne:", type(dim_reduce_tsne))
        if not np.isfinite(dim_reduce_tsne).all():
            print("dim_reduce_tsne contains non-finite values.")
        if np.isnan(dim_reduce_tsne).any() or np.isinf(dim_reduce_tsne).any():
            print("dim_reduce_tsne contains NaN or inf values.")
        print("dim_reduce_tsne:", dim_reduce_tsne.dtype)

        # Visualizations
        # grab list of models from self.llms
        model_names = [llm[1] for llm in self.llms]
        tsne_filename = self.generate_plot_filename(
            model_names, "tsne_embedding_responses"
        )
        approval_filename = self.generate_plot_filename(
            model_names, "approval_responses"
        )
        if "tsne" not in self.hide_plots:
            self.viz.plot_embedding_responses(
                dim_reduce_tsne, joint_embeddings_all_llms, model_names, tsne_filename
            )
        # plt.hist(labels, bins=50)
        # plt.show()
        # plt.close()

        # Analyze the clusters and get a summary table
        print("Compiling cluster table...")

        file_loaded, rows = load_pkl_or_not(
            "rows.pkl", self.pickle_dir, self.reuse_cluster_rows
        )
        if not file_loaded:
            # File doesn't exist or needs to be updated, generate new content
            rows = self.model_eval.analyze_clusters(
                chosen_clustering,
                joint_embeddings_all_llms,
                query_results=all_query_results,
            )
            # Save the new content
            with open(f"{self.pickle_dir}/rows.pkl", "wb") as f:
                pickle.dump(rows, f)

        # Save and display the results
        print("Saving and displaying results...")
        self.model_eval.save_and_display_results(chosen_clustering, rows)

        self.run_approvals_based_evaluation_and_plotting(approval_filename)
        self.clustering_obj = Clustering(self.args)
        statement_clustering = self.clustering_obj.cluster_persona_embeddings(
            self.statement_embeddings,
            n_clusters=120,
            plot=self.plot_statement_clustering,
        )
        print(f"labels: {labels}")
        self.clustering_obj.cluster_approval_stats(
            self.approvals_statements_and_embeddings,
            statement_clustering,
            all_model_info,
            self.reuse_cluster_rows,
        )
        print("Calculating hierarchical cluster data...")
        file_loaded, hierarchy_data = load_pkl_or_not(
            "hierarchy_data.pkl", self.pickle_dir, self.reuse_conditions
        )
        if not file_loaded:
            # File doesn't exist or needs to be updated, generate new content
            hierarchy_data = self.clustering_obj.calculate_hierarchical_cluster_data(
                statement_clustering,
                self.approvals_statements_and_embeddings,
                rows,
            )  # (Z, leaf_labels, original_cluster_sizes, merged_cluster_sizes)
            # Save the new content
            with open(f"{self.pickle_dir}/hierarchy_data.pkl", "wb") as f:
                pickle.dump(hierarchy_data, f)
        print("Visualizing hierarchical cluster...")
        if "hierarchical" not in self.hide_plots:
            self.viz.visualize_hierarchical_cluster(
                hierarchy_data,
                plot_type="approval",
                labels=labels,
            )

        with open(f"{self.pickle_dir}/conditions.pkl", "rb") as f:
            be_nice_conditions = pickle.load(f)

        pdb.set_trace()
        print(f"be_nice_conditions: {be_nice_conditions}")

        with open(
            f"{self.pickle_dir}/Anthropic_5000_random_statements_embs.pkl", "rb"
        ) as f:
            Anthropic_5000_random_statements_embs = pickle.load(f)
            Anthropic_5000_random_statements_embs = (
                Anthropic_5000_random_statements_embs[:n_statements]
            )

        data_include_statements_and_embeddings_4_prompts = [
            [[], s[0], s[2]] for s in Anthropic_5000_random_statements_embs
        ]  # [empty list for conditions, statement, statement_embedding]

        print(
            f"data_include_statements_and_embeddings_4_prompts[0]: {data_include_statements_and_embeddings_4_prompts[0]}"
        )

        pdb.set_trace()
        for condition in be_nice_conditions:
            for i, record in enumerate(condition):
                print(f"record: {record}")
                print(f"i: {i}")
                if record == 0:
                    data_include_statements_and_embeddings_4_prompts[i][0].append(0)
                elif record == 1:
                    data_include_statements_and_embeddings_4_prompts[i][0].append(1)
                else:
                    data_include_statements_and_embeddings_4_prompts[i][0].append(-1)

        pdb.set_trace()
        print(
            f"data_include_statements_and_embeddings_4_prompts: {data_include_statements_and_embeddings_4_prompts}"
        )
        print(
            f"len(data_include_statements_and_embeddings_4_prompts): {len(data_include_statements_and_embeddings_4_prompts)}"
        )
        print(
            f"len(data_include_statements_and_embeddings_4_prompts[0]): {len(data_include_statements_and_embeddings_4_prompts[0])}"
        )
        print(
            f"len(data_include_statements_and_embeddings_4_prompts[0][0]): {len(data_include_statements_and_embeddings_4_prompts[0][0])}"
        )
        print(
            f"len(data_include_statements_and_embeddings_4_prompts[0][1]): {len(data_include_statements_and_embeddings_4_prompts[0][1])}"
        )
        print(
            f"len(data_include_statements_and_embeddings_4_prompts[0][2]): {len(data_include_statements_and_embeddings_4_prompts[0][2])}"
        )

        statement_embeddings = np.array(
            [e[2] for e in data_include_statements_and_embeddings_4_prompts]
        )

        # TODO: What is the difference between this statement_clustering and the one above?
        statement_clustering = self.clustering_obj.cluster_persona_embeddings(
            self.statement_embeddings
        )

        if "awareness" not in self.hide_plots:
            self.viz.plot_approvals(
                dim_reduce_tsne,
                data_include_statements_and_embeddings_4_prompts,  # TODO: ???
                1,
                plot_type="awareness",
                filename="embedding_of_approvals_diff_chats.png",
                title=f"Embeddings of approvals for different chat modes",
            )

        print("Done. Please check the results directory for the plots.")

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
            combined_embeddings, iterations=2000, perplexity=self.perplexity
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
        file_loaded, self.approvals_statements_and_embeddings = load_pkl_or_not(
            "approvals_statements_and_embeddings_G_B_BE.pkl",
            self.pickle_dir,
            self.reuse_approvals,
        )

        if not file_loaded:
            print("Embedding statements...")
            statement_embeddings = embed_texts(
                self.text_subset.tolist(), model="text-embedding-ada-002"
            )
            print("Generating approvals...")
            print(self.text_subset)
            print(type(self.text_subset))
            all_condition_approvals = load_pkl_or_not(
                "all_condition_approvals.pkl",
                self.pickle_dir,
                self.reuse_conditions,
            )

            if not self.reuse_conditions:
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
                if not os.path.exists(f"{self.pickle_dir}/all_condition_approvals.pkl"):
                    pickle.dump(
                        all_condition_approvals,
                        open(
                            f"{self.pickle_dir}/all_condition_approvals.pkl",
                            "wb",
                        ),
                    )
            print(f"all_condition_approvals: {all_condition_approvals}")

            self.approvals_statements_and_embeddings = []
            for i in range(len(self.text_subset)):
                print(f"Approvals record {i}")
                record_approvals = [
                    condition_approvals[i]
                    for condition_approvals in all_condition_approvals
                ]
                print(f"record_approvals: {record_approvals}")
                record = [
                    record_approvals,
                    self.text_subset[i],
                    statement_embeddings[i],
                ]
                print(f"record: {record}")
                self.approvals_statements_and_embeddings.append(record)
                pickle.dump(
                    self.approvals_statements_and_embeddings,
                    open(
                        f"{self.pickle_dir}/approvals_statements_and_embeddings_G_B_BE.pkl",
                        "wb",
                    ),
                )

        # Perform dimensionality reduction on the embeddings part of the approvals data
        # assuming that the embeddings are the second element in the tuple
        print("Performing dimensionality reduction...")
        self.statement_embeddings = np.array(
            [approval[2] for approval in self.approvals_statements_and_embeddings]
        )
        # saving the np array of statement embeddings
        np.save(
            f"{os.getcwd()}/data/results/statement_embeddings.npy",
            self.statement_embeddings,
        )
        dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
            self.statement_embeddings, iterations=2000, perplexity=self.perplexity
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
        if "approvals" not in self.hide_plots:
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
                    plot_type="approval",
                    filename=approval_filename,
                    title=f"Embeddings of {condition_title} for different chat modes",
                )
