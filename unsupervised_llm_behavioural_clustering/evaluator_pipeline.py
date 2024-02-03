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
from clustering import Clustering, ClusteringArgs
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

    # Set up directories
    def setup_directories(self):
        dirs = [self.data_dir, self.evals_dir, self.results_dir, self.viz_dir]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def load_api_key(self):
        self.api_key = DataPreparation.load_api_key(self, "OPENAI_API_KEY")

    def clone_evals_repo(self):
        if not os.path.exists(
            os.path.join(self.evals_dir, "anthropic-model-written-evals")
        ):
            DataPreparation.clone_repo(
                self,
                "https://github.com/anthropics/evals.git",
                "anthropic-model-written-evals",
            )

    def load_evaluation_data(self):
        self.data_prep = DataPreparation()
        all_texts = self.data_prep.load_evaluation_data(self.data_prep.file_paths)
        return all_texts

    def process_reuse_data(self, reuse_data, all_data_types):
        reuse_types = set()

        if "all" in reuse_data:
            reuse_types = set(all_data_types)  # Ensure it's a set
            for item in reuse_data:
                if item.startswith("!"):
                    exclude_type = item[1:]
                    reuse_types.discard(exclude_type)  # discard method works on sets
        else:
            for item in reuse_data:
                if item in all_data_types:
                    reuse_types.add(item)  # add method to include in the set

        return reuse_types

    def set_reuse_flags(self):
        data_types = [
            "embedding_clustering",
            "joint_embeddings",
            "tsne",
            "approvals",
            "hierarchical_approvals",
            "hierarchical_awareness",
            "awareness",
            "cluster_rows",
            "conditions",
        ]

        reuse_data_types = self.process_reuse_data(self.args.reuse_data, data_types)
        print("Data types to reuse:", reuse_data_types)

        for data_type in data_types:
            setattr(self, f"reuse_{data_type}", data_type in reuse_data_types)

    def process_hide_plots(self, hide_plots):
        all_plot_types = [
            "tsne",
            "approval",
            "awareness",
            "hierarchical_approvals",
            "hierarchical_awareness",
            "spectral",
        ]
        hide_types = []

        if "all" not in hide_plots:
            hide_types = all_plot_types
        else:
            for plot_type in hide_plots:
                if plot_type.startswith("!"):
                    include_plot_type = plot_type[1:]
                    if include_plot_type in all_plot_types:
                        hide_types.remove(include_plot_type)

        return hide_types

    def save_results(self, data, file_name, sub_dir):
        # Save data to a pickle file
        if not os.path.exists(f"{self.results_dir}/{sub_dir}"):
            os.makedirs(f"{self.results_dir}/{sub_dir}")
        with open(f"{self.results_dir}/{sub_dir}/{file_name}", "wb") as f:
            pickle.dump(data, f)

    def load_results(self, file_name, sub_dir):
        # Load data from a pickle file
        try:
            with open(f"{self.results_dir}/{sub_dir}/{file_name}", "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def generate_plot_filename(self, model_names: list, plot_type):
        # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"{self.viz_dir}/"
        for model in model_names:
            filename += f"{model}-"
        filename += f"{plot_type}.png"
        return filename

    def generate_responses(self):
        n_statements = 300 if self.test_mode else self.args.n_statements
        self.perplexity = 3 if self.test_mode else 50
        text_subset = self.model_eval.load_and_preprocess_data(
            self.data_prep, n_statements=n_statements
        )

        if self.saved_query_results is None:
            all_query_results = self.generate_and_save_responses(n_statements)
        else:
            all_query_results = [self.saved_query_results]

        return text_subset, all_query_results

    def generate_and_save_responses(self, n_statements):
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
        return all_query_results

    def collect_model_info(self, all_query_results):
        print("Collecting model info...")
        print(f"self.all_query_results: {all_query_results}")
        all_model_info = [result["model_info"] for result in all_query_results]
        return all_model_info

    def perform_tsne_dimensionality_reduction(self, combined_embeddings):
        print("Performing t-SNE dimensionality reduction...")
        dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
            combined_embeddings, iterations=300, perplexity=self.perplexity
        )
        self.check_tsne_values(dim_reduce_tsne)
        return dim_reduce_tsne

    def check_tsne_values(self, dim_reduce_tsne):
        if not np.isfinite(dim_reduce_tsne).all():
            print("dim_reduce_tsne contains non-finite values.")
        if np.isnan(dim_reduce_tsne).any() or np.isinf(dim_reduce_tsne).any():
            print("dim_reduce_tsne contains NaN or inf values.")
        print("dim_reduce_tsne:", dim_reduce_tsne.dtype)

    def visualize_results(self, dim_reduce_tsne, joint_embeddings_all_llms):
        model_names = [llm[1] for llm in self.llms]
        tsne_filename = self.generate_plot_filename(
            model_names, "tsne_embedding_responses"
        )
        self.approval_filename = self.generate_plot_filename(
            model_names, "approval_responses"
        )
        if "tsne" not in self.hide_plots:
            self.viz.plot_embedding_responses(
                dim_reduce_tsne,
                joint_embeddings_all_llms,
                model_names,
                tsne_filename,
            )

    def embed_responses(self):
        print("Embedding responses...")
        file_loaded, joint_embeddings_all_llms = load_pkl_or_not(
            "joint_embeddings_all_llms.pkl",
            self.pickle_dir,
            self.reuse_joint_embeddings,
        )
        if not file_loaded:
            joint_embeddings_all_llms = self.generate_and_save_embeddings()

        combined_embeddings = np.array(
            [e[3] for e in joint_embeddings_all_llms]
        )  # grab the embeddings of the inputs + responses
        combined_embeddings = np.array(combined_embeddings, dtype=np.float64)
        if not np.isfinite(combined_embeddings).all():
            print("Embeddings contain non-finite values.")

        return joint_embeddings_all_llms, combined_embeddings

    def generate_and_save_embeddings(self):
        joint_embeddings_all_llms = self.model_eval.embed_responses(
            query_results=self.all_query_results, llms=self.llms
        )
        with open(f"{self.pickle_dir}/joint_embeddings_all_llms.pkl", "wb") as f:
            pickle.dump(joint_embeddings_all_llms, f)
        return joint_embeddings_all_llms

    def run_clustering(self, combined_embeddings):
        print("Running clustering...")
        self.model_eval.run_clustering(combined_embeddings)
        print("Analyzing clusters...")
        file_loaded, chosen_clustering = load_pkl_or_not(
            "chosen_clustering.pkl", self.pickle_dir, self.reuse_embedding_clustering
        )
        if not file_loaded:
            chosen_clustering = self.generate_and_save_chosen_clustering()
        return chosen_clustering

    def generate_and_save_chosen_clustering(self):
        chosen_clustering = self.model_eval.clustering_results["KMeans"]
        with open(f"{self.pickle_dir}/chosen_clustering.pkl", "wb") as f:
            pickle.dump(chosen_clustering, f)
        return chosen_clustering

    def analyze_clusters(self, chosen_clustering):
        print("Analyzing clusters...")
        file_loaded, rows = load_pkl_or_not(
            "rows.pkl", self.pickle_dir, self.reuse_cluster_rows
        )
        if not file_loaded:
            rows = self.generate_and_save_cluster_analysis()
        # Save and display the results
        print("Saving and displaying results...")
        self.model_eval.save_and_display_results(chosen_clustering, rows)
        return rows

    def generate_and_save_cluster_analysis(self):
        rows = self.model_eval.analyze_clusters(
            self.chosen_clustering,
            self.joint_embeddings_all_llms,
            self.all_query_results,
        )
        with open(f"{self.pickle_dir}/rows.pkl", "wb") as f:
            pickle.dump(rows, f)
        return rows

    def run_approvals_based_evaluation(self, approval_filename):
        # TODO: Refactor run_approvals_based_evaluation_and_plotting
        statement_embeddings = self.run_approvals_based_evaluation_and_plotting(
            self.approval_filename
        )  # ?????????
        self.clustering_obj = Clustering(self.args)
        statement_clustering = self.clustering_obj.cluster_persona_embeddings(
            statement_embeddings,
            n_clusters=120,
            spectral_plot=False if "spectral" in self.hide_plots else True,
        )
        self.clustering_obj.cluster_approval_stats(
            self.approvals_statements_and_embeddings,
            statement_clustering,
            self.all_model_info,
            self.reuse_cluster_rows,
        )

    def perform_hierarchical_clustering(self):
        print("Calculating hierarchical cluster data...")
        file_loaded, self.hierarchy_data = load_pkl_or_not(
            "hierarchy_approval_data.pkl",
            self.pickle_dir,
            self.reuse_hierarchical_approvals,
        )
        if not file_loaded:
            self.hierarchy_data = self.generate_and_save_hierarchical_data()

    def generate_and_save_hierarchical_data(self):
        hierarchy_data = self.clustering_obj.calculate_hierarchical_cluster_data(
            self.statement_clustering,
            self.approvals_statements_and_embeddings,
            self.rows,
        )
        with open(f"{self.pickle_dir}/hierarchy_approval_data.pkl", "wb") as f:
            pickle.dump(hierarchy_data, f)
        return hierarchy_data

    def visualize_hierarchical_clusters(self):
        print("Visualizing hierarchical cluster...")
        if "hierarchical_approvals" not in self.hide_plots:
            for model_name in self.model_names:
                filename = (
                    f"{self.viz_dir}/hierarchical_clustering_approval_{model_name}"
                )
                self.viz.visualize_hierarchical_cluster(
                    self.hierarchy_data,
                    plot_type="approval",
                    labels=self.labels,
                    bar_height=0.7,
                    bb_width=40,
                    x_leftshift=0,
                    y_downshift=0,
                    filename=filename,
                )

    def run_evaluations(self):
        # Set up
        self.setup_directories()
        self.load_api_key()
        self.clone_evals_repo()
        # Load data
        all_texts = self.load_evaluation_data()
        # Generate responses to statement prompts
        text_subset, all_query_results = self.generate_responses()
        self.all_model_info = self.collect_model_info(all_query_results)
        # Embed model responses to statement prompts
        joint_embeddings_all_llms, combined_embeddings = self.embed_responses()
        #
        chosen_clustering = self.run_clustering(combined_embeddings)

        labels = self.chosen_clustering.labels_
        print(f"labels: {labels}")
        plt.hist(labels, bins=3)
        # plt.show()
        # plt.close()

        dim_reduce_tsne = self.perform_tsne_dimensionality_reduction(
            combined_embeddings
        )
        self.visualize_results(dim_reduce_tsne, joint_embeddings_all_llms)

        clust_res = ClusteringArgs()
        # save as json
        with open(f"{self.pickle_dir}/clustering_result.json", "w") as f:
            json.dump(clust_res, f)

        rows = self.analyze_clusters(chosen_clustering)
        self.run_approvals_based_evaluation()
        self.perform_hierarchical_clustering()
        self.visualize_hierarchical_clusters()

        with open(f"{self.pickle_dir}/conditions.pkl", "rb") as f:
            be_nice_conditions = pickle.load(f)

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

        for i, condition in enumerate(be_nice_conditions):
            for record in condition:
                if record == 0:
                    data_include_statements_and_embeddings_4_prompts[i][0].append(0)
                elif record == 1:
                    data_include_statements_and_embeddings_4_prompts[i][0].append(1)
                else:
                    data_include_statements_and_embeddings_4_prompts[i][0].append(-1)

        statement_embeddings = np.array(
            [e[2] for e in data_include_statements_and_embeddings_4_prompts]
        )

        # TODO: What is the difference between this statement_clustering and the one above?
        statement_clustering = self.clustering_obj.cluster_persona_embeddings(
            statement_embeddings,
            spectral_plot=False if "spectral" in self.hide_plots else True,
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

        # hierarchical clustering for awareness
        print("Calculating hierarchical cluster data for awareness prompts...")
        file_loaded, hierarchy_data = load_pkl_or_not(
            "hierarchy_awareness_data.pkl",
            self.pickle_dir,
            self.reuse_hierarchical_awareness,
        )
        if not file_loaded:
            # File doesn't exist or needs to be updated, generate new content
            hierarchy_data = self.clustering_obj.calculate_hierarchical_cluster_data(
                statement_clustering,
                data_include_statements_and_embeddings_4_prompts,
                rows,
            )
            # Save the new content
            with open(f"{self.pickle_dir}/hierarchy_awareness_data.pkl", "wb") as f:
                pickle.dump(hierarchy_data, f)

        print("Visualizing hierarchical cluster...")
        if "hierarchical_awareness" not in self.hide_plots:
            for model_name in model_names:
                filename = (
                    f"{self.viz_dir}/hierarchical_clustering_awareness_{model_name}"
                )
                self.viz.visualize_hierarchical_cluster(
                    hierarchy_data,
                    plot_type="awareness",
                    labels=labels,
                    bar_height=0.7,
                    bb_width=40,
                    x_leftshift=0,
                    y_downshift=0,
                    filename=filename,
                )

        print("Done. Please check the results directory for the plots.")

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
        statement_embeddings = np.array(
            [approval[2] for approval in self.approvals_statements_and_embeddings]
        )
        # saving the np array of statement embeddings
        np.save(
            f"{os.getcwd()}/data/results/statement_embeddings.npy",
            statement_embeddings,
        )
        dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
            statement_embeddings, iterations=2000, perplexity=self.perplexity
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

        return statement_embeddings
