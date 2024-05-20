import os
from pathlib import Path
import json
import yaml
import numpy as np
import pickle
from datetime import datetime
import pdb
from matplotlib import pyplot as plt
from data_preparation import DataPreparation
from models import LocalModel
from model_evaluation import ModelEvaluation
from visualization import Visualization
from clustering import Clustering
from utils import (
    embed_texts,
    load_pkl_or_not,
    query_model_on_statements,
    check_gpu_availability,
    check_gpu_memory,
)
from typing import List, Dict, Any, Tuple
from config.run_settings import RunSettings


class EvaluatorPipeline:
    def __init__(self, run_settings: RunSettings):
        self.run_settings = run_settings
        self.data_dir = self.run_settings.directory_settings.data_dir
        self.evals_dir = self.run_settings.directory_settings.evals_dir
        self.results_dir = self.run_settings.directory_settings.results_dir
        self.pickle_dir = self.run_settings.directory_settings.pickle_dir
        self.viz_dir = self.run_settings.directory_settings.viz_dir
        self.tables_dir = self.run_settings.directory_settings.tables_dir

        # model information
        self.llms: List[Tuple[str, str]] = self.run_settings.model_settings.models
        self.model_names: List[str] = [model for _, model in self.llms]
        self.local_models: Dict[str, LocalModel] = {}
        for model_family, model in self.llms:
            if model_family == "local":
                self.local_models[model] = LocalModel(model_name_or_path=model)
        print(f"Models: {self.model_names}")
        self.embedding_model_name: str = (
            self.run_settings.embedding_settings.embedding_model
        )

        # Load approval prompts from same directory as this file
        self.test_mode = self.run_settings.test_mode
        self.n_statements = (
            300 if self.test_mode else self.run_settings.data_settings.n_statements
        )
        with open(f"{self.data_dir}/prompts/approval_prompts.json", "r") as file:
            # { "google_chat_desc": [ "prompt"], "bing_chat_desc": [...], ...}
            self.approval_prompts = json.load(file)
            self.persona_approval_prompts = [
                prompt for prompt in list(self.approval_prompts["personas"].values())
            ]
        # Create a loop that creates a new attribute which is a dictionary of the
        # persona prompt title and the prompt itself
        for key in self.approval_prompts.keys():
            setattr(self, key, list(self.approval_prompts[key].keys()))

        self.approval_question_prompt_template = (
            self.run_settings.prompt_settings.approval_prompt_template
        )
        self.plot_statement_clustering = False

        # Set up objects
        self.model_eval = ModelEvaluation(self.run_settings, self.llms)
        self.viz = Visualization(self.run_settings.plot_settings)
        self.clustering_obj = Clustering(self.run_settings.clustering_settings)

        if self.run_settings.data_settings.new_generation:
            self.saved_query_results = None
            if (self.pickle_dir / "query_results_per_model.pkl").exists():
                timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                (self.pickle_dir / "query_results_per_model.pkl").rename(
                    self.pickle_dir / f"query_results_per_model_{timestamp}.pkl"
                )
        else:
            if (self.pickle_dir / "query_results_per_model.pkl").exists():
                self.saved_query_results = self.load_results(
                    "query_results_per_model.pkl", "pickle_files"
                )
            else:
                self.saved_query_results = None

    def setup_evaluations(self):
        self.setup_directories()
        self.load_api_key()
        self.clone_evals_repo()

    def get_model_batches(self):
        model_batches = []
        batch = []
        total_memory = 0

        for model_name, local_model in self.local_models.items():
            model_memory = local_model.get_memory_usage()
            if total_memory + model_memory <= self.max_gpu_memory:
                batch.append((model_name, local_model))
                total_memory += model_memory
            else:
                model_batches.append(batch)
                batch = [(model_name, local_model)]
                total_memory = model_memory

        if batch:
            model_batches.append(batch)

        return model_batches

    # Set up directories
    def setup_directories(self):
        dirs = [
            self.data_dir,
            self.evals_dir,
            self.results_dir,
            self.viz_dir,
            self.tables_dir,
            self.pickle_dir,
        ]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def load_api_key(self):
        self.api_key = DataPreparation.load_api_key(self, "OPENAI_API_KEY")

    def clone_evals_repo(self, repos=None):
        if repos is None:
            repos = [
                (
                    "https://github.com/anthropics/evals.git",
                    "anthropic-model-written-evals",
                )
            ]

        for repo_url, folder_name in repos:
            if not (self.evals_dir / folder_name).exists():
                DataPreparation.clone_repo(self, repo_url, folder_name)

    def load_evaluation_data(self, dataset_names: List[str] = ["anthropic"]):
        self.dataset_names_filename: str = "_".join(dataset_names)
        self.data_prep = DataPreparation()
        all_texts = self.data_prep.load_evaluation_data(self.data_prep.file_paths)
        return all_texts

    def save_results(self, data, file_name, sub_dir):
        # Save data to a pickle file
        if not os.path.exists(self.results_dir / sub_dir):
            os.makedirs(self.results_dir / sub_dir)
        with open(self.results_dir / sub_dir / file_name, "wb") as f:
            pickle.dump(data, f)

    def load_results(self, file_name, sub_dir):
        # Load data from a pickle file
        try:
            with open(self.results_dir / sub_dir / file_name, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None

    def save_run_metadata_to_yaml(
        self,
        run_id: str,
        data_files: Dict[str, str],
    ):
        metadata = {
            "run_id": run_id,
            "dataset_names": self.dataset_names_filename,
            "model_names": self.model_names,
            "n_statements": self.n_statements,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_settings": self.run_settings.data_settings.__dict__,
            "model_settings": self.run_settings.model_settings.__dict__,
            "embedding_settings": self.run_settings.embedding_settings.__dict__,
            "prompt_settings": self.run_settings.prompt_settings.__dict__,
            "plot_settings": self.run_settings.plot_settings.__dict__,
            "clustering_settings": self.run_settings.clustering_settings.__dict__,
            "tsne_settings": self.run_settings.tsne_settings.__dict__,
            "test_mode": self.run_settings.test_mode,
            "skip_sections": self.run_settings.skip_sections,
            "data_files": data_files,
        }

        with open(self.run_settings.directory_settings.metadata_file, "a") as f:
            yaml.dump([metadata], f)
            f.write("\n")  # Add a newline separator between runs

    def get_or_create_data_file_path(
        self, data_file_id: str, data_file_dir: Path, mapping_file: Path, **kwargs
    ):
        with open(mapping_file, "r") as f:
            data_file_mapping = yaml.safe_load(f)

        # Generate the filename based on the provided arguments
        filename_parts = [data_file_id]
        for key, value in kwargs.items():
            filename_parts.append(f"{key}_{value}")
        filename = "_".join(filename_parts) + ".pkl"

        if filename in data_file_mapping:
            return data_file_mapping[filename]
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = data_file_dir / f"{filename}_{timestamp}.pkl"
            data_file_mapping[filename] = file_path

            with open(mapping_file, "w") as f:
                yaml.dump(data_file_mapping, f)

            return file_path

    def generate_plot_filename(self, model_names: list, plot_type: str):
        # timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        plot_type = plot_type.replace(" ", "_")
        filename = f"{str(self.viz_dir)}/"
        for model in model_names:
            filename += f"{model}-"
        filename += f"{plot_type}.png"
        return filename

    def generate_responses(
        self,
        n_statements=None,
        prompt_template=None,
        system_message=None,
        models=None,
        perplexity=None,
    ):
        perplexity = perplexity or (
            3 if self.test_mode else self.run_settings.tsne_settings.perplexity
        )

        n_statements = n_statements or self.run_settings.data_settings.n_statements
        text_subset = self.data_prep.load_and_preprocess_data(
            data_settings=self.run_settings.data_settings
        )

        if self.saved_query_results is None:
            prompt_template = (
                prompt_template
                or self.run_settings.prompt_settings.statements_prompt_template
            )
            system_message = (
                system_message
                or self.run_settings.prompt_settings.statements_system_message
            )
            models = models or self.llms

            query_results_per_model = self.generate_and_save_responses(
                text_subset,
                n_statements,
                prompt_template,
                system_message,
                models,
            )
        else:
            query_results_per_model = [self.saved_query_results]

        return text_subset, query_results_per_model

    def generate_and_save_responses(
        self,
        text_subset,
        n_statements=None,
        prompt_template=None,
        system_message=None,
        llms=None,
    ):
        llms = llms or self.llms
        prompt_template = (
            prompt_template
            or self.run_settings.prompt_settings.statements_prompt_template
        )
        system_message = (
            system_message
            or self.run_settings.prompt_settings.statements_system_message
        )
        n_statements = n_statements or self.run_settings.data_settings.n_statements
        query_results_per_model = []
        for model_family, model in llms:
            print(f"Generating responses for {model} from {model_family}...")
            file_name = f"{model_family}_{model}_reaction_to_{n_statements}_{self.dataset_names_filename}_statements.pkl"
            query_results = query_model_on_statements(
                text_subset, model_family, model, prompt_template, system_message
            )  # dictionary of inputs, responses, and model instance
            query_results_per_model.append(query_results)
            self.save_results(query_results, file_name, "pickle_files")
            print(f"{file_name} saved.")
        self.save_results(
            query_results_per_model, "query_results_per_model.pkl", "pickle_files"
        )  # last saved
        print(f"{file_name} saved.")
        return query_results_per_model

    def collect_model_info(self, query_results_per_model):
        print("Collecting model info...")
        print(f"query_results_per_model: {query_results_per_model}")
        model_info_list = [result["model_info"] for result in query_results_per_model]
        # print the keys
        for key in model_info_list[0].keys():
            print(key)
        return model_info_list

    def perform_tsne_dimensionality_reduction(
        self, combined_embeddings, perplexity=None
    ):
        print("Performing t-SNE dimensionality reduction...")
        perplexity = perplexity or self.run_settings.tsne_settings.perplexity
        dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
            combined_embeddings, iterations=300, perplexity=perplexity
        )
        self.check_tsne_values(dim_reduce_tsne)
        return dim_reduce_tsne

    def check_tsne_values(self, dim_reduce_tsne):
        if not np.isfinite(dim_reduce_tsne).all():
            print("dim_reduce_tsne contains non-finite values.")
        if np.isnan(dim_reduce_tsne).any() or np.isinf(dim_reduce_tsne).any():
            print("dim_reduce_tsne contains NaN or inf values.")
        print("dim_reduce_tsne:", dim_reduce_tsne.dtype)

    def visualize_results(
        self, dim_reduce_tsne, joint_embeddings_all_llms, model_names, tsne_filename
    ):
        # Check if 'tsne' plots are set to be hidden
        if "tsne" in self.run_settings.plot_settings.hide_plots:
            print("Skipping t-SNE plot visualization as per settings.")
            return

        # Proceed with visualization if not hidden
        self.viz.plot_embedding_responses(
            dim_reduce_tsne,
            joint_embeddings_all_llms,
            model_names,
            tsne_filename,
        )

    def embed_responses(self, query_results_per_model):
        print("Embedding responses...")
        file_loaded, joint_embeddings_all_llms = load_pkl_or_not(
            "joint_embeddings_all_llms.pkl",
            self.pickle_dir,
            self.run_settings.data_settings.reuse_joint_embeddings,
        )
        if not file_loaded:
            joint_embeddings_all_llms = self.create_embeddings(
                query_results_per_model, self.llms, self.run_settings.embedding_settings
            )

        combined_embeddings = np.array(
            [e[3] for e in joint_embeddings_all_llms]
        )  # grab the embeddings of the inputs + responses
        combined_embeddings = np.array(combined_embeddings, dtype=np.float64)
        if not np.isfinite(combined_embeddings).all():
            print("Embeddings contain non-finite values.")

        # save combined embeddings
        with open(self.pickle_dir / "combined_embeddings.pkl", "wb") as f:
            pickle.dump(combined_embeddings, f)

        return joint_embeddings_all_llms, combined_embeddings

    def create_embeddings(
        self,
        query_results_per_model,
        llms,
        embedding_settings,
        combine_statements=False,
        save=True,
    ):
        """Embed the responses generated by ."""
        joint_embeddings_all_llms = []

        for i, (model_family, model) in enumerate(llms):
            print(f"Embedding responses for LLM {i}...")
            inputs = query_results_per_model[i]["inputs"]  # list of statements
            responses = query_results_per_model[i][
                "responses"
            ]  # list of responses to the statements by the LLM number i
            print(f"inputs: {inputs}")
            print(f"responses: {responses}")
            if i == 0:
                inputs_embeddings = embed_texts(
                    texts=inputs,
                    embedding_settings=embedding_settings,
                )
                n_statements = len(inputs)
                with open(
                    self.pickle_dir
                    / f"{self.dataset_names_filename}_{n_statements}_statements_embs.pkl",
                    "wb",
                ) as f:
                    pickle.dump(inputs_embeddings, f)

            # Embed the model responses to the input statements of a given model
            if combine_statements:
                joint_embeddings = embed_texts(
                    texts=[
                        input + " " + response
                        for input, response in zip(inputs, responses)
                    ],
                    embedding_settings=embedding_settings,
                )
            else:
                responses_embeddings = embed_texts(
                    texts=responses,
                    embedding_settings=embedding_settings,
                )
                joint_embeddings = [
                    inp + r for inp, r in zip(inputs_embeddings, responses_embeddings)
                ]
            for input, response, embedding in zip(inputs, responses, joint_embeddings):
                joint_embeddings_all_llms.append([i, input, response, embedding, model])

        if save:
            with open(self.pickle_dir / "joint_embeddings_all_llms.pkl", "wb") as f:
                pickle.dump(joint_embeddings_all_llms, f)
        return joint_embeddings_all_llms

    def run_clustering(self, combined_embeddings):
        file_loaded, chosen_clustering = load_pkl_or_not(
            "chosen_clustering.pkl",
            self.pickle_dir,
            self.run_settings.data_settings.reuse_embedding_clustering,
        )
        if not file_loaded:
            print("Running clustering...")
            clustering_results = self.clustering_obj.cluster_embeddings(
                combined_embeddings, multiple=True
            )
            print("Choosing clustering method... (KMeans is default)")
            chosen_clustering = clustering_results["KMeans"]
            with open(self.pickle_dir / "chosen_clustering.pkl", "wb") as f:
                pickle.dump(chosen_clustering, f)
        return chosen_clustering

    def analyze_response_embeddings_clusters(
        self, chosen_clustering, joint_embeddings_all_llms, model_info_list
    ):
        print("Analyzing clusters...")
        file_loaded, rows = load_pkl_or_not(
            "rows.pkl",
            self.pickle_dir,
            self.run_settings.data_settings.reuse_cluster_rows,
        )
        if not file_loaded:
            rows = self.clustering_obj.compile_cluster_table(
                chosen_clustering,
                joint_embeddings_all_llms,
                "joint_embeddings",
                model_info_list,
            )
            with open(self.pickle_dir / "rows.pkl", "wb") as f:
                pickle.dump(rows, f)
        # Save and display the results
        # Save chosen clustering and rows using pickle for later use
        print("Saving results...")
        clustering_file_path = self.pickle_dir / "latest_clustering_reaction.pkl"
        rows_file_path = self.pickle_dir / "latest_clustering_rows.pkl"
        with open(clustering_file_path, "wb") as f:
            pickle.dump(chosen_clustering, f)
        with open(rows_file_path, "wb") as f:
            pickle.dump(rows, f)
        print("Results saved.")
        print("Displaying results...")
        self.model_eval.display_statement_themes(
            chosen_clustering, rows, self.model_info_list
        )
        return rows

    def load_or_generate_approvals_data(self, approvals_type, reuse_approvals):
        """
        Using the statements from a dataset, prompt the language model with a set of personas
        and ask them if they approve or disapprove of the statements. For the set of personas,
        generate the approval responses for each language model you want to evaluate.
        """
        reuse_conditions = reuse_approvals  # can set to True if you only want to debug after the conditions are generated
        pickle_filename = f"approvals_statements_and_embeddings_{approvals_type}.pkl"
        file_loaded, approvals_statements_and_embeddings = load_pkl_or_not(
            pickle_filename,
            self.pickle_dir,
            reuse_approvals,
        )

        if not file_loaded:
            print("Generating approvals...")

            # load statement embeddings
            with open(
                self.pickle_dir
                / f"{self.dataset_names_filename}_{self.n_statements}_statements_embs.pkl",
                "rb",
            ) as f:
                statement_embeddings = pickle.load(f)

            conditions_loaded, approval_results_per_model = load_pkl_or_not(
                "approval_results_per_model_{approvals_type}.pkl",
                self.pickle_dir,
                reuse_conditions,
            )

            if not reuse_conditions or not conditions_loaded:
                # note: self.args.models is a list of models
                # create a dictionary with the model as the key, and a list of persona approvals as the value
                approval_results_per_model = {model: [] for model in self.args.models}

                for model_family, model in self.llms:
                    model_approvals = []
                    for role_description in self.persona_approval_prompts:
                        list_of_approvals_for_statements = (
                            self.model_eval.get_model_approvals(
                                statements=self.text_subset,
                                prompt_template=self.approval_question_prompt_template,
                                model_family=model_family,
                                model=model,
                                system_message=role_description,
                            )
                        )  # [0, 1, 0, 1, -1, 0, ...]
                        model_approvals.append(list_of_approvals_for_statements)
                    approval_results_per_model[model] = (
                        model_approvals  # {model_1: [approval1, approval2, ...], model_2: [approval1, approval2, ...], ... }
                    )

                if not (
                    self.pickle_dir / f"approval_results_per_model_{approvals_type}.pkl"
                ).exists():
                    pickle.dump(
                        approval_results_per_model,
                        open(
                            self.pickle_dir
                            / f"approval_results_per_model_{approvals_type}.pkl",
                            "wb",
                        ),
                    )
            print(f"approval_results_per_model: {approval_results_per_model}")

            # Store the approvals, statements, and embeddings in a list
            # [ [{model_1: [approval_for_prompt_1, approval_for_prompt_2, ...], model_2: [approval_for_prompt_1, approval_for_prompt_2, ...], ...}, statement, embedding], ... ]
            approvals_statements_and_embeddings = []
            for i in range(len(self.text_subset)):
                print(f"Approvals record {i}")
                record_approvals = {}
                for model, model_approvals in approval_results_per_model.items():
                    record_approvals[model] = [
                        approval[i] for approval in model_approvals
                    ]
                print(f"record_approvals: {record_approvals}")
                record = [
                    record_approvals,
                    self.text_subset[i],
                    statement_embeddings[i],
                ]  # [ [{model_1: [0, 1, 0, 0]}, {model_2: [0, 0, 1, 0]}, ...], statement, embedding]
                print(f"record: {record}")
                approvals_statements_and_embeddings.append(record)
                pickle.dump(
                    approvals_statements_and_embeddings,
                    open(
                        self.pickle_dir / pickle_filename,
                        "wb",
                    ),
                )

        return approvals_statements_and_embeddings

    def perform_hierarchical_clustering(
        self,
        statement_clustering,
        approvals_statements_and_embeddings,
        rows,
        prompt_type,
        reuse_hierarchical_approvals,
    ):
        print("Calculating hierarchical cluster data...")
        file_loaded, hierarchy_data = load_pkl_or_not(
            f"hierarchy_approval_data_{prompt_type}.pkl",
            self.pickle_dir,
            reuse_hierarchical_approvals,
        )
        if not file_loaded:
            hierarchy_data = self.clustering_obj.calculate_hierarchical_cluster_data(
                statement_clustering,
                approvals_statements_and_embeddings,
                rows,
            )
            with open(
                self.pickle_dir / f"hierarchy_approval_data_{prompt_type}.pkl", "wb"
            ) as f:
                pickle.dump(hierarchy_data, f)

        return hierarchy_data

    def visualize_hierarchical_clusters(
        self, *, model_names, hierarchy_data, plot_type, labels
    ):
        """
        Visualizes hierarchical clusters for the given plot type.

        Parameters:
        - model_names: List of model names to include in the visualization.
        - hierarchy_data: The hierarchical data to be visualized.
        - plot_type: The type of plot to generate ('approval' or 'awareness').
        """
        if plot_type in self.run_settings.plot_settings.hide_plots:
            print(
                f"Skipping hierarchical cluster visualization for {plot_type} as per settings."
            )
            return

        for model_name in model_names:
            filename = (
                self.viz_dir / f"hierarchical_clustering_{plot_type}_{model_name}"
            )
            print(f"Visualizing hierarchical cluster for {model_name}...")
            self.viz.visualize_hierarchical_cluster(
                hierarchy_data,
                plot_type=plot_type,
                filename=filename,
                labels=labels,
                bar_height=0.7,
                bb_width=40,
                x_leftshift=0,
                y_downshift=0,
            )

    def visualize_approval_embeddings(
        self,
        model_names,
        dim_reduce_tsne,
        approval_data,
        prompt_approver_type,  # "personas", "awareness", etc.
    ):
        if prompt_approver_type in self.run_settings.plot_settings.hide_plots:
            print(
                f"Skipping approval embeddings visualization for {prompt_approver_type} as per settings."
            )
            return

        for model_name in model_names:
            for condition in [1, 0, -1]:
                condition_title = {
                    1: "approvals",
                    0: "disapprovals",
                    -1: "no response",
                }[condition]
                plot_type = prompt_approver_type + "-" + condition_title
                approval_filename = self.generate_plot_filename(
                    model_names=[model_name], plot_type=plot_type
                )
                self.viz.plot_approvals(
                    dim_reduce=dim_reduce_tsne,
                    approval_data=approval_data,
                    condition=condition,
                    model_name=model_name,
                    plot_type=(
                        "approval"
                        if prompt_approver_type == "personas"
                        else "awareness"
                    ),
                    filename=f"{approval_filename}",
                    title=f"Embeddings of {condition_title} for {model_name} {prompt_approver_type} responses",
                )

    def create_data_file_paths(self) -> Dict[str, str]:
        # Generate filenames based on relevant parameters
        joint_embeddings_filename = self.get_or_create_data_file_path(
            "joint_embeddings",
            self.pickle_dir,
            self.run_settings.directory_settings.data_file_mapping,
            models="_".join(self.model_names),
            embedding_model=self.embedding_model_name,
            n_statements=self.n_statements,
            dataset=self.dataset_names_filename,
            random_seed=self.run_settings.random_state,
        )

        combined_embeddings_filename = self.get_or_create_data_file_path(
            "combined_embeddings",
            self.pickle_dir,
            self.run_settings.directory_settings.data_file_mapping,
            models="_".join(self.model_names),
            embedding_model=self.embedding_model_name,
            n_statements=self.n_statements,
            dataset=self.dataset_names_filename,
            random_seed=self.run_settings.random_state,
        )

        chosen_clustering_filename = self.get_or_create_data_file_path(
            "chosen_clustering",
            self.pickle_dir,
            self.run_settings.directory_settings.data_file_mapping,
            clustering_algorithm=self.run_settings.clustering_settings.main_clustering_algorithm,
            n_clusters=self.run_settings.clustering_settings.n_clusters,
            random_seed=self.run_settings.random_state,
            dataset=self.dataset_names_filename,
        )

        rows_filename = self.get_or_create_data_file_path(
            "rows",
            self.pickle_dir,
            self.run_settings.directory_settings.data_file_mapping,
            clustering_algorithm=self.run_settings.clustering_settings.main_clustering_algorithm,
            n_clusters=self.run_settings.clustering_settings.n_clusters,
            random_seed=self.run_settings.random_state,
            dataset=self.dataset_names_filename,
        )

        # Load and visualize saved data for the current run
        data_file_paths = {
            "joint_embeddings": joint_embeddings_filename,
            "combined_embeddings": combined_embeddings_filename,
            "chosen_clustering": chosen_clustering_filename,
            "rows": rows_filename,
        }

        for prompt_type in self.approval_prompts.keys():
            data_file_paths[f"approvals_{prompt_type}"] = (
                self.get_or_create_data_file_path(
                    f"approvals_{prompt_type}",
                    self.pickle_dir,
                    self.run_settings.directory_settings.data_file_mapping,
                )
            )
            data_file_paths[f"hierarchy_data_{prompt_type}"] = (
                self.get_or_create_data_file_path(
                    f"hierarchy_data_{prompt_type}",
                    self.pickle_dir,
                    self.run_settings.directory_settings.data_file_mapping,
                )
            )
        return data_file_paths

    def load_and_visualize_saved_data(self, run_id):
        with open(self.run_settings.directory_settings.metadata_file, "r") as f:
            runs_metadata = yaml.safe_load(f)

        run_metadata = next(
            (run for run in runs_metadata if run["run_id"] == run_id), None
        )
        if run_metadata is None:
            print(f"Run with ID {run_id} not found.")
            return

        data_files = run_metadata["data_files"]

        # Load saved data
        joint_embeddings_all_llms = self.load_results(
            data_files["joint_embeddings"], "pickle_files"
        )
        combined_embeddings = self.load_results(
            data_files["combined_embeddings"], "pickle_files"
        )
        chosen_clustering = self.load_results(
            data_files["chosen_clustering"], "pickle_files"
        )
        rows = self.load_results(data_files["rows"], "pickle_files")

        # Refactored visualization into a separate method
        dim_reduce_tsne, labels = self.visualize_loaded_data(
            combined_embeddings, joint_embeddings_all_llms, chosen_clustering, rows
        )

        # Visualize approval embeddings for each prompt type
        for prompt_type in self.approval_prompts.keys():
            approvals_statements_and_embeddings = self.load_results(
                data_files[f"approvals_{prompt_type}"], "pickle_files"
            )
            self.visualize_approval_embeddings(
                self.model_names,
                dim_reduce_tsne,
                approvals_statements_and_embeddings,
                prompt_approver_type=prompt_type,
            )

            hierarchy_data = self.load_results(
                data_files[f"hierarchy_data_{prompt_type}"], "pickle_files"
            )
            self.visualize_hierarchical_clusters(
                model_names=self.model_names,
                hierarchy_data=hierarchy_data,
                plot_type=prompt_type,
                labels=list(self.approval_prompts[prompt_type].keys()),
            )

    def visualize_loaded_data(
        self, combined_embeddings, joint_embeddings_all_llms, chosen_clustering, rows
    ):
        # Visualize results
        tsne_filename = self.generate_plot_filename(
            self.model_names, "tsne_embedding_responses"
        )
        dim_reduce_tsne = self.perform_tsne_dimensionality_reduction(
            combined_embeddings
        )
        self.visualize_results(
            dim_reduce_tsne, joint_embeddings_all_llms, self.model_names, tsne_filename
        )

        labels = chosen_clustering.labels_
        print(f"labels: {labels}")
        plt.hist(labels, bins=5)

        clusters_desc_table = [
            ["ID", "N", "Inputs Themes", "Responses Themes", "Interaction Themes"]
        ]
        table_pickle_path = self.pickle_dir / "clusters_desc_table_personas.pkl"
        self.clustering_obj.create_cluster_table(
            clusters_desc_table, rows, table_pickle_path, "response_comparisons"
        )

        return dim_reduce_tsne, labels

    def run_evaluations(self):
        # Load data
        all_texts = self.load_evaluation_data(self.run_settings.data_settings.datasets)

        # Check if local models are included in the run
        has_local_models = any(model_family == "local" for model_family, _ in self.llms)

        if has_local_models:
            gpu_availability = check_gpu_availability()
            if gpu_availability == "multiple_gpus":
                # Run local models in parallel
                model_batches = self.get_model_batches()
                for model_batch in model_batches:
                    self.run_pipeline_for_models(
                        model_batch, gpu_availability="multiple_gpus"
                    )
            else:
                # Run local models sequentially
                for model_name, local_model in self.local_models.items():
                    if check_gpu_memory([(model_name, local_model)], buffer_factor=1.2):
                        local_model.load()
                        self.run_pipeline_for_models(
                            [(model_name, local_model)], gpu_availability="single_gpu"
                        )
                        local_model.unload()
                    else:
                        print(
                            f"Not enough GPU memory for {model_name} with buffer factor applied."
                        )
        else:
            # Run API models
            self.run_pipeline_for_models([])

    def run_pipeline_for_models(
        self, model_batch, gpu_availability="single_gpu", buffer_factor=1.2
    ):
        """Steps to run the evaluation pipeline.

        1. Compare how multiple LLMs fall into different clusters based on their responses to the same statement prompts.
        1.1. Load statement prompts to generate model responses.
        1.2. Generate responses to statement prompts.
        1.3. Embed model responses to statement prompts.
        1.4. Run clustering on the statement + response embeddings and visualize the clusters (with spectral clustering).
        1.5. Apply dimensionality reduction to the embeddings and visualize the results.
        1.6. Analyze the clusters by auto-labeling clusters with an LLM and print and save the cluster table results.

        2. Evaluation based on prompt types (e.g., personas, awareness): How do LLMs respond to different types of prompts?
        2.1. Load prompts (approval or awareness) and embeddings for each prompt type.
        2.2. Ask the LLMs if they approve or disapprove of certain statements based on the prompt type.
        2.3. Store the LLMs' approval or disapproval responses along with the statement embeddings.
        2.4. Utilize dimensionality reduction on the statement embeddings to visualize the responses (approve, disapprove, no response) for each prompt type.
        2.5. Conduct a comparison analysis between the different prompt types.
        2.6. Generate a table to compare the approval rates of the LLMs for each cluster, segmented by prompt type.
        2.7. Perform hierarchical clustering on the responses to each prompt type and visualize the resulting clusters.
        """
        # Generate responses to statement prompts
        # Given that we can only fit 1 model in the GPU at a time, we will need to loop over the models we want to evaluate.
        # For each loop, we'll run the entire pipeline for that model.
        # Then, we'll remove that model from the GPU and load the next model.
        # At the end, we'll plot the results for all models.
        self.text_subset, query_results_per_model = self.generate_responses()
        self.model_info_list = self.collect_model_info(query_results_per_model)

        if "model_comparison" not in self.run_settings.skip_sections:
            # Embed model responses to statement prompts
            # joint_embeddings_all_llms: [ [model_id, input, response, statement_embedding, model_name], ... ]
            joint_embeddings_all_llms, combined_embeddings = self.embed_responses(
                query_results_per_model
            )
            chosen_clustering = self.run_clustering(combined_embeddings)
            rows = self.analyze_response_embeddings_clusters(
                chosen_clustering, joint_embeddings_all_llms, self.model_info_list
            )
            dim_reduce_tsne, labels = self.visualize_loaded_data(
                combined_embeddings, joint_embeddings_all_llms, chosen_clustering, rows
            )

        # The Approval Persona prompts and Awareness prompts sections are similar, so we can refactor them into a single function where we loop over the type of prompts created in the json file. So, the following code should be able to run n number of prompts for m number of models and p number of prompt types (e.g. personas, awareness, etc.).
        # In order to do this, we will need to refactor some of the functions used for both prompt types to be more general. We will also need to allow for iterating over the models we want to evaluate.
        if "approvals" not in self.run_settings.skip_sections:
            for prompt_type in self.approval_prompts.keys():
                print(
                    f"prompt_type: {prompt_type}"
                )  # e.g. "personas", "awareness", etc.
                approvals_filename = f"approvals_{prompt_type}_{'_'.join(self.model_names)}_{self.run_settings.embedding_settings.embedding_model}_{self.n_statements}_{self.dataset_names_filename}.pkl"
                hierarchy_data_filename = f"hierarchy_data_{prompt_type}_{'_'.join(self.model_names)}_{self.run_settings.embedding_settings.embedding_model}_{self.n_statements}_{self.dataset_names_filename}.pkl"

                if not self.load_results(approvals_filename, "pickle_files"):
                    self.save_data(
                        approvals_filename, approvals_statements_and_embeddings
                    )
                if not self.load_results(hierarchy_data_filename, "pickle_files"):
                    self.save_data(hierarchy_data_filename, hierarchy_data)
                # approvals_statements_and_embeddings: # [ [{model_1: [approval1, approval2, ...], model_2: [approval1, approval2, ...], ...}, statement, embedding], ... ]
                # get boolean from run_settings to determine if we should reuse the hierarchical approvals for whichever prompt type we are currently evaluating
                reuse_approvals = self.run_settings.data_settings.reuse_approvals
                approvals_statements_and_embeddings = (
                    self.load_or_generate_approvals_data(
                        approvals_type=prompt_type, reuse_approvals=reuse_approvals
                    )
                )
                if "statement_embeddings" not in locals():
                    statement_embeddings = np.array(
                        [
                            approval[2]
                            for approval in approvals_statements_and_embeddings
                        ]
                    )

                if "dim_reduce_tsne" not in locals():
                    print("Performing dimensionality reduction...")
                    dim_reduce_tsne = self.model_eval.tsne_dimension_reduction(
                        statement_embeddings,
                        iterations=2000,
                        perplexity=self.run_settings.tsne_settings.perplexity,
                    )

                # Visualize approval embeddings
                if (
                    prompt_type not in self.run_settings.plot_settings.hide_approval
                    and not self.run_settings.plot_settings.visualize_at_end
                ):
                    self.visualize_approval_embeddings(
                        self.model_names,
                        dim_reduce_tsne,
                        approvals_statements_and_embeddings,
                        prompt_approver_type=prompt_type,
                    )
                prompt_dict = {
                    k: v for k, v in self.approval_prompts.items() if k == prompt_type
                }  # { "personas": { "google_chat_desc": [ "prompt"], "bing_chat_desc": [...], ...} }
                prompt_labels = list(prompt_dict.keys())

                print(f"Clustering statement embeddings...")
                n_clusters = 120
                if "statement_clustering" not in locals():
                    statement_clustering = self.clustering_obj.cluster_embeddings(
                        statement_embeddings,
                        clustering_algorithm="SpectralClustering",
                        n_clusters=n_clusters,
                        multiple=False,
                    )
                if "spectral" not in self.run_settings.plot_settings.hide_plots:
                    self.viz.plot_spectral_clustering(
                        statement_clustering.labels_,
                        n_clusters=n_clusters,
                        prompt_approver_type=prompt_type.capitalize(),
                    )
                self.clustering_obj.cluster_approval_stats(
                    approvals_statements_and_embeddings,
                    statement_clustering,
                    self.model_info_list,
                    prompt_dict=prompt_dict,
                    reuse_cluster_rows=self.run_settings.data_settings.reuse_cluster_rows,
                )
                print(
                    f"Calculating hierarchical cluster data for {prompt_type} prompts..."
                )
                hierarchy_data = self.perform_hierarchical_clustering(
                    statement_clustering,
                    approvals_statements_and_embeddings,
                    rows,
                    prompt_type,
                    reuse_hierarchical_approvals=self.run_settings.data_settings.reuse_hierarchical_approvals,
                )
                print(f"Visualizing hierarchical cluster for {prompt_type} prompts...")
                if (
                    self.run_settings.plot_settings.hide_hierarchical
                    and not self.run_settings.plot_settings.visualize_at_end
                ):
                    self.visualize_hierarchical_clusters(
                        model_names=self.model_names,
                        hierarchy_data=hierarchy_data,
                        plot_type=prompt_type,
                        labels=prompt_labels,
                    )

        # run id should include model names, dataset name, number of statements, and timestamp
        run_id = (
            "_".join(self.model_names)
            + "_"
            + self.dataset_names_filename
            + "_"
            + str(self.n_statements)
            + "_"
            + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )
        data_file_paths = self.create_data_file_paths()
        self.save_data(data_file_paths["joint_embeddings"], joint_embeddings_all_llms)
        self.save_data(data_file_paths["combined_embeddings"], combined_embeddings)
        self.save_data(data_file_paths["chosen_clustering"], chosen_clustering)
        self.save_data(data_file_paths["rows"], rows)
        # To make it easier to find the results of a specific run, we will save the run metadata to a yaml file
        self.save_run_metadata_to_yaml(run_id, data_file_paths)
        if self.run_settings.plot_settings.visualize_at_end:
            self.load_and_visualize_saved_data(run_id)

        print("Done. Please check the results directory for the plots.")
