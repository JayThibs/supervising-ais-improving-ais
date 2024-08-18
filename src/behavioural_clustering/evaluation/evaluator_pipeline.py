import os
from pathlib import Path
import json
import yaml
import numpy as np
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
from functools import lru_cache
import hashlib

from behavioural_clustering.config.run_settings import RunSettings, DataSettings
from behavioural_clustering.utils.data_preparation import DataPreparation, DataHandler
from behavioural_clustering.utils.visualization import Visualization
from behavioural_clustering.evaluation.clustering import Clustering, ClusterAnalyzer
from behavioural_clustering.evaluation.embeddings import embed_texts
from behavioural_clustering.evaluation.dimensionality_reduction import tsne_reduction
from behavioural_clustering.models.local_models import LocalModel
from behavioural_clustering.utils.resource_management import ResourceManager
from .model_evaluation_manager import ModelEvaluationManager
from .approval_evaluation_manager import ApprovalEvaluationManager
from .hierarchical_clustering_manager import HierarchicalClusteringManager
from behavioural_clustering.utils.caching_utils import cache_manager

class EvaluatorPipeline:
    def __init__(self, run_settings: RunSettings):
        """
        Initialize the EvaluatorPipeline with the given run settings.

        Args:
            run_settings (RunSettings): Configuration settings for the evaluation run.
        """
        self.run_settings = run_settings
        self.setup_directories()
        self.setup_models()
        self.setup_managers()

    def setup_directories(self) -> None:
        for dir_name in ['data_dir', 'evals_dir', 'results_dir', 'viz_dir', 'tables_dir', 'pickle_dir']:
            dir_path = getattr(self.run_settings.directory_settings, dir_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

    def setup_models(self) -> None:
        self.llms = self.run_settings.model_settings.models
        self.model_names = [model for _, model in self.llms]
        self.local_models = {model: LocalModel(model_name_or_path=model) for model_family, model in self.llms if model_family == "local"}
        self.embedding_model_name = self.run_settings.embedding_settings.embedding_model

    def setup_managers(self) -> None:
        self.data_prep = DataPreparation()
        self.data_handler = DataHandler(
            self.run_settings.directory_settings.results_dir,
            self.run_settings.directory_settings.pickle_dir,
            self.run_settings.directory_settings.data_file_mapping
        )
        self.viz = Visualization(self.run_settings.plot_settings)
        self.clustering = Clustering(self.run_settings)
        self.cluster_analyzer = ClusterAnalyzer(self.run_settings)
        self.model_eval_manager = ModelEvaluationManager(self.run_settings, self.llms)
        self.approval_eval_manager = ApprovalEvaluationManager(self.run_settings, self.model_eval_manager)
        self.hierarchical_clustering_manager = HierarchicalClusteringManager(self.run_settings, self.cluster_analyzer, self.viz)

    def run_evaluations(self) -> None:
        """
        Execute the main evaluation pipeline, including model comparison,
        prompt evaluations, and hierarchical clustering if specified in the run settings.
        """
        all_texts = self.load_evaluation_data(tuple(self.run_settings.data_settings.datasets))
        self.text_subset = self.load_and_preprocess_data(self.run_settings.data_settings)

        if "model_comparison" not in self.run_settings.skip_sections:
            self.run_model_comparison()
        
        self.approvals_data = {}
        for prompt_type in self.run_settings.approval_prompts.keys():
            if f"{prompt_type}_evaluation" not in self.run_settings.skip_sections:
                self.run_prompt_evaluation(prompt_type)

        if "hierarchical_clustering" not in self.run_settings.skip_sections:
            self.run_hierarchical_clustering()

        self.save_run_data()

    def run_model_comparison(self) -> None:
        """
        Perform model comparison by generating responses, creating embeddings,
        clustering the embeddings, and analyzing the clusters.
        """
        query_results = self.cached_generate_responses(self.text_subset)
        self.joint_embeddings_all_llms, self.combined_embeddings = self.cached_create_embeddings(query_results)
        self.chosen_clustering = self.cached_cluster_embeddings(self.combined_embeddings)
        self.rows = self.cached_analyze_response_embeddings_clusters(self.chosen_clustering, self.joint_embeddings_all_llms)
        self.visualize_model_comparison()

    def run_prompt_evaluation(self, prompt_type: str) -> None:
        """
        Evaluate a specific prompt type by generating approvals data,
        embedding texts, and analyzing cluster approval statistics.

        Args:
            prompt_type (str): The type of prompt to evaluate (e.g., 'personas', 'awareness').
        """
        self.approvals_data[prompt_type] = self.cached_load_or_generate_approvals_data(prompt_type, self.text_subset)
        embeddings = self.cached_embed_texts(self.text_subset)
        self.cluster_analyzer.cluster_approval_stats(
            self.approvals_data[prompt_type],
            embeddings,
            self.model_eval_manager.model_info_list,
            {prompt_type: self.run_settings.approval_prompts[prompt_type]}
        )
        self.visualize_approvals(prompt_type)

    def run_hierarchical_clustering(self) -> None:
        """
        Perform hierarchical clustering for each prompt type specified in the run settings.
        """
        for prompt_type in self.run_settings.approval_prompts.keys():
            self.hierarchical_clustering_manager.run_hierarchical_clustering(
                prompt_type,
                self.chosen_clustering,
                self.approvals_data[prompt_type],
                self.rows,
                self.model_names,
                self.run_settings.approval_prompts[prompt_type]
            )

    def visualize_model_comparison(self) -> None:
        """
        Create visualizations for the model comparison results, including
        t-SNE plots of embedding responses and displays of statement themes.
        """
        dim_reduce_tsne = self.cached_tsne_reduction(self.combined_embeddings)
        self.viz.plot_embedding_responses(dim_reduce_tsne, self.joint_embeddings_all_llms, self.model_names, self.generate_plot_filename(self.model_names, "tsne_embedding_responses"))
        self.cluster_analyzer.display_statement_themes(self.chosen_clustering, self.rows, self.model_eval_manager.model_info_list)

    def visualize_approvals(self, prompt_type: str) -> None:
        """
        Generate visualizations for approval data for a specific prompt type.

        Args:
            prompt_type (str): The type of prompt to visualize (e.g., 'personas', 'awareness').
        """
        approvals_embeddings = np.array([e[2] for e in self.approvals_data[prompt_type]])
        dim_reduce_tsne = self.cached_tsne_reduction(approvals_embeddings)
        for model_name in self.model_names:
            for condition in [1, 0, -1]:
                self.viz.plot_approvals(
                    dim_reduce_tsne,
                    self.approvals_data[prompt_type],
                    model_name,
                    condition,
                    "approval" if prompt_type == "personas" else "awareness",
                    self.generate_plot_filename([model_name], f"{prompt_type}-{condition}"),
                    f"Embeddings of {['approvals', 'disapprovals', 'no response'][condition]} for {model_name} {prompt_type} responses"
                )

    def generate_plot_filename(self, model_names: list, plot_type: str) -> str:
        plot_type = plot_type.replace(" ", "_")
        filename = f"{self.run_settings.directory_settings.viz_dir}/" + "-".join(model_names) + f"-{plot_type}.png"
        return filename

    def save_run_data(self) -> None:
        """
        Save the results of the evaluation run, including embeddings, clusterings,
        and other relevant data, along with metadata about the run.
        """
        run_id = self.data_handler.generate_run_id()
        data_file_paths = self.create_data_file_paths()
        self.data_handler.save_pickles(data_file_paths, [
            self.joint_embeddings_all_llms,
            self.combined_embeddings,
            self.chosen_clustering,
            self.rows,
            self.approvals_data.get('personas'),
            self.hierarchical_clustering_manager.hierarchy_data.get('personas'),
            self.approvals_data.get('awareness'),
            self.hierarchical_clustering_manager.hierarchy_data.get('awareness')
        ])
        
        metadata = {
            "run_id": run_id,
            "dataset_names": "_".join(self.run_settings.data_settings.datasets),
            "model_names": self.model_names,
            "n_statements": self.run_settings.data_settings.n_statements,
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
            "data_files": data_file_paths,
        }
        self.data_handler.save_run_metadata_to_yaml(run_id, metadata)

    def create_data_file_paths(self) -> dict:
        return {
            "joint_embeddings": self.data_handler.get_or_create_data_file_path("joint_embeddings", self.run_settings.directory_settings.pickle_dir, self.run_settings.directory_settings.data_file_mapping, models="_".join(self.model_names), embedding_model=self.embedding_model_name, n_statements=self.run_settings.data_settings.n_statements, dataset="_".join(self.run_settings.data_settings.datasets), random_seed=self.run_settings.random_state),
            "combined_embeddings": self.data_handler.get_or_create_data_file_path("combined_embeddings", self.run_settings.directory_settings.pickle_dir, self.run_settings.directory_settings.data_file_mapping, models="_".join(self.model_names), embedding_model=self.embedding_model_name, n_statements=self.run_settings.data_settings.n_statements, dataset="_".join(self.run_settings.data_settings.datasets), random_seed=self.run_settings.random_state),
            "chosen_clustering": self.data_handler.get_or_create_data_file_path("chosen_clustering", self.run_settings.directory_settings.pickle_dir, self.run_settings.directory_settings.data_file_mapping, clustering_algorithm=self.run_settings.clustering_settings.main_clustering_algorithm, n_clusters=self.run_settings.clustering_settings.n_clusters, random_seed=self.run_settings.random_state, dataset="_".join(self.run_settings.data_settings.datasets)),
            "rows": self.data_handler.get_or_create_data_file_path("rows", self.run_settings.directory_settings.pickle_dir, self.run_settings.directory_settings.data_file_mapping, clustering_algorithm=self.run_settings.clustering_settings.main_clustering_algorithm, n_clusters=self.run_settings.clustering_settings.n_clusters, random_seed=self.run_settings.random_state, dataset="_".join(self.run_settings.data_settings.datasets)),
            **{f"approvals_{prompt_type}": self.data_handler.get_or_create_data_file_path(f"approvals_{prompt_type}", self.run_settings.directory_settings.pickle_dir, self.run_settings.directory_settings.data_file_mapping) for prompt_type in self.run_settings.approval_prompts.keys()},
            **{f"hierarchy_data_{prompt_type}": self.data_handler.get_or_create_data_file_path(f"hierarchy_data_{prompt_type}", self.run_settings.directory_settings.pickle_dir, self.run_settings.directory_settings.data_file_mapping) for prompt_type in self.run_settings.approval_prompts.keys()}
        }

    @cache_manager.cached("generate_responses")
    def cached_generate_responses(self, text_subset: list) -> dict:
        """
        Generate and cache model responses for the given text subset.

        Args:
            text_subset (list): A subset of texts to generate responses for.

        Returns:
            dict: Generated responses from various models.
        """
        return self.model_eval_manager.generate_responses(text_subset)

    @cache_manager.cached("create_embeddings")
    def cached_create_embeddings(self, query_results: dict) -> tuple:
        """
        Create and cache embeddings for the generated responses.

        Args:
            query_results (dict): Results from model query responses.

        Returns:
            tuple: Joint embeddings for all models and combined embeddings.
        """
        return self.model_eval_manager.create_embeddings(query_results, self.llms, self.run_settings.embedding_settings)

    @cache_manager.cached("cluster_embeddings")
    def cached_cluster_embeddings(self, combined_embeddings: np.array) -> object:
        """
        Perform and cache clustering on the combined embeddings.

        Args:
            combined_embeddings (np.array): Combined embeddings from all models.

        Returns:
            object: Clustering results.
        """
        return self.clustering.cluster_embeddings(combined_embeddings)

    @cache_manager.cached("analyze_response_embeddings_clusters")
    def cached_analyze_response_embeddings_clusters(self, chosen_clustering: object, joint_embeddings_all_llms: dict) -> list:
        """
        Analyze and cache the response embeddings and clusters.

        Args:
            chosen_clustering (object): Clustering results.
            joint_embeddings_all_llms (dict): Joint embeddings for all models.

        Returns:
            list: Analyzed rows.
        """
        return self.cluster_analyzer.analyze_response_embeddings_clusters(chosen_clustering, joint_embeddings_all_llms, self.model_eval_manager.model_info_list)

    @cache_manager.cached("load_or_generate_approvals_data")
    def cached_load_or_generate_approvals_data(self, prompt_type: str, text_subset: list) -> list:
        """
        Load or generate and cache approvals data for the given prompt type and text subset.

        Args:
            prompt_type (str): The type of prompt (e.g., 'personas', 'awareness').
            text_subset (list): A subset of texts.

        Returns:
            list: Approvals data.
        """
        return self.approval_eval_manager.load_or_generate_approvals_data(prompt_type, text_subset)

    @cache_manager.cached("embed_texts")
    def cached_embed_texts(self, text_subset: list) -> np.array:
        """
        Embed and cache the given texts.

        Args:
            text_subset (list): A subset of texts to embed.

        Returns:
            np.array: Embedded texts.
        """
        return embed_texts(text_subset, self.run_settings.embedding_settings)

    @cache_manager.cached("tsne_reduction")
    def cached_tsne_reduction(self, embeddings: np.array) -> np.array:
        """
        Perform and cache t-SNE dimensionality reduction on the given embeddings.

        Args:
            embeddings (np.array): Embeddings to reduce.

        Returns:
            np.array: Reduced embeddings.
        """
        return tsne_reduction(embeddings, self.run_settings.tsne_settings, self.run_settings.random_state)

    @cache_manager.cached("load_and_preprocess_data")
    def load_and_preprocess_data(self, data_settings: DataSettings) -> list:
        """
        Load and cache the evaluation data and preprocess it according to the given data settings.

        Args:
            data_settings (DataSettings): Data settings.

        Returns:
            list: Preprocessed evaluation data.
        """
        return self.data_prep.load_and_preprocess_data(data_settings)

    @lru_cache(maxsize=None)
    def load_evaluation_data(self, datasets_tuple: tuple) -> list:
        """
        Load and cache evaluation data for the specified datasets.

        Args:
            datasets_tuple (tuple): Tuple of dataset names to load.

        Returns:
            list: Loaded evaluation data.
        """
        datasets = list(datasets_tuple)
        return self.data_prep.load_evaluation_data(self.run_settings.data_settings.datasets)