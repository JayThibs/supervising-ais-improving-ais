import os
import pickle
from pathlib import Path
import json
import yaml
import numpy as np
import pickle
from datetime import datetime
from matplotlib import pyplot as plt
from functools import lru_cache
import hashlib
import traceback
import uuid
from typing import List, Dict, Any

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
        self.run_id = self.generate_run_id()
        self.data_handler = DataHandler(self.run_settings.directory_settings.data_dir, self.run_id)
        self.run_metadata_file = self.run_settings.directory_settings.data_dir / "metadata" / "run_metadata.yaml"
        self.run_metadata = self.load_run_metadata()
        self.setup_managers()

    def generate_run_id(self) -> str:
        metadata_file = self.run_settings.directory_settings.data_dir / "metadata" / "run_metadata.yaml"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                existing_runs = yaml.safe_load(f) or {}
        else:
            existing_runs = {}
        
        run_number = 1
        while f"run_{run_number}" in existing_runs:
            run_number += 1
        
        return f"run_{run_number}"

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
        
        data_file_mapping = self.run_settings.directory_settings.data_file_mapping
        if not os.path.exists(data_file_mapping):
            print(f"Warning: data_file_mapping file {data_file_mapping} does not exist. Creating an empty file.")
            with open(data_file_mapping, 'w') as f:
                pass  # Create an empty file
        
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
        print(f"Data settings: {self.run_settings.data_settings}")
        self.text_subset = self.load_text_subset()
        
        if len(self.text_subset) == 0:
            raise ValueError("No texts loaded. Please check your dataset configuration.")

        print(f"Loaded {len(self.text_subset)} texts.")
        print("Sample texts:")
        for i in range(min(5, len(self.text_subset))):
            print(f"Text {i+1}: {self.text_subset[i][:100]}...")

        if "model_comparison" not in self.run_settings.skip_sections:
            self.run_model_comparison()
        
        self.approvals_data = {}
        for prompt_type in self.run_settings.approval_prompts.keys():
            if f"{prompt_type}_evaluation" not in self.run_settings.skip_sections:
                self.run_prompt_evaluation(prompt_type)

        if "hierarchical_clustering" not in self.run_settings.skip_sections:
            self.run_hierarchical_clustering()

        self.save_run_data()

    def load_text_subset(self) -> List[str]:
        config = self.create_current_metadata()
        return self.data_prep.load_and_preprocess_data(self.run_settings.data_settings)

    def run_model_comparison(self) -> None:
        config = self.create_current_metadata()

        self.query_results_per_model = self.load_data("all_query_results", config)
        if self.query_results_per_model is None:
            print("Generating new query results...")
            self.query_results_per_model = self.model_eval_manager.generate_responses(self.text_subset)
            self.data_handler.save_data(self.query_results_per_model, "all_query_results", config)
        else:
            print("Loaded existing query results.")

        if not self.query_results_per_model:
            print("Warning: No query results available. Check the generate_responses function.")
            return

        loaded_data = self.load_data("joint_embeddings_all_llms", config)
        if loaded_data is None:
            print("Generating new embeddings...")
            self.joint_embeddings_all_llms, self.combined_embeddings = self.model_eval_manager.create_embeddings(
                self.query_results_per_model,
                self.llms,
                self.run_settings.embedding_settings
            )
            self.data_handler.save_data(self.joint_embeddings_all_llms, "joint_embeddings_all_llms", config)
            self.data_handler.save_data(self.combined_embeddings, "combined_embeddings", config)
        else:
            print("Loaded existing embeddings.")
            self.joint_embeddings_all_llms = loaded_data
            self.combined_embeddings = self.load_data("combined_embeddings", config)

        if not self.combined_embeddings:
            print("Warning: combined_embeddings is empty. Check the create_embeddings function.")
            return

        print(f"Number of combined embeddings: {len(self.combined_embeddings)}")
        
        combined_embeddings_array = np.array(self.combined_embeddings)
        
        self.chosen_clustering = self.load_data("chosen_clustering", config)
        if self.chosen_clustering is None:
            print("Generating new clustering...")
            self.chosen_clustering = self.clustering.cluster_embeddings(combined_embeddings_array)
            self.data_handler.save_data(self.chosen_clustering, "chosen_clustering", config)
        else:
            print("Loaded existing clustering.")

        self.rows = self.load_data("compile_cluster_table", config)
        if self.rows is None:
            print("Generating new cluster table...")
            self.rows = self.cluster_analyzer.compile_cluster_table(
                clustering=self.chosen_clustering,
                data=self.joint_embeddings_all_llms,
                model_info_list=self.model_eval_manager.model_info_list,
                data_type="joint_embeddings",
                theme_summary_instructions=self.run_settings.prompt_settings.theme_summary_instructions,
                max_desc_length=self.run_settings.prompt_settings.max_desc_length
            )
            self.data_handler.save_data(self.rows, "compile_cluster_table", config)
        else:
            print("Loaded existing cluster table.")

        self.visualize_model_comparison()

    def run_prompt_evaluation(self, prompt_type: str) -> None:
        """
        Evaluate a specific prompt type by generating approvals data,
        embedding texts, and analyzing cluster approval statistics.

        Args:
            prompt_type (str): The type of prompt to evaluate (e.g., 'personas', 'awareness').
        """
        config = self.create_current_metadata()
        config["prompt_type"] = prompt_type

        # Load or generate approvals data
        self.approvals_data[prompt_type] = self.data_handler.load_data(f"approvals_statements_and_embeddings_{prompt_type}", config)
        if self.approvals_data[prompt_type] is None or not self.run_settings.data_settings.should_reuse_data(f"approvals_statements_and_embeddings_{prompt_type}"):
            print(f"Generating new approvals data for {prompt_type}...")
            self.approvals_data[prompt_type] = self.approval_eval_manager.load_or_generate_approvals_data(prompt_type, self.text_subset)
            self.data_handler.save_data(self.approvals_data[prompt_type], f"approvals_statements_and_embeddings_{prompt_type}", config)
        else:
            print(f"Loaded existing approvals data for {prompt_type}.")

        # Load or generate embeddings
        embeddings = self.data_handler.load_data("embed_texts", config)
        if embeddings is None or not self.run_settings.data_settings.should_reuse_data("embed_texts"):
            print("Generating new text embeddings...")
            embeddings = embed_texts(self.text_subset, self.run_settings.embedding_settings)
            self.data_handler.save_data(embeddings, "embed_texts", config)
        else:
            print("Loaded existing text embeddings.")

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
        dim_reduce_tsne = self.load_data(
            "tsne_reduction",
            self.create_current_metadata()
        )
        if dim_reduce_tsne is None:
            print("Generating new tsne_reduction...")
            dim_reduce_tsne = tsne_reduction(
                self.combined_embeddings,
                self.run_settings.tsne_settings,
                self.run_settings.random_state
            )
            self.data_handler.save_data(dim_reduce_tsne, "tsne_reduction", self.create_current_metadata())
        else:
            print("Loaded existing tsne_reduction.")

        self.viz.plot_embedding_responses(dim_reduce_tsne, self.joint_embeddings_all_llms, self.model_names, self.generate_plot_filename(self.model_names, "tsne_embedding_responses"))
        self.cluster_analyzer.display_statement_themes(self.chosen_clustering, self.rows, self.model_eval_manager.model_info_list)

    def visualize_approvals(self, prompt_type: str) -> None:
        """
        Generate visualizations for approval data for a specific prompt type.

        Args:
            prompt_type (str): The type of prompt to visualize (e.g., 'personas', 'awareness').
        """
        approvals_embeddings = np.array([e[2] for e in self.approvals_data[prompt_type]])
        dim_reduce_tsne = self.load_data(
            "tsne_reduction",
            self.create_current_metadata()
        )
        if dim_reduce_tsne is None:
            print("Generating new tsne_reduction...")
            dim_reduce_tsne = tsne_reduction(
                approvals_embeddings,
                self.run_settings.tsne_settings,
                self.run_settings.random_state
            )
            self.data_handler.save_data(dim_reduce_tsne, "tsne_reduction", self.create_current_metadata())
        else:
            print("Loaded existing tsne_reduction.")

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
        metadata = {
            "dataset_names": "_".join(self.run_settings.data_settings.datasets),
            "model_names": self.model_names,
            "n_statements": self.run_settings.data_settings.n_statements,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data_settings": self.run_settings.data_settings.to_dict(),
            "model_settings": self.run_settings.model_settings.__dict__,
            "embedding_settings": self.run_settings.embedding_settings.__dict__,
            "prompt_settings": self.run_settings.prompt_settings.__dict__,
            "plot_settings": self.run_settings.plot_settings.__dict__,
            "clustering_settings": self.run_settings.clustering_settings.__dict__,
            "tsne_settings": self.run_settings.tsne_settings.__dict__,
            "test_mode": self.run_settings.test_mode,
            "skip_sections": self.run_settings.skip_sections,
        }
        
        self.save_run_metadata(metadata)
        print(f"Run metadata saved to {self.run_metadata_file}")

    def load_data(self, data_type: str, config: Dict[str, Any]) -> Any:
        if self.run_settings.data_settings.should_reuse_data(data_type):
            return self.data_handler.load_data(data_type, config)
        return None

    def create_current_metadata(self) -> dict:
        def set_to_list(obj):
            if isinstance(obj, set):
                return list(obj)
            return obj

        metadata = {
            "data_settings": self.run_settings.data_settings.to_dict(),
            "model_settings": self.run_settings.model_settings.__dict__,
            "embedding_settings": self.run_settings.embedding_settings.__dict__,
            "prompt_settings": self.run_settings.prompt_settings.__dict__,
            "clustering_settings": self.run_settings.clustering_settings.__dict__,
            "run_id": self.run_id
        }
        metadata_str = json.dumps(metadata, sort_keys=True, default=set_to_list)
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        metadata["metadata_hash"] = metadata_hash
        return metadata

    def load_run_metadata(self) -> Dict[str, Any]:
        if self.run_metadata_file.exists():
            with open(self.run_metadata_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def save_run_metadata(self, metadata: Dict[str, Any]):
        self.run_metadata[self.run_id] = metadata
        with open(self.run_metadata_file, 'w') as f:
            yaml.dump(self.run_metadata, f, default_flow_style=False)