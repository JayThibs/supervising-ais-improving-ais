import os
import json
import yaml
import numpy as np
from datetime import datetime
import hashlib
from typing import Dict, Any
from pathlib import Path

from behavioural_clustering.config.run_settings import RunSettings
from behavioural_clustering.utils.data_preparation import DataPreparation, DataHandler
from behavioural_clustering.utils.visualization import Visualization
from behavioural_clustering.evaluation.clustering import Clustering, ClusterAnalyzer
from behavioural_clustering.evaluation.embeddings import embed_texts, create_embeddings
from behavioural_clustering.evaluation.dimensionality_reduction import tsne_reduction
from behavioural_clustering.models.local_models import LocalModel
from behavioural_clustering.evaluation.model_evaluation_manager import ModelEvaluationManager
from behavioural_clustering.evaluation.approval_evaluation_manager import ApprovalEvaluationManager

class EvaluatorPipeline:
    def __init__(self, run_settings: RunSettings):
        """
        Initialize the EvaluatorPipeline with the given run settings.

        Args:
            run_settings (RunSettings): Configuration settings for the evaluation run.
        """
        self.run_settings = run_settings
        self.run_sections = run_settings.run_only if run_settings.run_only else run_settings.run_sections
        self.setup_directories()
        self.setup_models()
        self.run_id = self.generate_run_id()
        self.data_handler = DataHandler(self.run_settings.directory_settings.data_dir, self.run_id)
        self.run_metadata_file = self.run_settings.directory_settings.data_dir / "metadata" / "run_metadata.yaml"
        self.run_metadata = self.load_run_metadata()
        self.setup_managers()
        self.load_approval_prompts()

    def load_approval_prompts(self):
        approval_prompts_path = self.run_settings.directory_settings.data_dir / "prompts" / "approval_prompts.json"
        with open(approval_prompts_path, 'r') as f:
            self.approval_prompts = json.load(f)

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
            dir_path = getattr(self.run_settings.directory_settings, dir_name, None)
            if dir_path:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
            else:
                print(f"Warning: {dir_name} not found in directory_settings")

    def setup_models(self) -> None:
        self.llms = self.run_settings.model_settings.models
        self.model_names = [model for _, model in self.llms]
        self.local_models = {model: LocalModel(model_name_or_path=model) for model_family, model in self.llms if model_family == "local"}
        self.embedding_model_name = self.run_settings.embedding_settings.embedding_model

    def setup_managers(self) -> None:
        self.data_prep = DataPreparation()
        self.viz = Visualization(self.run_settings.plot_settings)
        self.clustering = Clustering(self.run_settings)
        self.cluster_analyzer = ClusterAnalyzer(self.run_settings)
        self.model_eval_manager = ModelEvaluationManager(self.run_settings, self.llms)
        self.approval_eval_manager = ApprovalEvaluationManager(self.run_settings, self.model_eval_manager)

    def run_evaluations(self) -> None:
        """
        Execute the main evaluation pipeline, including model comparison,
        prompt evaluations, and hierarchical clustering if specified in the run settings.
        """
        print(f"Data settings: {self.run_settings.data_settings}")
        self.text_subset = self.data_prep.load_and_preprocess_data(self.run_settings.data_settings)
        
        if len(self.text_subset) == 0:
            raise ValueError("No texts loaded. Please check your dataset configuration.")

        print(f"Loaded {len(self.text_subset)} texts.")
        print("Sample texts:")
        for i in range(min(5, len(self.text_subset))):
            print(f"Text {i+1}: {self.text_subset[i][:100]}...")

        # Create metadata once
        metadata_config = self.create_current_metadata()

        if "model_comparison" in self.run_sections:
            self.run_model_comparison(metadata_config)
        
        self.approvals_data = {}
        for prompt_type in self.approval_prompts.keys():
            if f"{prompt_type}_evaluation" in self.run_sections:
                self.run_prompt_evaluation(prompt_type, metadata_config)

        if "hierarchical_clustering" in self.run_sections:
            self.run_hierarchical_clustering(metadata_config)

        self.save_run_data()

    def run_model_comparison(self, metadata_config: Dict[str, Any]) -> None:
        self.query_results_per_model = self.load_data("all_query_results", metadata_config)
        if self.query_results_per_model is None:
            print("Generating new query results...")
            self.query_results_per_model = self.model_eval_manager.generate_responses(self.text_subset)
            self.data_handler.save_data(self.query_results_per_model, "all_query_results", metadata_config)
        else:
            print("Loaded existing query results.")

        if not self.query_results_per_model:
            print("Warning: No query results available. Check the generate_responses function.")
            return

        self.joint_embeddings_all_llms = self.load_data("joint_embeddings_all_llms", metadata_config)
        if self.joint_embeddings_all_llms is None:
            print("Generating new embeddings...")
            # joint_embeddings_all_llms is a list of dictionaries, each containing:
            # {"model_num": model_num, "statement": input, "response": response, "embedding": embedding, "model_name": model_name}
            self.joint_embeddings_all_llms = create_embeddings(
                self.query_results_per_model,
                self.llms,
                self.run_settings.embedding_settings
            )
            self.combined_embeddings = [e["embedding"] for e in self.joint_embeddings_all_llms]
            self.data_handler.save_data(self.joint_embeddings_all_llms, "joint_embeddings_all_llms", metadata_config)
        else:
            print("Loaded existing embeddings.")
            self.combined_embeddings = [e["embedding"] for e in self.joint_embeddings_all_llms]
        
        combined_embeddings_array = np.array(self.combined_embeddings)
        
        self.spectral_clustering = self.load_data("spectral_clustering", metadata_config)
        if self.spectral_clustering is None:
            print("Generating new spectral clustering of embeddings...")
            self.spectral_clustering = self.clustering.cluster_embeddings(
                combined_embeddings_array,
                clustering_algorithm="SpectralClustering",
                n_clusters=self.run_settings.clustering_settings.n_clusters,
                affinity=self.run_settings.clustering_settings.affinity
            )
            self.data_handler.save_data(self.spectral_clustering, "spectral_clustering", metadata_config)
        else:
            print("Loaded existing spectral clustering.")

        self.chosen_clustering = self.load_data("chosen_clustering", metadata_config)
        if self.chosen_clustering is None:
            print("Generating new clustering...")
            self.chosen_clustering = self.clustering.cluster_embeddings(combined_embeddings_array)
            self.data_handler.save_data(self.chosen_clustering, "chosen_clustering", metadata_config)
        else:
            print("Loaded existing clustering.")

        self.rows = self.load_data("compile_cluster_table", metadata_config)
        if self.rows is None:
            print("Generating new cluster table...")
            self.rows = self.cluster_analyzer.compile_cluster_table(
                clustering=self.chosen_clustering,
                data=self.joint_embeddings_all_llms,
                model_info_list=self.model_eval_manager.model_info_list,
                data_type="joint_embeddings",
                max_desc_length=self.run_settings.prompt_settings.max_desc_length
            )
            self.data_handler.save_data(self.rows, "compile_cluster_table", metadata_config)
        else:
            print("Loaded existing cluster table.")

        self.visualize_model_comparison(metadata_config)

    def run_prompt_evaluation(self, prompt_type: str, metadata_config: Dict[str, Any]) -> None:
        """
        Evaluate a specific prompt type by generating approvals data,
        embedding texts, and analyzing cluster approval statistics.

        Args:
            prompt_type (str): The type of prompt to evaluate (e.g., 'personas', 'awareness').
        """
        metadata_config = metadata_config.copy()
        metadata_config["prompt_type"] = prompt_type
        print(f"Config for {prompt_type}: {metadata_config.keys()}")

        # Load or generate approvals data
        approvals_key = f"approvals_statements_{prompt_type}"
        self.approvals_data[prompt_type] = self.load_data(approvals_key, metadata_config)
        if self.approvals_data[prompt_type] is None or not self.run_settings.data_settings.should_reuse_data(approvals_key):
            print(f"Generating new approvals data for {prompt_type}...")
            self.approvals_data[prompt_type] = self.approval_eval_manager.load_or_generate_approvals_data(prompt_type, self.text_subset)
            self.data_handler.save_data(self.approvals_data[prompt_type], approvals_key, metadata_config)
        else:
            print(f"Loaded existing approvals data for {prompt_type}.")

        # Load or generate embeddings
        embeddings_key = "embed_texts"
        print(f"Attempting to load embeddings with key: {embeddings_key}")
        embeddings = self.load_data(embeddings_key, metadata_config)
        if embeddings is None or not self.run_settings.data_settings.should_reuse_data(embeddings_key):
            print("Generating new text embeddings...")
            statements = [item['statement'] for item in self.approvals_data[prompt_type]]
            embeddings = embed_texts(statements, self.run_settings.embedding_settings)
            print(f"Saving embeddings with key: {embeddings_key}")
            self.data_handler.save_data(embeddings, embeddings_key, metadata_config)
        else:
            print("Loaded existing text embeddings.")

        # Combine approvals data with embeddings
        print(f"Adding embeddings to approvals data...")
        for item, embedding in zip(self.approvals_data[prompt_type], embeddings):
            item['embedding'] = embedding

        # Analyze cluster approval statistics and visualize approvals
        self.cluster_analyzer.cluster_approval_stats(
            self.approvals_data[prompt_type],
            embeddings,
            self.model_eval_manager.model_info_list,
            {prompt_type: self.approval_prompts[prompt_type]}
        )
        self.visualize_approvals(prompt_type, metadata_config)

    def run_hierarchical_clustering(self, metadata_config: Dict[str, Any]) -> None:
        """
        Perform hierarchical clustering for each prompt type specified in the run settings.
        """
        self.hierarchy_data = {}

        for prompt_type in self.approval_prompts.keys():
            print(f"Running hierarchical clustering for {prompt_type}...")
            
            # Update metadata_config with prompt_type
            metadata_config_prompt = metadata_config.copy()
            metadata_config_prompt["prompt_type"] = prompt_type
            
            # Attempt to load existing hierarchical data
            hierarchical_key = f"hierarchical_clustering_{prompt_type}"
            hierarchy_data = self.load_data(hierarchical_key, metadata_config_prompt)
            
            if hierarchy_data is None:
                print(f"Generating new hierarchical clustering data for {prompt_type}...")
                hierarchy_data = self.cluster_analyzer.calculate_hierarchical_cluster_data(
                    self.chosen_clustering,
                    self.approvals_data[prompt_type],
                    self.rows
                )
                # Save the newly generated hierarchical data
                self.data_handler.save_data(hierarchy_data, hierarchical_key, metadata_config_prompt)
            else:
                print(f"Loaded existing hierarchical clustering data for {prompt_type}.")
            
            self.hierarchy_data[prompt_type] = hierarchy_data
            
            labels = list(self.approval_prompts[prompt_type].keys())
            
            for model_name in self.model_names:
                self.viz.visualize_hierarchical_plot(
                    hierarchy_data=hierarchy_data,
                    plot_type=prompt_type,
                    filename=f"{self.run_settings.directory_settings.viz_dir}/hierarchical_clustering_{prompt_type}_{model_name}",
                    labels=labels,
                    show_plot=not self.run_settings.plot_settings.hide_hierarchical
                )

            print(f"Hierarchical clustering for {prompt_type} completed.")

    def visualize_model_comparison(self, metadata_config: Dict[str, Any]) -> None:
        """
        Create visualizations for the model comparison results, including
        t-SNE plots of embedding responses, displays of statement themes,
        and spectral clustering visualization.
        """
        dim_reduce_tsne = self.load_data(
            "tsne_reduction",
            metadata_config
        )
        if dim_reduce_tsne is None:
            print("Generating new tsne_reduction...")
            dim_reduce_tsne = tsne_reduction(
                combined_embeddings=self.combined_embeddings,
                tsne_settings=self.run_settings.tsne_settings,
                random_state=self.run_settings.random_state
            )
            self.data_handler.save_data(dim_reduce_tsne, "tsne_reduction", metadata_config)
        else:
            print("Loaded existing tsne_reduction.")

        # Always generate and save the plot, but only display if not hidden
        self.viz.plot_embedding_responses(
            dim_reduce_tsne, 
            self.joint_embeddings_all_llms, 
            self.model_names, 
            self.generate_plot_filename(self.model_names, "tsne_embedding_responses"),
            show_plot=not self.run_settings.plot_settings.hide_model_comparison
        )
        self.cluster_analyzer.display_statement_themes(self.chosen_clustering, self.rows, self.model_eval_manager.model_info_list)

        # Add spectral clustering visualization
        if not self.run_settings.plot_settings.hide_spectral:
            self.viz.plot_spectral_clustering(
                self.spectral_clustering.labels_,
                self.run_settings.clustering_settings.n_clusters,
                self.generate_plot_filename(self.model_names, "spectral_clustering"),
            )

    def visualize_approvals(self, prompt_type: str, metadata_config: Dict[str, Any]) -> None:
        """
        Generate visualizations for approval data for a specific prompt type.

        Args:
            prompt_type (str):  The type of prompt to visualize. Examples: 'personas', 'awareness'. 
                                Found in approval_prompts.json.
        """
        approvals_embeddings = np.array([e['embedding'] for e in self.approvals_data[prompt_type]])
        dim_reduce_tsne = self.load_data(
            "tsne_reduction",
            metadata_config
        ) # may have already been created in run_model_comparison, unless it was skipped
        if dim_reduce_tsne is None:
            print("Generating new tsne_reduction...")
            dim_reduce_tsne = tsne_reduction(
                combined_embeddings=approvals_embeddings,
                tsne_settings=self.run_settings.tsne_settings,
                random_state=self.run_settings.random_state
            )
            self.data_handler.save_data(dim_reduce_tsne, "tsne_reduction", metadata_config)
        else:
            print("Loaded existing tsne_reduction.")

        for model_name in self.model_names:
            for condition in [1, 0, -1]:
                self.viz.plot_approvals(
                    dim_reduce= dim_reduce_tsne,
                    approval_data=self.approvals_data[prompt_type],
                    model_name=model_name,
                    condition=condition,
                    plot_type=prompt_type,
                    filename=self.generate_plot_filename([model_name], f"{prompt_type}-{condition}"),
                    title=f"Embeddings of {['approvals', 'disapprovals', 'no response'][condition]} for {model_name} {prompt_type} responses",
                    show_plot=not self.run_settings.plot_settings.should_hide_approval_plot(prompt_type)
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
            "run_settings": self.run_settings.to_dict(),
            "test_mode": self.run_settings.test_mode,
            "run_sections": self.run_settings.run_sections,
            "data_file_ids": {}
        }
        
        for data_type in self.data_handler.data_metadata:
            if data_type in self.data_handler.data_metadata:
                file_id = list(self.data_handler.data_metadata[data_type].keys())[-1]  # Get the latest file ID
                metadata["data_file_ids"][data_type] = file_id
        
        self.save_run_metadata(metadata)
        print(f"Run metadata saved to {self.run_metadata_file}")

    def load_data(self, data_type: str, metadata_config: Dict[str, Any]) -> Any:
        if self.run_settings.data_settings.should_reuse_data(data_type):
            return self.data_handler.load_saved_data(data_type, metadata_config)
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