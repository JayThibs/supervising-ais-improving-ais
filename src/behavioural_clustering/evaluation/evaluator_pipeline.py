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
from behavioural_clustering.utils.embedding_manager import EmbeddingManager

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
        self.embedding_manager = EmbeddingManager(run_settings.directory_settings.data_dir)
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
        self.model_eval_manager = ModelEvaluationManager(self.run_settings, self.llms, self.embedding_manager)
        self.approval_eval_manager = ApprovalEvaluationManager(self.run_settings, self.model_eval_manager)

    def run_evaluations(self) -> None:
        """
        Execute the main evaluation pipeline.

        This method orchestrates the entire evaluation process, including:
        1. Loading and preprocessing the dataset.
        2. Running model comparisons if specified.
        3. Performing prompt evaluations for each prompt type.
        4. Conducting hierarchical clustering if specified.

        The specific evaluations run are determined by the run_sections specified
        in the run settings.

        Side effects:
            - Loads and preprocesses the dataset.
            - Populates self.approvals_data with results from prompt evaluations.
            - Triggers model comparison, prompt evaluations, and hierarchical clustering as specified.
            - Saves the run data upon completion.
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
        """
        Perform model comparison analysis.

        This method handles the generation or loading of query results and embeddings
        for all models. It then performs clustering on the combined embeddings and
        prepares data for visualization.

        Args:
            metadata_config (Dict[str, Any]): Configuration metadata for the current run.

        Side effects:
            - Generates or loads query results for all models.
            - Creates or loads embeddings for the query results.
            - Performs spectral and chosen clustering on the embeddings.
            - Compiles a cluster table for analysis.
            - Triggers visualization of the model comparison results.
        """
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
            self.joint_embeddings_all_llms = create_embeddings(
                self.query_results_per_model,
                self.llms,
                self.run_settings.embedding_settings,
                self.embedding_manager
            )
            self.combined_embeddings = [e["embedding"] for e in self.joint_embeddings_all_llms]
            # Convert to list for JSON serialization before saving
            serializable_embeddings = [{**e, "embedding": e["embedding"].tolist()} for e in self.joint_embeddings_all_llms]
            self.data_handler.save_data(serializable_embeddings, "joint_embeddings_all_llms", metadata_config)
        else:
            print("Loaded existing embeddings.")
            # Convert loaded embeddings back to numpy arrays
            self.joint_embeddings_all_llms = [{**e, "embedding": np.array(e["embedding"])} for e in self.joint_embeddings_all_llms]
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
        Run the evaluation process for a specific prompt type.

        This method handles the generation or loading of approvals data and embeddings
        for a given prompt type. It then combines this data and performs cluster analysis
        and visualization.

        Args:
            prompt_type (str): The type of prompt to evaluate (e.g., 'personas', 'awareness').
            metadata_config (Dict[str, Any]): Configuration metadata for the current run.

        Side effects:
            - Updates self.approvals_data with new or loaded data for the prompt type.
            - Generates or loads embeddings for the approval statements.
            - Performs cluster analysis on the approval data.
            - Triggers visualization of the approval data.
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
        embeddings_key = f"embed_texts_{prompt_type}"
        print(f"Attempting to load embeddings with key: {embeddings_key}")
        embeddings = self.load_data(embeddings_key, metadata_config)
        if embeddings is None or not self.run_settings.data_settings.should_reuse_data(embeddings_key):
            print("Generating new text embeddings...")
            statements = [item['statement'] for item in self.approvals_data[prompt_type]]
            embeddings = self.embedding_manager.get_or_create_embeddings(statements, self.run_settings.embedding_settings)
            print(f"Type of embeddings after get_or_create_embeddings: {type(embeddings)}")
            print(f"Type of first embedding: {type(embeddings[0])}")
            print(f"Saving embeddings with key: {embeddings_key}")
            self.data_handler.save_data(embeddings, embeddings_key, metadata_config)
        else:
            print("Loaded existing text embeddings.")

        # Combine approvals data with embeddings
        print(f"Adding embeddings to approvals data...")
        for item, embedding in zip(self.approvals_data[prompt_type], embeddings):
            if isinstance(embedding, np.ndarray):
                item['embedding'] = embedding.tolist()
            elif isinstance(embedding, list):
                item['embedding'] = embedding
            else:
                print(f"Unexpected embedding type: {type(embedding)}")
                item['embedding'] = list(embedding)

        # Analyze cluster approval statistics and visualize approvals
        header_labels = ["ID", "N"]
        for model_info in self.model_eval_manager.model_info_list:
            model_name = model_info['model_name']
            for prompt in self.approval_prompts[prompt_type]:
                header_labels.append(f"{model_name} - {prompt}")
        header_labels.append("Inputs Themes")

        clusters_desc_table = [header_labels]
        
        self.cluster_analyzer.cluster_approval_stats(
            self.approvals_data[prompt_type],
            np.array(embeddings),  # Convert back to numpy array for clustering
            self.model_eval_manager.model_info_list,
            {prompt_type: self.approval_prompts[prompt_type]},
            clusters_desc_table
        )
        self.visualize_approvals(prompt_type, metadata_config)

    def run_hierarchical_clustering(self, metadata_config: Dict[str, Any]) -> None:
        """
        Perform hierarchical clustering for each prompt type and model.

        This method conducts hierarchical clustering analysis on the approval data
        for each prompt type and model specified in the run settings. It generates
        or loads clustering data and creates interactive treemaps for visualization.

        Args:
            metadata_config (Dict[str, Any]): Configuration metadata for the current run.

        Side effects:
            - Populates self.hierarchy_data with hierarchical clustering results for each prompt type.
            - Generates or loads hierarchical clustering data for each prompt type and model.
            - Creates interactive treemaps for visualization if not disabled in settings.
        """
        self.hierarchy_data = {}

        for prompt_type in self.approval_prompts.keys():
            print(f"Running hierarchical clustering for {prompt_type}...")
            
            metadata_config_prompt = metadata_config.copy()
            metadata_config_prompt["prompt_type"] = prompt_type
            
            hierarchical_key = f"hierarchical_clustering_{prompt_type}"
            hierarchy_data = self.load_data(hierarchical_key, metadata_config_prompt)
            
            if hierarchy_data is None:
                print(f"Generating new hierarchical clustering data for {prompt_type}...")
                
                # Extract embeddings for the statements (only once)
                statements = [item['statement'] for item in self.approvals_data[prompt_type]]
                statement_embeddings = self.embedding_manager.get_or_create_embeddings(statements, self.run_settings.embedding_settings)
                # Ensure statement_embeddings are numpy arrays
                statement_embeddings = [np.array(embedding) if isinstance(embedding, list) else embedding for embedding in statement_embeddings]
                
                # Perform clustering once for all models
                clustering = self.clustering.cluster_embeddings(
                    np.array(statement_embeddings),
                    n_clusters=self.run_settings.clustering_settings.n_clusters
                )
                
                hierarchy_data = {}
                for model_name in self.model_names:
                    # Generate rows for this model's clustering results
                    model_rows = self.cluster_analyzer.compile_cluster_table(
                        clustering=clustering,
                        data=self.approvals_data[prompt_type],
                        model_info_list=[{'model_name': model_name}],
                        data_type="approvals",
                        max_desc_length=self.run_settings.prompt_settings.max_desc_length
                    )
                    
                    hierarchy_data[model_name] = self.cluster_analyzer.calculate_hierarchical_cluster_data(
                        clustering,
                        self.approvals_data[prompt_type],
                        model_rows,
                        model_name
                    )
                self.data_handler.save_data(hierarchy_data, hierarchical_key, metadata_config_prompt)
            else:
                print(f"Loaded existing hierarchical clustering data for {prompt_type}.")
            
            self.hierarchy_data[prompt_type] = hierarchy_data
            
            # Create interactive treemap for all models
            if not self.run_settings.plot_settings.hide_interactive_treemap:
                interactive_treemap_data = self.viz.create_interactive_treemap(
                    hierarchy_data=hierarchy_data,
                    plot_type=prompt_type,
                    model_names=self.model_names,
                    filename=f"{self.run_settings.directory_settings.viz_dir}/interactive_treemap_{prompt_type}"
                )
                self.data_handler.save_data(
                    interactive_treemap_data,
                    f"interactive_treemap_{prompt_type}",
                    metadata_config_prompt
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
        
        # Use a different key for approvals t-SNE reduction
        tsne_key = f"tsne_reduction_approvals_{prompt_type}"
        dim_reduce_tsne = self.load_data(tsne_key, metadata_config)
        
        if dim_reduce_tsne is None:
            print(f"Generating new tsne_reduction for {prompt_type} approvals...")
            dim_reduce_tsne = tsne_reduction(
                combined_embeddings=approvals_embeddings,
                tsne_settings=self.run_settings.tsne_settings,
                random_state=self.run_settings.random_state
            )
            self.data_handler.save_data(dim_reduce_tsne, tsne_key, metadata_config)
        else:
            print(f"Loaded existing tsne_reduction for {prompt_type} approvals.")

        for model_name in self.model_names:
            for condition in [1, 0, -1]:
                condition_str = {1: "approved", 0: "disapproved", -1: "no_response"}[condition]
                self.viz.plot_approvals(
                    dim_reduce=dim_reduce_tsne,
                    approval_data=self.approvals_data[prompt_type],
                    model_name=model_name,
                    condition=condition,
                    plot_type=prompt_type,
                    filename=self.generate_plot_filename([model_name], f"{prompt_type}-{condition_str}"),
                    title=f"Embeddings of {condition_str} for {model_name} {prompt_type} responses",
                    show_plot=not self.run_settings.plot_settings.should_hide_approval_plot(prompt_type)
                )

    def generate_plot_filename(self, model_names: list, plot_type: str) -> str:
        plot_type = plot_type.replace(" ", "_")
        
        # Replace condition numbers with descriptive terms
        condition_map = {"-1": "no_response", "0": "disapproved", "1": "approved"}
        for num, desc in condition_map.items():
            plot_type = plot_type.replace(f"-{num}", f"_{desc}")
        
        # Add number of statements to the filename
        n_statements = self.run_settings.data_settings.n_statements
        
        # Generate the new filename
        filename = f"{self.run_settings.directory_settings.viz_dir}/" + "-".join(model_names) + f"-{plot_type}-{n_statements}_statements.png"
        
        # Check if the file already exists
        if os.path.exists(filename):
            # If it does, rename the existing file with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = filename.replace(".png", f"_{timestamp}.png")
            os.rename(filename, new_filename)
            print(f"Existing file renamed to: {new_filename}")
        
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