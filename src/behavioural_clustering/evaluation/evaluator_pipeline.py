import os
import json
import yaml
import numpy as np
from datetime import datetime
import hashlib
from typing import Dict, Any, List
from pathlib import Path
from dataclasses import asdict
import logging
from termcolor import colored
from tqdm import tqdm

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
from behavioural_clustering.utils.cache_management import CacheManager, CacheMetadata, ModelParams, EmbeddingParams, DataParams
from behavioural_clustering.utils.embedding_data import JointEmbeddings

# Import the new IterativeAnalyzer class
from behavioural_clustering.evaluation.iterative_analysis import IterativeAnalyzer

logger = logging.getLogger(__name__)

class EvaluatorPipeline:
    """
    The EvaluatorPipeline orchestrates the evaluation process, including data preparation,
    model comparisons, prompt evaluations, hierarchical clustering, and optionally uses
    an iterative approach to discover behavioral differences between models.
    """

    def __init__(self, run_settings: RunSettings):
        """
        Args:
            run_settings (RunSettings): Configuration settings for the evaluation run.
        """
        self.run_settings = run_settings
        self.run_sections = run_settings.run_only if run_settings.run_only else run_settings.run_sections
        self.setup_directories()
        self.llms = self.run_settings.model_settings.models
        self.model_names = [model for _, model in self.llms]
        self.embedding_model_name = self.run_settings.embedding_settings.embedding_model

        self.model_evaluation_manager = ModelEvaluationManager(run_settings, self.llms)
        self.model_info_list = self.model_evaluation_manager.model_info_list
        self.run_id = self.generate_run_id()
        self.data_handler = DataHandler(self.run_settings.directory_settings.data_dir, self.run_id)
        self.run_metadata_file = self.run_settings.directory_settings.data_dir / "metadata" / "run_metadata.yaml"
        self.run_metadata = self.load_run_metadata()
        self.cluster_table_ids = {}
        self.embedding_manager = EmbeddingManager(run_settings.directory_settings.data_dir)

        # Create the iterative analyzer instance with proper path
        iterative_prompts_path = (
            self.run_settings.directory_settings.data_dir / 
            "iterative" / 
            "prompts" / 
            "newly_generated_prompts.json"
        )
        self.iterative_analyzer = IterativeAnalyzer(self.run_settings)

        self.setup_managers()
        self.load_approval_prompts()

    def load_approval_prompts(self):
        """
        Load the approval prompts from JSON in the data directory.
        """
        approval_prompts_path = self.run_settings.directory_settings.data_dir / "prompts" / "approval_prompts.json"
        with open(approval_prompts_path, 'r') as f:
            self.approval_prompts = json.load(f)

    def generate_run_id(self) -> str:
        """
        Generate a unique run ID based on existing metadata.
        """
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
        """
        Ensure that required directories exist for saving data.
        """
        for dir_name in ['data_dir', 'evals_dir', 'results_dir', 'viz_dir', 'tables_dir', 'pickle_dir']:
            dir_path = getattr(self.run_settings.directory_settings, dir_name, None)
            if dir_path:
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
            else:
                print(f"Warning: {dir_name} not found in directory_settings")

    def setup_managers(self) -> None:
        """
        Initialize helper classes for data, visualization, clustering, and approvals.
        """
        self.data_prep = DataPreparation()
        self.viz = Visualization(self.run_settings.plot_settings)
        self.clustering = Clustering(self.run_settings)
        self.cluster_analyzer = ClusterAnalyzer(self.run_settings)
        self.approval_eval_manager = ApprovalEvaluationManager(self.run_settings, self.model_evaluation_manager)

    def run_evaluations(self) -> None:
        """
        Execute the evaluation pipeline, including data loading, model comparison,
        persona/awareness evaluations, and hierarchical clustering.
        """
        try:
            self.model_evaluation_manager.unload_all_models()

            logger.info(colored("\nLoading and preprocessing data...", "cyan"))
            self.text_subset = self.data_prep.load_and_preprocess_data(self.run_settings.data_settings)

            if len(self.text_subset) == 0:
                raise ValueError("No texts loaded. Please check your dataset configuration.")

            logger.info(colored(f"Loaded {len(self.text_subset)} texts", "green"))
            logger.info(colored("\nSample texts:", "cyan"))
            for i in range(min(5, len(self.text_subset))):
                logger.info(colored(f"Text {i+1}: {self.text_subset[i][:100]}...", "cyan"))

            # Save initial results
            results = {
                "texts": self.text_subset,
                "settings": self.run_settings.to_dict(),
                "timestamp": datetime.now().isoformat()
            }
            results_path = Path(self.run_settings.directory_settings.results_dir) / "initial_results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

            metadata_config = self.create_current_metadata()

            # Create progress bar for pipeline sections
            sections_to_run = []
            if "model_comparison" in self.run_sections:
                sections_to_run.append(("Model Comparison", lambda: self.run_model_comparison(metadata_config)))
            
            for prompt_type in self.approval_prompts.keys():
                if f"{prompt_type}_evaluation" in self.run_sections:
                    sections_to_run.append(
                        (f"{prompt_type.title()} Evaluation", 
                         lambda pt=prompt_type: self.run_prompt_evaluation(pt, metadata_config))
                    )
            
            if "hierarchical_clustering" in self.run_sections:
                sections_to_run.append(("Hierarchical Clustering", lambda: self.run_hierarchical_clustering(metadata_config)))
            
            if "iterative_evaluation" in self.run_sections:
                sections_to_run.append(("Iterative Evaluation", lambda: self.run_iterative_evaluation()))
                
            if "report_cards" in self.run_sections:
                sections_to_run.append(("Report Cards", lambda: self.run_report_cards()))

            # Run each section with progress bar
            section_pbar = tqdm(sections_to_run, desc="Pipeline Progress", unit="section")
            for section_name, section_func in section_pbar:
                section_pbar.set_description(f"Running {section_name}")
                section_func()
                
                # Save model evaluation results if available after model comparison
                if section_name == "Model Comparison" and hasattr(self, 'query_results_per_model') and self.query_results_per_model:
                    model_results = {
                        "model_responses": self.query_results_per_model,
                        "timestamp": datetime.now().isoformat()
                    }
                    model_results_path = Path(self.run_settings.directory_settings.results_dir) / "model_results.json"
                    with open(model_results_path, "w", encoding="utf-8") as f:
                        json.dump(model_results, f, indent=2)
                
                section_pbar.set_postfix({"status": "completed"})

            self.save_run_data()
            logger.info(colored("\nEvaluation pipeline completed successfully", "green"))
            
        except Exception as e:
            logger.error(colored(f"Error in evaluation pipeline: {str(e)}", "red"))
            raise

    def run_model_comparison(self, metadata_config: Dict[str, Any]) -> None:
        """
        Generate or load query results and embeddings, run clustering, and prepare data for visualization.
        """
        logger.info(colored("\nStarting model comparison...", "cyan"))
        
        # Create visualization directory if it doesn't exist
        viz_dir = Path(self.run_settings.directory_settings.viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Create cache manager
        cache_dir = Path(self.run_settings.directory_settings.data_dir) / "saved_data"
        cache_manager = CacheManager(cache_dir)
        
        # Create cache metadata for query results
        query_results_metadata = CacheMetadata(
            data_params=DataParams(
                n_statements=self.run_settings.data_settings.n_statements,
                datasets=self.run_settings.data_settings.datasets,
                random_seed=self.run_settings.data_settings.random_state,
                new_generation=self.run_settings.data_settings.new_generation,
                reuse_data=self.run_settings.data_settings.reuse_data
            ),
            model_params=[
                ModelParams(
                    model_family=family,
                    model_name=name,
                    temperature=self.run_settings.model_settings.temperature,
                    max_tokens=self.run_settings.model_settings.generate_responses_max_tokens,
                    system_message=msg
                )
                for (family, name), msg in zip(
                    self.llms,
                    self.run_settings.model_settings.model_system_messages
                )
            ],
            embedding_params=EmbeddingParams(
                model_name=self.run_settings.embedding_settings.embedding_model,
                batch_size=self.run_settings.embedding_settings.batch_size,
                max_retries=self.run_settings.embedding_settings.max_retries,
                initial_sleep_time=self.run_settings.embedding_settings.initial_sleep_time
            ),
            cache_type="all_query_results"
        )
        
        # Try to load cached query results
        cached_results = cache_manager.load_cache(query_results_metadata)
        if cached_results is not None:
            self.query_results_per_model = cached_results[0]
            logger.info(colored("Loaded cached query results", "green"))
        else:
            logger.info(colored("Generating new query results...", "cyan"))
            self.query_results_per_model = self.model_evaluation_manager.generate_responses(self.text_subset)
            cache_manager.save_cache(self.query_results_per_model, query_results_metadata)

        if not self.query_results_per_model:
            logger.warning(colored("No query results available. Check the generate_responses function.", "yellow"))
            return

        # Create embeddings with proper ordering
        logger.info(colored("\nGenerating embeddings...", "cyan"))
        model_tuples = [(family, name) for family, name in self.llms]
        self.joint_embeddings_all_llms = create_embeddings(
            self.query_results_per_model,
            model_tuples,
            self.run_settings.embedding_settings,
            self.embedding_manager,
            cache_manager
        )

        # Convert embeddings for visualization
        combined_embeddings = self.joint_embeddings_all_llms.get_embedding_matrix()

        # Generate t-SNE visualization
        logger.info(colored("\nGenerating t-SNE visualization...", "cyan"))
        dim_reduce_tsne = tsne_reduction(combined_embeddings)
        
        # Save t-SNE plot
        plot_filename = viz_dir / "model_comparison.png"
        self.viz.plot_embedding_responses(
            dim_reduce_tsne,
            self.joint_embeddings_all_llms.get_all_embeddings(),
            [name for _, name in self.llms],
            plot_filename,
            show_plot=False
        )
        logger.info(colored("Saved t-SNE plot", "green"))

        # Run clustering
        logger.info(colored("\nRunning spectral clustering...", "cyan"))
        self.spectral_clustering = self.clustering.cluster_embeddings(
            combined_embeddings,
            clustering_algorithm="SpectralClustering",
            n_clusters=self.run_settings.clustering_settings.n_clusters,
            affinity=self.run_settings.clustering_settings.affinity
        )

        # Save spectral clustering plot
        plot_filename = viz_dir / "spectral_clustering.png"
        # Extract labels from the clustering object
        if hasattr(self.spectral_clustering, 'labels_'):
            cluster_labels = self.spectral_clustering.labels_
        else:
            cluster_labels = self.spectral_clustering  # In case it's already the labels array
            
        self.viz.plot_spectral_clustering_plotly(
            cluster_labels,
            self.run_settings.clustering_settings.n_clusters,
            "Spectral Clustering Results"
        ).write_image(str(plot_filename))
        logger.info(colored("Saved spectral clustering plot", "green"))

        # Run chosen clustering
        logger.info(colored("\nRunning main clustering algorithm...", "cyan"))
        self.chosen_clustering = self.clustering.cluster_embeddings(combined_embeddings)

        # Compile cluster table
        logger.info(colored("\nCompiling cluster analysis...", "cyan"))
        self.rows = self.cluster_analyzer.compile_cluster_table(
            clustering=self.chosen_clustering,
            data=self.joint_embeddings_all_llms.get_all_embeddings(),
            model_info_list=self.model_info_list,
            data_type="joint_embeddings",
            max_desc_length=self.run_settings.prompt_settings.max_desc_length,
            run_settings=self.run_settings
        )

        # Display cluster themes
        try:
            logger.info(colored("\nAnalyzing statement themes...", "cyan"))
            self.cluster_analyzer.display_statement_themes(
                self.chosen_clustering,
                self.rows,
                self.model_info_list
            )
        except Exception as e:
            logger.error(colored(f"Error displaying statement themes: {str(e)}", "red"))
            logger.error(colored("Continuing with pipeline despite theme display error...", "yellow"))
            
        logger.info(colored("\nModel comparison completed successfully", "green"))

    def run_prompt_evaluation(self, prompt_type: str, metadata_config: Dict[str, Any]) -> None:
        """
        Evaluate a specific prompt type, including approvals data and embedding generation.
        """
        logger.info(colored(f"\nStarting {prompt_type} evaluation...", "cyan"))
        metadata_config = metadata_config.copy()
        metadata_config["prompt_type"] = prompt_type

        # Initialize approvals data if not exists
        if not hasattr(self, 'approvals_data'):
            self.approvals_data = {}

        # Load or generate approvals data
        approvals_key = f"approvals_statements_{prompt_type}"
        self.approvals_data[prompt_type] = self.load_data(approvals_key, metadata_config)
        if (
            self.approvals_data[prompt_type] is None
            or not self.run_settings.data_settings.should_reuse_data(approvals_key)
        ):
            logger.info(colored(f"Generating new approvals data for {prompt_type}...", "cyan"))
            self.approvals_data[prompt_type] = self.approval_eval_manager.load_or_generate_approvals_data(
                prompt_type, self.text_subset
            )
            self.data_handler.save_data(self.approvals_data[prompt_type], approvals_key, metadata_config)
        else:
            logger.info(colored(f"Loaded existing approvals data for {prompt_type}", "green"))

        # Generate embeddings
        embeddings_key = f"embed_texts_{prompt_type}"
        logger.info(colored("\nProcessing embeddings...", "cyan"))
        embeddings = self.load_data(embeddings_key, metadata_config)
        if (
            embeddings is None
            or not self.run_settings.data_settings.should_reuse_data(embeddings_key)
        ):
            logger.info(colored("Generating new text embeddings...", "cyan"))
            statements = [item['statement'] for item in self.approvals_data[prompt_type]]
            
            # Create progress bar for embedding generation
            statements_pbar = tqdm(statements, desc="Generating embeddings", unit="text")
            embeddings = []
            for statement in statements_pbar:
                embedding = self.embedding_manager.get_or_create_embeddings(
                    [statement], self.run_settings.embedding_settings
                )[0]
                embeddings.append(embedding)
                
            logger.info(colored(f"Generated {len(embeddings)} embeddings", "green"))
            self.data_handler.save_data(embeddings, embeddings_key, metadata_config)
        else:
            logger.info(colored(f"Loaded {len(embeddings)} existing embeddings", "green"))

        # Add embeddings to approvals data
        logger.info(colored("\nProcessing approval data...", "cyan"))
        for item, embedding in tqdm(zip(self.approvals_data[prompt_type], embeddings), 
                                  desc="Adding embeddings to approvals", 
                                  total=len(embeddings)):
            if isinstance(embedding, np.ndarray):
                item['embedding'] = embedding.tolist()
            elif isinstance(embedding, list):
                item['embedding'] = embedding
            else:
                logger.warning(colored(f"Unexpected embedding type: {type(embedding)}", "yellow"))
                item['embedding'] = list(embedding)

        # Prepare cluster table headers
        header_labels = ["ID", "N"]
        for model_info in self.model_info_list:
            model_name = model_info['model_name']
            for prompt in self.approval_prompts[prompt_type]:
                header_labels.append(f"{model_name} - {prompt}")
        header_labels.append("Inputs Themes")

        clusters_desc_table = [header_labels]

        # Run cluster analysis
        logger.info(colored("\nRunning cluster analysis...", "cyan"))
        csv_file_path = self.cluster_analyzer.cluster_approval_stats(
            self.approvals_data[prompt_type],
            np.array(embeddings),
            self.model_info_list,
            {prompt_type: self.approval_prompts[prompt_type]},
            clusters_desc_table,
        )

        csv_key = f"prompt_cluster_table_csv_{prompt_type}"
        self.cluster_table_ids[csv_key] = csv_file_path

        # Generate visualizations
        logger.info(colored("\nGenerating visualizations...", "cyan"))
        self.visualize_approvals(prompt_type, metadata_config)
        
        logger.info(colored(f"\n{prompt_type} evaluation completed successfully", "green"))

    def run_hierarchical_clustering(self, metadata_config: Dict[str, Any]) -> None:
        """
        Perform hierarchical clustering for each prompt type and store or visualize results.
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
                statements = [item['statement'] for item in self.approvals_data[prompt_type]]
                statement_embeddings = self.embedding_manager.get_or_create_embeddings(
                    statements, self.run_settings.embedding_settings
                )
                statement_embeddings = [
                    np.array(embedding) if isinstance(embedding, list) else embedding
                    for embedding in statement_embeddings
                ]

                clustering = self.clustering.cluster_embeddings(
                    np.array(statement_embeddings),
                    n_clusters=self.run_settings.clustering_settings.n_clusters
                )

                hierarchy_data = {}
                for model_name in self.model_names:
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
        Create visualizations for the model comparison results (t-SNE embeddings, statement themes, spectral clustering).
        """
        dim_reduce_tsne = self.load_data("tsne_reduction", metadata_config)
        if dim_reduce_tsne is None:
            print("Generating new tsne_reduction...")
            combined_embeddings = np.array([e["embedding"] for e in self.joint_embeddings_all_llms])
            dim_reduce_tsne = tsne_reduction(
                combined_embeddings=combined_embeddings,
                tsne_settings=self.run_settings.tsne_settings,
                random_state=self.run_settings.random_state
            )
            self.data_handler.save_data(dim_reduce_tsne, "tsne_reduction", metadata_config)
        else:
            print("Loaded existing tsne_reduction.")

        self.viz.plot_embedding_responses(
            dim_reduce_tsne,
            self.joint_embeddings_all_llms,
            self.model_names,
            self.generate_plot_filename(self.model_names, "tsne_embedding_responses"),
            show_plot=not self.run_settings.plot_settings.hide_model_comparison
        )
        self.cluster_analyzer.display_statement_themes(
            self.chosen_clustering, self.rows, self.model_info_list
        )

        if not self.run_settings.plot_settings.hide_spectral:
            if hasattr(self.spectral_clustering, "labels_"):
                sc_labels = self.spectral_clustering.labels_
            else:
                sc_labels = self.spectral_clustering

            self.viz.plot_spectral_clustering_plotly(
                sc_labels,
                self.run_settings.clustering_settings.n_clusters,
                self.generate_plot_filename(self.model_names, "spectral_clustering")
            )

    def visualize_approvals(self, prompt_type: str, metadata_config: Dict[str, Any]) -> None:
        """
        Generate t-SNE visualizations for approvals data for each model, color-coded by approval condition.
        """
        approvals_embeddings = np.array(
            [e['embedding'] for e in self.approvals_data[prompt_type]]
        )

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
        """
        Generate a filename for saving plots, substituting special condition markers.
        """
        plot_type = plot_type.replace(" ", "_")
        condition_map = {"-1": "no_response", "0": "disapproved", "1": "approved"}
        for num, desc in condition_map.items():
            plot_type = plot_type.replace(f"-{num}", f"_{desc}")

        n_statements = self.run_settings.data_settings.n_statements
        model_names_str = "-".join([name.replace("/", "_") for name in model_names])
        filename = f"{model_names_str}-{plot_type}-{n_statements}_statements.png"

        full_path = self.run_settings.directory_settings.viz_dir / filename
        if full_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = full_path.with_name(f"{full_path.stem}_{timestamp}{full_path.suffix}")
            full_path.rename(new_filename)
            print(f"Existing file renamed to: {new_filename}")

        return str(full_path)

    def save_run_data(self) -> None:
        """
        Save the results of the evaluation run and record metadata.
        """
        metadata = {
            "dataset_names": "_".join(self.run_settings.data_settings.datasets),
            "model_names": self.model_names,
            "n_statements": self.run_settings.data_settings.n_statements,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "run_settings": self.run_settings.to_dict(),
            "test_mode": self.run_settings.test_mode,
            "run_sections": self.run_settings.run_sections,
            "data_file_ids": {},
        }

        for data_type, file_id in self.data_handler.data_file_ids.items():
            metadata["data_file_ids"][data_type] = file_id

        self.save_run_metadata(metadata)
        print(f"Run metadata saved to {self.run_metadata_file}")

    def load_data(self, data_type: str, metadata_config: Dict[str, Any]) -> Any:
        """
        Load saved data if reuse is enabled and the data exists in the data handler.
        """
        if self.run_settings.data_settings.should_reuse_data(data_type):
            loaded_data = self.data_handler.load_saved_data(data_type, metadata_config)
            if loaded_data is not None:
                print(f"Loaded existing {data_type} data.")
                return loaded_data
        print(f"No existing {data_type} data found or reuse not allowed. Will generate new data.")
        return None

    def create_current_metadata(self) -> dict:
        """
        Capture current metadata for this run, including data, model, embedding, prompt, and clustering settings.
        """
        metadata = {
            "data_settings": self.run_settings.data_settings.to_dict(),
            "model_settings": asdict(self.run_settings.model_settings),
            "embedding_settings": asdict(self.run_settings.embedding_settings),
            "prompt_settings": asdict(self.run_settings.prompt_settings),
            "clustering_settings": asdict(self.run_settings.clustering_settings),
            "run_id": self.run_id
        }
        metadata_str = json.dumps(metadata, sort_keys=True, default=lambda o: o.__dict__)
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        metadata["metadata_hash"] = metadata_hash
        return metadata

    def load_run_metadata(self) -> Dict[str, Any]:
        """
        Load overall run metadata from a YAML file if present.
        """
        if self.run_metadata_file.exists():
            with open(self.run_metadata_file, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def save_run_metadata(self, metadata: Dict[str, Any]):
        """
        Update run metadata YAML with the given metadata dictionary.
        """
        self.run_metadata[self.run_id] = metadata

        csv_files = {}
        for key, value in metadata.items():
            if key.startswith('cluster_table_csv_'):
                csv_files[key] = value

        if csv_files:
            self.run_metadata[self.run_id]['cluster_table_csv_files'] = csv_files

        with open(self.run_metadata_file, 'w') as f:
            yaml.dump(self.run_metadata, f, default_flow_style=False)

    def run_iterative_evaluation(self) -> None:
        """
        Run the iterative evaluation pipeline with proper error handling.
        """
        try:
            logger.info(colored("Starting iterative evaluation...", "cyan"))
            
            # Load initial prompts
            initial_prompts = self.data_prep.load_and_preprocess_data(self.run_settings.data_settings)
            if not initial_prompts:
                raise ValueError("No initial prompts loaded. Check data settings.")
            
            logger.info(colored(f"Loaded {len(initial_prompts)} initial prompts", "green"))

            # Run iterative evaluation
            results = self.iterative_analyzer.run_iterative_evaluation(
                initial_prompts=initial_prompts,
                model_evaluation_manager=self.model_evaluation_manager,
                data_prep=self.data_prep,
                run_settings=self.run_settings
            )

            # Save results
            results_path = (
                self.run_settings.directory_settings.results_dir / 
                f"iterative_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {
                        k: v.tolist() if hasattr(v, 'tolist') else v 
                        for k, v in value.items()
                    }
                else:
                    serializable_results[key] = value.tolist() if hasattr(value, 'tolist') else value

            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(colored(f"Saved iterative analysis results to {results_path}", "green"))

            # Save metadata
            metadata = self.create_current_metadata()
            metadata["iterative_evaluation"] = {
                "completed": True,
                "timestamp": datetime.now().isoformat(),
                "results_file": str(results_path),
                "total_differences": results.get("total_differences", 0),
                "iterations_completed": results.get("iterations_completed", 0)
            }
            self.save_run_metadata(metadata)
            
            logger.info(colored("Iterative evaluation completed successfully", "green"))

        except Exception as e:
            logger.error(colored(f"Error in iterative evaluation: {str(e)}", "red"))
            raise
            
    def run_report_cards(self) -> None:
        """
        Run the Report Cards evaluation pipeline to generate qualitative evaluations of models.
        """
        try:
            logger.info(colored("\nStarting Report Cards evaluation...", "cyan"))
            
            logger.info(colored("Loading and preprocessing data...", "cyan"))
            statements = self.data_prep.load_and_preprocess_data(self.run_settings.data_settings)
            
            if len(statements) == 0:
                logger.error(colored("No statements loaded. Please check your dataset configuration.", "red"))
                return
                
            logger.info(colored(f"Loaded {len(statements)} statements", "green"))
            
            from behavioural_clustering.evaluation.report_cards import ReportCardGenerator
            
            logger.info(colored("Initializing Report Card Generator...", "cyan"))
            report_card_generator = ReportCardGenerator(run_settings=self.run_settings)
            
            logger.info(colored("Setting PRESS parameters...", "cyan"))
            report_card_generator.set_press_parameters(
                progression_set_size=self.run_settings.report_cards_settings.progression_set_size,
                progression_batch_size=self.run_settings.report_cards_settings.progression_batch_size,
                iterations=self.run_settings.report_cards_settings.iterations,
                word_limit=self.run_settings.report_cards_settings.word_limit,
                max_subtopics=self.run_settings.report_cards_settings.max_subtopics,
                merge_threshold=self.run_settings.report_cards_settings.merge_threshold
            )
            
            logger.info(colored("Setting evaluator model...", "cyan"))
            report_card_generator.set_evaluator_model(
                evaluator_model_family=self.run_settings.report_cards_settings.evaluator_model_family,
                evaluator_model_name=self.run_settings.report_cards_settings.evaluator_model_name
            )
            
            logger.info(colored("Generating Report Cards comparison...", "cyan"))
            comparison_results = report_card_generator.compare_models(
                model_evaluation_manager=self.model_evaluation_manager,
                model1_family=self.run_settings.model_settings.models[0][0],
                model1_name=self.run_settings.model_settings.models[0][1],
                model2_family=self.run_settings.model_settings.models[1][0],
                model2_name=self.run_settings.model_settings.models[1][1],
                statements=statements,
                report_progress=True
            )
            
            # Save results
            logger.info(colored("Saving Report Cards results...", "cyan"))
            results_dir = Path(self.run_settings.directory_settings.results_dir) / "report_cards"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            text_path = results_dir / "report.md"
            with open(text_path, "w") as f:
                f.write("# Model Comparison Report Cards\n\n")
                f.write(f"## Model 1: {comparison_results['model1']['model']}\n\n")
                f.write(comparison_results['model1']['report_card'])
                f.write("\n\n")
                f.write(f"## Model 2: {comparison_results['model2']['model']}\n\n")
                f.write(comparison_results['model2']['report_card'])
                f.write("\n\n")
                f.write("## Comparison Summary\n\n")
                f.write(comparison_results['comparison_summary'])
            
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Comparison Report Cards</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .summary {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                    .report-card {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .model1 {{ background-color: #e6f7ff; }}
                    .model2 {{ background-color: #fff7e6; }}
                    .comparison {{ background-color: #e6ffe6; }}
                </style>
            </head>
            <body>
                <h1>Model Comparison Report Cards</h1>
                
                <h2>Model 1: {comparison_results['model1']['model']}</h2>
                <div class="report-card model1">
                    <pre>{comparison_results['model1']['report_card']}</pre>
                </div>
                
                <h2>Model 2: {comparison_results['model2']['model']}</h2>
                <div class="report-card model2">
                    <pre>{comparison_results['model2']['report_card']}</pre>
                </div>
                
                <h2>Comparison Summary</h2>
                <div class="report-card comparison">
                    <pre>{comparison_results['comparison_summary']}</pre>
                </div>
            </body>
            </html>
            """
            
            html_path = results_dir / "report.html"
            with open(html_path, "w") as f:
                f.write(html_content)
            
            json_path = results_dir / "report.json"
            with open(json_path, "w") as f:
                json.dump(comparison_results, f, indent=2)
            
            # Save metadata
            metadata = self.create_current_metadata()
            metadata["report_cards"] = {
                "completed": True,
                "timestamp": datetime.now().isoformat(),
                "results_dir": str(results_dir),
                "models_compared": [
                    f"{self.run_settings.model_settings.models[0][0]}/{self.run_settings.model_settings.models[0][1]}",
                    f"{self.run_settings.model_settings.models[1][0]}/{self.run_settings.model_settings.models[1][1]}"
                ]
            }
            self.save_run_metadata(metadata)
            
            logger.info(colored(f"Report Cards saved to: {results_dir}", "green"))
            logger.info(colored("\nReport Cards evaluation completed successfully", "green"))
            
        except Exception as e:
            logger.error(colored(f"Error in iterative evaluation: {str(e)}", "red"))
            raise
