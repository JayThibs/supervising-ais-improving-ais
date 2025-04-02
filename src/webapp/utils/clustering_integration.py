"""
Integration module for connecting the behavioral clustering utilities with the webapp.
This module provides functions for loading, processing, and visualizing clustering results.
"""

import os
import sys
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple, Union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.behavioural_clustering.utils.dataset_loader import (
    create_default_registry, DatasetLoader
)
from src.behavioural_clustering.utils.clustering_algorithms import (
    ClusteringFactory, evaluate_clustering, find_optimal_clusters
)
from src.behavioural_clustering.utils.hardware_detection import (
    get_hardware_info, configure_models_for_hardware
)
from src.behavioural_clustering.utils.plotly_visualizations import (
    plot_embedding_responses_plotly, plot_approvals_plotly,
    plot_cluster_sizes_plotly, plot_model_comparison_plotly
)
from src.behavioural_clustering.utils.embedding_data import JointEmbeddings
from src.behavioural_clustering.evaluation.evaluator_pipeline import EvaluatorPipeline
from src.behavioural_clustering.evaluation.clustering import ClusterAnalyzer
from src.behavioural_clustering.config.run_settings import RunSettings
from src.behavioural_clustering.config.run_configuration_manager import RunConfigurationManager

logger = logging.getLogger(__name__)


class ClusteringIntegration:
    """
    Integration class for connecting behavioral clustering with the webapp.
    """
    
    def __init__(self, data_dir: Optional[Union[str, Path]] = None, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the clustering integration.
        
        Args:
            data_dir: Directory for datasets and results
            config_path: Path to configuration file
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.config_path = Path(config_path) if config_path else None
        
        self.dataset_registry = create_default_registry(self.data_dir)
        self.dataset_loader = DatasetLoader(self.dataset_registry)
        
        self.hardware_info = get_hardware_info()
        self.model_configs = configure_models_for_hardware(self.config_path)
        
        self.run_config_manager = RunConfigurationManager()
            
    def list_available_datasets(self) -> List[str]:
        """
        List all available datasets.
        
        Returns:
            List of dataset names
        """
        return self.dataset_registry.list_datasets()
        
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models.
        
        Returns:
            List of model configurations
        """
        return self.model_configs['available_models']
        
    def get_optimal_model(self) -> Dict[str, Any]:
        """
        Get the optimal model configuration for the current hardware.
        
        Returns:
            Optimal model configuration
        """
        return self.model_configs['optimal_model']
        
    def get_parallel_configs(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get parallel configurations for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of model configurations for parallel execution
        """
        return self.model_configs['parallel_configs'].get(model_name, [])
        
    def load_dataset(self, dataset_name: str, 
                    max_length: Optional[int] = None,
                    min_length: Optional[int] = None,
                    categories: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Load and filter a dataset.
        
        Args:
            dataset_name: Name of the dataset
            max_length: Maximum statement length
            min_length: Minimum statement length
            categories: List of categories to include
            
        Returns:
            List of statements with metadata
        """
        statements, _, _ = self.dataset_loader.load_dataset(dataset_name)
        
        if max_length or min_length or categories:
            statements = self.dataset_loader.filter_statements(
                statements,
                max_length=max_length,
                min_length=min_length,
                categories=categories
            )
            
        return statements
        
    def run_clustering(self, embeddings: np.ndarray, 
                      algorithm: str = 'kmeans',
                      n_clusters: Optional[int] = None,
                      find_optimal: bool = True,
                      min_clusters: int = 2,
                      max_clusters: int = 20,
                      **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Run clustering on embeddings.
        
        Args:
            embeddings: Embedding matrix
            algorithm: Clustering algorithm to use
            n_clusters: Number of clusters (if not finding optimal)
            find_optimal: Whether to find the optimal number of clusters
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
            **kwargs: Additional parameters for the clustering algorithm
            
        Returns:
            Tuple of (cluster labels, clustering results)
        """
        if find_optimal:
            optimal_n_clusters, labels, metrics = find_optimal_clusters(
                embeddings,
                algorithm=algorithm,
                min_clusters=min_clusters,
                max_clusters=max_clusters,
                **kwargs
            )
            
            results = {
                'algorithm': algorithm,
                'n_clusters': optimal_n_clusters,
                'metrics': metrics
            }
            
            return labels, results
        else:
            if n_clusters is None:
                n_clusters = min(10, len(embeddings) // 5)  # Default to 10 or fewer clusters
                
            clustering_algo = ClusteringFactory.create_algorithm(
                algorithm,
                n_clusters=n_clusters,
                **kwargs
            )
            
            labels = clustering_algo.fit(embeddings)
            
            metrics = evaluate_clustering(embeddings, labels)
            
            results = {
                'algorithm': algorithm,
                'n_clusters': n_clusters,
                'metrics': metrics
            }
            
            return labels, results
            
    def create_embedding_visualization(self, embeddings_2d: np.ndarray,
                                     labels: np.ndarray,
                                     responses: List[str],
                                     model_names: List[str],
                                     title: str = "Model Response Embeddings") -> go.Figure:
        """
        Create an interactive visualization of embeddings.
        
        Args:
            embeddings_2d: 2D embeddings
            labels: Cluster labels
            responses: Model responses
            model_names: Model names
            title: Plot title
            
        Returns:
            Plotly figure
        """
        return plot_embedding_responses_plotly(
            embeddings_2d,
            labels,
            responses,
            model_names,
            title=title
        )
        
    def create_approval_visualization(self, approval_matrix: np.ndarray,
                                    model_names: List[str],
                                    cluster_labels: Optional[np.ndarray] = None,
                                    title: str = "Model Approval Patterns") -> go.Figure:
        """
        Create an interactive visualization of approval patterns.
        
        Args:
            approval_matrix: Approval matrix
            model_names: Model names
            cluster_labels: Cluster labels
            title: Plot title
            
        Returns:
            Plotly figure
        """
        return plot_approvals_plotly(
            approval_matrix,
            model_names,
            cluster_labels,
            title=title
        )
        
    def load_run_results(self, run_id: str) -> Dict[str, Any]:
        """
        Load results from a previous run.
        
        Args:
            run_id: Run ID
            
        Returns:
            Dictionary of run results
        """
        return self.run_config_manager.get_run_metadata(run_id)
        
    def list_runs(self) -> List[Dict[str, Any]]:
        """
        List all available runs.
        
        Returns:
            List of run information
        """
        run_metadata = self.run_config_manager.load_run_metadata()
        return [{"run_id": run_id, "metadata": metadata} for run_id, metadata in run_metadata.items()]
        
    def get_hardware_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the hardware.
        
        Returns:
            Dictionary of hardware information
        """
        return {
            'system': self.hardware_info.system_info,
            'cpu': self.hardware_info.cpu_info,
            'memory': self.hardware_info.memory_info,
            'gpu': self.hardware_info.gpu_info,
            'optimal_device': self.hardware_info.get_optimal_device()
        }
        
    def create_joint_embeddings_from_run(self, run_id: str) -> Optional[JointEmbeddings]:
        """
        Create a JointEmbeddings object from a run.
        
        Args:
            run_id: Run ID
            
        Returns:
            JointEmbeddings object or None if not available
        """
        run_results = self.load_run_results(run_id)
        
        if not run_results or 'embeddings' not in run_results:
            logger.warning(f"No embeddings found in run {run_id}")
            return None
            
        embeddings_data = run_results['embeddings']
        
        if 'model_order' not in embeddings_data:
            logger.warning(f"No model order found in embeddings for run {run_id}")
            return None
            
        joint_embeddings = JointEmbeddings(embeddings_data['model_order'])
        
        for entry in embeddings_data.get('entries', []):
            joint_embeddings.add_embedding(
                entry['model_idx'],
                entry['statement'],
                entry['response'],
                np.array(entry['embedding'])
            )
            
        return joint_embeddings
        
    def create_evaluator_pipeline_from_run(self, run_id: str) -> Optional[EvaluatorPipeline]:
        """
        Create an EvaluatorPipeline object from a run.
        
        Args:
            run_id: Run ID
            
        Returns:
            EvaluatorPipeline object or None if not available
        """
        run_results = self.load_run_results(run_id)
        
        if not run_results:
            logger.warning(f"No results found for run {run_id}")
            return None
            
        if 'settings' in run_results:
            run_settings = RunSettings.from_dict(run_results['settings'])
        else:
            logger.warning(f"No settings found in run {run_id}, using default settings")
            run_settings = RunSettings()
            
        pipeline = EvaluatorPipeline(run_settings)
        
        joint_embeddings = self.create_joint_embeddings_from_run(run_id)
        if joint_embeddings:
            pipeline.joint_embeddings_all_llms = joint_embeddings
        
        if 'clustering' in run_results:
            clustering_results = run_results['clustering']
            
            if 'labels' in clustering_results:
                pipeline.spectral_labels = np.array(clustering_results['labels'])
                
        return pipeline
        
    def create_cluster_analyzer_from_run(self, run_id: str) -> Optional[ClusterAnalyzer]:
        """
        Create a ClusterAnalyzer object from a run.
        
        Args:
            run_id: Run ID
            
        Returns:
            ClusterAnalyzer object or None if not available
        """
        pipeline = self.create_evaluator_pipeline_from_run(run_id)
        
        if not pipeline:
            return None
            
        analyzer = ClusterAnalyzer(pipeline.run_settings)
        
        return analyzer
