"""
Benchmark Analysis Page for the Behavioral Clustering Webapp

This page provides tools for analyzing benchmark datasets and comparing
model behaviors across different benchmarks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import json
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.webapp.utils.clustering_integration import ClusteringIntegration
from src.behavioural_clustering.utils.benchmark_integration import (
    BenchmarkConverter, download_and_prepare_benchmarks, create_benchmark_config
)
from src.behavioural_clustering.utils.plotly_visualizations import (
    plot_embedding_responses_plotly, plot_approvals_plotly,
    plot_cluster_sizes_plotly, plot_model_comparison_plotly
)

logger = logging.getLogger(__name__)


def app():
    """
    Main function for the benchmark analysis page.
    """
    st.title("Benchmark Analysis")
    
    st.markdown("""
    This page allows you to analyze benchmark datasets and compare model behaviors
    across different benchmarks. You can download benchmark datasets, run analyses,
    and visualize the results.
    """)
    
    integration = ClusteringIntegration()
    
    tab1, tab2, tab3 = st.tabs(["Benchmark Management", "Run Analysis", "Results Visualization"])
    
    with tab1:
        benchmark_management_section(integration)
        
    with tab2:
        benchmark_analysis_section(integration)
        
    with tab3:
        benchmark_visualization_section(integration)


def benchmark_management_section(integration: ClusteringIntegration):
    """
    Section for managing benchmark datasets.
    
    Args:
        integration: ClusteringIntegration instance
    """
    st.header("Benchmark Management")
    
    st.markdown("""
    This section allows you to download and prepare benchmark datasets for analysis.
    You can download standard benchmarks like Anthropic Model-Written Evaluations and
    TruthfulQA, or upload your own custom benchmark datasets.
    """)
    
    st.subheader("Available Datasets")
    
    available_datasets = integration.list_available_datasets()
    
    if available_datasets:
        st.write("The following datasets are available:")
        
        for dataset in available_datasets:
            st.write(f"- {dataset}")
    else:
        st.write("No datasets are currently available.")
    
    st.subheader("Download Benchmark Datasets")
    
    col1, col2 = st.columns(2)
    
    with col1:
        download_anthropic = st.button("Download Anthropic Benchmark")
        download_truthfulqa = st.button("Download TruthfulQA Benchmark")
        
    with col2:
        create_contrastive = st.button("Create Contrastive Hypotheses")
        create_config = st.button("Create Benchmark Config")
    
    if download_anthropic or download_truthfulqa:
        with st.spinner("Downloading benchmark datasets..."):
            try:
                output_dir = Path("data/benchmarks")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                converter = BenchmarkConverter(output_dir)
                
                if download_anthropic:
                    anthropic_path = converter.download_anthropic_benchmark()
                    converter.convert_anthropic_to_statements(anthropic_path)
                    st.success(f"Downloaded and converted Anthropic benchmark to {output_dir}")
                    
                if download_truthfulqa:
                    truthfulqa_path = converter.download_truthfulqa()
                    converter.convert_truthfulqa_to_statements(truthfulqa_path)
                    st.success(f"Downloaded and converted TruthfulQA benchmark to {output_dir}")
                    
                registry = integration.dataset_loader.registry
                converter.register_benchmark_datasets(registry)
                
                available_datasets = integration.list_available_datasets()
                
            except Exception as e:
                st.error(f"Error downloading benchmark datasets: {e}")
    
    if create_contrastive:
        with st.spinner("Creating contrastive hypotheses..."):
            try:
                output_dir = Path("data/benchmarks")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                converter = BenchmarkConverter(output_dir)
                
                anthropic_path = output_dir / "anthropic_statements.jsonl"
                truthfulqa_path = output_dir / "truthfulqa_statements.jsonl"
                
                if anthropic_path.exists():
                    converter.create_contrastive_hypotheses(anthropic_path)
                    st.success(f"Created contrastive hypotheses for Anthropic benchmark")
                    
                if truthfulqa_path.exists():
                    converter.create_contrastive_hypotheses(truthfulqa_path)
                    st.success(f"Created contrastive hypotheses for TruthfulQA benchmark")
                    
                registry = integration.dataset_loader.registry
                converter.register_benchmark_datasets(registry)
                
                available_datasets = integration.list_available_datasets()
                
            except Exception as e:
                st.error(f"Error creating contrastive hypotheses: {e}")
    
    if create_config:
        with st.spinner("Creating benchmark configuration..."):
            try:
                config_path = create_benchmark_config()
                st.success(f"Created benchmark configuration at {config_path}")
                
            except Exception as e:
                st.error(f"Error creating benchmark configuration: {e}")
    
    st.subheader("Upload Custom Benchmark Dataset")
    
    uploaded_file = st.file_uploader("Upload a custom benchmark dataset (JSONL format)", type=["jsonl", "json"])
    
    if uploaded_file is not None:
        try:
            output_dir = Path("data/benchmarks")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = output_dir / uploaded_file.name
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            st.success(f"Uploaded custom benchmark dataset to {file_path}")
            
            dataset_name = st.text_input("Dataset Name", value=f"custom_{uploaded_file.name.split('.')[0]}")
            dataset_description = st.text_input("Dataset Description", value="Custom benchmark dataset")
            
            if st.button("Register Dataset"):
                registry = integration.dataset_loader.registry
                
                registry.register_dataset(
                    dataset_name,
                    str(file_path),
                    description=dataset_description,
                    categories=["custom"]
                )
                
                st.success(f"Registered custom benchmark dataset as {dataset_name}")
                
                available_datasets = integration.list_available_datasets()
                
        except Exception as e:
            st.error(f"Error uploading custom benchmark dataset: {e}")


def benchmark_analysis_section(integration: ClusteringIntegration):
    """
    Section for running benchmark analyses.
    
    Args:
        integration: ClusteringIntegration instance
    """
    st.header("Run Benchmark Analysis")
    
    st.markdown("""
    This section allows you to run analyses on benchmark datasets. You can select
    which models to use, which datasets to analyze, and configure the analysis
    parameters.
    """)
    
    st.subheader("Select Benchmark Datasets")
    
    available_datasets = integration.list_available_datasets()
    
    if not available_datasets:
        st.warning("No datasets are available. Please download or upload benchmark datasets first.")
        return
        
    selected_datasets = st.multiselect(
        "Select datasets to analyze",
        options=available_datasets,
        default=available_datasets[:1] if available_datasets else []
    )
    
    if not selected_datasets:
        st.warning("Please select at least one dataset to analyze.")
        return
        
    st.subheader("Select Models")
    
    available_models = integration.list_available_models()
    
    if not available_models:
        st.warning("No models are available. Please check your configuration.")
        return
        
    model_options = [f"{model['model_family']}/{model['model_name']}" for model in available_models]
    
    selected_models = st.multiselect(
        "Select models to analyze",
        options=model_options,
        default=model_options[:2] if len(model_options) >= 2 else model_options
    )
    
    if not selected_models:
        st.warning("Please select at least one model to analyze.")
        return
        
    st.subheader("Configure Analysis Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_statements = st.number_input(
            "Number of statements to analyze",
            min_value=10,
            max_value=1000,
            value=100,
            step=10
        )
        
        n_clusters = st.number_input(
            "Number of clusters",
            min_value=2,
            max_value=50,
            value=10,
            step=1
        )
        
    with col2:
        clustering_algorithm = st.selectbox(
            "Clustering algorithm",
            options=["kmeans", "spectral", "agglomerative", "dbscan"],
            index=0
        )
        
        find_optimal = st.checkbox("Find optimal number of clusters", value=True)
    
    if st.button("Run Analysis"):
        with st.spinner("Running benchmark analysis..."):
            try:
                models = []
                
                for model_str in selected_models:
                    model_family, model_name = model_str.split("/")
                    
                    models.append({
                        "model_family": model_family,
                        "model_name": model_name
                    })
                
                all_statements = []
                
                for dataset_name in selected_datasets:
                    statements = integration.load_dataset(dataset_name)
                    
                    if n_statements < len(statements):
                        statements = statements[:n_statements]
                        
                    all_statements.extend(statements)
                
                st.info("Analysis functionality is not fully implemented yet. This would run the behavioral clustering pipeline on the selected datasets and models.")
                
                st.subheader("Sample Statements")
                
                sample_statements = all_statements[:5]
                
                for i, statement in enumerate(sample_statements):
                    st.write(f"**Statement {i+1}:** {statement.get('statement', '')}")
                
                st.session_state.benchmark_results = {
                    "datasets": selected_datasets,
                    "models": selected_models,
                    "n_statements": n_statements,
                    "n_clusters": n_clusters,
                    "clustering_algorithm": clustering_algorithm,
                    "find_optimal": find_optimal,
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                st.success("Benchmark analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Error running benchmark analysis: {e}")


def benchmark_visualization_section(integration: ClusteringIntegration):
    """
    Section for visualizing benchmark analysis results.
    
    Args:
        integration: ClusteringIntegration instance
    """
    st.header("Results Visualization")
    
    st.markdown("""
    This section allows you to visualize the results of benchmark analyses. You can
    view embeddings, cluster distributions, and model comparisons.
    """)
    
    if not hasattr(st.session_state, "benchmark_results"):
        st.warning("No benchmark analysis results are available. Please run an analysis first.")
        return
        
    st.subheader("Analysis Summary")
    
    results = st.session_state.benchmark_results
    
    st.write(f"**Datasets:** {', '.join(results['datasets'])}")
    st.write(f"**Models:** {', '.join(results['models'])}")
    st.write(f"**Number of statements:** {results['n_statements']}")
    st.write(f"**Number of clusters:** {results['n_clusters']}")
    st.write(f"**Clustering algorithm:** {results['clustering_algorithm']}")
    st.write(f"**Find optimal clusters:** {results['find_optimal']}")
    st.write(f"**Timestamp:** {results['timestamp']}")
    
    st.subheader("Visualizations")
    
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["Embeddings", "Clusters", "Model Comparison"])
    
    with viz_tab1:
        st.markdown("### Embedding Visualization")
        
        n_points = 100
        n_clusters = results['n_clusters']
        
        embeddings_2d = np.random.randn(n_points, 2)
        labels = np.random.randint(0, n_clusters, n_points)
        
        responses = [f"Response {i}" for i in range(n_points)]
        
        model_names = [results['models'][i % len(results['models'])] for i in range(n_points)]
        
        fig = plot_embedding_responses_plotly(
            embeddings_2d,
            labels,
            responses,
            model_names,
            title="Model Response Embeddings (Placeholder)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with viz_tab2:
        st.markdown("### Cluster Visualization")
        
        fig = plot_cluster_sizes_plotly(
            labels,
            title="Cluster Sizes (Placeholder)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        n_models = len(results['models'])
        
        approval_matrix = np.random.rand(n_models, n_clusters)
        
        fig = plot_approvals_plotly(
            approval_matrix,
            [model.split("/")[1] for model in results['models']],
            labels,
            title="Model Approval Patterns (Placeholder)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with viz_tab3:
        st.markdown("### Model Comparison")
        
        model_indices = np.array([i % n_models for i in range(n_points)])
        
        fig = plot_model_comparison_plotly(
            embeddings_2d,
            model_indices,
            [model.split("/")[1] for model in results['models']],
            responses,
            title="Model Response Comparison (Placeholder)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
