import streamlit as st
from behavioural_clustering.evaluation.evaluator_pipeline import EvaluatorPipeline
from webapp.components.sidebar import get_selected_models
from pathlib import Path
import yaml

def get_run_metadata():
    base_dir = Path(__file__).resolve().parents[3] / "data"
    metadata_path = base_dir / "metadata" / "run_metadata.yaml"
    
    with open(metadata_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_session_state(run_settings):
    if 'n_statements' not in st.session_state:
        st.session_state.n_statements = run_settings.data_settings.n_statements
    if 'max_desc_length' not in st.session_state:
        st.session_state.max_desc_length = run_settings.prompt_settings.max_desc_length
    if 'n_clusters' not in st.session_state:
        st.session_state.n_clusters = run_settings.clustering_settings.n_clusters
    if 'hide_plots' not in st.session_state:
        st.session_state.hide_plots = run_settings.plot_settings.hide_plots

def show():
    st.header("Run Analysis")
    
    config_manager = st.session_state.config_manager
    run_settings = config_manager.get_configuration(st.session_state.get('selected_config', 'Default Run'))
    
    initialize_session_state(run_settings)
    
    run_type = st.session_state.get("run_type", "Full Evaluation")
    
    # Add input fields for customizable settings
    st.session_state.n_statements = st.number_input("Number of Statements", value=st.session_state.n_statements, min_value=1)
    st.session_state.max_desc_length = st.number_input("Max Description Length", value=st.session_state.max_desc_length, min_value=1)
    st.session_state.n_clusters = st.number_input("Number of Clusters", value=st.session_state.n_clusters, min_value=1)
    
    # Handle the 'hide_plots' multiselect
    plot_options = ["tsne", "approvals", "hierarchical", "spectral"]
    default_hide_plots = [opt for opt in st.session_state.hide_plots if opt in plot_options]
    st.session_state.hide_plots = st.multiselect(
        "Hide Plots",
        options=plot_options,
        default=default_hide_plots
    )
    
    if st.button("Start Analysis"):
        # Update run settings with customized values
        run_settings.data_settings.n_statements = st.session_state.n_statements
        run_settings.model_settings.models = get_selected_models()
        run_settings.prompt_settings.max_desc_length = st.session_state.max_desc_length
        run_settings.clustering_settings.n_clusters = st.session_state.n_clusters
        run_settings.plot_settings.hide_plots = st.session_state.hide_plots
        
        evaluator = EvaluatorPipeline(run_settings)
        
        with st.spinner("Running analysis..."):
            if run_type == "Full Evaluation":
                evaluator.run_evaluations()
            elif run_type == "Model Comparison":
                metadata_config = evaluator.create_current_metadata()
                evaluator.run_model_comparison(metadata_config)
            elif run_type == "Approval Prompts":
                metadata_config = evaluator.create_current_metadata()
                for prompt_type in evaluator.approval_prompts.keys():
                    evaluator.run_prompt_evaluation(prompt_type, metadata_config)
        
        st.success("Analysis completed successfully!")
    
    # Display current settings
    st.subheader("Current Run Settings")
    st.write(f"Run Type: {run_type}")
    st.write(f"Number of Statements: {st.session_state.n_statements}")
    st.write(f"Selected Models: {', '.join([f'{m[0]}/{m[1]}' for m in get_selected_models()])}")
    st.write(f"Max Description Length: {st.session_state.max_desc_length}")
    st.write(f"Number of Clusters: {st.session_state.n_clusters}")
    st.write(f"Hidden Plots: {', '.join(st.session_state.hide_plots)}")
