import streamlit as st
from typing import List, Tuple

def sidebar():
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to", ["Run Analysis", "View Results", "Compare Models"])
        
        if page == "Run Analysis":
            st.subheader("Run Options")
            
            # Use the 'key' parameter to update session state automatically
            st.selectbox(
                "Select run type",
                ["Full Evaluation", "Model Comparison", "Approval Prompts"],
                key="run_type"
            )
            
            # Add configuration selection
            config_manager = st.session_state.config_manager
            config_names = config_manager.list_configurations()
            st.selectbox("Select configuration", config_names, key="selected_config")
            
            # Load the selected configuration
            run_settings = config_manager.get_configuration(st.session_state.selected_config)
            
            # Add customization options
            st.subheader("Customize Run")
            
            # Data settings
            st.number_input("Number of statements", min_value=10, max_value=10000, value=run_settings.data_settings.n_statements, key="n_statements")
            
            # Model settings
            available_models = [("openai", "gpt-3.5-turbo"), ("openai", "gpt-4"), ("anthropic", "claude-v1")]
            default_models = [m for m in run_settings.model_settings.models if m in available_models]
            st.multiselect("Select models", available_models, default=default_models, key="selected_models")
            
            # Prompt settings
            st.number_input("Max description length", min_value=50, max_value=1000, value=run_settings.prompt_settings.max_desc_length, key="max_desc_length")
            
            # Clustering settings
            st.number_input("Number of clusters", min_value=2, max_value=1000, value=run_settings.clustering_settings.n_clusters, key="n_clusters")
            
            # Plot settings
            hide_plots_options = ["none"] + HIDEABLE_PLOT_TYPES + ["all"]
            hide_plots_default = (
                ["all"] if "all" in run_settings.plot_settings.hide_plots
                else ["none"] if not run_settings.plot_settings.hide_plots
                else run_settings.plot_settings.hide_plots
            )
            st.multiselect("Hide plots", hide_plots_options, default=hide_plots_default, key="hide_plots")
        
    return page

def get_selected_models() -> List[Tuple[str, str]]:
    return st.session_state.get('selected_models', [])

HIDEABLE_PLOT_TYPES = ["tsne", "approvals", "hierarchical", "spectral"]