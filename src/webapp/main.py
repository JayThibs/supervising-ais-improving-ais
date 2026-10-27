import streamlit as st
import sys
from pathlib import Path

st.set_page_config(page_title="Behavioural Clustering Analysis", layout="wide")

src_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(src_dir))

from webapp.pages import run_analysis, view_results, compare_models, view_tables, view_treemap
from webapp.components.sidebar import sidebar
from behavioural_clustering.config.run_configuration_manager import RunConfigurationManager
from behavioural_clustering.utils.data_accessor import DataAccessor
from behavioural_clustering.evaluation.evaluator_pipeline import EvaluatorPipeline

@st.cache_resource(ttl=7200)
def get_data_accessor():
    base_dir = src_dir / "data"
    return DataAccessor(base_dir)

@st.cache_resource(ttl=7200)
def get_config_manager():
    config_manager = RunConfigurationManager()
    config_manager.load_configurations()
    return config_manager

@st.cache_resource
def get_evaluator_pipeline(run_settings):
    return EvaluatorPipeline(run_settings)

def main():
    st.title("Behavioural Clustering Analysis")

    # Initialize configuration manager and data accessor
    st.session_state.config_manager = get_config_manager()
    st.session_state.data_accessor = get_data_accessor()
    
    # Force reload of metadata
    st.session_state.data_accessor.load_metadata()

    # Get the default run settings
    default_run_settings = st.session_state.config_manager.get_configuration('Default Run')
    
    if default_run_settings is None:
        st.error("No 'Default Run' configuration found. Please check your configuration file.")
        return

    # Initialize the evaluator pipeline with default settings
    try:
        st.session_state.evaluator_pipeline = get_evaluator_pipeline(default_run_settings)
    except TypeError as e:
        st.error(f"Error initializing EvaluatorPipeline: {str(e)}")
        st.error("Please check that EvaluatorPipeline.__init__() is correctly implemented.")
        return

    # Initialize session state variables
    if 'selected_config' not in st.session_state:
        st.session_state.selected_config = 'Default Run'

    # Sidebar navigation
    page = sidebar()

    # Main content
    if page == "Run Analysis":
        run_analysis.show()
    elif page == "View Results":
        view_results.show()
    elif page == "Compare Models":
        compare_models.show()
    elif page == "View Tables":
        view_tables.show()
    elif page == "View Treemap":
        view_treemap.show()

if __name__ == "__main__":
    main()