import streamlit as st
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parents[1]
sys.path.append(str(project_root))

from webapp.pages import run_analysis, view_results, compare_models
from src.webapp.components.sidebar import sidebar
from src.webapp.auth.user_auth import login, create_account, is_authenticated
from src.behavioural_clustering.config.run_configuration_manager import RunConfigurationManager

st.set_page_config(page_title="Behavioral Clustering Analysis", layout="wide")

def main():
    st.title("Behavioral Clustering Analysis")

    # User Authentication
    if 'username' not in st.session_state:
        st.session_state.username = None

    if not is_authenticated(st.session_state.username):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                st.session_state.username = login()
        with col2:
            if st.button("Create Account"):
                st.session_state.username = create_account()
        if not is_authenticated(st.session_state.username):
            st.stop()

    # Initialize configuration manager
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = RunConfigurationManager()
        st.session_state.config_manager.load_configurations()

    # Sidebar
    page = sidebar()

    # Main content
    if page == "Run Analysis":
        run_analysis.show()
    elif page == "View Results":
        view_results.show()
    elif page == "Compare Models":
        compare_models.show()

if __name__ == "__main__":
    main()