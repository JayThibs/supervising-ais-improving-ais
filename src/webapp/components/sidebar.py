import streamlit as st
from typing import List, Tuple

def sidebar():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Run Analysis", "View Results", "Compare Models", "View Tables", "View Treemap"])
    return page

def get_selected_models() -> List[Tuple[str, str]]:
    return st.session_state.get('selected_models', [])

HIDEABLE_PLOT_TYPES = ["tsne", "approvals", "hierarchical", "spectral"]