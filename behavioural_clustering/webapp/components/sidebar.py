import streamlit as st
from webapp.components.model_selector import select_models
from src.webapp.components.config_manager import save_custom_configuration

def sidebar():
    with st.sidebar:
        st.header("Configuration")
        
        config_manager = st.session_state.config_manager
        
        # Load predefined configuration or create a new one
        config_option = st.selectbox("Choose configuration", 
                                     ["Custom"] + list(config_manager.configurations.keys()))
        
        if config_option == "Custom":
            run_settings = config_manager.create_custom_configuration()
        else:
            run_settings = config_manager.get_configuration(config_option)
        
        # Model selection
        selected_models = select_models(run_settings.model_settings)
        
        # Clustering options
        st.subheader("Clustering Options")
        clustering_algorithm = st.selectbox("Clustering Algorithm", 
                                            run_settings.clustering_settings.all_clustering_algorithms,
                                            index=run_settings.clustering_settings.all_clustering_algorithms.index(run_settings.clustering_settings.main_clustering_algorithm))
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=1000, 
                               value=run_settings.clustering_settings.n_clusters)
        
        # Visualization options
        st.subheader("Visualization Options")
        show_tsne = st.checkbox("Show t-SNE plot", value=not run_settings.plot_settings.hide_tsne)
        show_umap = st.checkbox("Show UMAP plot", value=True)  # Add this option to RunSettings if needed
        show_hierarchical = st.checkbox("Show hierarchical clustering", value=not run_settings.plot_settings.hide_hierarchical)
        show_approval = st.checkbox("Show approval plots", value=not run_settings.plot_settings.hide_approval)
        
        # Advanced options
        with st.expander("Advanced Options"):
            perplexity = st.slider("t-SNE Perplexity", min_value=5, max_value=100, value=run_settings.tsne_settings.perplexity)
            learning_rate = st.slider("t-SNE Learning Rate", min_value=10, max_value=1000, value=int(run_settings.tsne_settings.learning_rate))
            n_neighbors = st.slider("UMAP n_neighbors", min_value=2, max_value=100, value=15)
            min_dist = st.slider("UMAP min_dist", min_value=0.0, max_value=1.0, value=0.1)
        
        # Save configuration
        if st.button("Save Configuration"):
            save_custom_configuration(config_manager, run_settings, clustering_algorithm, n_clusters, show_tsne, show_umap, 
                                      show_hierarchical, show_approval, perplexity, learning_rate, n_neighbors, min_dist)
            st.success("Configuration saved!")

        # Page selection
        st.header("Navigation")
        page = st.radio("Go to", ["Run Analysis", "View Results", "Compare Models"])

    return page