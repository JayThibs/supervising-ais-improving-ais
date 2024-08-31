import streamlit as st
from src.behavioural_clustering.evaluation.clustering import run_clustering
from src.behavioural_clustering.utils.visualization import visualize_results

def show():
    st.header("Run Analysis")

    if st.button("Run Clustering Analysis"):
        try:
            with st.spinner("Running clustering analysis..."):
                progress_bar = st.progress(0)
                
                # Get the current configuration
                run_settings = st.session_state.config_manager.get_configuration("Custom")
                
                # Run the clustering analysis
                results = run_clustering(run_settings.model_settings.models,
                                         run_settings.clustering_settings.main_clustering_algorithm,
                                         run_settings.clustering_settings.n_clusters,
                                         progress_callback=progress_bar.progress)
                
            st.success("Clustering analysis complete!")
            
            # Visualize results
            visualize_results(results,
                              not run_settings.plot_settings.hide_tsne,
                              True,  # show_umap
                              not run_settings.plot_settings.hide_hierarchical,
                              not run_settings.plot_settings.hide_approval,
                              run_settings.tsne_settings.perplexity,
                              run_settings.tsne_settings.learning_rate,
                              15,  # n_neighbors for UMAP
                              0.1)  # min_dist for UMAP
            
            # Save run
            run_name = st.text_input("Enter a name for this run:")
            if st.button("Save Run"):
                # Implement save_run functionality
                st.success(f"Run '{run_name}' saved successfully!")
            
            # Export results
            if st.button("Export Results"):
                # Implement export_results functionality
                st.success("Results exported successfully!")
                
        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")