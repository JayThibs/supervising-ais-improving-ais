import streamlit as st
from src.behavioural_clustering.utils.visualization import visualize_results
from src.webapp.utils.data_processing import load_previous_runs

def show():
    st.header("View Results")

    previous_runs = load_previous_runs(st.session_state.username)
    
    if not previous_runs:
        st.warning("No previous runs found.")
        return

    selected_run = st.selectbox("Select a run to view", list(previous_runs.keys()))

    if selected_run:
        run_data = previous_runs[selected_run]
        st.subheader(f"Results for run: {selected_run}")

        # Display run configuration
        st.write("Run Configuration:")
        st.json(run_data['config'])

        # Visualize results
        visualize_results(
            run_data['results'],
            show_tsne=run_data['config']['plot_settings']['show_tsne'],
            show_umap=run_data['config']['plot_settings']['show_umap'],
            show_hierarchical=run_data['config']['plot_settings']['show_hierarchical'],
            show_approval=run_data['config']['plot_settings']['show_approval'],
            perplexity=run_data['config']['tsne_settings']['perplexity'],
            learning_rate=run_data['config']['tsne_settings']['learning_rate'],
            n_neighbors=run_data['config']['umap_settings']['n_neighbors'],
            min_dist=run_data['config']['umap_settings']['min_dist']
        )

        # Add option to export results
        if st.button("Export Results"):
            # Implement export functionality
            st.success("Results exported successfully!")