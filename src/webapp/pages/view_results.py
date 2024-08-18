import streamlit as st
import os
from behavioural_clustering.utils.visualization import Visualization
from behavioural_clustering.config.run_settings import PlotSettings
from webapp.utils.data_processing import load_previous_runs

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

        # Create PlotSettings instance
        plot_settings = PlotSettings(
            plot_dim=run_data['config']['plot_settings']['plot_dim'],
            save_path=os.path.join(os.getcwd(), 'data', 'results', 'plots'),
            colors=run_data['config']['plot_settings']['colors'],
            shapes=run_data['config']['plot_settings']['shapes'],
            plot_aesthetics=run_data['config']['plot_settings']['plot_aesthetics']
        )

        # Create Visualization instance
        visualizer = Visualization(plot_settings)

        # Visualize results
        if run_data['config']['plot_settings']['show_tsne']:
            visualizer.plot_embedding_responses(
                run_data['results']['tsne'],
                run_data['results']['joint_embeddings'],
                run_data['results']['model_names'],
                'tsne_plot.png'
            )
            st.image('data/results/plots/tsne_plot.png')

        if run_data['config']['plot_settings']['show_approvals']:
            for plot_type in ['approval', 'awareness']:
                visualizer.plot_approvals(
                    run_data['results']['tsne'],
                    run_data['results']['approval_data'],
                    run_data['results']['model_names'][0],  # Assuming single model for simplicity
                    1,  # Condition (you may need to adjust this)
                    plot_type,
                    f'{plot_type}_plot.png',
                    f'{plot_type.capitalize()} Plot'
                )
                st.image(f'data/results/plots/{plot_type}_plot.png')

        if run_data['config']['plot_settings']['show_hierarchical']:
            visualizer.visualize_hierarchical_plot(
                run_data['results']['hierarchy_data'],
                'approval',  # or 'awareness', depending on your needs
                'hierarchical_cluster',
                labels=run_data['results']['model_names']
            )
            st.image('data/results/plots/hierarchical_cluster.png')

        if 'spectral_clustering' in run_data['results']:
            visualizer.plot_spectral_clustering(
                run_data['results']['spectral_clustering']['labels'],
                run_data['results']['spectral_clustering']['n_clusters'],
                'approval'  # or 'awareness', depending on your needs
            )
            st.image('data/results/plots/spectral_clustering_approval_statements.png')

        # Add option to export results
        if st.button("Export Results"):
            # Implement export functionality
            st.success("Results exported successfully!")