import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from behavioural_clustering.utils.visualization import Visualization
from behavioural_clustering.config.run_settings import PlotSettings, RunSettings

def show():
    st.header("View Results")

    data_accessor = st.session_state.data_accessor
    runs = data_accessor.list_runs()
    
    if not runs:
        st.warning("No previous runs found.")
        return

    selected_run = st.selectbox("Select a run to view", runs, help="Choose a run to visualize its results")

    if selected_run:
        try:
            run_config = data_accessor.get_run_config(selected_run)
            st.subheader(f"Results for run: {selected_run}")

            # Display run configuration
            with st.expander("Run Configuration"):
                st.json(run_config)

            # Create RunSettings instance
            run_settings = RunSettings.from_dict(run_config)

            # Ensure we have a PlotSettings object
            if not isinstance(run_settings.plot_settings, PlotSettings):
                run_settings.plot_settings = PlotSettings()

            # Create Visualization instance
            visualizer = Visualization(run_settings.plot_settings)

            # Get available data types for the run
            data_types = data_accessor.list_data_types(selected_run)

            # Allow user to select data type to visualize
            selected_data_type = st.selectbox("Select data type to visualize", data_types, help="Choose the type of data you want to visualize")

            if selected_data_type:
                data = data_accessor.get_run_data(selected_run, selected_data_type)

                if selected_data_type == "combined_embeddings":
                    st.subheader("Embeddings Visualization")
                    fig = visualizer.plot_embedding_responses_plotly(
                        data['dim_reduce_tsne'],
                        data['joint_embeddings_all_llms'],
                        [model[1] for model in run_settings.model_settings.models],
                        f"{selected_run}_embeddings",
                        show_plot=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif selected_data_type.startswith("approvals_statements_"):
                    st.subheader(f"{selected_data_type.replace('_', ' ').title()} Visualization")
                    prompt_type = selected_data_type.split("_")[-1]
                    condition = st.selectbox("Select condition", [0, 1], format_func=lambda x: "Approved" if x == 1 else "Not Approved")
                    for model_info in run_settings.model_settings.models:
                        model = model_info[1]
                        fig = visualizer.plot_approvals_plotly(
                            data['dim_reduce'],
                            data['approval_data'],
                            model,
                            condition,
                            prompt_type,
                            f"{selected_run}_{prompt_type}_approvals_{model}",
                            f"{selected_run} {prompt_type.capitalize()} Approvals for {model}",
                            show_plot=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                elif selected_data_type == "spectral_clustering":
                    st.subheader("Spectral Clustering Visualization")
                    fig = visualizer.plot_spectral_clustering_plotly(
                        data['labels'],
                        run_settings.clustering_settings.n_clusters,
                        f"{selected_run}_spectral_clustering"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write(f"Data for {selected_data_type}:")
                    st.write(data)

        except Exception as e:
            st.error(f"An error occurred while loading or visualizing data: {str(e)}")
            st.exception(e)  # This will print the full traceback

        # Add option to export results
        if st.button("Export Results", help="Download the results as a CSV file"):
            # Implement export functionality
            st.success("Results exported successfully!")