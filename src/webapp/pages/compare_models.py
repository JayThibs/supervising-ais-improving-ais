import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from behavioural_clustering.utils.visualization import Visualization
from behavioural_clustering.config.run_settings import RunSettings, PlotSettings

def show():
    st.header("Compare Models")

    data_accessor = st.session_state.data_accessor
    runs = data_accessor.list_runs()
    
    if not runs:
        st.warning("No previous runs found.")
        return

    selected_runs = st.multiselect("Select runs to compare", runs, help="Choose one or more runs to compare")

    if not selected_runs:
        st.warning("Please select at least one run to compare.")
        return

    comparison_type = st.radio("Select comparison type", ["Model Comparison", "Approval Prompts"])

    if comparison_type == "Model Comparison":
        compare_models(data_accessor, selected_runs)
    elif comparison_type == "Approval Prompts":
        compare_approval_prompts(data_accessor, selected_runs)

def compare_models(data_accessor, selected_runs):
    st.subheader("Model Comparison")

    # Create a dataframe to store comparison metrics
    comparison_df = pd.DataFrame()

    for run in selected_runs:
        try:
            run_config = data_accessor.get_run_config(run)
            run_settings = RunSettings.from_dict(run_config)
            
            # Extract relevant metrics for comparison
            metrics = {
                'Run': run,
                'Models': ', '.join([model[1] for model in run_settings.model_settings.models]),
                'Dataset': ', '.join(run_settings.data_settings.datasets),
                'N Statements': run_settings.data_settings.n_statements,
                'Clustering Algorithm': run_settings.clustering_settings.main_clustering_algorithm,
                'N Clusters': run_settings.clustering_settings.n_clusters,
            }
            
            comparison_df = comparison_df.append(metrics, ignore_index=True)
        except Exception as e:
            st.warning(f"Error loading data for run {run}: {str(e)}")

    # Display the comparison dataframe
    st.write(comparison_df)

    # Visualize comparison
    st.subheader("Visualization")
    
    # Compare embeddings
    if st.checkbox("Show Embeddings Comparison", help="Visualize and compare embeddings for selected runs"):
        fig = plot_interactive_embeddings_comparison(data_accessor, selected_runs)
        st.plotly_chart(fig, use_container_width=True)

    # Compare spectral clustering
    if st.checkbox("Show Spectral Clustering Comparison", help="Compare spectral clustering across runs"):
        fig = plot_interactive_spectral_clustering_comparison(data_accessor, selected_runs)
        st.plotly_chart(fig, use_container_width=True)

def compare_approval_prompts(data_accessor, selected_runs):
    st.subheader("Approval Prompts Comparison")

    prompt_types = ["personas", "awareness"]  # Add more prompt types if needed
    selected_prompt_type = st.selectbox("Select prompt type", prompt_types)

    models = set()
    for run in selected_runs:
        run_config = data_accessor.get_run_config(run)
        run_settings = RunSettings.from_dict(run_config)
        models.update([model[1] for model in run_settings.model_settings.models])

    selected_model = st.selectbox("Select model", list(models))

    condition = st.selectbox("Select condition", [1, 0, -1], format_func=lambda x: "Approved" if x == 1 else "Disapproved" if x == 0 else "No Response")

    fig = plot_interactive_approval_comparison(data_accessor, selected_runs, selected_prompt_type, selected_model, condition)
    st.plotly_chart(fig, use_container_width=True)

def plot_interactive_embeddings_comparison(data_accessor, selected_runs):
    fig = go.Figure()
    for run in selected_runs:
        try:
            embeddings_data = data_accessor.get_run_data(run, "combined_embeddings")
            run_config = data_accessor.get_run_config(run)
            run_settings = RunSettings.from_dict(run_config)
            visualizer = Visualization(run_settings.plot_settings)
            
            dim_reduce_tsne = data_accessor.get_run_data(run, "tsne_reduction")
            
            plot_fig = visualizer.plot_embedding_responses_plotly(
                dim_reduce_tsne,
                embeddings_data,
                [model[1] for model in run_settings.model_settings.models],
                f"{run}_embeddings",
                show_plot=False
            )
            
            for trace in plot_fig.data:
                trace.name = f"{run} - {trace.name}"
                fig.add_trace(trace)
        except Exception as e:
            st.warning(f"Error visualizing embeddings for run {run}: {str(e)}")
    
    fig.update_layout(
        title="Embeddings Comparison Across Runs",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        legend_title="Runs and Models",
        hovermode="closest"
    )
    return fig

def plot_interactive_spectral_clustering_comparison(data_accessor, selected_runs):
    fig = go.Figure()
    for run in selected_runs:
        try:
            spectral_clustering_data = data_accessor.get_run_data(run, "spectral_clustering")
            run_config = data_accessor.get_run_config(run)
            run_settings = RunSettings.from_dict(run_config)
            visualizer = Visualization(run_settings.plot_settings)
            
            plot_fig = visualizer.plot_spectral_clustering_plotly(
                spectral_clustering_data,
                run_settings.clustering_settings.n_clusters,
                f"{run}_spectral_clustering"
            )
            
            for trace in plot_fig.data:
                trace.name = f"{run}"
                fig.add_trace(trace)
        except Exception as e:
            st.warning(f"Error visualizing spectral clustering for run {run}: {str(e)}")
    
    fig.update_layout(
        title="Spectral Clustering Comparison Across Runs",
        xaxis_title="Cluster",
        yaxis_title="Count",
        barmode='group'
    )
    return fig

def plot_interactive_approval_comparison(data_accessor, selected_runs, prompt_type, model_name, condition):
    fig = go.Figure()
    for run in selected_runs:
        try:
            approvals_data = data_accessor.get_run_data(run, f"approvals_statements_{prompt_type}")
            run_config = data_accessor.get_run_config(run)
            run_settings = RunSettings.from_dict(run_config)
            visualizer = Visualization(run_settings.plot_settings)
            
            dim_reduce_tsne = data_accessor.get_run_data(run, "tsne_reduction")
            
            plot_fig = visualizer.plot_approvals_plotly(
                dim_reduce_tsne,
                approvals_data,
                model_name,
                condition,
                prompt_type,
                f"{run}_{prompt_type}_approvals_{model_name}",
                f"{run} {prompt_type.capitalize()} Approvals for {model_name}",
                show_plot=False
            )
            
            for trace in plot_fig.data:
                trace.name = f"{run} - {trace.name}"
                fig.add_trace(trace)
        except Exception as e:
            st.warning(f"Error visualizing approvals for run {run}: {str(e)}")
    
    fig.update_layout(
        title=f"Approval Comparison Across Runs ({prompt_type.capitalize()}, {model_name})",
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        legend_title="Runs and Approval Status",
        hovermode="closest"
    )
    return fig