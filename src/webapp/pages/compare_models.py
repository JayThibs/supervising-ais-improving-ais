import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import textwrap
import yaml
import traceback
from pathlib import Path
from behavioural_clustering.utils.visualization import Visualization
from behavioural_clustering.config.run_settings import PlotSettings
from behavioural_clustering.evaluation.analysis import analyze_approval_patterns, summarize_findings
from behavioural_clustering.models.model_factory import initialize_model

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

    # Load run metadata
    base_dir = Path(__file__).resolve().parents[3] / "data"
    metadata_path = base_dir / "metadata" / "run_metadata.yaml"
    
    with open(metadata_path, 'r') as file:
        run_metadata = yaml.safe_load(file)

    # Create a dataframe to store comparison metrics
    comparison_data = []

    for run in selected_runs:
        try:
            run_data = run_metadata.get(run, {})
            
            # Extract relevant metrics for comparison
            metrics = {
                'Run': run,
                'Models': ', '.join(run_data.get('model_names', [])),
                'Dataset': data_accessor.get_dataset_names(run),
                'N Statements': run_data.get('n_statements', ''),
                'Clustering Algorithm': run_data.get('run_settings', {}).get('clustering_settings', {}).get('main_clustering_algorithm', ''),
                'N Clusters': run_data.get('run_settings', {}).get('clustering_settings', {}).get('n_clusters', ''),
            }
            
            comparison_data.append(metrics)
        except Exception as e:
            st.warning(f"Error loading data for run {run}: {str(e)}")

    comparison_df = pd.DataFrame(comparison_data)

    # Display the comparison dataframe
    st.write(comparison_df)

    # Visualize comparison
    st.subheader("Visualization")
    
    # Compare embeddings
    if st.checkbox("Show Embeddings Comparison", help="Visualize and compare embeddings for selected runs"):
        fig = plot_interactive_embeddings_comparison(data_accessor, selected_runs, run_metadata)
        st.plotly_chart(fig, use_container_width=True)

    # Compare spectral clustering
    if st.checkbox("Show Spectral Clustering Comparison", help="Compare spectral clustering across runs"):
        fig = plot_interactive_spectral_clustering_comparison(data_accessor, selected_runs, run_metadata)
        st.plotly_chart(fig, use_container_width=True)

def plot_interactive_embeddings_comparison(data_accessor, selected_runs, run_metadata):
    df_list = []
    
    for run in selected_runs:
        try:
            joint_embeddings = data_accessor.get_run_data(run, "joint_embeddings_all_llms")
            dim_reduce_tsne = data_accessor.get_run_data(run, "tsne_reduction")
            
            if not isinstance(joint_embeddings, list) or not isinstance(dim_reduce_tsne, np.ndarray):
                raise ValueError(f"Invalid data format for run {run}.")
            
            if len(joint_embeddings) != len(dim_reduce_tsne):
                raise ValueError(f"Mismatch in data lengths for run {run}.")
            
            run_df = pd.DataFrame({
                'x': dim_reduce_tsne[:, 0],
                'y': dim_reduce_tsne[:, 1],
                'model': [e["model_name"] for e in joint_embeddings],
                'statement': [e['statement'] for e in joint_embeddings],
                'response': [e['response'] for e in joint_embeddings],
                'run': run
            })
            df_list.append(run_df)
            
        except Exception as e:
            st.warning(f"Error processing data for run {run}: {str(e)}")
    
    if not df_list:
        st.error("No valid data to visualize.")
        return None
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Function to wrap text
    def wrap_text(text, width=50):
        return '<br>'.join(textwrap.wrap(text, width=width))
    
    # Create hover text
    df['hover_text'] = df.apply(lambda row: f"Model: {row['model']}<br>Statement: {wrap_text(row['statement'])}<br>Response: {wrap_text(row['response'])}", axis=1)
    
    run_data = run_metadata.get(selected_runs[0], {})
    plot_settings = PlotSettings(**run_data.get('run_settings', {}).get('plot_settings', {}))
    visualizer = Visualization(plot_settings)
    
    fig = visualizer.create_interactive_embedding_scatter(
        df, 'x', 'y', 'model', 'run', 'hover_text',
        "Embeddings Comparison Across Runs", "Dimension 1", "Dimension 2"
    )
    return fig

def plot_interactive_spectral_clustering_comparison(data_accessor, selected_runs, run_metadata):
    fig = go.Figure()
    for run in selected_runs:
        try:
            spectral_clustering_data = data_accessor.get_run_data(run, "spectral_clustering")
            
            if not isinstance(spectral_clustering_data, (list, np.ndarray, pd.Series)):
                raise ValueError(f"Invalid data format for run {run}. Expected list, numpy array, or pandas Series.")
            
            run_data = run_metadata.get(run, {})
            plot_settings = PlotSettings(**run_data.get('run_settings', {}).get('plot_settings', {}))
            visualizer = Visualization(plot_settings)
            
            n_clusters = run_data.get('run_settings', {}).get('clustering_settings', {}).get('n_clusters', 10)
            
            plot_fig = visualizer.plot_spectral_clustering_plotly(
                spectral_clustering_data,
                n_clusters,
                f"Spectral Clustering - {run}"
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

def compare_approval_prompts(data_accessor, selected_runs):
    st.subheader("Approval Prompts Comparison")

    prompt_types = set()
    models = set()
    for run in selected_runs:
        run_settings = data_accessor.get_run_config(run).get('run_settings', {})
        prompt_types.update(run_settings.get('approval_prompts', {}).keys())
        run_models = data_accessor.get_model_names(run)
        models.update(run_models)

    selected_prompt_type = st.selectbox("Select prompt type", list(prompt_types))
    selected_model = st.selectbox("Select model", list(models))

    view_option = st.radio("Select view option", ["All Responses", "Disagreements Only"])

    if view_option == "All Responses":
        condition = st.selectbox("Select condition", [1, 0, -1], format_func=lambda x: "Approved" if x == 1 else "Disapproved" if x == 0 else "No Response")
    else:
        condition = None  # We'll handle disagreements separately

    try:
        fig = plot_interactive_approval_comparison(data_accessor, selected_runs, selected_prompt_type, selected_model, condition, view_option)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting approval comparison: {str(e)}")
        st.write("Traceback:", traceback.format_exc())

    # Add table with filters
    st.subheader("Approval Responses Table")
    approval_data = get_approval_data(data_accessor, selected_runs, selected_prompt_type)
    
    # Table view options
    table_view = st.radio("Table view", ["By Statement and Label", "By Statement and Model"])
    
    if table_view == "By Statement and Label":
        pivot_data = approval_data.pivot_table(
            index=['Run', 'Statement', 'Label'], 
            columns='Model', 
            values='Approval', 
            aggfunc=lambda x: ' | '.join(x)
        ).reset_index()
    else:
        pivot_data = approval_data.pivot_table(
            index=['Run', 'Statement', 'Model'], 
            columns='Label', 
            values='Approval', 
            aggfunc=lambda x: ' | '.join(x)
        ).reset_index()
    
    # Filter options
    st.subheader("Table Filters")
    filter_models = st.multiselect("Filter by models", list(models), default=list(models))
    
    # Apply filters
    if table_view == "By Statement and Label":
        filtered_data = pivot_data[pivot_data.columns[pivot_data.columns.isin(['Run', 'Statement', 'Label'] + filter_models)]]
    else:
        filtered_data = pivot_data[pivot_data['Model'].isin(filter_models)]
    
    if view_option == "Disagreements Only":
        if table_view == "By Statement and Label":
            filtered_data = filtered_data[filtered_data.iloc[:, 3:].apply(lambda row: len(set(row.dropna())) > 1, axis=1)]
        else:
            filtered_data = filtered_data[filtered_data.iloc[:, 3:].apply(lambda row: len(set(row.dropna())) > 1, axis=1)]
    
    # Display table
    st.dataframe(filtered_data)

    # Add a new section for pattern analysis
    st.subheader("Approval Pattern Analysis")
    if st.button("Analyze Approval Patterns"):
        with st.spinner("Analyzing approval patterns..."):
            # Get the full approval data
            approval_data = get_approval_data(data_accessor, selected_runs, selected_prompt_type)
            
            # Filter for disagreements
            disagreement_data = approval_data[approval_data.groupby(['Statement', 'Label'])['Approval'].transform('nunique') > 1]
            
            # Perform the analysis
            analysis_results = analyze_approval_patterns(disagreement_data)
            
            # Summarize the findings
            model_info = {
                "model_family": "anthropic",
                "model_name": "claude-3-opus-20240229",
                "system_message": "You are an AI assistant tasked with summarizing findings from approval pattern analysis."
            }
            summary_model = initialize_model(model_info)
            summary = summarize_findings(analysis_results, summary_model)
            
            # Display the results
            st.subheader("Analysis Results")
            st.write("Model Disagreement Analysis:")
            st.write(analysis_results['model_disagreement_analysis'])
            st.write("Label Disagreement Analysis:")
            st.write(analysis_results['label_disagreement_analysis'])
            
            st.subheader("Summary of Findings")
            st.write(summary)

def get_approval_data(data_accessor, selected_runs, prompt_type):
    all_data = []
    for run in selected_runs:
        try:
            approvals_data = data_accessor.get_run_data(run, f"approvals_statements_{prompt_type}")
            for item in approvals_data:
                statement = item['statement']
                for model, approvals in item['approvals'].items():
                    for label, approval in approvals.items():
                        all_data.append({
                            'Run': run,
                            'Statement': statement,
                            'Model': model,
                            'Label': label,
                            'Approval': 'Approved' if approval == 1 else 'Disapproved' if approval == 0 else 'No Response'
                        })
        except Exception as e:
            st.warning(f"Error processing data for run {run}: {str(e)}")
    
    return pd.DataFrame(all_data)

def plot_interactive_approval_comparison(data_accessor, selected_runs, prompt_type, model_name, condition, view_option):
    df_list = []
    condition_map = {-1: "No Response", 0: "Disapproved", 1: "Approved"}
    
    for run in selected_runs:
        try:
            approvals_data = data_accessor.get_run_data(run, f"approvals_statements_{prompt_type}")
            dim_reduce_tsne = data_accessor.get_run_data(run, "tsne_reduction")
            
            if not isinstance(approvals_data, list) or not isinstance(dim_reduce_tsne, np.ndarray):
                raise ValueError(f"Invalid data format for run {run}.")
            
            # Handle data length mismatch
            min_length = min(len(approvals_data), len(dim_reduce_tsne))
            approvals_data = approvals_data[:min_length]
            dim_reduce_tsne = dim_reduce_tsne[:min_length]
            
            # Create DataFrame
            df = pd.DataFrame({
                'x': dim_reduce_tsne[:, 0],
                'y': dim_reduce_tsne[:, 1],
                'statement': [item['statement'] for item in approvals_data],
                'run': run
            })
            
            # Add approval columns for each label
            labels = approvals_data[0]['approvals'][model_name].keys()
            for label in labels:
                df[label] = [item['approvals'][model_name][label] for item in approvals_data]
            
            df_list.append(df)
            
        except Exception as e:
            st.warning(f"Error processing data for run {run}: {str(e)}")
    
    if not df_list:
        st.error("No valid data to visualize.")
        return None
    
    df = pd.concat(df_list, ignore_index=True)
    
    # Filter based on the selected view option
    label_columns = [col for col in df.columns if col in labels]
    if view_option == "All Responses":
        df_filtered = df[df[label_columns].eq(condition).any(axis=1)].copy()
    else:  # Disagreements Only
        # Check if there's any disagreement among the labels
        df_filtered = df[df[label_columns].nunique(axis=1) > 1].copy()
    
    # Function to wrap text
    def wrap_text(text, width=50):
        return '<br>'.join(textwrap.wrap(text, width=width))
    
    # Create hover text with full wrapped statement
    df_filtered['hover_text'] = df_filtered.apply(lambda row: 
        f"Run: {row['run']}<br>Statement: {wrap_text(row['statement'])}<br>" + 
        "<br>".join([f"{label}: {condition_map[row[label]]}" for label in label_columns]), 
        axis=1
    )
    
    # Melt the dataframe to create 'label' and 'value' columns
    df_melted = df_filtered.melt(id_vars=['x', 'y', 'run', 'statement', 'hover_text'],
                                 value_vars=label_columns,
                                 var_name='label', value_name='value')
    
    # Set visibility based on view option
    if view_option == "All Responses":
        df_melted['visible'] = df_melted['value'] == condition
    else:
        df_melted['visible'] = True
    
    run_data = data_accessor.get_run_config(selected_runs[0])
    plot_settings = PlotSettings(**run_data.get('run_settings', {}).get('plot_settings', {}))
    visualizer = Visualization(plot_settings)
    
    title = (f"Approval Comparison Across Runs ({prompt_type.capitalize()}, {model_name})"
             f"{', ' + condition_map[condition] if condition is not None else ', Disagreements Only'}")
    
    fig = visualizer.create_interactive_approval_scatter(
        df_melted, 'x', 'y', 'run', 'label', 'hover_text',
        title, "Dimension 1", "Dimension 2"
    )
    return fig