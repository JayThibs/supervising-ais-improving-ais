"""
Standalone Plotly visualization functions for behavioral clustering.
These functions provide interactive visualizations without requiring the Visualization class.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Any, Optional, Union, Tuple
import plotly.io as pio


def plot_embedding_responses_plotly(
    embeddings_2d: np.ndarray,
    labels: np.ndarray,
    responses: List[str],
    model_names: List[str],
    title: str = "Model Response Embeddings"
) -> go.Figure:
    """
    Create an interactive Plotly scatter plot of embeddings for different model responses.
    
    Args:
        embeddings_2d: 2D embeddings (e.g., from t-SNE or UMAP)
        labels: Cluster labels for each point
        responses: List of model responses
        model_names: List of model names
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'cluster': labels,
        'response': responses
    })
    
    if len(model_names) > 0:
        df['model'] = model_names
    
    fig = go.Figure()
    
    marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star', 'pentagon', 'hexagon']
    color_palette = px.colors.qualitative.Plotly
    
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_df = df[df['cluster'] == cluster_id]
        
        hover_text = [
            f"Cluster: {cluster_id}<br>Response: {resp[:100]}..." if len(resp) > 100 else f"Cluster: {cluster_id}<br>Response: {resp}"
            for resp in cluster_df['response']
        ]
        
        fig.add_trace(go.Scatter(
            x=cluster_df['x'],
            y=cluster_df['y'],
            mode='markers',
            marker=dict(
                size=8,
                symbol=marker_symbols[cluster_id % len(marker_symbols)],
                color=color_palette[cluster_id % len(color_palette)],
                line=dict(width=1, color='DarkSlateGrey')
            ),
            name=f"Cluster {cluster_id}",
            text=hover_text,
            hoverinfo='text',
            showlegend=True
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        legend_title="Clusters",
        hovermode="closest",
        template="plotly_white"
    )
    
    return fig


def plot_approvals_plotly(
    approval_matrix: np.ndarray,
    model_names: List[str],
    cluster_labels: Optional[np.ndarray] = None,
    title: str = "Model Approval Patterns"
) -> go.Figure:
    """
    Create an interactive Plotly heatmap of approval patterns across models and clusters.
    
    Args:
        approval_matrix: Matrix of approval values (models x clusters or statements)
        model_names: List of model names
        cluster_labels: Optional cluster labels for each statement
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    if cluster_labels is not None:
        n_clusters = len(np.unique(cluster_labels))
        cluster_approval = np.zeros((len(model_names), n_clusters))
        
        for i, model in enumerate(range(len(model_names))):
            for j, cluster in enumerate(range(n_clusters)):
                mask = cluster_labels == cluster
                if np.any(mask):
                    cluster_approval[i, j] = np.mean(approval_matrix[i, mask])
        
        df = pd.DataFrame(cluster_approval, index=model_names, columns=[f"Cluster {i}" for i in range(n_clusters)])
    else:
        df = pd.DataFrame(approval_matrix, index=model_names)
    
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale='RdBu_r',
        zmid=0.5,  # Center the color scale at 0.5
        text=df.round(2).values,
        texttemplate="%{text}",
        textfont={"size": 10},
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Clusters" if cluster_labels is not None else "Statements",
        yaxis_title="Models",
        template="plotly_white",
        height=400 + 30 * len(model_names),  # Adjust height based on number of models
        width=800,
        margin=dict(l=150, r=50, t=100, b=50)
    )
    
    return fig


def plot_cluster_sizes_plotly(
    labels: np.ndarray,
    title: str = "Cluster Sizes"
) -> go.Figure:
    """
    Create an interactive Plotly bar chart of cluster sizes.
    
    Args:
        labels: Cluster labels
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    df = pd.DataFrame({
        'cluster': [f"Cluster {label}" for label in unique_labels],
        'size': counts
    })
    
    df = df.sort_values('cluster')
    
    fig = go.Figure(data=go.Bar(
        x=df['cluster'],
        y=df['size'],
        text=df['size'],
        textposition='auto',
        marker_color=px.colors.qualitative.Plotly[:len(df)]
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Cluster",
        yaxis_title="Number of Statements",
        template="plotly_white"
    )
    
    return fig


def plot_model_comparison_plotly(
    embeddings_2d: np.ndarray,
    model_indices: np.ndarray,
    model_names: List[str],
    responses: Optional[List[str]] = None,
    title: str = "Model Response Comparison"
) -> go.Figure:
    """
    Create an interactive Plotly scatter plot comparing responses from different models.
    
    Args:
        embeddings_2d: 2D embeddings (e.g., from t-SNE or UMAP)
        model_indices: Array of model indices for each point
        model_names: List of model names
        responses: Optional list of model responses
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'model_idx': model_indices
    })
    
    if responses is not None:
        df['response'] = responses
    
    fig = go.Figure()
    
    marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star', 'pentagon', 'hexagon']
    color_palette = px.colors.qualitative.Plotly
    
    for i, model_name in enumerate(model_names):
        model_df = df[df['model_idx'] == i]
        
        if model_df.empty:
            continue
        
        hover_text = [
            f"Model: {model_name}<br>Response: {resp[:100]}..." if len(resp) > 100 else f"Model: {model_name}<br>Response: {resp}"
            for resp in model_df['response']
        ] if responses is not None else [f"Model: {model_name}" for _ in range(len(model_df))]
        
        fig.add_trace(go.Scatter(
            x=model_df['x'],
            y=model_df['y'],
            mode='markers',
            marker=dict(
                size=8,
                symbol=marker_symbols[i % len(marker_symbols)],
                color=color_palette[i % len(color_palette)],
                line=dict(width=1, color='DarkSlateGrey')
            ),
            name=model_name,
            text=hover_text,
            hoverinfo='text',
            showlegend=True
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        legend_title="Models",
        hovermode="closest",
        template="plotly_white"
    )
    
    return fig


def save_plotly_figure(fig: go.Figure, filename: str, formats: List[str] = ['html', 'png']) -> Dict[str, str]:
    """
    Save a Plotly figure in multiple formats.
    
    Args:
        fig: Plotly figure object
        filename: Base filename (without extension)
        formats: List of formats to save (html, png, jpg, pdf, svg)
        
    Returns:
        Dictionary of saved file paths
    """
    saved_files = {}
    
    for fmt in formats:
        if fmt.lower() == 'html':
            output_file = f"{filename}.html"
            pio.write_html(fig, output_file)
            saved_files['html'] = output_file
        elif fmt.lower() == 'png':
            output_file = f"{filename}.png"
            pio.write_image(fig, output_file)
            saved_files['png'] = output_file
        elif fmt.lower() == 'jpg' or fmt.lower() == 'jpeg':
            output_file = f"{filename}.jpg"
            pio.write_image(fig, output_file)
            saved_files['jpg'] = output_file
        elif fmt.lower() == 'pdf':
            output_file = f"{filename}.pdf"
            pio.write_image(fig, output_file)
            saved_files['pdf'] = output_file
        elif fmt.lower() == 'svg':
            output_file = f"{filename}.svg"
            pio.write_image(fig, output_file)
            saved_files['svg'] = output_file
    
    return saved_files
