"""
Visualization Utilities Module

This module provides tools for visualizing circuit differences, behavioral
patterns, and analysis results. Visualization is crucial for understanding
and interpreting the mechanistic findings.

Key Features:
    - Circuit difference heatmaps
    - Feature activation plots
    - Hypothesis validation summaries
    - Interactive network graphs
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..circuit_analysis import CircuitDifference, FeatureDifference, ConnectionDifference
from ..hypothesis_generation import BehavioralHypothesis
from ..behavioral_validation import ValidationResult


class CircuitVisualizer:
    """
    Visualizes circuit differences and analysis results.
    
    This class creates various plots to help understand mechanistic
    and behavioral differences between models.
    """
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
    def visualize_circuit_difference(self,
                                   circuit_diff: CircuitDifference,
                                   save_name: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create comprehensive visualization of a circuit difference.
        
        Args:
            circuit_diff: Circuit difference to visualize
            save_name: Optional filename to save plot
            
        Returns:
            Figure and axes objects
        """
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Feature activation differences heatmap
        self._plot_feature_heatmap(circuit_diff.feature_differences, axes[0, 0])
        
        # 2. Connection weight changes
        self._plot_connection_changes(circuit_diff.connection_differences, axes[0, 1])
        
        # 3. Summary statistics
        self._plot_summary_stats(circuit_diff.statistics, axes[1, 0])
        
        # 4. Top changed features
        self._plot_top_features(circuit_diff.feature_differences, axes[1, 1])
        
        # Overall title
        fig.suptitle(f"Circuit Analysis: {circuit_diff.prompt[:50]}...", fontsize=16)
        
        plt.tight_layout()
        
        if save_name:
            save_path = self.output_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
            
        return fig, axes
        
    def _plot_feature_heatmap(self, 
                            feature_diffs: List[FeatureDifference],
                            ax: plt.Axes):
        """Plot heatmap of feature activation differences."""
        if not feature_diffs:
            ax.text(0.5, 0.5, 'No feature differences', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Feature Activation Differences")
            return
            
        # Organize by layer
        layer_features = {}
        for fd in feature_diffs:
            if fd.layer not in layer_features:
                layer_features[fd.layer] = []
            layer_features[fd.layer].append(fd)
            
        # Create matrix
        layers = sorted(layer_features.keys())
        max_features_per_layer = max(len(features) for features in layer_features.values())
        
        matrix = np.zeros((len(layers), min(max_features_per_layer, 50)))  # Limit width
        
        for i, layer in enumerate(layers):
            for j, feat_diff in enumerate(layer_features[layer][:50]):
                matrix[i, j] = feat_diff.activation_delta
                
        # Plot heatmap
        im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Labels
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Layer")
        ax.set_title("Feature Activation Differences\n(Red = Higher in Intervention)")
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        
        # Colorbar
        plt.colorbar(im, ax=ax)
        
    def _plot_connection_changes(self,
                               conn_diffs: List[ConnectionDifference],
                               ax: plt.Axes):
        """Plot connection weight changes."""
        if not conn_diffs:
            ax.text(0.5, 0.5, 'No connection differences',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Connection Weight Changes")
            return
            
        # Get top changes
        top_changes = sorted(conn_diffs, key=lambda c: c.weight_change, reverse=True)[:20]
        
        # Prepare data
        labels = []
        base_weights = []
        int_weights = []
        
        for conn in top_changes:
            label = f"{conn.source[:10]}→{conn.target[:10]}"
            labels.append(label)
            base_weights.append(conn.base_weight)
            int_weights.append(conn.intervention_weight)
            
        # Create grouped bar chart
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, base_weights, width, label='Base', alpha=0.8)
        ax.bar(x + width/2, int_weights, width, label='Intervention', alpha=0.8)
        
        ax.set_xlabel('Connection')
        ax.set_ylabel('Weight')
        ax.set_title('Top Connection Weight Changes')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        
    def _plot_summary_stats(self, stats: Dict[str, float], ax: plt.Axes):
        """Plot summary statistics."""
        # Select key statistics
        key_stats = {
            'Feature Jaccard': stats.get('feature_jaccard', 0),
            'New Connections': stats.get('n_new_connections', 0),
            'Removed Connections': stats.get('n_removed_connections', 0),
            'Mean Weight Change': stats.get('mean_weight_change', 0),
            'Total Change Score': stats.get('total_change_score', 0)
        }
        
        # Create bar chart
        ax.bar(key_stats.keys(), key_stats.values())
        ax.set_ylabel('Value')
        ax.set_title('Summary Statistics')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, (k, v) in enumerate(key_stats.items()):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
    def _plot_top_features(self,
                         feature_diffs: List[FeatureDifference],
                         ax: plt.Axes):
        """Plot top changed features."""
        if not feature_diffs:
            ax.text(0.5, 0.5, 'No feature differences',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Top Changed Features")
            return
            
        # Get top features by absolute change
        top_features = sorted(
            feature_diffs,
            key=lambda f: abs(f.activation_delta),
            reverse=True
        )[:15]
        
        # Prepare data
        labels = [f"L{f.layer}_F{f.feature_idx}" for f in top_features]
        deltas = [f.activation_delta for f in top_features]
        colors = ['red' if d > 0 else 'blue' for d in deltas]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(labels))
        ax.barh(y_pos, deltas, color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Activation Delta')
        ax.set_title('Top Changed Features')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
    def visualize_hypothesis_validation(self,
                                      validation_results: List[ValidationResult],
                                      save_name: Optional[str] = None) -> plt.Figure:
        """
        Visualize hypothesis validation results.
        
        Args:
            validation_results: List of validation results
            save_name: Optional filename to save
            
        Returns:
            Figure object
        """
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('P-values Distribution', 'Effect Sizes',
                          'Validation Success by Type', 'Confidence vs Effect Size'),
            specs=[[{'type': 'histogram'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]]
        )
        
        # 1. P-value distribution
        p_values = [r.p_value for r in validation_results]
        fig.add_trace(
            go.Histogram(x=p_values, nbinsx=20, name='P-values'),
            row=1, col=1
        )
        fig.add_vline(x=0.05, line_dash="dash", line_color="red",
                     annotation_text="α=0.05", row=1, col=1)
        
        # 2. Effect sizes
        effect_sizes = [r.effect_size for r in validation_results]
        hypothesis_types = [r.hypothesis.hypothesis_type.value for r in validation_results]
        
        fig.add_trace(
            go.Bar(x=hypothesis_types, y=effect_sizes, name='Effect Size'),
            row=1, col=2
        )
        
        # 3. Validation success by type
        type_counts = {}
        type_validated = {}
        
        for result in validation_results:
            h_type = result.hypothesis.hypothesis_type.value
            type_counts[h_type] = type_counts.get(h_type, 0) + 1
            if result.is_validated():
                type_validated[h_type] = type_validated.get(h_type, 0) + 1
                
        success_rates = {
            h_type: type_validated.get(h_type, 0) / count
            for h_type, count in type_counts.items()
        }
        
        fig.add_trace(
            go.Bar(x=list(success_rates.keys()), 
                  y=list(success_rates.values()),
                  name='Success Rate'),
            row=2, col=1
        )
        
        # 4. Confidence vs Effect Size scatter
        confidences = [r.confidence for r in validation_results]
        colors = ['green' if r.is_validated() else 'red' for r in validation_results]
        
        fig.add_trace(
            go.Scatter(
                x=effect_sizes,
                y=confidences,
                mode='markers',
                marker=dict(color=colors, size=8),
                text=[r.hypothesis.description[:50] for r in validation_results],
                name='Results'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Hypothesis Validation Results",
            showlegend=False,
            height=800
        )
        
        fig.update_xaxes(title_text="P-value", row=1, col=1)
        fig.update_xaxes(title_text="Hypothesis Type", row=1, col=2)
        fig.update_xaxes(title_text="Hypothesis Type", row=2, col=1)
        fig.update_xaxes(title_text="Effect Size", row=2, col=2)
        
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Effect Size", row=1, col=2)
        fig.update_yaxes(title_text="Success Rate", row=2, col=1)
        fig.update_yaxes(title_text="Confidence", row=2, col=2)
        
        if save_name:
            save_path = self.output_dir / save_name
            fig.write_html(save_path)
            print(f"Saved interactive plot to {save_path}")
            
        return fig
        
    def create_circuit_network_graph(self,
                                   circuit_diff: CircuitDifference,
                                   save_name: Optional[str] = None) -> go.Figure:
        """
        Create an interactive network graph of circuit differences.
        
        This visualizes how information flow changes between models.
        
        Args:
            circuit_diff: Circuit difference to visualize
            save_name: Optional filename to save
            
        Returns:
            Plotly figure object
        """
        # Build graph from circuit differences
        G = nx.DiGraph()
        
        # Add nodes for features that changed
        for feat_diff in circuit_diff.feature_differences[:50]:  # Limit for clarity
            node_id = f"L{feat_diff.layer}_F{feat_diff.feature_idx}"
            G.add_node(
                node_id,
                layer=feat_diff.layer,
                feature_idx=feat_diff.feature_idx,
                activation_delta=feat_diff.activation_delta
            )
            
        # Add edges for connection differences
        for conn_diff in circuit_diff.connection_differences[:100]:  # Limit
            if conn_diff.weight_change > 0.1:  # Threshold
                G.add_edge(
                    conn_diff.source[:20],  # Truncate for readability
                    conn_diff.target[:20],
                    weight_change=conn_diff.weight_change,
                    base_weight=conn_diff.base_weight,
                    int_weight=conn_diff.intervention_weight
                )
                
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Create edge trace
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            edge_data = G.edges[edge]
            width = min(edge_data.get('weight_change', 0.1) * 5, 5)
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=width, color='gray'),
                    hoverinfo='none'
                )
            )
            
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            node_data = G.nodes[node]
            delta = node_data.get('activation_delta', 0)
            node_color.append(delta)
            
            # Hover text
            text = f"{node}<br>"
            text += f"Layer: {node_data.get('layer', 'N/A')}<br>"
            text += f"Activation Δ: {delta:.3f}"
            node_text.append(text)
            
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=[n[:10] for n in G.nodes()],  # Truncated labels
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='RdBu_r',
                size=10,
                color=node_color,
                colorbar=dict(
                    thickness=15,
                    title='Activation Delta',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            title='Circuit Difference Network',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        if save_name:
            save_path = self.output_dir / save_name
            fig.write_html(save_path)
            print(f"Saved network graph to {save_path}")
            
        return fig
        
    def create_summary_dashboard(self,
                               analysis_report,
                               save_name: str = "analysis_dashboard.html"):
        """
        Create a comprehensive dashboard summarizing the analysis.
        
        Args:
            analysis_report: Complete analysis report
            save_name: Filename for saving dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Circuit Changes Overview',
                'Hypothesis Types Distribution',
                'Validation Success Rates',
                'Effect Size Distribution',
                'Runtime Breakdown',
                'Top Validated Findings'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'bar'}, {'type': 'histogram'}],
                [{'type': 'pie'}, {'type': 'table'}]
            ],
            row_heights=[0.3, 0.3, 0.4]
        )
        
        # 1. Circuit changes overview
        circuit_stats = {
            'Total Differences': len(analysis_report.circuit_differences),
            'Systematic Features': analysis_report.systematic_patterns.get('n_systematic_features', 0),
            'Feature Clusters': len(analysis_report.systematic_patterns.get('feature_clusters', [])),
        }
        
        fig.add_trace(
            go.Bar(x=list(circuit_stats.keys()), y=list(circuit_stats.values())),
            row=1, col=1
        )
        
        # 2. Hypothesis types distribution
        hypothesis_counts = {}
        for h in analysis_report.hypotheses:
            h_type = h.hypothesis_type.value
            hypothesis_counts[h_type] = hypothesis_counts.get(h_type, 0) + 1
            
        fig.add_trace(
            go.Pie(labels=list(hypothesis_counts.keys()),
                  values=list(hypothesis_counts.values())),
            row=1, col=2
        )
        
        # 3. Validation success rates
        validated = [r for r in analysis_report.validation_results if r.is_validated()]
        success_data = {
            'Total Tested': len(analysis_report.validation_results),
            'Validated': len(validated),
            'Failed': len(analysis_report.validation_results) - len(validated)
        }
        
        fig.add_trace(
            go.Bar(x=list(success_data.keys()), y=list(success_data.values())),
            row=2, col=1
        )
        
        # 4. Effect size distribution
        effect_sizes = [r.effect_size for r in analysis_report.validation_results]
        
        fig.add_trace(
            go.Histogram(x=effect_sizes, nbinsx=15),
            row=2, col=2
        )
        
        # 5. Runtime breakdown
        runtime_data = analysis_report.runtime_info
        
        fig.add_trace(
            go.Pie(labels=list(runtime_data.keys()),
                  values=list(runtime_data.values())),
            row=3, col=1
        )
        
        # 6. Top validated findings table
        top_findings = sorted(
            validated,
            key=lambda r: r.effect_size,
            reverse=True
        )[:5]
        
        table_data = {
            'Hypothesis': [r.hypothesis.description[:50] + '...' for r in top_findings],
            'Type': [r.hypothesis.hypothesis_type.value for r in top_findings],
            'P-value': [f"{r.p_value:.4f}" for r in top_findings],
            'Effect Size': [f"{r.effect_size:.3f}" for r in top_findings],
            'Confidence': [f"{r.confidence:.2f}" for r in top_findings]
        }
        
        fig.add_trace(
            go.Table(
                header=dict(values=list(table_data.keys())),
                cells=dict(values=list(table_data.values()))
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Mechanistic Behavioral Analysis Dashboard",
            showlegend=False,
            height=1200
        )
        
        # Save
        save_path = self.output_dir / save_name
        fig.write_html(save_path)
        print(f"Saved dashboard to {save_path}")
        
        return fig


def create_comparison_plots(circuit_differences: List[CircuitDifference],
                          output_dir: str = "./visualizations") -> None:
    """
    Create comparison plots across multiple circuit differences.
    
    This is useful for identifying systematic patterns.
    
    Args:
        circuit_differences: List of circuit differences to compare
        output_dir: Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature change frequency heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Count feature changes
    feature_counts = {}
    max_layer = 0
    
    for diff in circuit_differences:
        for feat_diff in diff.feature_differences:
            key = (feat_diff.layer, feat_diff.feature_idx)
            feature_counts[key] = feature_counts.get(key, 0) + 1
            max_layer = max(max_layer, feat_diff.layer)
            
    # Create matrix
    max_features = max(feat_idx for _, feat_idx in feature_counts.keys())
    matrix = np.zeros((max_layer + 1, min(max_features + 1, 100)))
    
    for (layer, feat_idx), count in feature_counts.items():
        if feat_idx < 100:  # Limit for visualization
            matrix[layer, feat_idx] = count
            
    # Plot
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Layer')
    ax.set_title(f'Feature Change Frequency Across {len(circuit_differences)} Prompts')
    plt.colorbar(im, ax=ax, label='Change Count')
    
    plt.tight_layout()
    plt.savefig(output_path / 'feature_change_frequency.png', dpi=300)
    plt.close()
    
    # 2. Connection change patterns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count new and removed connections
    new_conn_counts = {}
    removed_conn_counts = {}
    
    for diff in circuit_differences:
        for conn in diff.connection_differences:
            if conn.is_new_connection():
                key = f"{conn.source_type}→{conn.target_type}"
                new_conn_counts[key] = new_conn_counts.get(key, 0) + 1
            elif conn.is_removed_connection():
                key = f"{conn.source_type}→{conn.target_type}"
                removed_conn_counts[key] = removed_conn_counts.get(key, 0) + 1
                
    # Plot new connections
    if new_conn_counts:
        ax1.bar(new_conn_counts.keys(), new_conn_counts.values(), color='green', alpha=0.7)
        ax1.set_title('New Connections by Type')
        ax1.set_xlabel('Connection Type')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot removed connections
    if removed_conn_counts:
        ax2.bar(removed_conn_counts.keys(), removed_conn_counts.values(), color='red', alpha=0.7)
        ax2.set_title('Removed Connections by Type')
        ax2.set_xlabel('Connection Type')
        ax2.set_ylabel('Count')
        ax2.tick_params(axis='x', rotation=45)
        
    plt.tight_layout()
    plt.savefig(output_path / 'connection_changes.png', dpi=300)
    plt.close()
    
    print(f"Comparison plots saved to {output_path}")