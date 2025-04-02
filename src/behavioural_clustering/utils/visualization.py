import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from behavioural_clustering.config.run_settings import PlotSettings
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import plotly.io as pio
import textwrap
from typing import List
from behavioural_clustering.utils.embedding_data import EmbeddingEntry

class Visualization:
    def __init__(self, plot_settings: PlotSettings):
        """
        Initialize the visualization class with the plot settings and approval prompts.
        """
        self.plot_settings = plot_settings
        self.plot_dim = self.plot_settings.plot_dim
        self.save_path = self.plot_settings.save_path
        self.colors = self.plot_settings.colors
        self.shapes = self.plot_settings.shapes
        self.plot_aesthetics = self.plot_settings.plot_aesthetics

        # Load approval prompts from the data directory
        approval_prompts_path = Path(self.save_path).parent.parent / "prompts" / "approval_prompts.json"
        if approval_prompts_path.exists():
            with open(approval_prompts_path, "r", encoding="utf-8") as file:
                self.approval_prompts = json.load(file)
        else:
            print(f"Warning: approval_prompts.json not found at {approval_prompts_path}")
            self.approval_prompts = {}

        # Update plot aesthetics for each category
        for category, prompts in self.approval_prompts.items():
            num_prompts = len(prompts)
            self.plot_aesthetics[f"{category}_approvals"] = {
                "colors": self.colors[:num_prompts],
                "shapes": self.shapes[:num_prompts],
                "labels": list(prompts.keys()),
                "sizes": [self.plot_settings.plot_aesthetics["approvals"]["marker_size"]] * num_prompts,
                "order": None,
                "font_size": self.plot_settings.plot_aesthetics["approvals"]["font_size"],
                "legend_font_size": self.plot_settings.plot_aesthetics["approvals"]["legend_font_size"],
                "marker_size": self.plot_settings.plot_aesthetics["approvals"]["marker_size"],
                "alpha": self.plot_settings.plot_aesthetics["approvals"]["alpha"],
            }

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.setup_plot_directories()

    def setup_plot_directories(self):
        """Create subdirectories for different plot types."""
        plot_types = ['model_comparison', 'approvals', 'hierarchical', 'spectral']
        for plot_type in plot_types:
            subdir = os.path.join(self.save_path, plot_type)
            if not os.path.exists(subdir):
                os.makedirs(subdir)

    def save_plot(self, filename: str, plot_type: str):
        """
        Save the plot, moving older versions to subdirectories.

        Args:
            filename (str): The full path of the file to save.
            plot_type (str): The type of plot (e.g., 'model_comparison', 'approvals', etc.)
        """
        # Ensure the base directory exists
        self.save_path.mkdir(parents=True, exist_ok=True)

        # Create the subdirectory for the specific plot type
        subdir = self.save_path / plot_type
        subdir.mkdir(parents=True, exist_ok=True)

        base_filename = Path(filename).name
        new_filename = subdir / base_filename

        # Check if the file already exists in the subdirectory
        if new_filename.exists():
            # If it does, rename the existing file with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_filename = new_filename.with_name(f"{new_filename.stem}_{timestamp}{new_filename.suffix}")
            new_filename.rename(archived_filename)
            print(f"Existing file moved to: {archived_filename}")

        # Save the new plot
        plt.savefig(new_filename, bbox_inches='tight')
        print(f"New plot saved to: {new_filename}")

    def plot_embedding_responses(
        self, dim_reduce_tsne, joint_embeddings_all_llms: List[EmbeddingEntry], model_names, filename, show_plot=True
    ):
        # Adjust figure size and DPI
        plt.figure(figsize=self.plot_dim)
        plt.rcParams["font.size"] = self.plot_aesthetics["approvals"]["font_size"]

        for i, model_name in enumerate(model_names):
            mask = np.array([e.model_name == model_name for e in joint_embeddings_all_llms])
            x_values = dim_reduce_tsne[:, 0][mask]
            y_values = dim_reduce_tsne[:, 1][mask]

            plt.scatter(
                x_values,
                y_values,
                c=self.colors[i % len(self.colors)],  # Choose color
                label=model_name,
                s=20,
                alpha=0.5,
            )

        plt.legend()
        plt.title(f"Embeddings of {', '.join(model_names)} responses", wrap=True)
        plt.tight_layout()  # Ensure all elements fit within the figure
        self.save_plot(filename, 'model_comparison')
        if show_plot:
            plt.show()
        plt.close()

    def plot_approvals(
        self,
        dim_reduce,
        approval_data,
        model_name,
        condition: int,
        plot_type: str,
        filename: str,
        title: str,
        show_plot=True,
    ):
        # Adjust figure size and DPI
        fig, ax = plt.subplots(figsize=(14, 10), dpi=100)
        aesthetics = self.plot_aesthetics[f"{plot_type}_approvals"]
        colors, shapes, labels, sizes, order, fontsize, legend_fontsize, marker_size, alpha = aesthetics.values()
        
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        prompt_type_dict = self.approval_prompts[plot_type]
        # Example prompt_type_dict:
        # {"Bing Chat": "You are Bing Chat, the chat model of Bing search. Your purpose is to help users by 
        # providing clear and concise answers to any questions.",
        # "Bing Chat Emoji": "You are Bing Chat, the chat model of Bing search. 😊 Your purpose is to help users 
        # by providing clear and concise answers to any questions.",}
        
        # Create masks for each label
        label_masks = {
            label: np.array([
                e['approvals'][model_name][label] == condition if model_name in e['approvals'] else False
                for e in approval_data
            ])
            for label in labels
        }
        
        # Plot colored points for each persona
        for i, label in enumerate(labels):
            mask = label_masks[label]
            if np.any(mask):
                x = dim_reduce[:, 0][mask]
                y = dim_reduce[:, 1][mask]
                ax.scatter(x, y, marker=shapes[i], c=colors[i], label=label, s=marker_size, alpha=alpha)
                print(f"Plotted {np.sum(mask)} points for {label}")
            else:
                print(f"No data points for {label} with condition {condition}")

        # Plot grey points for data that doesn't match any persona's condition
        unmatched_mask = ~np.any(list(label_masks.values()), axis=0)
        if np.any(unmatched_mask):
            x_unmatched = dim_reduce[:, 0][unmatched_mask]
            y_unmatched = dim_reduce[:, 1][unmatched_mask]
            ax.scatter(x_unmatched, y_unmatched, c="grey", s=marker_size, alpha=alpha/2, label="Unmatched")
            print(f"Plotted {np.sum(unmatched_mask)} unmatched points in grey")

        condition_str = {1: "approved", 0: "disapproved", -1: "no response"}[condition]
        
        ax.set_title(title, fontsize=fontsize+2, wrap=True)
        
        # Create a custom legend
        legend_elements = [Line2D([0], [0], marker=shapes[i], color='w', label=label,
                                  markerfacecolor=colors[i], markersize=8)
                           for i, label in enumerate(labels)]
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Unmatched',
                                      markerfacecolor='grey', markersize=8))
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                  fontsize=legend_fontsize, title=plot_type.capitalize(), title_fontsize=legend_fontsize+2)
        
        plt.tight_layout()
        self.save_plot(filename, 'approvals')
        if show_plot:
            plt.show()
        plt.close(fig)

    def visualize_hierarchical_plot(
        self,
        hierarchy_data: tuple,
        plot_type: str,
        filename,
        bar_height=0.7,
        bb_width=40,
        x_leftshift=0,
        y_downshift=0,
        figsize=(20, 15),
        labels=None,
        model_name=None,
        show_plot=True,
    ):
        Z, leaf_labels, original_cluster_sizes, merged_cluster_sizes, n_clusters = hierarchy_data

        fig, ax = plt.subplots(figsize=figsize)
        dendrogram(
            Z,
            ax=ax,
            orientation='left',
            labels=leaf_labels,
            leaf_rotation=0,
            leaf_font_size=8,
            show_contracted=True,
        )

        ax.set_title(f'Hierarchical Clustering Dendrogram - {plot_type} - {model_name}')
        ax.set_xlabel('Distance')

        # Update the filename to include the model name
        filename = f"{filename}_{model_name}.png"
        self.save_plot(filename, 'hierarchical')
        if show_plot:
            plt.show()
        plt.close()

    def plot_spectral_clustering_plotly(self, labels, n_clusters, title):
        """
        Create a bar plot of cluster sizes using plotly.
        
        Args:
            labels: Array of cluster labels
            n_clusters: Number of clusters
            title: Plot title
        """
        # Convert labels to DataFrame with explicit index
        df = pd.DataFrame({'cluster': labels, 'count': 1}, index=range(len(labels)))
        
        # Group by cluster and count
        cluster_counts = df.groupby('cluster')['count'].sum().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(x=cluster_counts.index.astype(str), 
                  y=cluster_counts.values,
                  text=cluster_counts.values,
                  textposition='auto')
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Cluster",
            yaxis_title="Count",
            bargap=0.2,
            bargroupgap=0.1,
            showlegend=False
        )
        
        return fig

    def create_interactive_embedding_scatter(self, df, x_col, y_col, color_col, symbol_col, hover_data, title, x_label, y_label):
        marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star', 'pentagon', 'hexagon']
        color_palette = px.colors.qualitative.Plotly

        fig = go.Figure()

        for i, (model, group) in enumerate(df.groupby(color_col)):
            fig.add_trace(go.Scatter(
                x=group[x_col],
                y=group[y_col],
                mode='markers',
                marker=dict(
                    size=8,
                    symbol=marker_symbols[i % len(marker_symbols)],
                    color=color_palette[i % len(color_palette)],
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                name=model,
                text=group[hover_data],
                hoverinfo='text',
                showlegend=True
            ))

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend_title=color_col,
            hovermode="closest"
        )

        return fig

    def create_interactive_approval_scatter(self, df, x_col, y_col, color_col, symbol_col, hover_data, title, x_label, y_label):
        marker_symbols = ['circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down', 'star', 'pentagon', 'hexagon']
        color_palette = px.colors.qualitative.Plotly

        fig = go.Figure()

        all_labels = df[symbol_col].unique()
        all_runs = df[color_col].unique()

        for run_index, run in enumerate(all_runs):
            for label_index, label in enumerate(all_labels):
                group_df = df[(df[color_col] == run) & (df[symbol_col] == label) & df['visible']]
                
                marker_color = color_palette[label_index % len(color_palette)]
                marker_symbol = marker_symbols[label_index % len(marker_symbols)]
                
                if not group_df.empty:
                    fig.add_trace(go.Scatter(
                        x=group_df[x_col],
                        y=group_df[y_col],
                        mode='markers',
                        marker=dict(
                            size=8,
                            symbol=marker_symbol,
                            color=marker_color,
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        name=f"{run} - {label}",
                        text=group_df[hover_data],
                        hoverinfo='text',
                        showlegend=True
                    ))
                else:
                    # Add an empty trace to ensure the label appears in the legend
                    fig.add_trace(go.Scatter(
                        x=[None],
                        y=[None],
                        mode='markers',
                        marker=dict(
                            size=8,
                            symbol=marker_symbol,
                            color=marker_color,
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        name=f"{run} - {label}",
                        showlegend=True
                    ))

        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            legend_title=f"{color_col} and {symbol_col}",
            hovermode="closest",
            legend=dict(
                itemsizing='constant',
                title_font_family='Arial',
                font=dict(family='Arial', size=10),
                borderwidth=1
            )
        )

        return fig

    def create_interactive_treemap(self, hierarchy_data, plot_type, model_names, filename):
        def parse_hierarchical_data(hierarchical_data):
            linkage_matrix, descriptions, original_counts, merged_counts, n_clusters = hierarchical_data

            tree_dict = {}
            for i, (desc, counts) in enumerate(zip(descriptions, original_counts)):
                tree_dict[i] = {
                    "name": f"Cluster_{i}",
                    "description": desc,
                    "value": counts[0],
                    "proportions": {
                        "Unaware": counts[1] / counts[0],
                        "Other AI": counts[2] / counts[0],
                        "Aware": counts[3] / counts[0],
                        "Human": counts[4] / counts[0],
                    },
                }

            for i, link in enumerate(linkage_matrix):
                left, right, _, _ = link
                node_id = i + n_clusters
                left_child = tree_dict[int(left)]
                right_child = tree_dict[int(right)]

                tree_dict[node_id] = {
                    "name": f"Cluster_{node_id}",
                    "children": [left_child, right_child],
                    "value": left_child["value"] + right_child["value"],
                    "proportions": {
                        k: (left_child["value"] * left_child["proportions"][k] + right_child["value"] * right_child["proportions"][k])
                        / (left_child["value"] + right_child["value"])
                        for k in left_child["proportions"]
                    },
                }

            return tree_dict[node_id]

        def format_tree(node, parent_name="", level=0):
            results = []
            node_name = node["name"]
            results.append({
                "name": node_name,
                "parent": parent_name,
                "value": node["value"],
                "level": level,
                "proportions": node["proportions"],
                "description": node.get("description", ""),
            })

            if "children" in node:
                for child in node["children"]:
                    results.extend(format_tree(child, node_name, level + 1))

            return results

        def create_treemap(df, max_level, color_by, model_name):
            fig = go.Figure(go.Treemap(
                labels=[item["name"] for item in df if item["level"] <= max_level],
                parents=[item["parent"] for item in df if item["level"] <= max_level],
                values=[item["value"] for item in df if item["level"] <= max_level],
                branchvalues="total",
                hovertemplate="<b>%{label}</b><br>Value: %{value}<br>Proportions:<br>%{customdata}<extra></extra>",
                customdata=[
                    "<br>".join([f"{k}: {v:.2f}" for k, v in item["proportions"].items()])
                    + ("<br><br>" + "<br>".join(textwrap.wrap(item["description"], width=50)) if item["description"] else "")
                    for item in df if item["level"] <= max_level
                ],
                marker=dict(
                    colors=[item["proportions"][color_by] for item in df if item["level"] <= max_level],
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=f"{color_by} Proportion"),
                ),
                texttemplate="<b>%{label}</b><br>%{text}",
                text=[
                    "<br>".join([f"{k}: {v:.2f}" for k, v in item["proportions"].items()])
                    for item in df if item["level"] <= max_level
                ],
                textposition="middle center",
            ))

            fig.update_layout(
                title_text=f"Hierarchical Clustering Treemap - {plot_type} - {model_name}",
                width=1000,
                height=800,
            )

            return fig

        tree_data = {}
        max_levels = {}
        for model_name in model_names:
            if model_name in hierarchy_data:
                root = parse_hierarchical_data(hierarchy_data[model_name])
                tree_data[model_name] = format_tree(root)
                max_levels[model_name] = max(item["level"] for item in tree_data[model_name])

        # Generate HTML file
        html_content = """
        <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; }
                .controls { margin-bottom: 20px; }
                #treemap { width: 100%; height: 800px; }
            </style>
        </head>
        <body>
            <div class="controls">
                <label for="modelSelect">Select Model:</label>
                <select id="modelSelect"></select>
                <label for="maxLevelSlider">Max Level:</label>
                <input type="range" id="maxLevelSlider" min="1" max="10" value="2">
                <label for="colorBySelect">Color by:</label>
                <select id="colorBySelect">
                    <option value="Unaware">Unaware</option>
                    <option value="Other AI">Other AI</option>
                    <option value="Aware">Aware</option>
                    <option value="Human">Human</option>
                </select>
            </div>
            <div id="treemap"></div>
            <script>
                const treeData = {tree_data_json};
                const maxLevels = {max_levels_json};
                let currentModel = Object.keys(treeData)[0];
                let currentMaxLevel = 2;
                let currentColorBy = 'Aware';

                function updateTreemap() {
                    const data = treeData[currentModel];
                    const fig = createTreemap(data, currentMaxLevel, currentColorBy, currentModel);
                    Plotly.newPlot('treemap', fig.data, fig.layout);
                }

                function createTreemap(df, maxLevel, colorBy, modelName) {
                    const filteredData = df.filter(item => item.level <= maxLevel);
                    return {{
                        data: [{{
                            type: 'treemap',
                            labels: filteredData.map(item => item.name),
                            parents: filteredData.map(item => item.parent),
                            values: filteredData.map(item => item.value),
                            branchvalues: 'total',
                            hovertemplate: '<b>%{{label}}</b><br>Value: %{{value}}<br>Proportions:<br>%{{customdata}}<extra></extra>',
                            customdata: filteredData.map(item => 
                                Object.entries(item.proportions).map(([k, v]) => `${{k}}: ${{v.toFixed(2)}}`).join('<br>') +
                                (item.description ? '<br><br>' + item.description.split(' ').reduce((acc, word, i) => 
                                    acc + (i > 0 && i % 10 === 0 ? word + '<br>' : word + ' '), '') : '')
                            ),
                            marker: {{
                                colors: filteredData.map(item => item.proportions[colorBy]),
                                colorscale: 'Viridis',
                                showscale: true,
                                colorbar: {{title: `${{colorBy}} Proportion`}},
                            }},
                            texttemplate: '<b>%{{label}}</b><br>%{{text}}',
                            text: filteredData.map(item => 
                                Object.entries(item.proportions).map(([k, v]) => `${{k}}: ${{v.toFixed(2)}}`).join('<br>')
                            ),
                            textposition: 'middle center',
                        }}],
                        layout: {{
                            title: `Hierarchical Clustering Treemap - {plot_type} - ${{modelName}}`,
                            width: 1000,
                            height: 800,
                        }}
                    }};
                }

                // Initialize controls
                const modelSelect = document.getElementById('modelSelect');
                Object.keys(treeData).forEach(model => {{
                    const option = document.createElement('option');
                    option.value = model;
                    option.textContent = model;
                    modelSelect.appendChild(option);
                }});

                const maxLevelSlider = document.getElementById('maxLevelSlider');
                const colorBySelect = document.getElementById('colorBySelect');

                // Event listeners
                modelSelect.addEventListener('change', (e) => {{
                    currentModel = e.target.value;
                    maxLevelSlider.max = maxLevels[currentModel];
                    updateTreemap();
                }});

                maxLevelSlider.addEventListener('input', (e) => {{
                    currentMaxLevel = parseInt(e.target.value);
                    updateTreemap();
                }});

                colorBySelect.addEventListener('change', (e) => {{
                    currentColorBy = e.target.value;
                    updateTreemap();
                }});

                // Initial update
                updateTreemap();
            </script>
        </body>
        </html>
        """

        html_content = html_content.replace("{tree_data_json}", json.dumps(tree_data))
        html_content = html_content.replace("{max_levels_json}", json.dumps(max_levels))
        html_content = html_content.replace("{plot_type}", plot_type)

        with open(f"{filename}.html", "w") as f:
            f.write(html_content)

        print(f"Interactive treemap HTML saved to {filename}.html")

        return {
            "tree_data": tree_data,
            "max_levels": max_levels,
            "plot_type": plot_type,
            "model_names": model_names,
        }
