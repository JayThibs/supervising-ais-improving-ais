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

        with open(f"{os.getcwd()}/data/prompts/approval_prompts.json", "r") as file:
            self.approval_prompts = json.load(file)

        # Update plot aesthetics for each category
        for category, prompts in self.approval_prompts.items():
            num_prompts = len(prompts)
            self.plot_aesthetics[f"{category}_approvals"] = {
                "colors": self.colors[:num_prompts],
                "shapes": self.shapes[:num_prompts],
                "labels": list(prompts.keys()),
                "sizes": [self.plot_settings.plot_aesthetics["approvals"]["marker_size"]] * num_prompts,  # Use a single size
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
        self, dim_reduce_tsne, joint_embeddings_all_llms, model_names, filename, show_plot=True
    ):
        # Adjust figure size and DPI
        plt.figure(figsize=(10, 8), dpi=100)
        plt.rcParams["font.size"] = self.plot_aesthetics["approvals"]["font_size"]

        for i, model_name in enumerate(model_names):
            mask = np.array([e["model_name"] == model_name for e in joint_embeddings_all_llms])
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
                                  markerfacecolor=colors[i], markersize=10)
                           for i, label in enumerate(labels)]
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Unmatched',
                                      markerfacecolor='grey', markersize=10))
        
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

        # ... (rest of the method remains the same)

        # Update the filename to include the model name
        filename = f"{filename}_{model_name}.png"
        self.save_plot(filename, 'hierarchical')
        if show_plot:
            plt.show()
        plt.close()

    def plot_spectral_clustering(self, labels, n_clusters, filename):
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.hist(labels, bins=n_clusters)
        ax.set_title(
            f"Spectral Clustering of Statement Responses",
            fontsize=16,
            wrap=True
        )
        plt.tight_layout()
        self.save_plot(filename, 'spectral')
        plt.close("all")

    def plot_embedding_responses_plotly(
        self, dim_reduce_tsne, joint_embeddings_all_llms, model_names, filename, show_plot=True
    ):
        df = pd.DataFrame({
            'x': dim_reduce_tsne[:, 0],
            'y': dim_reduce_tsne[:, 1],
            'model': [e['model_name'] for e in joint_embeddings_all_llms],
            'statement': [e['statement'] for e in joint_embeddings_all_llms],
            'response': [e['response'] for e in joint_embeddings_all_llms]
        })
        
        fig = px.scatter(df, x='x', y='y', color='model', hover_data=['statement', 'response'])
        
        fig.update_layout(
            title=f"Embeddings of {', '.join(model_names)} responses",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            legend_title="Models",
            font=dict(size=self.plot_aesthetics['approvals']['font_size'])
        )
        
        fig.update_traces(marker=dict(size=self.plot_aesthetics['approvals']['marker_size'],
                                      opacity=self.plot_aesthetics['approvals']['alpha']))
        
        if show_plot:
            fig.show()
        
        fig.write_html(f"{filename}.html")
        return fig

    def plot_approvals_plotly(
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
        aesthetics = self.plot_aesthetics[f"{plot_type}_approvals"]
        labels = aesthetics['labels']
        
        df = pd.DataFrame({
            'x': dim_reduce[:, 0],
            'y': dim_reduce[:, 1],
            'statement': [e['statement'] for e in approval_data],
            'approval': ['Unmatched'] * len(approval_data)
        })
        
        for label in labels:
            mask = np.array([
                e['approvals'][model_name][label] == condition if model_name in e['approvals'] else False
                for e in approval_data
            ])
            df.loc[mask, 'approval'] = label
        
        fig = px.scatter(df, x='x', y='y', color='approval', hover_data=['statement'],
                         color_discrete_map={**{label: aesthetics['colors'][i] for i, label in enumerate(labels)},
                                             'Unmatched': 'grey'})
        
        fig.update_traces(marker=dict(size=aesthetics['marker_size'], opacity=aesthetics['alpha']))
        
        fig.update_layout(
            title=title,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            legend_title=f"{plot_type.capitalize()} Approval",
            font=dict(size=aesthetics['font_size'])
        )
        
        if show_plot:
            fig.show()
        
        fig.write_html(f"{filename}.html")
        return fig

    def plot_spectral_clustering_plotly(self, labels, n_clusters, filename):
        df = pd.DataFrame({'cluster': labels})
        fig = px.histogram(df, x='cluster', nbins=n_clusters)
        fig.update_layout(
            title=f"Spectral Clustering of Statement Responses (n_clusters={n_clusters})",
            xaxis_title="Cluster",
            yaxis_title="Count",
            font=dict(size=self.plot_aesthetics['approvals']['font_size'])
        )
        
        fig.write_html(f"{filename}.html")
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
                    + ("<br><br>" + item["description"] if item["description"] else "")
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

        # Prepare the output data
        output = {
            "tree_data": tree_data,
            "max_levels": max_levels,
            "plot_type": plot_type,
            "model_names": model_names,
            "create_treemap": create_treemap  # Include the create_treemap function
        }

        # Save the output as a JSON file (excluding the create_treemap function)
        json_output = {k: v for k, v in output.items() if k != "create_treemap"}
        with open(f"{filename}.json", "w") as f:
            json.dump(json_output, f)

        return output