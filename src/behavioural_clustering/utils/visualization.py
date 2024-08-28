import os
import yaml
import pdb
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram
from behavioural_clustering.config.run_settings import PlotSettings
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

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
                "sizes": self.plot_aesthetics["approvals"]["sizes"][:num_prompts],
                "order": None,
                "font_size": self.plot_aesthetics["approvals"]["font_size"],
                "legend_font_size": self.plot_aesthetics["approvals"]["legend_font_size"],
                "marker_size": self.plot_aesthetics["approvals"]["marker_size"],
                "alpha": self.plot_aesthetics["approvals"]["alpha"],
            }

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

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
        plt.savefig(os.path.join(self.save_path, filename), bbox_inches='tight')
        print(f"Saved plot to {os.path.join(self.save_path, filename)}")
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
        fig, ax = plt.subplots(figsize=(14, 10), dpi=100)  # Increased figure size
        aesthetics = self.plot_aesthetics[f"{plot_type}_approvals"]
        colors, shapes, labels, sizes, order, fontsize, legend_fontsize, marker_size, alpha = aesthetics.values()
        
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        prompt_type_dict = self.approval_prompts[plot_type]
        # Example prompt_type_dict:
        # {"Bing Chat": "You are Bing Chat, the chat model of Bing search. Your purpose is to help users by 
        # providing clear and concise answers to any questions.",
        # "Bing Chat Emoji": "You are Bing Chat, the chat model of Bing search. 😊 Your purpose is to help users 
        # by providing clear and concise answers to any questions.",}
        
        # Create a mask for points that don't match any persona's condition
        all_masks = np.zeros(len(approval_data), dtype=bool)
        
        # Plot colored points for each persona
        for i, label in enumerate(labels):
            mask = np.array([e['approvals'][model_name][label] == condition for e in approval_data])
            all_masks |= mask  # Update the overall mask
            if np.any(mask):
                x = dim_reduce[:, 0][mask]
                y = dim_reduce[:, 1][mask]
                ax.scatter(x, y, marker=shapes[i], c=colors[i], label=label, s=marker_size, alpha=alpha)
                print(f"Plotted {np.sum(mask)} points for {label}")
            else:
                print(f"No data points for {label} with condition {condition}")

        # Plot grey points for data that doesn't match any persona's condition
        unmatched_mask = ~all_masks
        if np.any(unmatched_mask):
            x_unmatched = dim_reduce[:, 0][unmatched_mask]
            y_unmatched = dim_reduce[:, 1][unmatched_mask]
            ax.scatter(x_unmatched, y_unmatched, c="grey", s=marker_size//2, alpha=alpha/2, label="Unmatched")
            print(f"Plotted {np.sum(unmatched_mask)} unmatched points in grey")

        ax.set_title(title, fontsize=fontsize+2, wrap=True)
        
        # Create a custom legend
        legend_elements = [Line2D([0], [0], marker=shapes[i], color='w', label=label,
                                  markerfacecolor=colors[i], markersize=sizes[i]//5)
                           for i, label in enumerate(labels)]
        legend_elements.append(Line2D([0], [0], marker='o', color='w', label='Unmatched',
                                      markerfacecolor='grey', markersize=sizes[0]//10))
        
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                  fontsize=legend_fontsize, title=plot_type.capitalize(), title_fontsize=legend_fontsize+2)
        
        plt.tight_layout()
        fig.savefig(os.path.join(self.save_path, filename), bbox_inches='tight')
        print(f"Saved plot to {os.path.join(self.save_path, filename)}")
        if show_plot:
            plt.show()
        plt.close(fig)  # Close the figure to free up memory

    def visualize_hierarchical_plot(
        self,
        hierarchy_data: tuple,
        plot_type: str,
        filename,
        bar_height=0.7,
        bb_width=40,
        x_leftshift=0,
        y_downshift=0,
        figsize=(20, 15),  # Reduced figure size
        labels=None,
        show_plot=True,
    ):
        colors = self.plot_aesthetics[f"{plot_type}_approvals"]["colors"]

        # Unpack hierarchy data
        (
            Z,
            leaf_labels,
            original_cluster_sizes,
            merged_cluster_sizes,
            n_clusters,
        ) = hierarchy_data

        # def llf(id):
        #     if id < n_clusters:
        #         return leaf_labels[id]
        #     else:
        #         return "Error: id too high."

        def llf(id):
            if id < len(leaf_labels):
                return leaf_labels[id]
            else:
                # Adjust this part to handle IDs for merged clusters appropriately
                return f"Cluster {id}"

        # font size
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=100)
        ax.tick_params(axis="both", which="major", labelsize=18)
        dn = dendrogram(
            Z, ax=ax, leaf_rotation=-90, leaf_font_size=20, leaf_label_func=llf
        )

        ii = np.argsort(np.array(dn["dcoord"])[:, 1])
        for j, (icoord, dcoord) in enumerate(zip(dn["icoord"], dn["dcoord"])):
            x = 0.5 * sum(icoord[1:3])
            y = dcoord[1]
            ind = np.nonzero(ii == j)[0][0]
            s = merged_cluster_sizes[ind]

            for i in range(len(colors)):
                ax.add_patch(
                    Rectangle(
                        (
                            x - bb_width / 2 - x_leftshift,
                            y
                            - y_downshift
                            - i * bar_height
                            + bar_height * (len(colors) - 1),
                        ),
                        bb_width * s[i + 1] / s[0],
                        bar_height,
                        facecolor=colors[i],
                    )
                )

            ax.add_patch(
                Rectangle(
                    (x - bb_width / 2 - x_leftshift, y - y_downshift),
                    bb_width,
                    bar_height * len(colors),
                    facecolor="none",
                    ec="k",
                    lw=1,
                )
            )

        if labels is not None:
            patch_colors = [
                mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)
            ]
            ax.legend(handles=patch_colors)

        plt.tight_layout()
        plt.savefig(f"{filename}.png", bbox_inches='tight')
        plt.savefig(f"{filename}.svg", format="svg", bbox_inches='tight')
        print(f"Saved hierarchical plot to {filename}.png and {filename}.svg")
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
        plt.show()
        filename = f"{os.getcwd()}/data/results/plots/{filename}.png"
        fig.savefig(
            filename, bbox_inches="tight", dpi=100
        )  # Use bbox_inches='tight' to fit the entire content
        print(f"Saved spectral clustering plot to {filename}")
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
            legend_title="Models"
        )
        
        if show_plot:
            fig.show()
        
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
        df = pd.DataFrame({
            'x': dim_reduce[:, 0],
            'y': dim_reduce[:, 1],
            'statement': [e['statement'] for e in approval_data],
            'approval': [e['approvals'][model_name][plot_type] == condition for e in approval_data]
        })
        
        fig = px.scatter(df, x='x', y='y', color='approval', hover_data=['statement'])
        fig.update_layout(
            title=title,
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            legend_title=f"{plot_type.capitalize()} Approval"
        )
        
        if show_plot:
            fig.show()
        
        return fig

    def plot_spectral_clustering_plotly(self, labels, n_clusters, filename):
        df = pd.DataFrame({'cluster': labels})
        fig = px.histogram(df, x='cluster', nbins=n_clusters)
        fig.update_layout(
            title=f"Spectral Clustering of Statement Responses (n_clusters={n_clusters})",
            xaxis_title="Cluster",
            yaxis_title="Count"
        )
        return fig