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


class Visualization:
    def __init__(self, plot_settings: PlotSettings, approval_prompts=[]):
        """
        Initialize the visualization class with the plot settings and approval prompts.
        """
        self.plot_dim = plot_settings.plot_dim
        self.save_path = plot_settings.save_path
        self.colors = plot_settings.colors
        self.shapes = plot_settings.shapes
        self.plot_aesthetics = plot_settings.plot_aesthetics

        with open(f"{os.getcwd()}/data/prompts/approval_prompts.json", "r") as file:
            self.approval_prompts = json.load(file)
            for key in self.approval_prompts.keys():
                setattr(self, key, list(self.approval_prompts[key].keys()))

        self.plot_aesthetics["approval"]["colors"] = self.colors[: len(self.personas)]
        self.plot_aesthetics["approval"]["shapes"] = self.shapes[: len(self.personas)]
        self.plot_aesthetics["approval"]["labels"] = self.personas
        self.plot_aesthetics["awareness"]["colors"] = self.colors[: len(self.awareness)]
        self.plot_aesthetics["awareness"]["shapes"] = self.shapes[: len(self.awareness)]
        self.plot_aesthetics["awareness"]["labels"] = self.awareness

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def plot_embedding_responses(
        self, dim_reduce_tsne, joint_embeddings_all_llms, model_names, filename
    ):
        # Adjust figure size and DPI
        plt.figure(figsize=(10, 8), dpi=100)
        plt.rcParams["font.size"] = self.plot_aesthetics["approval"]["font_size"]

        for i, model_name in enumerate(model_names):
            mask = np.array([e[4] == model_name for e in joint_embeddings_all_llms])
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
    ):
        # Adjust figure size and DPI
        fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
        colors, shapes, labels, sizes, order, fontsize = self.plot_aesthetics[
            plot_type
        ].values()
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        n_persona = len(labels)
        if order is None:
            order = [i for i in range(n_persona)]
            
        # Create a mask for points that don't match any persona's condition
        all_masks = np.zeros(len(approval_data), dtype=bool)
        
        # Plot colored points for each persona
        for i in order:
            mask = np.array([e[0][model_name][i] == condition for e in approval_data])
            all_masks |= mask  # Update the overall mask
            if np.any(mask):
                x = dim_reduce[:, 0][mask]
                y = dim_reduce[:, 1][mask]
                ax.scatter(x, y, marker=shapes[i], c=colors[i], label=labels[i], s=sizes[i], alpha=0.7)
                print(f"Plotted {np.sum(mask)} points for {labels[i]}")
            else:
                print(f"No data points for {labels[i]} with condition {condition}")
        # Plot grey points for data that doesn't match any persona's condition
        unmatched_mask = ~all_masks
        if np.any(unmatched_mask):
            x_unmatched = dim_reduce[:, 0][unmatched_mask]
            y_unmatched = dim_reduce[:, 1][unmatched_mask]
            ax.scatter(x_unmatched, y_unmatched, c="grey", s=10, alpha=0.3, label="Unmatched")
            print(f"Plotted {np.sum(unmatched_mask)} unmatched points in grey")

        ax.set_title(title, wrap=True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        fig.savefig(os.path.join(self.save_path, filename), bbox_inches='tight')
        print(f"Saved plot to {os.path.join(self.save_path, filename)}")
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
    ):
        colors = self.plot_aesthetics[plot_type]["colors"]

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
 

 
 
    def plot_spectral_clustering(self, labels, n_clusters, prompt_approver_type):
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100)  # Adjusted figure size
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.hist(labels, bins=n_clusters)
        ax.set_title(
            f"Spectral Clustering of {prompt_approver_type} Statement Responses",
            fontsize=16,
            wrap=True
        )
        # Adjust layout to fit all elements
        plt.tight_layout()
        plt.show()
        filename = f"{os.getcwd()}/data/results/plots/spectral_clustering_{prompt_approver_type}_statements.png"
        fig.savefig(
            filename, bbox_inches="tight", dpi=100
        )  # Use bbox_inches='tight' to fit the entire content
        print(f"Saved spectral clustering plot to {filename}")
        plt.close("all")
