import os
import yaml
import pdb
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram
from config.run_settings import PlotSettings


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
        plt.figure(figsize=self.plot_dim)
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
        plt.title(f"Embeddings of {', '.join(model_names)} responses")
        plt.savefig(os.path.join(self.save_path, filename))
        print(f"Saved plot to {os.path.join(self.save_path, filename)}")
        # Commented out plt.show() to prevent displaying the plot
        # plt.show()
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
        fig, ax = plt.subplots(figsize=self.plot_dim)
        colors, shapes, labels, sizes, order, fontsize = self.plot_aesthetics[
            plot_type
        ].values()
        ax.tick_params(axis="both", which="major", labelsize=fontsize)
        n_persona = len(labels)
        if order is None:
            order = [i for i in range(n_persona)]
        masks = []
        for i in range(n_persona):
            print(f"num i: {i}")
            for e in approval_data:
                print(f"e[0]: {e[0][model_name]}")
                print(f"e[0][i]: {e[0][model_name][i]}")
            mask = np.array([e[0][model_name][i] == condition for e in approval_data])
            masks.append(mask)
            print(f"mask for persona {i}: {mask}")
        ax.scatter(dim_reduce[:, 0], dim_reduce[:, 1], c="grey", s=10, alpha=0.5)
        for i in order:
            ax.scatter(
                dim_reduce[:, 0][masks[i]],
                dim_reduce[:, 1][masks[i]],
                marker=shapes[i],
                c=colors[i],
                label=labels[i],
                s=sizes[i],
            )
        ax.set_title(title)
        ax.legend()
        fig.savefig(os.path.join(self.save_path, filename))
        print(f"Saved plot to {os.path.join(self.save_path, filename)}")
        plt.show()

    def visualize_hierarchical_cluster(
        self,
        hierarchy_data: tuple,
        plot_type: str,
        filename,
        bar_height=0.7,
        bb_width=40,
        x_leftshift=0,
        y_downshift=0,
        figsize=(35, 35),
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
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=120)
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
        plt.savefig(f"{filename}.png")
        plt.savefig(f"{filename}.svg", format="svg")

    def plot_spectral_clustering(self, labels, n_clusters, prompt_approver_type):
        # Set the figsize parameter to increase the figure size
        fig, ax = plt.subplots(figsize=(10, 6))  # You can adjust these values as needed
        ax.tick_params(axis="both", which="major", labelsize=15)
        ax.hist(labels, bins=n_clusters)
        ax.set_title(
            f"Spectral Clustering of {prompt_approver_type} Statement Responses",
            fontsize=20,
        )
        # Adjust layout to fit all elements
        fig.tight_layout()
        plt.show()
        filename = f"{os.getcwd()}/data/results/plots/spectral_clustering_{prompt_approver_type}_statements.png"
        fig.savefig(
            filename, bbox_inches="tight"
        )  # Use bbox_inches='tight' to fit the entire content
        print(f"Saved spectral clustering plot to {filename}")
        plt.close("all")