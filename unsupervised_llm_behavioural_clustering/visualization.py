import os
import pdb
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
import numpy as np


class Visualization:
    def __init__(self, plot_dim=(25, 25), save_path="data/plots"):
        self.plot_dim = plot_dim
        self.save_path = save_path
        # Define plotting aesthetics for statement approval plots
        self.plot_aesthetics = {
            "approval": {
                "colors": ["red", "black", "green", "blue"],
                "shapes": ["o", "o", "*", "+"],
                "labels": [
                    "Google Chat",
                    "Bing Chat",
                    "Bing Chat Emoji",
                    "Bing Chat Janus",
                ],
                "sizes": [5, 30, 200, 300],
                "order": None,
            },
            "awareness": {
                "colors": ["red", "black", "green", "blue"],
                "shapes": ["o", "o", "*", "+"],
                "labels": ["Unaware", "Other AI", "Aware", "Other human"],
                "sizes": [5, 30, 200, 300],
                "order": [2, 1, 3, 0],
            },
        }
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def save_plot(self, filename):
        plt.savefig(os.path.join(self.save_path, filename))

    def plot_embedding_responses(self, dim_reduce_tsne, labels, model_names, filename):
        plt.figure(figsize=self.plot_dim)
        unique_labels = np.unique(labels)
        print("unique_labels", unique_labels)

        # If only one model name is provided, use it for all labels
        if len(model_names) == 1:
            model_names = [model_names[0] for _ in unique_labels]
        elif len(model_names) < len(unique_labels):
            # If there are more labels than model names, raise an error or handle it as needed
            raise ValueError(
                "Number of model names is less than the number of unique labels"
            )

        for label in unique_labels:
            mask = labels == label
            x_values = dim_reduce_tsne[:, 0][mask]
            y_values = dim_reduce_tsne[:, 1][mask]

            model_label = model_names[label]  # Assuming label can be used as an index

            plt.scatter(
                x_values,
                y_values,
                label=model_label,
                s=20,
                alpha=0.5,
            )
        plt.legend()
        plt.title(f"Embeddings of {', '.join(model_names)} responses")
        plt.savefig(filename)  # Saving the plot
        plt.close()

    def plot_approvals(
        self,
        dim_reduce,
        approval_data,
        condition: int,
        plot_type: str,
        filename: str,
        title: str,
    ):
        plt.figure(figsize=self.plot_dim)
        colors, shapes, labels, sizes, order = self.plot_aesthetics[plot_type].values()
        n_persona = len(labels)
        if order is None:
            order = [i for i in range(n_persona)]
        masks = [
            np.array([e[0][i] == condition for e in approval_data])
            for i in range(n_persona)
        ]
        plt.scatter(dim_reduce[:, 0], dim_reduce[:, 1], c="grey", s=10, alpha=0.5)
        for i in order:
            plt.scatter(
                dim_reduce[:, 0][masks[i]],
                dim_reduce[:, 1][masks[i]],
                marker=shapes[i],
                c=colors[i],
                label=labels[i],
                s=sizes[i],
            )
        plt.title(title)
        plt.legend()
        plt.show()
        plt.save_plot(filename)
        # plt.close()

    def visualize_hierarchical_cluster(
        self,
        Z,
        leaf_labels,
        original_cluster_sizes,
        merged_cluster_sizes,
        bar_height=1,
        bb_width=10,
        x_leftshift=0,
        y_downshift=0,
        figsize=(35, 35),
        labels=None,
        filename="hierarchical_clustering.pdf",
    ):
        colors = self.aware_plot_aesthetics["approval"]["colors"]

        def llf(id):
            if id < len(leaf_labels):
                return leaf_labels[id]
            else:
                return "Error: id too high."

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=120)
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
        plt.savefig(filename)

    def visualize_awareness(
        self,
        dim_reduce_tsne,
        data_include_statements_and_embeddings_4_prompts,
        condition,
        title,
        filename="awareness.png",
    ):
        colors, shapes, labels, sizes, order = self.plot_aesthetics[
            "awareness"
        ].values()

        self.plot_approvals(
            dim_reduce_tsne,
            data_include_statements_and_embeddings_4_prompts,
            1,
            "awareness",
            "Embeddings of approvals for different chat modes",
            order=order,
        )
