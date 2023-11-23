import os
import pdb
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
import numpy as np


class Visualization:
    def __init__(self, plot_dim=(16, 16), save_path="data/plots"):
        self.plot_dim = plot_dim
        self.save_path = save_path
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
        dim_reduce_data,
        approval_data,
        filename,
        condition,
        colors,
        shapes,
        labels,
        sizes,
        title,
    ):
        plt.figure(figsize=self.plot_dim)

        plt.rcParams["figure.figsize"] = [25, 25]
        plt.rcParams["font.size"] = 25

        for idx, (color, shape, label, size) in enumerate(
            zip(colors, shapes, labels, sizes)
        ):
            mask = [e[0] == condition for e in approval_data]
            plt.scatter(
                dim_reduce_data[:, 0][mask],
                dim_reduce_data[:, 1][mask],
                c=color,
                label=label,
                s=size,
                marker=shape,
                alpha=0.5,
            )

        plt.legend()
        plt.title(title)
        plt.show()
        plt.save_plot(filename)
        # plt.close()

    def visualize_hierarchical_clustering(
        cluster_data, filename, method="ward", metric="euclidean", color_threshold=None
    ):
        """
        Visualize hierarchical clustering of data using dendrograms.

        Parameters:
        - cluster_data: ndarray
            The data used for clustering. Each row is a data point and each column is a feature.
        - method: str, optional
            The linkage algorithm to use. Default is 'ward'.
        - metric: str, optional
            The distance metric to use. Default is 'euclidean'.
        - color_threshold: float, optional
            The threshold to apply when coloring the branches. Default is None, meaning natural coloring.

        Returns:
        - None

        Example usage:
            import numpy as np
            random_data = np.random.rand(50, 4)  # 50 data points, each with 4 features
            visualize_hierarchical_clustering(random_data, color_threshold=1.5)
        """
        try:
            # Perform hierarchical/agglomerative clustering.
            Z = linkage(cluster_data, method=method, metric=metric)

            # Plot dendrogram
            plt.figure(figsize=(10, 7))
            dendrogram(Z, color_threshold=color_threshold)
            plt.title("Hierarchical Clustering Dendrogram")
            plt.xlabel("Data point index (or cluster index)")
            plt.ylabel("Distance")

            if color_threshold is not None:
                plt.axhline(y=color_threshold, c="black", lw=1, linestyle="dashed")

            plt.show()
            plt.save_plot(filename)
            # plt.close()

        except Exception as e:
            print(
                f"An error occurred while generating the hierarchical clustering visualization: {e}"
            )
