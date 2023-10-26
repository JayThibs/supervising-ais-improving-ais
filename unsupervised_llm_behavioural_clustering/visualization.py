import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.manifold import TSNE
import numpy as np


class Visualization:
    def __init__(self, data, plot_dim: str, save_path: str):
        self.data = data
        self.plot_dim = plot_dim
        self.save_path = save_path

    def plot_dimension_reduction(
        dim_reduce_tsne, joint_embeddings_all_llms, plot_colors=None
    ):
        """
        Plot the dimension reduction.
        """
        if plot_colors is None:
            plot_colors = [
                "plum",
                "salmon",
                "darkgreen",
                "cyan",
                "slategrey",
                "purple",
                "black",
                "yellow",
                "slategrey",
                "darkgreen",
            ]

        plt.rcParams["figure.figsize"] = [16, 16]
        plt.rcParams["font.size"] = 25

        mask_002 = [e[0] == 0 for e in joint_embeddings_all_llms]
        mask_003 = [e[0] == 1 for e in joint_embeddings_all_llms]

        plt.scatter(
            dim_reduce_tsne[:, 0][mask_003],
            dim_reduce_tsne[:, 1][mask_003],
            c="blue",
            label="003",
            s=20,
            alpha=0.5,
        )
        plt.scatter(
            dim_reduce_tsne[:, 0][mask_002],
            dim_reduce_tsne[:, 1][mask_002],
            c="red",
            label="002",
            s=20,
            alpha=0.5,
        )
        plt.legend()
        plt.title("Embeddings of Davinci 002 and 003 responses")
        plt.show()

    def plot_embedding_responses(
        self,
        joint_embeddings_all_llms,
        model_names=["002", "003"],
        colors=["red", "blue"],
    ):
        # Prepare the data
        combined_embeddings = np.array([e[3] for e in joint_embeddings_all_llms])

        # Dimensionality reduction
        iterations = 2000
        p = 50
        dim_reduce_tsne = TSNE(
            perplexity=p,
            n_iter=iterations,
            angle=0.8,
            init="pca",
            early_exaggeration=22,
            learning_rate="auto",
            random_state=42,
        ).fit_transform(X=combined_embeddings)

        # Plotting
        plt.rcParams["figure.figsize"] = [16, 16]
        plt.rcParams["font.size"] = 25

        for idx, model_name in enumerate(model_names):
            mask = [e[0] == idx for e in joint_embeddings_all_llms]
            plt.scatter(
                dim_reduce_tsne[:, 0][mask],
                dim_reduce_tsne[:, 1][mask],
                c=colors[idx],
                label=model_name,
                s=20,
                alpha=0.5,
            )

        plt.legend()
        plt.title(
            f"Embeddings of Davinci {model_names[0]} and {model_names[1]} responses"
        )
        plt.show()

    def visualize_hierarchical_clustering(
        cluster_data, method="ward", metric="euclidean", color_threshold=None
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

        except Exception as e:
            print(
                f"An error occurred while generating the hierarchical clustering visualization: {e}"
            )
