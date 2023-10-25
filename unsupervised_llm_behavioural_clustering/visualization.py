import matplotlib.pyplot as plt


def plot_dimension_reduction(dim_reduce_tsne, joint_embeddings_all_llms):
    """
    Plot the dimension reduction.
    """
    plt.rcParams["figure.figsize"] = [16, 16]
    plt.rcParams["font.size"] = 25
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


def visualize_hierarchical_clustering(
    clustering, approvals_statements_and_embeddings, rows, colors
):
    """
    Visualize the hierarchical clustering.
    """
    # Implementation omitted for brevity
