import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from sklearn.cluster import OPTICS, SpectralClustering, AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from utils import lookup_cid_pos_in_rows, identify_theme


class Clustering:
    def __init__(self, embeddings, args):
        self.embeddings = embeddings
        self.args = args

    def perform_multiple_clustering(self):
        cluster_algorithms = {
            "OPTICS": OPTICS(min_samples=2, xi=0.12),
            "Spectral": SpectralClustering(100 if not self.args.test_mode else 2),
            "Agglomerative": AgglomerativeClustering(
                100 if not self.args.test_mode else 2
            ),
            "KMeans": KMeans(
                n_clusters=200 if not self.args.test_mode else 2, random_state=42
            ),
            # Add more as needed
        }

        print("Embeddings shape:", self.embeddings.shape)
        print("Embeddings:", self.embeddings)
        clustering_results = {}
        for name, algorithm in cluster_algorithms.items():
            print(f"Running {name} clustering...")
            clustering_results[name] = algorithm.fit(self.embeddings)

        return clustering_results

    def get_cluster_centroids(self, embeddings, cluster_labels):
        centroids = []
        for i in range(max(cluster_labels) + 1):
            c = np.mean(embeddings[cluster_labels == i], axis=0).tolist()
            centroids.append(c)
        return np.array(centroids)

    def hierarchical_cluster(
        self,
        clustering,
        approvals_statements_and_embeddings,
        rows,
        colors,
        labels=None,
        bar_height=1,
        bb_width=10,
        x_leftshift=0,
        y_downshift=0,
        figsize=(35, 35),
        filename="hierarchical_clustering.pdf",
    ):
        def llf(id):
            if id < n_clusters:
                return leaf_labels[id]
            else:
                return "Error: id too high."

        statement_embeddings = np.array(
            [e[2] for e in approvals_statements_and_embeddings]
        )
        centroids = self.get_cluster_centroids(statement_embeddings, clustering.labels_)
        Z = linkage(centroids, "ward")

        n_clusters = max(clustering.labels_) + 1
        cluster_labels = []
        for i in range(n_clusters):
            pos = lookup_cid_pos_in_rows(rows, i)
            if pos >= 0:
                cluster_labels.append(rows[pos][-1])
            else:
                cluster_labels.append("(Label missing)")

        all_cluster_sizes = []
        for i in range(n_clusters):
            inputs, model_approval_fractions = self.get_cluster_approval_stats(
                approvals_statements_and_embeddings, clustering.labels_, i
            )
            n = len(inputs)
            all_cluster_sizes.append(
                [n] + [int(f * n) for f in model_approval_fractions]
            )

        for merge in Z:
            m1 = int(merge[0])
            m2 = int(merge[1])
            m1_sizes = all_cluster_sizes[m1]
            m2_sizes = all_cluster_sizes[m2]
            merged_sizes = [
                int(m1_entry + m2_entry)
                for m1_entry, m2_entry in zip(m1_sizes, m2_sizes)
            ]
            all_cluster_sizes.append(merged_sizes)

        original_cluster_sizes = all_cluster_sizes[:n_clusters]
        merged_cluster_sizes = all_cluster_sizes[n_clusters:]

        # leaf_labels=[":".join([str(condition_size) for condition_size in s]) + " : " + l for s,l in zip(original_cluster_sizes,cluster_labels)]
        leaf_labels = [
            str(s[0]) + " : " + l
            for s, l in zip(original_cluster_sizes, cluster_labels)
        ]

        # adapted from: https://stackoverflow.com/questions/30317688/annotating-dendrogram-nodes-in-scipy-matplotlib

        Z[:, 2] = np.arange(1.0, len(Z) + 1)
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
            # ax.annotate(merged_cluster_labels[ind], (x,y), va='top', ha='center')
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
        if not labels is None:
            patch_colors = [
                mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)
            ]
            ax.legend(handles=patch_colors)

        plt.tight_layout()
        plt.savefig(filename)

    def compile_cluster_table(
        self,
        clustering,
        approvals_statements_and_embeddings,
        theme_summary_instructions="Briefly list the common themes of the following texts:",
        max_desc_length=250,
    ):
        n_clusters = max(clustering.labels_) + 1

        rows = []
        for cluster_id in tqdm(range(n_clusters)):
            row = [str(cluster_id)]
            cluster_indices = np.arange(len(clustering.labels_))[
                clustering.labels_ == cluster_id
            ]
            row.append(len(cluster_indices))
            inputs, model_approval_fractions = self.get_cluster_approval_stats(
                approvals_statements_and_embeddings, clustering.labels_, cluster_id
            )
            for frac in model_approval_fractions:
                row.append(str(round(100 * frac, 1)) + "%")
            cluster_inputs_themes = [
                identify_theme(
                    inputs,
                    sampled_texts=10,
                    max_tokens=70,
                    temp=0.5,
                    instructions=theme_summary_instructions,
                )[:max_desc_length].replace("\n", " ")
                for _ in range(1)
            ]
            inputs_themes_str = "\n".join(cluster_inputs_themes)

            row.append(inputs_themes_str)
            rows.append(row)
        rows = sorted(rows, key=lambda x: x[1], reverse=True)
        return rows
