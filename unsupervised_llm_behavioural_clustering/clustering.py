import os
import numpy as np
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import sklearn
from terminaltables import AsciiTable
from sklearn.cluster import OPTICS, SpectralClustering, AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from utils import lookup_cid_pos_in_rows, identify_theme, compare_response_pair


class Clustering:
    def __init__(self, args):
        self.args = args

    def perform_multiple_clustering(self, embeddings):
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

        print("Embeddings shape:", embeddings.shape)
        print("Embeddings:", embeddings)
        clustering_results = {}
        for name, algorithm in cluster_algorithms.items():
            print(f"Running {name} clustering...")
            clustering_results[name] = algorithm.fit(embeddings)

        return clustering_results

    def get_cluster_centroids(self, embeddings, cluster_labels):
        centroids = []
        for i in range(max(cluster_labels) + 1):
            c = np.mean(embeddings[cluster_labels == i], axis=0).tolist()
            centroids.append(c)
        return np.array(centroids)

    def cluster_statement_embeddings(self, statement_embeddings):
        print("Running clustering on statement embeddings...")
        clustering = sklearn.cluster.SpectralClustering(120, random_state=42).fit(
            statement_embeddings
        )

        plt.hist(
            clustering.labels_, bins=120, title="Spectral Clustering of Statements"
        )
        plt.show()
        return clustering

    def cluster_approval_stats(
        self,
        approvals_statements_and_embeddings,
        statement_clustering,
        labels,
    ):
        # Calculating the confusion matrix and pearson correlation between responses
        chat_modes = ["Bing Chat", "Google Chat", "Bing Chat Emoji", "Bing Chat Janus"]
        response_types = ["approve", "disapprove"]

        for response_type in response_types:
            for i in range(len(chat_modes)):
                for j in range(i + 1, len(chat_modes)):
                    compare_response_pair(
                        approvals_statements_and_embeddings,
                        chat_modes[i],
                        chat_modes[j],
                        labels,
                        response_type,
                    )
            print("\n\n")

        # rows = pickle.load(open("chat_mode_approvals_spectral_clustering_rows.pkl", "rb"))
        rows = self.compile_cluster_table(
            statement_clustering,
            approvals_statements_and_embeddings,
            theme_summary_instructions="Briefly list the common themes of the following texts:",
            max_desc_length=250,
        )

        pickle.dump(
            rows,
            open(
                f"{os.getcwd()}/data/results/pickle_files/rows_chatbots_G_B_BE_BJ.pkl",
                "wb",
            ),
        )
        pickle.dump(
            statement_clustering,
            open(
                f"{os.getcwd()}/data/results/pickle_files/clustering_chatbots_G_B_BE_BJ.pkl",
                "wb",
            ),
        )

        # rows = pickle.load(open("chat_mode_approvals_spectral_clustering_rows.pkl", "rb"))
        clusters_desc_table = [
            [
                "ID",
                "N",
                "Google Chat",
                "Bing Chat",
                "Bing Chat Emojis",
                "Bing Chat Janus",
                "Inputs Themes",
            ]
        ]
        for r in rows:
            clusters_desc_table.append(r)
        t = AsciiTable(clusters_desc_table)
        t.inner_row_border = True
        print(t.table)
        print("\n\n")
        print("Saving table to file...")
        pickle.dump(
            clusters_desc_table,
            open(
                f"{os.getcwd()}/data/results/pickle_files/clusters_desc_table_chatbots_G_B_BE_BJ.pkl",
                "wb",
            ),
        )

    def calculate_hierarchical_cluster_data(
        self, clustering, approvals_statements_and_embeddings, rows
    ):
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
        leaf_labels = [
            str(s[0]) + " : " + l
            for s, l in zip(original_cluster_sizes, cluster_labels)
        ]

        return Z, leaf_labels, original_cluster_sizes, merged_cluster_sizes

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
