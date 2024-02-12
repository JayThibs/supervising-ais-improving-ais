import os
import csv
import numpy as np
from tqdm import tqdm
import pdb
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import sklearn
from terminaltables import AsciiTable
from sklearn.cluster import OPTICS, SpectralClustering, AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from utils import lookup_cid_pos_in_rows, identify_theme, compare_response_pair
from dataclasses import dataclass


from dataclasses import dataclass, asdict
import json


class Clustering:
    def __init__(self, args):
        self.args = args

    def perform_multiple_clustering(self, embeddings):
        cluster_algorithms = {
            "OPTICS": OPTICS(min_samples=2, xi=0.12),
            "Spectral": SpectralClustering(
                100 if not self.args.test_mode else 10, random_state=42
            ),
            "Agglomerative": AgglomerativeClustering(
                100 if not self.args.test_mode else 10
            ),
            "KMeans": KMeans(
                n_clusters=200 if not self.args.test_mode else 10, random_state=42
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

    def cluster_persona_embeddings(
        self,
        statement_embeddings,
        prompt_approver_type,
        n_clusters=120,
        spectral_plot=True,
    ):
        print("Running clustering on statement embeddings...")
        clustering = sklearn.cluster.SpectralClustering(
            n_clusters, random_state=42
        ).fit(statement_embeddings)
        print("clustering.labels_", clustering.labels_)
        print("n_clusters:", n_clusters)

        if spectral_plot:
            # Set the figsize parameter to increase the figure size
            fig, ax = plt.subplots(
                figsize=(10, 6)
            )  # You can adjust these values as needed
            ax.tick_params(axis="both", which="major", labelsize=15)
            ax.hist(clustering.labels_, bins=n_clusters)
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
            print(f"Saved plot to {filename}")
            plt.close("all")
        return clustering

    def get_cluster_approval_stats(
        self, approvals_statements_and_embeddings, cluster_labels, cluster_ID
    ):
        """Analyzes a cluster and extracts aggregated approval statistics."""
        inputs = []
        responses = []
        cluster_size = 0
        n_conditions = len(approvals_statements_and_embeddings[0][0])
        approval_fractions = [0 for _ in range(n_conditions)]
        for e, l in zip(approvals_statements_and_embeddings, cluster_labels):
            if l != cluster_ID:
                continue
            for i in range(n_conditions):
                if e[0][i] == 1:
                    approval_fractions[i] += 1
            cluster_size += 1
            inputs.append(e[1])
        return inputs, [f / cluster_size for f in approval_fractions]

    def cluster_approval_stats(
        self,
        approvals_statements_and_embeddings,
        statement_clustering,
        all_model_info,
        reuse_cluster_rows=False,
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
                        chat_modes,
                        response_type,
                    )
            print("\n\n")

        pickle_base_path = f"{os.getcwd()}/data/results/pickle_files"
        rows_pickle_path = f"{pickle_base_path}/rows_chatbots_G_B_BE_BJ.pkl"
        clustering_pickle_path = f"{pickle_base_path}/clustering_chatbots_G_B_BE_BJ.pkl"
        table_pickle_path = (
            f"{pickle_base_path}/clusters_desc_table_chatbots_G_B_BE_BJ.pkl"
        )

        if reuse_cluster_rows:
            print("Loading rows from file...")
            with open(rows_pickle_path, "rb") as file:
                rows = pickle.load(file)
        else:
            print("Calculating rows...")
            rows = self.compile_cluster_table(
                statement_clustering,
                approvals_statements_and_embeddings,
                all_model_info,
                theme_summary_instructions="Briefly list the common themes of the following texts:",
                max_desc_length=250,
            )

            with open(rows_pickle_path, "wb") as file:
                pickle.dump(rows, file)
            with open(clustering_pickle_path, "wb") as file:
                pickle.dump(statement_clustering, file)

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
        self.create_cluster_table(
            clusters_desc_table, rows, table_pickle_path, "personas"
        )

    def create_cluster_table(
        self, clusters_desc_table, rows, table_pickle_path, table_type
    ):
        clusters_desc_table = clusters_desc_table + rows
        t = AsciiTable(clusters_desc_table)
        t.inner_row_border = True
        print(t.table)
        print("\n\n")
        print("Saving table to file...")
        with open(table_pickle_path, "wb") as file:
            pickle.dump(clusters_desc_table, file)

        # Save the table in a CSV format for easy visualization in VSCode
        csv_file_path = (
            f"{os.getcwd()}/data/results/tables/cluster_results_table_{table_type}.csv"
        )
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(clusters_desc_table)
        print(f"Table also saved in CSV format at {csv_file_path}")

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
        print(len(rows))
        print(rows)
        for i in range(n_clusters):
            pos = lookup_cid_pos_in_rows(rows, i)
            if pos >= 0:
                cluster_labels.append(rows[pos][-1])
            else:
                cluster_labels.append("(Label missing)")

        all_cluster_sizes = []
        for i in range(n_clusters):
            print("Cluster", i)
            print(approvals_statements_and_embeddings)
            print("cluster_indices", clustering.labels_)
            inputs, model_approval_fractions = self.get_cluster_approval_stats(
                approvals_statements_and_embeddings, clustering.labels_, i
            )  # TODO: Fix this
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
        Z[:, 2] = np.arange(1.0, len(Z) + 1)

        return (
            Z,
            leaf_labels,
            original_cluster_sizes,
            merged_cluster_sizes,
            n_clusters,
        )

    def compile_cluster_table(
        self,
        clustering,
        approvals_statements_and_embeddings,
        all_model_info,
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
            print("Cluster", cluster_id)
            print(approvals_statements_and_embeddings)
            print("cluster_indices", clustering.labels_)
            inputs, model_approval_fractions = self.get_cluster_approval_stats(
                approvals_statements_and_embeddings, clustering.labels_, cluster_id
            )
            for frac in model_approval_fractions:
                row.append(str(round(100 * frac, 1)) + "%")
            cluster_inputs_themes = [
                identify_theme(
                    inputs,
                    all_model_info[i],
                    sampled_texts=10,
                    max_tokens=70,
                    temp=0.5,
                    instructions=theme_summary_instructions,
                )[:max_desc_length].replace("\n", " ")
                for i in range(len(all_model_info))
            ]
            inputs_themes_str = "\n".join(cluster_inputs_themes)

            row.append(inputs_themes_str)
            rows.append(row)
        rows = sorted(rows, key=lambda x: x[1], reverse=True)
        return rows
