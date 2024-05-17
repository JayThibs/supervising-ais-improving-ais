import os
import csv
import numpy as np
from tqdm import tqdm
import pdb
import pickle
from terminaltables import AsciiTable
from sklearn.cluster import OPTICS, SpectralClustering, AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
from utils import lookup_cid_pos_in_rows, identify_theme, compare_response_pair
from config.run_settings import ClusteringSettings


class Clustering:
    def __init__(self, clustering_settings: ClusteringSettings):
        self.settings = clustering_settings

    def cluster_embeddings(
        self,
        embeddings,
        clustering_algorithm="SpectralClustering",
        n_clusters=None,
        multiple=False,
        **kwargs,
    ):
        if multiple:
            cluster_algorithms = {
                "OPTICS": OPTICS(min_samples=2, xi=0.12),
                "SpectralClustering": SpectralClustering(
                    self.settings.n_clusters if not self.settings.test_mode else 10,
                    random_state=42,
                ),
                "AgglomerativeClustering": AgglomerativeClustering(
                    self.settings.n_clusters if not self.settings.test_mode else 10
                ),
                "KMeans": KMeans(
                    n_clusters=(
                        self.settings.n_clusters if not self.settings.test_mode else 10
                    ),
                    random_state=42,
                ),
                # Add more as needed
            }
            clustering_results = {}
            for name, algorithm in cluster_algorithms.items():
                print(f"Running {name} clustering...")
                clustering_results[name] = algorithm.fit(embeddings)
            return clustering_results
        else:
            print(f"Running {clustering_algorithm} on embeddings...")
            if clustering_algorithm == "SpectralClustering":
                clustering = SpectralClustering(
                    n_clusters=n_clusters or self.settings.n_clusters,
                    random_state=42,
                    **kwargs,
                ).fit(embeddings)
            elif clustering_algorithm == "KMeans":
                clustering = KMeans(
                    n_clusters=n_clusters or self.settings.n_clusters,
                    random_state=42,
                    **kwargs,
                ).fit(embeddings)
            elif clustering_algorithm == "AgglomerativeClustering":
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters or self.settings.n_clusters, **kwargs
                ).fit(embeddings)
            elif clustering_algorithm == "OPTICS":
                clustering = OPTICS(**kwargs).fit(embeddings)
            else:
                raise ValueError(
                    f"Unsupported clustering algorithm: {clustering_algorithm}"
                )

            print("clustering.labels_", clustering.labels_)
            print("n_clusters:", n_clusters or "Auto")

            return clustering

    def get_cluster_centroids(self, embeddings, cluster_labels):
        centroids = []
        for i in range(max(cluster_labels) + 1):
            c = np.mean(embeddings[cluster_labels == i], axis=0).tolist()
            centroids.append(c)
        return np.array(centroids)

    def cluster_approval_stats(
        self,
        approvals_statements_and_embeddings,
        statement_clustering,
        model_info_list,
        prompt_dict,
        reuse_cluster_rows=False,
    ):
        # Calculating the confusion matrix and pearson correlation between responses
        prompt_approver_type = list(prompt_dict.keys())[0]
        prompt_labels = list(prompt_dict[prompt_approver_type].keys())
        response_types = ["approve", "disapprove"]

        for response_type in response_types:
            for i in range(len(prompt_labels)):
                for j in range(i + 1, len(prompt_labels)):
                    compare_response_pair(
                        approvals_statements_and_embeddings,
                        prompt_labels[i],
                        prompt_labels[j],
                        prompt_labels,
                        response_type,
                    )
            print("\n\n")

        pickle_base_path = f"{os.getcwd()}/data/results/pickle_files"
        rows_pickle_path = f"{pickle_base_path}/rows_{prompt_approver_type}.pkl"
        clustering_pickle_path = (
            f"{pickle_base_path}/clustering_{prompt_approver_type}.pkl"
        )
        table_pickle_path = (
            f"{pickle_base_path}/clusters_desc_table_{prompt_approver_type}.pkl"
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
                model_info_list,
                theme_summary_instructions="Briefly list the common themes of the following texts:",
                max_desc_length=250,
            )

            with open(rows_pickle_path, "wb") as file:
                pickle.dump(rows, file)
            with open(clustering_pickle_path, "wb") as file:
                pickle.dump(statement_clustering, file)

        header_labels = ["ID", "N"] + prompt_labels + ["Inputs Themes"]
        clusters_desc_table = [header_labels]
        self.create_cluster_table(
            clusters_desc_table, rows, table_pickle_path, prompt_approver_type
        )

    def create_cluster_table(
        self, clusters_desc_table, rows, table_pickle_path, prompt_approver_type
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
        csv_file_path = f"{os.getcwd()}/data/results/tables/cluster_results_table_{prompt_approver_type}.csv"
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

    def compile_cluster_table(self, clustering, data, data_type, model_info_list):
        """Compiles a table of cluster statistics and themes.

        Args:
            clustering: A clustering object.
            data: The data used for clustering. This can be joint_embeddings_all_llms or approvals_statements_and_embeddings.
            data_type: The type of data used for clustering. Either "joint_embeddings" or "approvals".
            model_info_list: A list of dictionaries containing information about each model.

        Returns:
            A list of lists where each row represents a cluster's statistics and themes.
        """
        include_responses_and_interactions = data_type in ["joint_embeddings"]

        rows = []
        n_clusters = max(clustering.labels_) + 1
        print(f"n_clusters: {n_clusters}")

        print("Compiling cluster table...")
        for cluster_id in tqdm(range(n_clusters)):
            print(f"Processing cluster {cluster_id}...")
            row = self.get_cluster_row(
                cluster_id,
                clustering.labels_,
                data,
                model_info_list,
                data_type=data_type,
                include_responses_and_interactions=include_responses_and_interactions,
            )
            rows.append(row)

        rows = sorted(rows, key=lambda x: x[1], reverse=True)
        print(f"Cluster table rows: {rows}")
        return rows

    def get_cluster_row(
        self,
        cluster_id,
        labels,
        data,  # joint_embeddings_all_llms or approvals_statements_and_embeddings
        model_info_list,
        data_type="joint_embeddings",  # or "approvals"
        include_responses_and_interactions=True,
    ):
        """Generates a row that represents a cluster's statistics and themes."""
        row = [str(cluster_id)]

        # Find the indices of items in this cluster
        cluster_indices = np.where(labels == cluster_id)[0]
        row.append(len(cluster_indices))  # Number of items in the cluster

        # Extract the inputs, responses (if applicable), and fractions of each model/approval prompt in the cluster
        if data_type == "joint_embeddings":
            # fractions are the fraction of each model in the cluster
            inputs, responses, fractions = self.get_cluster_stats(
                data,
                labels,
                cluster_id,
                data_type=data_type,
                include_responses=include_responses_and_interactions,
            )
        else:  # data_type == "approvals"
            # fractions are the fraction of each approval prompt in the cluster
            inputs, fractions = self.get_cluster_stats(
                data, labels, cluster_id, data_type=data_type
            )
            responses = None

        # Add the fractions to the row
        for frac in fractions:
            row.append(f"{round(100 * frac, 1)}%")

        # Identify themes within this cluster
        n_llms = len(model_info_list)
        for i in range(n_llms):  # loop through llms
            print(f"Identifying themes for LLM {i}...")
            model_info = model_info_list[i]
            print(f"inputs: {inputs}")
            print(f"model_info: {model_info}")
            inputs_themes_str = identify_theme(inputs, model_info)

            # Add input themes to the row
            row.append(inputs_themes_str)

            if include_responses_and_interactions and responses is not None:
                print(f"responses: {responses}")
                responses_themes_str = identify_theme(responses, model_info)

                interactions = [
                    f'(Statement: "{input}", Response: "{response}")'
                    for input, response in zip(inputs, responses)
                ]
                interactions_themes_str = identify_theme(interactions, model_info)

                print(f"interactions: {interactions}")

                # Add response and interaction themes to the row
                row.append(responses_themes_str)
                row.append(interactions_themes_str)

        print(f"Cluster row {cluster_id}: {row}")
        return row

    def get_cluster_stats(
        self,
        data,
        cluster_labels,
        cluster_ID,
        data_type="joint_embeddings",
        include_responses=True,
    ):
        """
        Analyzes a cluster and extracts aggregated statistics.

        data_type can be "joint_embeddings" or "approvals".
        """
        inputs = []
        responses = []
        cluster_size = 0

        if data_type == "joint_embeddings":
            n_categories = int(max([e[0] for e in data])) + 1  # number of models
        elif data_type == "approvals":
            n_categories = len(
                list(data[0][0].values())[0]
            )  # number of approval prompts
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        fractions = [0 for _ in range(n_categories)]

        for e, l in zip(data, cluster_labels):
            if l != cluster_ID:
                continue

            cluster_size += 1
            inputs.append(e[1])

            if data_type == "joint_embeddings":
                if e[0] >= 0:
                    fractions[e[0]] += 1
                if include_responses:
                    responses.append(e[2])
            elif data_type == "approvals":
                for i in range(n_categories):
                    if e[0][i] == 1:
                        fractions[i] += 1

        fractions = [f / cluster_size for f in fractions]

        if data_type == "joint_embeddings" and include_responses:
            return inputs, responses, fractions
        else:
            return inputs, fractions

    # def compile_cluster_table(
    #     self,
    #     clustering,
    #     approvals_statements_and_embeddings,
    #     model_info_list,
    #     theme_summary_instructions="Briefly list the common themes of the following texts:",
    #     max_desc_length=250,
    # ):
    #     rows = []
    #     n_clusters = max(clustering.labels_) + 1
    #     print("cluster_indices", clustering.labels_)

    #     for cluster_id in tqdm(range(n_clusters)):
    #         print("Cluster", cluster_id)
    #         row = [str(cluster_id)]
    #         cluster_indices = np.arange(len(clustering.labels_))[
    #             clustering.labels_ == cluster_id
    #         ]
    #         row.append(len(cluster_indices))
    #         inputs, model_approval_fractions = self.get_cluster_stats(
    #             approvals_statements_and_embeddings,
    #             clustering.labels_,
    #             cluster_id,
    #             data_type="approvals",
    #             include_responses=False,
    #         )
    #         for frac in model_approval_fractions:
    #             row.append(str(round(100 * frac, 1)) + "%")
    #         cluster_inputs_themes = [
    #             identify_theme(
    #                 inputs,
    #                 model_info_list[i],
    #                 sampled_texts=10,
    #                 max_tokens=70,
    #                 temp=0.5,
    #                 instructions=theme_summary_instructions,
    #             )[:max_desc_length].replace("\n", " ")
    #             for i in range(len(model_info_list))
    #         ]
    #         inputs_themes_str = "\n".join(cluster_inputs_themes)

    #         row.append(inputs_themes_str)
    #         rows.append(row)
    #     rows = sorted(rows, key=lambda x: x[1], reverse=True)
    #     return rows
