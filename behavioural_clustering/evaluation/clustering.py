import os
import csv
import yaml
import random
import time
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm
import pickle
from terminaltables import AsciiTable
import sklearn.metrics
import scipy.stats
from sklearn.cluster import OPTICS, SpectralClustering, AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import linkage
import logging
from utils.model_utils import initialize_model
from config.run_settings import RunSettings, ClusteringSettings

logger = logging.getLogger(__name__)

class Clustering:
    def __init__(self, run_settings: RunSettings):
        self.settings = run_settings.clustering_settings
        self.algorithm_map = {
            "SpectralClustering": SpectralClustering,
            "KMeans": KMeans,
            "AgglomerativeClustering": AgglomerativeClustering,
            "OPTICS": OPTICS,
        }

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        clustering_algorithm: str = None,
        n_clusters: Optional[int] = None,
        multiple: bool = False,
        **kwargs
    ) -> Union[Dict[str, object], object]:
        clustering_algorithm = clustering_algorithm or self.settings.main_clustering_algorithm
        n_clusters = n_clusters or self.settings.n_clusters
        if multiple:
            return self._run_multiple_clustering(embeddings, n_clusters, **kwargs)
        else:
            return self._run_single_clustering(embeddings, clustering_algorithm, n_clusters, **kwargs)

    def _run_multiple_clustering(self, embeddings: np.ndarray, n_clusters: int, **kwargs) -> Dict[str, object]:
        return {alg: self._run_single_clustering(embeddings, alg, n_clusters, **kwargs) for alg in self.algorithm_map.keys()}

    def _run_single_clustering(
        self, 
        embeddings: np.ndarray, 
        clustering_algorithm: str, 
        n_clusters: Optional[int], 
        **kwargs
    ) -> object:
        if clustering_algorithm not in self.algorithm_map:
            raise ValueError(f"Unsupported clustering algorithm: {clustering_algorithm}")
        
        algorithm_class = self.algorithm_map[clustering_algorithm]
        
        if clustering_algorithm != "OPTICS":
            kwargs["n_clusters"] = n_clusters
        
        clustering = algorithm_class(**kwargs).fit(embeddings)
        
        logger.info(f"Clustering completed using {clustering_algorithm}")
        logger.info(f"Number of clusters: {n_clusters if n_clusters is not None else 'auto'}")
        
        return clustering

    @staticmethod
    def get_cluster_centroids(embeddings: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        centroids = []
        for i in range(max(cluster_labels) + 1):
            c = np.mean(embeddings[cluster_labels == i], axis=0).tolist()
            centroids.append(c)
        return np.array(centroids)


class ClusterAnalyzer:
    def __init__(self, run_settings: RunSettings):
        self.run_settings = run_settings
        self.clustering = Clustering(run_settings)

    def cluster_approval_stats(
        self,
        approvals_statements_and_embeddings: List,
        embeddings: np.ndarray,
        model_info_list: List[Dict],
        prompt_dict: Dict,
        reuse_cluster_rows: bool = False,
    ) -> None:
        prompt_approver_type = list(prompt_dict.keys())[0]
        prompt_labels = list(prompt_dict[prompt_approver_type].keys())
        response_types = ["approve", "disapprove"]

        for response_type in response_types:
            for i in range(len(prompt_labels)):
                for j in range(i + 1, len(prompt_labels)):
                    self.compare_response_pair(
                        approvals_statements_and_embeddings,
                        prompt_labels[i],
                        prompt_labels[j],
                        prompt_labels,
                        response_type,
                    )
            logger.info("\n")

        pickle_base_path = self.run_settings.directory_settings.pickle_dir
        rows_pickle_path = os.path.join(pickle_base_path, f"rows_{prompt_approver_type}.pkl")
        clustering_pickle_path = os.path.join(pickle_base_path, f"clustering_{prompt_approver_type}.pkl")
        table_pickle_path = os.path.join(pickle_base_path, f"clusters_desc_table_{prompt_approver_type}.pkl")

        if reuse_cluster_rows and os.path.exists(rows_pickle_path):
            logger.info("Loading rows from file...")
            with open(rows_pickle_path, "rb") as file:
                rows = pickle.load(file)
        else:
            logger.info("Calculating rows...")
            clustering_result = self.clustering.cluster_embeddings(embeddings)
            rows = self.compile_cluster_table(
                clustering_result,
                approvals_statements_and_embeddings,
                model_info_list,
                theme_summary_instructions=self.run_settings.prompt_settings.theme_summary_instructions,
                max_desc_length=self.run_settings.prompt_settings.max_desc_length
            )

            with open(rows_pickle_path, "wb") as file:
                pickle.dump(rows, file)
            with open(clustering_pickle_path, "wb") as file:
                pickle.dump(clustering_result, file)

        header_labels = ["ID", "N"] + prompt_labels + ["Inputs Themes"]
        clusters_desc_table = [header_labels]
        self.create_cluster_table(
            clusters_desc_table, rows, table_pickle_path, prompt_approver_type, self.run_settings
        )

    @staticmethod
    def create_cluster_table(
        clusters_desc_table: List[List[str]],
        rows: List[List[str]],
        table_pickle_path: str,
        prompt_approver_type: str,
        run_settings: RunSettings
    ) -> None:
        clusters_desc_table = clusters_desc_table + rows
        t = AsciiTable(clusters_desc_table)
        t.inner_row_border = True
        logger.info(t.table)
        logger.info("\n")
        logger.info("Saving table to file...")
        with open(table_pickle_path, "wb") as file:
            pickle.dump(clusters_desc_table, file)

        csv_file_path = os.path.join(run_settings.directory_settings.tables_dir, f"cluster_results_table_{prompt_approver_type}.csv")
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(clusters_desc_table)
        logger.info(f"Table also saved in CSV format at {csv_file_path}")

    @staticmethod
    def lookup_name_index(labels: List[str], name: str) -> Optional[int]:
        for i, l in enumerate(labels):
            if name == l:
                return i
        logger.warning(f"Invalid name provided: {name}")
        return None

    @staticmethod
    def lookup_response_type_int(response_type: str) -> Optional[int]:
        response_type = str.lower(response_type)
        if response_type in ["approve", "approval", "a"]:
            return 1
        elif response_type in ["disapprove", "disapproval", "d"]:
            return 0
        elif response_type in ["no response", "no decision", "nr", "nd"]:
            return -1
        logger.warning(f"Invalid response type provided: {response_type}")
        return None

    @staticmethod
    def lookup_cid_pos_in_rows(rows: List[List[str]], cid: int) -> int:
        for i, row in enumerate(rows):
            if int(row[0]) == cid:
                return i
        return -1

    def compare_response_pair(
        self,
        approvals_statements_and_embeddings: List,
        r_1_name: str,
        r_2_name: str,
        labels: List[str],
        response_type: str
    ) -> None:
        response_type_int = self.lookup_response_type_int(response_type)
        r_1_index = self.lookup_name_index(labels, r_1_name)
        r_2_index = self.lookup_name_index(labels, r_2_name)

        if response_type_int is None or r_1_index is None or r_2_index is None:
            return

        r_1_mask = np.array([e[0][r_1_index] == response_type_int for e in approvals_statements_and_embeddings])
        r_2_mask = np.array([e[0][r_2_index] == response_type_int for e in approvals_statements_and_embeddings])

        logger.info(f'{r_1_name} "{response_type}" responses: {sum(r_1_mask)}')
        logger.info(f'{r_2_name} "{response_type}" responses: {sum(r_2_mask)}')

        logger.info(f'Intersection matrix for {r_1_name} and {r_2_name} "{response_type}" responses:')
        conf_matrix = sklearn.metrics.confusion_matrix(r_1_mask, r_2_mask)
        conf_rows = [["", f"Not In {r_1_name}", f"In {r_1_name}"]]
        conf_rows.append([f"Not In {r_2_name}", conf_matrix[0][0], conf_matrix[1][0]])
        conf_rows.append([f"In {r_2_name}", conf_matrix[0][1], conf_matrix[1][1]])
        t = AsciiTable(conf_rows)
        t.inner_row_border = True
        logger.info(t.table)

        pearson_r = scipy.stats.pearsonr(r_1_mask, r_2_mask)
        logger.info(f'Pearson correlation between "{response_type}" responses for {r_1_name} and {r_2_name}: {round(pearson_r.correlation, 5)}')
        logger.info(f"(P-value {round(pearson_r.pvalue, 5)})")

    def calculate_hierarchical_cluster_data(
        self,
        clustering: object,
        approvals_statements_and_embeddings: List,
        rows: List[List[str]]
    ) -> Tuple[np.ndarray, List[str], List[List[int]], List[List[int]], int]:
        statement_embeddings = np.array([e[2] for e in approvals_statements_and_embeddings])
        centroids = Clustering.get_cluster_centroids(statement_embeddings, clustering.labels_)
        Z = linkage(centroids, "ward")

        n_clusters = max(clustering.labels_) + 1
        cluster_labels = []
        for i in range(n_clusters):
            pos = self.lookup_cid_pos_in_rows(rows, i)
            cluster_labels.append(rows[pos][-1] if pos >= 0 else "(Label missing)")

        all_cluster_sizes = []
        for i in range(n_clusters):
            inputs, model_approval_fractions = self.get_cluster_approval_stats(
                approvals_statements_and_embeddings, clustering.labels_, i
            )
            n = len(inputs)
            all_cluster_sizes.append([n] + [int(f * n) for f in model_approval_fractions])

        for merge in Z:
            m1, m2 = int(merge[0]), int(merge[1])
            merged_sizes = [
                int(m1_entry + m2_entry)
                for m1_entry, m2_entry in zip(all_cluster_sizes[m1], all_cluster_sizes[m2])
            ]
            all_cluster_sizes.append(merged_sizes)

        original_cluster_sizes = all_cluster_sizes[:n_clusters]
        merged_cluster_sizes = all_cluster_sizes[n_clusters:]
        leaf_labels = [
            f"{s[0]} : {l}"
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
        clustering: object,
        data: List,
        model_info_list: List[Dict],
        data_type: str = "joint_embeddings",
        theme_summary_instructions: str = "Briefly list the common themes of the following texts:",
        max_desc_length: int = 250
    ) -> List[List[str]]:
        include_responses_and_interactions = data_type in ["joint_embeddings"]

        rows = []
        n_clusters = max(clustering.labels_) + 1
        logger.info(f"n_clusters: {n_clusters}")

        logger.info("Compiling cluster table...")
        for cluster_id in tqdm(range(n_clusters)):
            logger.info(f"Processing cluster {cluster_id}...")
            row = self.get_cluster_row(
                cluster_id,
                clustering.labels_,
                data,
                model_info_list,
                data_type=data_type,
                include_responses_and_interactions=include_responses_and_interactions,
                theme_summary_instructions=theme_summary_instructions,
                max_desc_length=max_desc_length
            )
            rows.append(row)

        rows = sorted(rows, key=lambda x: x[1], reverse=True)
        logger.info(f"Cluster table rows: {rows}")
        return rows

    def get_cluster_row(
        self,
        cluster_id: int,
        labels: np.ndarray,
        data: List,
        model_info_list: List[Dict],
        data_type: str = "joint_embeddings",
        include_responses_and_interactions: bool = True,
        theme_summary_instructions: str = "Briefly list the common themes of the following texts:",
        max_desc_length: int = 250
    ) -> List[str]:
        row = [str(cluster_id)]

        cluster_indices = np.where(labels == cluster_id)[0]
        row.append(len(cluster_indices))

        if data_type == "joint_embeddings":
            inputs, responses, fractions = self.get_cluster_stats(
                data,
                labels,
                cluster_id,
                data_type=data_type,
                include_responses=include_responses_and_interactions,
            )
        else:  # data_type == "approvals"
            inputs, fractions = self.get_cluster_stats(
                data, labels, cluster_id, data_type=data_type
            )
            responses = None

        for frac in fractions:
            row.append(f"{round(100 * frac, 1)}%")

        for model_info in model_info_list:
            inputs_themes_str = self.identify_theme(
                inputs, model_info, self.run_settings, instructions=theme_summary_instructions
            )[:max_desc_length]
            row.append(inputs_themes_str)

            if include_responses_and_interactions and responses is not None:
                responses_themes_str = self.identify_theme(
                    responses, model_info, self.run_settings, instructions=theme_summary_instructions
                )[:max_desc_length]

                interactions = [
                    f'(Statement: "{input}", Response: "{response}")'
                    for input, response in zip(inputs, responses)
                ]
                interactions_themes_str = self.identify_theme(
                    interactions, model_info, self.run_settings, instructions=theme_summary_instructions
                )[:max_desc_length]

                row.append(responses_themes_str)
                row.append(interactions_themes_str)

        return row

    def get_cluster_stats(
        self,
        data: List,
        cluster_labels: np.ndarray,
        cluster_ID: int,
        data_type: str = "joint_embeddings",
        include_responses: bool = True,
    ) -> Union[Tuple[List[str], List[str], List[float]], Tuple[List[str], List[float]]]:
        inputs = []
        responses = []
        cluster_size = 0

        if data_type == "joint_embeddings":
            n_categories = int(max([e[0] for e in data])) + 1
        elif data_type == "approvals":
            n_categories = len(list(data[0][0].values())[0])
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

    @staticmethod
    def identify_theme(
        texts: List[str],
        model_info: Dict,
        run_settings: RunSettings,
        sampled_texts: int = 5,
        temp: float = 0.5,
        max_tokens: int = 70,
        max_total_tokens: int = 250,    
        instructions: str = None,
    ) -> str:
        instructions = instructions or run_settings.prompt_settings.theme_summary_instructions
        model_info["system_message"] = ""
        sampled_texts = random.sample(texts, min(len(texts), sampled_texts))
        theme_identify_prompt = instructions + "\n\n"
        for i, text in enumerate(sampled_texts):
            theme_identify_prompt += f"Text {i + 1}: {str(text)}\n"
        theme_identify_prompt += "\nTheme:"
        model_instance = initialize_model(model_info, temp, max_tokens)
        for _ in range(20):
            try:
                completion = model_instance.generate(theme_identify_prompt)[:max_total_tokens].replace("\n", " ")
                return completion
            except Exception as e:
                logger.error(f"API error: {e}")
                time.sleep(2)
        return "Failed to identify theme"

    @staticmethod
    def print_cluster(
        cid: int,
        labels: np.ndarray,
        joint_embeddings_all_llms: List,
        rows: Optional[List[List[str]]] = None
    ) -> None:
        logger.info("####################################################################################")
        if rows is not None:
            cid_pos = ClusterAnalyzer.lookup_cid_pos_in_rows(rows, cid)
            logger.info("### Input desc:")
            logger.info(rows[cid_pos][4])
            logger.info("### Response desc:")
            logger.info(rows[cid_pos][5])
            logger.info("### Interaction desc:")
            logger.info(rows[cid_pos][6])
            logger.info(f"{rows[cid_pos][2]} / {rows[cid_pos][3]}")
        for i, label in enumerate(labels):
            if label == cid:
                logger.info(f"============================================================\nPoint {i}: ({2 + joint_embeddings_all_llms[i][0]})")
                logger.info(joint_embeddings_all_llms[i][1])
                logger.info(joint_embeddings_all_llms[i][2])

    @staticmethod
    def print_cluster_approvals(
        cid: int,
        labels: np.ndarray,
        approvals_statements_and_embeddings: List,
        rows: Optional[List[List[str]]] = None
    ) -> None:
        logger.info("####################################################################################")
        if rows is not None:
            cid_pos = ClusterAnalyzer.lookup_cid_pos_in_rows(rows, cid)
            logger.info("### Input desc:")
            logger.info(rows[cid_pos][-1])
            logger.info(rows[cid_pos][1:-1])
        for i, label in enumerate(labels):
            if label == cid:
                logger.info(f"============================================================\nPoint {i}:")
                logger.info(approvals_statements_and_embeddings[i][1])
