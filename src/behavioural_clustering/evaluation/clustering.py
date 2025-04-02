import os
from pathlib import Path
import csv
import random
import time
from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
from tqdm import tqdm
import pickle
from terminaltables import AsciiTable
from prettytable import PrettyTable
import sklearn.metrics
import scipy.stats
from itertools import combinations
from sklearn.cluster import OPTICS, SpectralClustering, AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import linkage
import logging
from behavioural_clustering.models.model_factory import initialize_model
from behavioural_clustering.config.run_settings import RunSettings
from datetime import datetime
from behavioural_clustering.utils.embedding_data import EmbeddingEntry, JointEmbeddings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Clustering:
    def __init__(self, run_settings: RunSettings):
        self.clustering_settings = run_settings.clustering_settings
        self.algorithm_map = {
            "SpectralClustering": SpectralClustering,
            "KMeans": KMeans,
            "AgglomerativeClustering": AgglomerativeClustering,
            "OPTICS": OPTICS,
            "k-LLMmeans": "k-LLMmeans",  # Special handling in _run_single_clustering
            "SPILL": "SPILL",  # Special handling in _run_single_clustering
        }

    def cluster_embeddings(
        self,
        embeddings: np.ndarray,
        clustering_algorithm: Optional[str] = None,
        n_clusters: Optional[int] = None,
        multiple: bool = False,
        **kwargs
    ) -> Union[Dict[str, object], object]:
        clustering_algorithm = clustering_algorithm or self.clustering_settings.main_clustering_algorithm
        n_clusters = n_clusters or self.clustering_settings.n_clusters
        if multiple:
            return self._run_multiple_clustering(embeddings, n_clusters, **kwargs)
        else:
            return self._run_single_clustering(embeddings, clustering_algorithm, n_clusters, **kwargs)

    def _run_multiple_clustering(self, embeddings: np.ndarray, n_clusters: int, **kwargs) -> Dict[str, object]:
        return {
            alg: self._run_single_clustering(embeddings, alg, n_clusters, **kwargs)
            for alg in self.algorithm_map.keys()
        }

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
        
        if clustering_algorithm in ["k-LLMmeans", "SPILL"]:
            from behavioural_clustering.utils.clustering_algorithms import ClusteringFactory
            
            if n_clusters is not None:
                kwargs["n_clusters"] = n_clusters
                
            clustering_algo = ClusteringFactory.create_algorithm(clustering_algorithm, **kwargs)
            
            if "texts" in kwargs:
                labels = clustering_algo.fit(embeddings, kwargs["texts"])
            else:
                logger.warning(f"No texts provided for {clustering_algorithm}. Results may be less interpretable.")
                labels = clustering_algo.fit(embeddings)
                
            clustering = type('DummyModel', (), {'labels_': labels})
            
        else:
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
        self.theme_identification_model_info = {
            "model_name": self.run_settings.clustering_settings.theme_identification_model_name,
            "model_family": self.run_settings.clustering_settings.theme_identification_model_family,
            "system_message": self.run_settings.clustering_settings.theme_identification_system_message
        }

    def cluster_approval_stats(
        self,
        approvals_statements_and_embeddings: List,
        embeddings: Union[np.ndarray, List[np.ndarray]],
        model_info_list: List[Dict],
        prompt_dict: Dict,
        clusters_desc_table: List[List[str]],
        reuse_cluster_rows: bool = False,
    ) -> Path:
        
        prompt_approver_type = list(prompt_dict.keys())[0]
        prompt_labels = list(prompt_dict[prompt_approver_type].keys())
        response_types = ["approve", "disapprove"]

        pickle_base_path = self.run_settings.directory_settings.pickle_dir
        rows_pickle_path = os.path.join(pickle_base_path, f"rows_{prompt_approver_type}.pkl")
        clustering_pickle_path = os.path.join(pickle_base_path, f"clustering_{prompt_approver_type}.pkl")
        table_pickle_path = os.path.join(pickle_base_path, f"clusters_desc_table_{prompt_approver_type}.pkl")

        if reuse_cluster_rows and os.path.exists(rows_pickle_path):
            logger.info("Loading rows from file...")
            with open(rows_pickle_path, "rb") as file:
                rows = pickle.load(file)
        else:
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)

            for model_info in model_info_list:
                model_name = model_info['model_name']
                for response_type in response_types:
                    for i in range(len(prompt_labels)):
                        for j in range(i + 1, len(prompt_labels)):
                            print(f"{model_name}: {prompt_labels[i]} vs {prompt_labels[j]}")
                            self.compare_responses(
                                approvals_statements_and_embeddings,
                                [prompt_labels[i], prompt_labels[j]],
                                response_type,
                                model_name
                            )
                logger.info("\n")

            logger.info("Calculating rows...")
            clustering_result = self.clustering.cluster_embeddings(embeddings, n_clusters=self.run_settings.clustering_settings.n_clusters)
            rows = self.compile_cluster_table(
                clustering=clustering_result,
                data=approvals_statements_and_embeddings,
                model_info_list=model_info_list,
                data_type="approvals",
                max_desc_length=self.run_settings.prompt_settings.max_desc_length
            )

            with open(rows_pickle_path, "wb") as file:
                pickle.dump(rows, file)
            with open(clustering_pickle_path, "wb") as file:
                pickle.dump(clustering_result, file)

        header_labels = ["ID", "N"]
        for model_info in model_info_list:
            model_name = model_info['model_name']
            for prompt in prompt_dict[prompt_approver_type]:
                header_labels.append(f"{model_name} - {prompt}")
        header_labels.append("Inputs Themes")
        
        clusters_desc_table = [header_labels]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_names_str = "-".join([model_info["model_name"].replace("/", "_") for model_info in model_info_list])
        n_clusters = self.run_settings.clustering_settings.n_clusters
        n_statements = self.run_settings.data_settings.n_statements
        clustering_algorithm = self.run_settings.clustering_settings.main_clustering_algorithm

        filename = f"cluster_results_table_{prompt_approver_type}_{model_names_str}_{n_clusters}clusters_{n_statements}statements_{clustering_algorithm}_{timestamp}.csv"
        csv_file_path = self.run_settings.directory_settings.tables_dir / filename.replace("/", "_")

        self.create_cluster_table(
            clusters_desc_table, rows, table_pickle_path, csv_file_path
        )

        return csv_file_path

    def get_cluster_approval_stats(self, approvals_statements_and_embeddings, approvals_prompts_dict, cluster_labels, cluster_ID, model_name):
        inputs = []
        cluster_size = 0
        approval_counts = {prompt_name: 0 for prompt_name in approvals_prompts_dict.keys()}
        
        for e, l in zip(approvals_statements_and_embeddings, cluster_labels):
            if l != cluster_ID:
                continue
            for prompt_name, approval in e['approvals'][model_name].items():
                if approval == 1:
                    approval_counts[prompt_name] += 1
            cluster_size += 1
            inputs.append(e['statement'])
        
        approval_fractions = {
            prompt_name: count / cluster_size if cluster_size > 0 else 0 
            for prompt_name, count in approval_counts.items()
        }
        return inputs, approval_counts, approval_fractions, cluster_size
    
    def create_cluster_table(
        self,
        clusters_desc_table: List[List[str]],
        rows: List[Dict[str, Any]],
        table_pickle_path: str,
        csv_file_path: Path
    ) -> None:
        """
        Creates the final cluster table and writes it to disk. We no longer slice rows
        because rows is a list of dictionaries and slicing it would cause a type error.
        """
        # Combine the table headers with the actual row data. The row data is a list of dicts.
        # We do not attempt to slice or modify them based on columns.
        clusters_desc_table = clusters_desc_table + self._convert_rows_for_table(rows)
        t = AsciiTable(clusters_desc_table)
        t.inner_row_border = True
        logger.info(t.table)
        logger.info("\n")
        logger.info("Saving table to file...")
        with open(table_pickle_path, "wb") as file:
            pickle.dump(clusters_desc_table, file)

        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(clusters_desc_table)
        logger.info(f"Table also saved in CSV format at {csv_file_path}")

    def _convert_rows_for_table(self, rows: List[Dict[str, Any]]) -> List[List[str]]:
        """
        Convert the list of dictionaries (rows) into a list of lists that can be displayed as a table.
        We'll keep it simple: each row has ID, N, plus placeholders for model stats, and an 'Inputs Themes' field.
        """
        converted_rows = []
        for row_data in rows:
            # We'll try to read cluster_id, size, statements, responses, model_responses
            # but it may vary depending on how compile_cluster_table was invoked.
            cluster_id = str(row_data.get("cluster_id", ""))
            size = str(row_data.get("size", ""))

            # We might store model approval data or fractions. If absent, we fill with placeholders
            # for the user to customize. The user is free to expand or adapt to their table format.
            row_list = [cluster_id, size]

            # We do not attempt the slicing anymore. We'll just populate the row with placeholders
            # if we don't have exact data from row_data.
            row_list.extend(["N/A"] * (len(self.run_settings.model_info_list) * len(self.approval_prompts_in_row(row_data))))
            
            # For the "Inputs Themes" column
            inputs_sample = row_data.get("statements", [])
            if len(inputs_sample) > 3:
                inputs_sample = inputs_sample[:3]
            row_list.append("; ".join(str(s) for s in inputs_sample))

            converted_rows.append(row_list)

        return converted_rows

    def approval_prompts_in_row(self, row_data: Dict[str, Any]) -> List[str]:
        """
        Attempt to find the set of prompts used in row_data, if any. 
        For now, we return a placeholder list if we can't detect them.
        """
        return ["Bing Chat", "Bing Chat Emoji", "Bing Chat Janus"]


    def get_cluster_row(
        self,
        cluster_id: int,
        labels: np.ndarray,
        data: List,
        model_info_list: List[Dict],
        theme_identification_model_info: Dict,
        data_type: str = "joint_embeddings",
        include_responses_and_interactions: bool = True,
        max_desc_length: int = 250,
        run_settings: Optional[RunSettings] = None
    ) -> List[str]:
        try:
            row = [str(cluster_id)]
            cluster_indices = np.where(labels == cluster_id)[0]
            row.append(str(len(cluster_indices)))

            result = self.get_cluster_stats(
                data,
                labels,
                cluster_id,
                data_type=data_type,
                include_responses=include_responses_and_interactions if data_type == "joint_embeddings" else False,
            )
            
            if data_type == "joint_embeddings" and include_responses_and_interactions:
                if len(result) == 3:
                    inputs, responses, fractions = result
                    if isinstance(fractions, list):
                        for frac in fractions:
                            row.append(f"{round(100 * frac, 1)}%")
                else:
                    inputs, fractions = result
                    responses = []
            else:
                if len(result) == 2:
                    inputs, fractions = result
                    responses = None
                else:
                    inputs = result[0] if len(result) > 0 else []
                    fractions = result[1] if len(result) > 1 else {}
                    responses = None
                
                if isinstance(fractions, dict):
                    for model_name in fractions:
                        if isinstance(fractions[model_name], dict):
                            for prompt, frac in fractions[model_name].items():
                                row.append(f"{round(100 * frac, 1)}%")

            if inputs:
                inputs_theme = self.identify_theme(
                    texts=inputs,
                    theme_identification_model_info=theme_identification_model_info,
                    sampled_texts_count=5,
                    max_tokens=self.run_settings.model_settings.identify_theme_max_tokens,
                    max_total_tokens=self.run_settings.model_settings.identify_theme_max_total_tokens
                )
                inputs_theme = str(inputs_theme)[:max_desc_length] if inputs_theme else ""
                row.append(inputs_theme)
            else:
                row.append("No inputs available")

            if include_responses_and_interactions and responses:
                responses_theme = self.identify_theme(
                    texts=responses,
                    theme_identification_model_info=theme_identification_model_info,
                    sampled_texts_count=5,
                    max_tokens=self.run_settings.model_settings.identify_theme_max_tokens,
                    max_total_tokens=self.run_settings.model_settings.identify_theme_max_total_tokens
                )
                responses_theme = str(responses_theme)[:max_desc_length] if responses_theme else ""
                row.append(responses_theme)

                interactions = [
                    f'(Statement: "{input}", Response: "{response}")'
                    for input, response in zip(inputs, responses)
                ]
                interactions_theme = self.identify_theme(
                    texts=interactions,
                    theme_identification_model_info=theme_identification_model_info,
                    sampled_texts_count=5,
                    max_tokens=self.run_settings.model_settings.identify_theme_max_tokens,
                    max_total_tokens=self.run_settings.model_settings.identify_theme_max_total_tokens
                )
                interactions_theme = str(interactions_theme)[:max_desc_length] if interactions_theme else ""
                row.append(interactions_theme)

            return row

        except Exception as e:
            logger.error(f"Error in get_cluster_row for cluster {cluster_id}: {str(e)}")
            logger.error(f"Data type: {data_type}")
            logger.error(f"Sample data: {data[0] if data else 'No data'}")
            raise

    def get_cluster_stats(
        self,
        data: List,
        cluster_labels: np.ndarray,
        cluster_ID: int,
        data_type: str = "joint_embeddings",
        include_responses: bool = True,
    ) -> Union[
        Tuple[List[str], List[str], List[float]],  # joint_embeddings with responses and list fractions
        Tuple[List[str], List[str], Dict[str, Dict[str, float]]],  # joint_embeddings with responses and dict fractions
        Tuple[List[str], List[float]],  # joint_embeddings without responses or approvals with list fractions
        Tuple[List[str], Dict[str, Dict[str, float]]]  # approvals with dict fractions
    ]:
        """
        Get statistics for a specific cluster.
        
        Args:
            data: List of data entries (EmbeddingEntry or dict)
            cluster_labels: Array of cluster labels
            cluster_ID: ID of the cluster to analyze
            data_type: Type of data ("joint_embeddings" or "approvals")
            include_responses: Whether to include responses in the result
            
        Returns:
            For joint_embeddings with include_responses=True: (inputs, responses, fractions)
            For joint_embeddings with include_responses=False or approvals: (inputs, fractions)
        """
        inputs = []
        responses = []
        cluster_size = 0

        if data_type == "joint_embeddings":
            n_categories = max([e.model_idx for e in data]) + 1
            fractions = [0 for _ in range(n_categories)]
        elif data_type == "approvals":
            fractions = {}
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        for e, l in zip(data, cluster_labels):
            if l != cluster_ID:
                continue

            cluster_size += 1
            
            if data_type == "joint_embeddings":
                inputs.append(e.statement)
                if e.model_idx >= 0:
                    fractions[e.model_idx] += 1
                if include_responses:
                    responses.append(e.response)
            else:
                inputs.append(e.get('statement'))
                for model_name, approvals in e.get('approvals', {}).items():
                    if model_name not in fractions:
                        fractions[model_name] = {prompt: 0 for prompt in approvals}
                    for prompt, approval in approvals.items():
                        if approval == 1:
                            fractions[model_name][prompt] += 1

        if data_type == "joint_embeddings":
            if isinstance(fractions, list) and cluster_size > 0:
                fractions = [f / cluster_size for f in fractions]
        else:
            if isinstance(fractions, dict) and cluster_size > 0:
                for model_name in fractions:
                    if isinstance(fractions[model_name], dict):
                        for prompt in fractions[model_name]:
                            fractions[model_name][prompt] /= cluster_size

        inputs_str: List[str] = [str(i) for i in inputs]
        
        if data_type == "joint_embeddings" and include_responses:
            responses_str: List[str] = [str(r) for r in responses]
            if isinstance(fractions, list):
                fractions_float: List[float] = [float(f) for f in fractions]
                return inputs_str, responses_str, fractions_float
            else:
                return inputs_str, responses_str, fractions
        else:
            if isinstance(fractions, list):
                fractions_float: List[float] = [float(f) for f in fractions]
                return inputs_str, fractions_float
            else:
                return inputs_str, fractions

    def identify_theme(
        self,
        texts: List[str],
        theme_identification_model_info: Dict,
        sampled_texts_count: int = 5,
        max_tokens: Optional[int] = None,
        max_total_tokens: Optional[int] = None,
    ) -> str:
        max_tokens = max_tokens or self.run_settings.clustering_settings.theme_identification_max_tokens
        max_total_tokens = max_total_tokens or self.run_settings.clustering_settings.theme_identification_max_total_tokens
        prompt = self.run_settings.clustering_settings.theme_identification_prompt
        
        sample_size = min(len(texts), sampled_texts_count)
        sampled_texts = random.sample(texts, sample_size)
        
        theme_identify_prompt = prompt + "\n\n"
        for i, text in enumerate(sampled_texts):
            theme_identify_prompt += f"Text {i + 1}: {str(text)}\n"
        theme_identify_prompt += "\nTheme:"

        model_instance = initialize_model(
            theme_identification_model_info,
            temperature=self.run_settings.clustering_settings.theme_identification_temperature,
            max_tokens=max_tokens
        )

        for _ in range(20):
            try:
                start_time = time.time()
                while True:
                    try:
                        completion = model_instance.generate(theme_identify_prompt)
                        break
                    except Exception as e:
                        if time.time() - start_time > 20:
                            raise e
                        print(f"Exception: {type(e).__name__}, {str(e)}")
                        print("Retrying generation due to exception...")
                        time.sleep(2)
                return completion[:max_total_tokens].replace("\n", " ")
            except Exception as e:
                logger.error(f"API error: {e}")
                time.sleep(2)
        return "Failed to identify theme"

    def display_statement_themes(self, chosen_clustering, rows, model_info_list):
        logger.info("Displaying statement themes...")
        
        model_columns = [model_info["model_name"] for model_info in model_info_list]
        table_headers = (
            [
                "ID",
                "N",
            ]
            + model_columns
            + [
                "Inputs Themes",
                "Responses Themes",
                "Interaction Themes",
            ]
        )

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_names_str = "-".join([model_info["model_name"].replace("/", "_") for model_info in model_info_list])
            n_clusters = len(set(chosen_clustering.labels_))
            n_statements = len(rows)
            clustering_algorithm = self.run_settings.clustering_settings.main_clustering_algorithm

            filename = f"cluster_results_table_statement_responses_{model_names_str}_{n_clusters}clusters_{n_statements}statements_{clustering_algorithm}_{timestamp}.csv"

            csv_file_path = self.run_settings.directory_settings.tables_dir / filename.replace("/", "_")

            formatted_rows = []
            for row_data in rows:
                try:
                    formatted_row = [
                        str(row_data["cluster_id"]),
                        str(row_data["size"])
                    ]
                    
                    for model_name in model_columns:
                        responses = row_data["model_responses"].get(model_name, [])
                        if isinstance(responses, list):
                            formatted_row.append(str(len(responses)))
                        else:
                            formatted_row.append("0")
                    
                    statements = row_data.get("statements", [])
                    if statements:
                        sample_size = min(5, len(statements))
                        sampled_statements = random.sample(statements, sample_size)
                        formatted_row.append("; ".join(str(s) for s in sampled_statements))
                    else:
                        formatted_row.append("No statements")

                    responses = row_data.get("responses", [])
                    if responses:
                        sample_size = min(5, len(responses))
                        sampled_responses = random.sample(responses, sample_size)
                        formatted_row.append("; ".join(str(r) for r in sampled_responses))
                        
                        interactions = [
                            f'(Statement: "{s}", Response: "{r}")'
                            for s, r in zip(sampled_statements[:sample_size], sampled_responses)
                        ]
                        formatted_row.append("; ".join(interactions))
                    else:
                        formatted_row.extend(["No responses", "No interactions"])

                    formatted_rows.append(formatted_row)
                except Exception as e:
                    logger.error(f"Error formatting row {row_data.get('cluster_id', 'unknown')}: {str(e)}")
                    logger.error(f"Row data: {row_data}")
                    continue

            with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(table_headers)
                writer.writerows(formatted_rows)

            logger.info(f"Cluster results table saved to: {csv_file_path}")

            t = PrettyTable()
            t.field_names = table_headers
            for row in formatted_rows:
                display_row = [
                    str(cell)[:100] + "..." if len(str(cell)) > 100 else str(cell)
                    for cell in row
                ]
                t.add_row(display_row)
            print(t)
            
        except Exception as e:
            logger.error(f"Error in display_statement_themes: {str(e)}")
            logger.error(f"Rows sample: {rows[0] if rows else 'No rows'}")
            raise

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
                logger.info(f"============================================================\nPoint {i}: ({2 + joint_embeddings_all_llms[i]['model_num']})")
                logger.info(joint_embeddings_all_llms[i]['statement'])
                logger.info(joint_embeddings_all_llms[i]['response'])

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
                logger.info(approvals_statements_and_embeddings[i]["statement"])

    @staticmethod
    def lookup_cid_pos_in_rows(rows: List[List[str]], cid: int) -> int:
        for i, row in enumerate(rows):
            if int(row[0]) == cid:
                return i
        return -1

    def compare_responses(
        self,
        approvals_statements_and_embeddings: List,
        persona_names: List[str],
        response_type: str,
        model_name: str
    ) -> None:
        response_type_int = self.lookup_response_type_int(response_type)
        if response_type_int is None:
            raise ValueError(f"Invalid response type: {response_type}. Please use 'approve', 'disapprove', or 'no response'.")

        masks = [
            np.array([
                int(e['approvals'][model_name][persona_name] == response_type_int)
                for e in approvals_statements_and_embeddings
            ], dtype=np.int32)
            for persona_name in persona_names
        ]

        for name, mask in zip(persona_names, masks):
            logger.info(f'{model_name} - {name} "{response_type}" responses: {sum(mask)}')

        name1, name2 = persona_names
        logger.info(f'Intersection matrix for {model_name} - {name1} and {name2} "{response_type}" responses:')
        conf_matrix = sklearn.metrics.confusion_matrix(masks[0], masks[1])
        
        if conf_matrix.shape == (1, 2):
            conf_matrix = np.vstack([conf_matrix, [0, 0]])
        elif conf_matrix.shape == (2, 1):
            conf_matrix = np.hstack([conf_matrix, [[0], [0]]])
        elif conf_matrix.shape == (1, 1):
            conf_matrix = np.array([[conf_matrix[0, 0], 0], [0, 0]])
        
        conf_rows = [
            ["", f"Not In {name1}", f"In {name1}"],
            [f"Not In {name2}", conf_matrix[0][0], conf_matrix[1][0]],
            [f"In {name2}", conf_matrix[0][1], conf_matrix[1][1]]
        ]
        t = AsciiTable(conf_rows)
        t.inner_row_border = True
        logger.info(t.table)

        mask1 = masks[0].astype(np.float64)
        mask2 = masks[1].astype(np.float64)
        pearson_r = scipy.stats.pearsonr(mask1, mask2)
        
        if hasattr(pearson_r, 'correlation'):
            correlation = pearson_r.correlation
            pvalue = 0.0  # Default value
            pvalue = getattr(pearson_r, "pvalue", 0.0)
        else:
            correlation = pearson_r[0]
            pvalue = pearson_r[1]
            
        logger.info(f'Pearson correlation between "{response_type}" responses for {model_name} - {name1} and {name2}: {round(correlation, 5)}')
        logger.info(f"(P-value {round(pvalue, 5)})")

    @staticmethod
    def lookup_response_type_int(response_type: str) -> Optional[int]:
        response_type = str.lower(response_type)
        if response_type in ["approvals", "approval", "a", "approve"]:
            return 1
        elif response_type in ["disapprovals", "disapproval", "d", "disapprove"]:
            return 0
        elif response_type in ["no response", "no decision", "nr", "nd"]:
            return -1
        logger.warning(f"Invalid response type provided: {response_type}")
        return None

    def compile_cluster_table(
        self,
        clustering,
        data: Union[List[EmbeddingEntry], JointEmbeddings, List[Dict]],
        model_info_list: List[Dict],
        data_type: str = "joint_embeddings",
        max_desc_length: int = 250,
        run_settings: Optional[RunSettings] = None
    ) -> List[Dict[str, Any]]:
        """
        Compile cluster information into a table format.
        
        Args:
            clustering: The clustering object or labels
            data: The data that was clustered (EmbeddingEntry list, JointEmbeddings, or dict list)
            model_info_list: List of model information dictionaries
            data_type: Type of data being clustered ("joint_embeddings" or "approvals")
            max_desc_length: Maximum length for theme descriptions
            run_settings: Optional run settings
            
        Returns:
            List of dictionaries containing cluster information
        """
        if hasattr(clustering, 'labels_'):
            labels = clustering.labels_
        else:
            labels = clustering

        n_clusters = len(set(labels))
        rows = []

        # Convert JointEmbeddings to list if needed
        if isinstance(data, JointEmbeddings):
            data = data.get_all_embeddings()

        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_size = len(cluster_indices)
            
            cluster_data = {
                'cluster_id': cluster_id,
                'size': cluster_size,
                'statements': [],
                'responses': [],
                'model_responses': {model_info['model_name']: [] for model_info in model_info_list}
            }

            if data_type == "joint_embeddings":
                for idx in cluster_indices:
                    entry = data[idx]
                    if isinstance(entry, EmbeddingEntry):
                        cluster_data['statements'].append(entry.statement)
                        cluster_data['responses'].append(entry.response)
                        if entry.model_name in cluster_data['model_responses']:
                            cluster_data['model_responses'][entry.model_name].append(entry.response)
                    elif isinstance(entry, dict):
                        cluster_data['statements'].append(entry.get('statement', ''))
                        cluster_data['responses'].append(entry.get('response', ''))
                        model_name = entry.get('model_name', '')
                        if model_name and model_name in cluster_data['model_responses']:
                            cluster_data['model_responses'][model_name].append(entry.get('response', ''))
            else:  # approvals data type
                for idx in cluster_indices:
                    entry = data[idx]
                    if isinstance(entry, dict):
                        cluster_data['statements'].append(entry.get('statement', ''))
                        for model_name in cluster_data['model_responses'].keys():
                            if 'approvals' in entry and model_name in entry['approvals']:
                                cluster_data['model_responses'][model_name].append(entry['approvals'][model_name])

            rows.append(cluster_data)

        return rows

    def calculate_hierarchical_cluster_data(
        self,
        clustering,
        data: Union[List[Dict], List[EmbeddingEntry]],
        cluster_rows: List[Dict],
        model_name: str
    ) -> Dict[str, Any]:
        """
        Calculate hierarchical clustering data for visualization.
        
        Args:
            clustering: The clustering object or labels
            data: The data that was clustered
            cluster_rows: Previously compiled cluster information
            model_name: Name of the model to analyze
            
        Returns:
            Dictionary containing hierarchical clustering information
        """
        if hasattr(clustering, 'labels_'):
            labels = clustering.labels_
        else:
            labels = clustering

        n_clusters = len(set(labels))
        
        # Create hierarchical structure
        hierarchy_data = {
            "name": "root",
            "children": []
        }
        
        # Process each cluster
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            
            # Find the corresponding cluster row
            cluster_row = next(
                (row for row in cluster_rows if row['cluster_id'] == cluster_id),
                None
            )
            
            if not cluster_row:
                continue
                
            # Calculate cluster statistics
            cluster_size = len(cluster_indices)
            if cluster_size == 0:
                continue
                
            # Get responses/approvals for this cluster
            responses = cluster_row['model_responses'].get(model_name, [])
            if not responses:
                continue
                
            # Calculate approval statistics if they exist
            approval_stats = {}
            if isinstance(responses[0], dict) and 'approvals' in responses[0]:
                for resp in responses:
                    for prompt, value in resp['approvals'].items():
                        if prompt not in approval_stats:
                            approval_stats[prompt] = {'approve': 0, 'disapprove': 0, 'no_response': 0}
                        if value == 1:
                            approval_stats[prompt]['approve'] += 1
                        elif value == 0:
                            approval_stats[prompt]['disapprove'] += 1
                        else:
                            approval_stats[prompt]['no_response'] += 1
            
            # Create cluster node
            cluster_node = {
                "name": f"Cluster {cluster_id}",
                "size": cluster_size,
                "statements": cluster_row['statements'][:5],  # Sample of statements
                "approval_stats": approval_stats
            }
            
            hierarchy_data["children"].append(cluster_node)
        
        return hierarchy_data
