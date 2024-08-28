import os
import csv
import random
import time
from typing import List, Dict, Optional, Tuple, Union
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # This will output to console
        # Uncomment the next line to also log to a file
        # logging.FileHandler("my_log_file.log")
    ]
)
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
        self.settings = run_settings
        self.clustering = Clustering(run_settings)
        self.theme_identification_model_info = {}
        self.theme_identification_model_info["model_name"] = self.settings.clustering_settings.theme_identification_model_name
        self.theme_identification_model_info["model_family"] = self.settings.clustering_settings.theme_identification_model_family
        self.theme_identification_model_info["system_message"] = self.settings.clustering_settings.theme_identification_system_message

    def cluster_approval_stats(
        self,
        approvals_statements_and_embeddings: List,
        embeddings: Union[np.ndarray, List[np.ndarray]],
        model_info_list: List[Dict],
        prompt_dict: Dict,
        reuse_cluster_rows: bool = False,
    ) -> None:
        
        prompt_approver_type = list(prompt_dict.keys())[0]
        prompt_labels = list(prompt_dict[prompt_approver_type].keys())
        response_types = ["approve", "disapprove"]

        pickle_base_path = self.settings.directory_settings.pickle_dir
        rows_pickle_path = os.path.join(pickle_base_path, f"rows_{prompt_approver_type}.pkl")
        clustering_pickle_path = os.path.join(pickle_base_path, f"clustering_{prompt_approver_type}.pkl")
        table_pickle_path = os.path.join(pickle_base_path, f"clusters_desc_table_{prompt_approver_type}.pkl")

        if reuse_cluster_rows and os.path.exists(rows_pickle_path):
            logger.info("Loading rows from file...")
            with open(rows_pickle_path, "rb") as file:
                rows = pickle.load(file)
        else:
            # Convert embeddings to numpy array if it's a list
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)

            for response_type in response_types:
                for model_info in model_info_list:
                    model_name = model_info['model_name']
                    for i in range(len(prompt_labels)):
                        for j in range(i + 1, len(prompt_labels)):
                            print(f"{model_name}: {prompt_labels[i]} vs {prompt_labels[j]}")
                            self.compare_responses(
                                approvals_statements_and_embeddings,
                                [prompt_labels[i], prompt_labels[j]],
                                response_type,
                            )
                logger.info("\n")
            logger.info("Calculating rows...")
            clustering_result = self.clustering.cluster_embeddings(embeddings, n_clusters=self.settings.clustering_settings.n_clusters)
            rows = self.compile_cluster_table(
                clustering=clustering_result,
                data=approvals_statements_and_embeddings,
                model_info_list=model_info_list,
                data_type="approvals",
                max_desc_length=self.settings.prompt_settings.max_desc_length
            )

            with open(rows_pickle_path, "wb") as file:
                pickle.dump(rows, file)
            with open(clustering_pickle_path, "wb") as file:
                pickle.dump(clustering_result, file)

        header_labels = ["ID", "N"]
        for model_name in prompt_dict:
            header_labels.extend([f"{model_name} - {prompt}" for prompt in prompt_dict[model_name]])
        header_labels.append("Inputs Themes")
        
        clusters_desc_table = [header_labels]
        self.create_cluster_table(
            clusters_desc_table, rows, table_pickle_path, prompt_approver_type
        )
      
    def get_cluster_approval_stats(self, approvals_statements_and_embeddings, approvals_prompts_dict, cluster_labels, cluster_ID):
        inputs = []
        cluster_size = 0
        n_conditions = len(list(approvals_statements_and_embeddings[0]['approvals'].values())[0])
        approval_counts = {prompt_name: 0 for prompt_name in approvals_prompts_dict.keys()}
        
        for e, l in zip(approvals_statements_and_embeddings, cluster_labels):
            if l != cluster_ID:
                continue
            approvals = list(e['approvals'].values())[0]
            for prompt_name in approvals_prompts_dict.keys():
                if approvals[prompt_name] == 1:
                    approval_counts[prompt_name] += 1
            cluster_size += 1
            inputs.append(e['statement'])
        
        approval_fractions = {prompt_name: count / cluster_size if cluster_size > 0 else 0 for prompt_name, count in approval_counts.items()}
        return inputs, approval_counts, approval_fractions, cluster_size
    
    def create_cluster_table(
        self,
        clusters_desc_table: List[List[str]],
        rows: List[List[str]],
        table_pickle_path: str,
        prompt_approver_type: str
    ) -> None:
        clusters_desc_table = clusters_desc_table + rows
        t = AsciiTable(clusters_desc_table)
        t.inner_row_border = True
        logger.info(t.table)
        logger.info("\n")
        logger.info("Saving table to file...")
        with open(table_pickle_path, "wb") as file:
            pickle.dump(clusters_desc_table, file)

        csv_file_path = os.path.join(self.settings.directory_settings.tables_dir, f"cluster_results_table_{prompt_approver_type}.csv")
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(clusters_desc_table)
        logger.info(f"Table also saved in CSV format at {csv_file_path}")

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
        response_type: str
    ) -> None:
        """
        Compare the responses across multiple models or conditions for a specific response type.

        This method analyzes the agreement and correlation among multiple sets of responses,
        providing insights into how similarly or differently the models/conditions behave.

        Args:
            approvals_statements_and_embeddings (List): List of approval data, statements, and embeddings.
                ex: [[{'model1': {'Google Chat': 1, 'Bing Chat': 0, 'Bing Chat Emoji': 1}, "statement1", embedding], [{'model1': {'Google Chat': 1, 'Bing Chat': 0, 'Bing Chat Emoji': 0}, "statement2", embedding]]
            persona_names (List[str]): Names of the models/conditions to compare.
                ex: ["Google Chat", "Bing Chat"]
            response_type (str): Type of response to analyze (e.g., "approve", "disapprove").

        Returns:
            None: Results are logged rather than returned.
        """
        # Convert response type to integer representation
        # For ex: "approve" -> 1, "disapprove" -> 0, "no response" -> -1
        response_type_int = self.lookup_response_type_int(response_type)
        if response_type_int is None:
            # Error handling for invalid response type
            raise ValueError(f"Invalid response type: {response_type}. Please use 'approve', 'disapprove', or 'no response'. Or, create a custom function to handle this.")

        # Create boolean masks for responses of each model/condition
        masks = [
            np.array([
                list(e['approvals'].values())[0][persona_name] == response_type_int 
                for e in approvals_statements_and_embeddings
            ]) 
            for persona_name in persona_names
        ]

        # Log the number of responses for each model/condition
        for name, mask in zip(persona_names, masks):
            logger.info(f'{name} "{response_type}" responses: {sum(mask)}')

        # Create and log a confusion matrix and Pearson correlation for each pair of models
        for name1, name2 in combinations(persona_names, 2):
            i = persona_names.index(name1)
            j = persona_names.index(name2)
            
            logger.info(f'Intersection matrix for {name1} and {name2} "{response_type}" responses:')
            conf_matrix = sklearn.metrics.confusion_matrix(masks[i], masks[j])
            conf_rows = [
                ["", f"Not In {name1}", f"In {name1}"],
                [f"Not In {name2}", conf_matrix[0][0], conf_matrix[1][0]],
                [f"In {name2}", conf_matrix[0][1], conf_matrix[1][1]]
            ]
            t = AsciiTable(conf_rows)
            t.inner_row_border = True
            logger.info(t.table)

            # Calculate and log Pearson correlation
            pearson_r = scipy.stats.pearsonr(masks[i], masks[j])
            logger.info(f'Pearson correlation between "{response_type}" responses for {name1} and {name2}: {round(pearson_r.correlation, 5)}')
            logger.info(f"(P-value {round(pearson_r.pvalue, 5)})")

    def calculate_hierarchical_cluster_data(
        self,
        clustering: object,
        approvals_statements_and_embeddings: List,
        rows: List[List[str]]
    ) -> Tuple[np.ndarray, List[str], List[List[int]], List[List[int]], int]:
        statement_embeddings = np.array([e['embedding'] for e in approvals_statements_and_embeddings])
        centroids = Clustering.get_cluster_centroids(statement_embeddings, clustering.labels_)
        Z = linkage(centroids, "ward") # the linkage matrix Z represents the process of merging clusters

        n_clusters = max(clustering.labels_) + 1
        cluster_labels = []
        for i in range(n_clusters):
            pos = self.lookup_cid_pos_in_rows(rows, i)
            cluster_labels.append(rows[pos][-1] if pos >= 0 else "(Label missing)")

        all_cluster_sizes = []
        approvals_prompts_dict = list(approvals_statements_and_embeddings[0]['approvals'].values())[0]
        for i in range(n_clusters):
            inputs, approval_counts, approval_fractions, cluster_size = self.get_cluster_approval_stats(
                approvals_statements_and_embeddings, approvals_prompts_dict, clustering.labels_, i
            )
            approval_counts_list = [approval_counts[prompt_name] for prompt_name in sorted(approval_counts.keys())]
            all_cluster_sizes.append([cluster_size] + approval_counts_list)

        # Handle merged clusters
        merged_cluster_sizes = all_cluster_sizes.copy()
        for i, merge in enumerate(Z):
            m1, m2 = int(merge[0]), int(merge[1])
            merged_sizes = [
                m1_entry + m2_entry
                for m1_entry, m2_entry in zip(merged_cluster_sizes[m1], merged_cluster_sizes[m2])
            ]
            merged_cluster_sizes.append(merged_sizes)

        original_cluster_sizes = all_cluster_sizes[:n_clusters]
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
                theme_identification_model_info=self.theme_identification_model_info,
                data_type=data_type,
                include_responses_and_interactions=include_responses_and_interactions,
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
        theme_identification_model_info: Dict,
        data_type: str = "joint_embeddings", # or "approvals"
        include_responses_and_interactions: bool = True,
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

        if data_type == "joint_embeddings":
            for frac in fractions:
                row.append(f"{round(100 * frac, 1)}%")
        else:  # data_type == "approvals"
            for model_name in fractions:
                for prompt, frac in fractions[model_name].items():
                    row.append(f"{round(100 * frac, 1)}%")

        for model_info in model_info_list:
            inputs_themes_str = self.identify_theme(
                texts=inputs, theme_identification_model_info=theme_identification_model_info, sampled_texts=5
            )[:max_desc_length]
            row.append(inputs_themes_str)

            if include_responses_and_interactions and responses is not None:
                responses_themes_str = self.identify_theme(
                    texts=responses, theme_identification_model_info=theme_identification_model_info, sampled_texts=5
                )[:max_desc_length]

                interactions = [
                    f'(Statement: "{input}", Response: "{response}")'
                    for input, response in zip(inputs, responses)
                ]
                interactions_themes_str = self.identify_theme(
                    texts=interactions, theme_identification_model_info=theme_identification_model_info, sampled_texts=5
                )[:max_desc_length]

                row.append(responses_themes_str)
                row.append(interactions_themes_str)

        return row

    def get_cluster_stats(
        self,
        data: List,
        cluster_labels: np.ndarray,
        cluster_ID: int,
        data_type: str = "joint_embeddings", # or "approvals"
        include_responses: bool = True,
    ) -> Union[Tuple[List[str], List[str], List[float]], Tuple[List[str], Dict[str, Dict[str, float]]]]:
        inputs = []
        responses = []
        cluster_size = 0

        if data_type == "joint_embeddings":
            n_categories = int(max([e['model_num'] for e in data])) + 1
            fractions = [0 for _ in range(n_categories)]
        elif data_type == "approvals":
            fractions = {}
        else:
            raise ValueError(f"Unsupported data type: {data_type}")

        for e, l in zip(data, cluster_labels):
            if l != cluster_ID:
                continue

            cluster_size += 1
            inputs.append(e['statement'])

            if data_type == "joint_embeddings":
                if e['model_num'] >= 0:
                    fractions[e['model_num']] += 1
                if include_responses:
                    responses.append(e['response'])
            elif data_type == "approvals":
                for model_name, approvals in e['approvals'].items():
                    if model_name not in fractions:
                        fractions[model_name] = {prompt: 0 for prompt in approvals}
                    for prompt, approval in approvals.items():
                        if approval == 1:
                            fractions[model_name][prompt] += 1

        if data_type == "joint_embeddings":
            fractions = [f / cluster_size for f in fractions]
        else:  # data_type == "approvals"
            for model_name in fractions:
                for prompt in fractions[model_name]:
                    fractions[model_name][prompt] /= cluster_size

        if data_type == "joint_embeddings" and include_responses:
            return inputs, responses, fractions
        else:
            return inputs, fractions

    def identify_theme(
        self,
        texts: List[str],
        theme_identification_model_info: Dict,
        sampled_texts: int = 5,
        temp: float = 0.5,
        max_tokens: int = 70,
        max_total_tokens: int = 250,
    ) -> str:
        prompt = theme_identification_model_info.get("theme_identification_prompt", self.settings.clustering_settings.theme_identification_prompt)
        sampled_texts = random.sample(texts, min(len(texts), sampled_texts))
        theme_identify_prompt = prompt + "\n\n"
        for i, text in enumerate(sampled_texts):
            theme_identify_prompt += f"Text {i + 1}: {str(text)}\n"
        theme_identify_prompt += "\nTheme:"
        model_instance = initialize_model(theme_identification_model_info, temp, max_tokens)
        for _ in range(20):
            try:
                completion = model_instance.generate(theme_identify_prompt)[:max_total_tokens].replace("\n", " ")
                return completion
            except Exception as e:
                logger.error(f"API error: {e}")
                time.sleep(2)
        return "Failed to identify theme"

    def display_statement_themes(self, chosen_clustering, rows, model_info_list):
        print(f"Chosen clustering: {chosen_clustering}")
        print(f"Rows: {rows}")
        # Create a table and save it in a readable format (CSV) for easy visualization in VSCode
        model_columns = [
            model_info["model_name"] for model_info in model_info_list
        ]  # Extract model names from model_info_list
        table_headers = (
            [
                "ID",  # cluster ID
                "N",  # number of items in the cluster
            ]
            + model_columns
            + [  # Add model names dynamically
                "Inputs Themes",  # LLM says the theme of the input
                "Responses Themes",  # LLM says the theme of the response
                "Interaction Themes",  # LLM says the theme of the input and response together
            ]
        )
        csv_file_path = (
            self.settings.directory_settings.tables_dir
            / "cluster_results_table_statement_responses.csv"
        )
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(table_headers)
            writer.writerows(rows)

        # Display the table in the console
        t = PrettyTable()
        t.field_names = table_headers
        for row in rows:
            t.add_row(row)
        print(t)

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
