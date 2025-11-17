from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np
import pickle
import scipy.stats as stats
from scipy.stats import pearsonr, binomtest
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, BitsAndBytesConfig
from anthropic import Anthropic
from openai import OpenAI
from google.genai import Client
import random
import sys
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import pandas as pd
import copy
from saffron_implementation import SAFFRON

sys.path.append("..")
sys.path.append("../interventions/auto_finetune_eval")
from auto_finetuning_helpers import make_api_request, extract_json_from_string, collect_dataset_from_api, rephrase_description, parallel_make_api_requests, permutation_test_auc
from baseline_discriminator import baseline_discrimination, LogisticBoWDiscriminator

from typing import List, Tuple, Dict, Optional, Union, Any

import warnings
warnings.filterwarnings('ignore', message='You have modified the pretrained model configuration to control generation.*')

from structlog._config import BoundLoggerLazyProxy



def attach_cluster_metrics_to_graph(
    G: nx.Graph,
    base_cluster_ids: List[int],
    finetuned_cluster_ids: List[int],
    base_mauve_scores: List[float],
    finetuned_mauve_scores: List[float],
    base_kl_scores: List[float],
    finetuned_kl_scores: List[float],
    base_entropy_scores: List[float],
    finetuned_entropy_scores: List[float]
):
    """
    Attach per-cluster metrics as node attributes in the graph.
    Node names are in the form '1_{cluster_id}' or '2_{cluster_id}'.
    """
    # For each base cluster node '1_{cid}', attach the relevant metrics
    for cid in base_cluster_ids:
        node_name = f"1_{cid}"
        if node_name in G.nodes():
            G.nodes[node_name]["mauve"] = base_mauve_scores[cid]
            G.nodes[node_name]["kl_div"] = base_kl_scores[cid]
            G.nodes[node_name]["mean_entropy"] = base_entropy_scores[cid]
    
    # For each finetuned cluster node '2_{cid}', attach the relevant metrics
    for cid in finetuned_cluster_ids:
        node_name = f"2_{cid}"
        if node_name in G.nodes():
            G.nodes[node_name]["mauve"] = finetuned_mauve_scores[cid]
            G.nodes[node_name]["kl_div"] = finetuned_kl_scores[cid]
            G.nodes[node_name]["mean_entropy"] = finetuned_entropy_scores[cid]


def analyze_metric_differences_vs_similarity(G: nx.Graph):
    rows = []
    for u, v, data in G.edges(data=True):
        # Suppose the 'weight' attribute is actually the "similarity_score" 
        # (depending on your naming, it might be 1 - accuracy, etc.)
        similarity_score = data["weight"]
        
        # Get the node metrics
        u_mauve = G.nodes[u].get("mauve", np.nan)
        v_mauve = G.nodes[v].get("mauve", np.nan)
        u_kl = G.nodes[u].get("kl_div", np.nan)
        v_kl = G.nodes[v].get("kl_div", np.nan)
        u_entropy = G.nodes[u].get("mean_entropy", np.nan)
        v_entropy = G.nodes[v].get("mean_entropy", np.nan)
        
        row = {
            "node_u": u,
            "node_v": v,
            "similarity_score": similarity_score,
            "diff_mauve": u_mauve - v_mauve,
            "diff_kl": u_kl - v_kl,
            "diff_entropy": u_entropy - v_entropy,
            "u_mauve": u_mauve,
            "v_mauve": v_mauve,
            "u_kl": u_kl,
            "v_kl": v_kl,
            "u_entropy": u_entropy,
            "v_entropy": v_entropy,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Now do a correlation analysis
    # For instance, does bigger absolute difference in mauve correlate with lower or higher similarity?
    corr_mauve, p_mauve = pearsonr(df["diff_mauve"], df["similarity_score"])
    print(f"Pearson correlation between diff_mauve & similarity: {corr_mauve:.3f} (p={p_mauve:.3g})")
    
    corr_kl, p_kl = pearsonr(df["diff_kl"], df["similarity_score"])
    print(f"Pearson correlation between diff_kl & similarity: {corr_kl:.3f} (p={p_kl:.3g})")
    
    corr_ent, p_ent = pearsonr(df["diff_entropy"], df["similarity_score"])
    print(f"Pearson correlation between diff_entropy & similarity: {corr_ent:.3f} (p={p_ent:.3g})")

    corr_u_mauve, p_u_mauve = pearsonr(df["u_mauve"], df["similarity_score"])
    print(f"Pearson correlation between u_mauve & similarity: {corr_u_mauve:.3f} (p={p_u_mauve:.3g})")
    
    corr_v_mauve, p_v_mauve = pearsonr(df["v_mauve"], df["similarity_score"])
    print(f"Pearson correlation between v_mauve & similarity: {corr_v_mauve:.3f} (p={p_v_mauve:.3g})")
    
    corr_u_kl, p_u_kl = pearsonr(df["u_kl"], df["similarity_score"])
    print(f"Pearson correlation between u_kl & similarity: {corr_u_kl:.3f} (p={p_u_kl:.3g})")
    
    corr_v_kl, p_v_kl = pearsonr(df["v_kl"], df["similarity_score"])
    print(f"Pearson correlation between v_kl & similarity: {corr_v_kl:.3f} (p={p_v_kl:.3g})")
    
    corr_u_entropy, p_u_entropy = pearsonr(df["u_entropy"], df["similarity_score"])
    print(f"Pearson correlation between u_entropy & similarity: {corr_u_entropy:.3f} (p={p_u_entropy:.3g})")
    
    corr_v_entropy, p_v_entropy = pearsonr(df["v_entropy"], df["similarity_score"])
    print(f"Pearson correlation between v_entropy & similarity: {corr_v_entropy:.3f} (p={p_v_entropy:.3g})")
    
    # Or do a multiple regression if you want
    # from statsmodels.formula.api import ols
    # model = ols("similarity_score ~ diff_mauve + diff_kl + diff_entropy", data=df).fit()
    # print(model.summary())
    
    return df

def analyze_node_metric_vs_neighbor_similarity(G: nx.Graph):
    data_rows = []
    for node in G.nodes():
        # Get this node's metric
        node_mauve = G.nodes[node].get("mauve", np.nan)
        node_kl = G.nodes[node].get("kl_div", np.nan)
        node_entropy = G.nodes[node].get("mean_entropy", np.nan)
        
        # Get edges for neighbors
        neighbors = G.neighbors(node)
        edge_similarity_scores = []
        for nbr in neighbors:
            if G.has_edge(node, nbr):
                edge_data = G[node][nbr]
                edge_similarity_scores.append(edge_data["weight"])
        
        mean_similarity = np.mean(edge_similarity_scores) if len(edge_similarity_scores) > 0 else np.nan
        
        data_rows.append({
            "node": node,
            "mauve": node_mauve,
            "kl_div": node_kl,
            "mean_entropy": node_entropy,
            "mean_neighbor_similarity": mean_similarity
        })

        if node_mauve < 0.3 and node_kl > 30:
            print("Candidate node found:")
            print(f"Node {node} has mauve {node_mauve} and kl_div {node_kl}")

    
    df_nodes = pd.DataFrame(data_rows)
    
    # Correlation of node metric vs. mean neighbor similarity
    corr_mauve, p_mauve = pearsonr(df_nodes["mauve"], df_nodes["mean_neighbor_similarity"])
    print(f"Correlation: node mauve vs. mean neighbor similarity: {corr_mauve:.3f} (p={p_mauve:.3g})")
    
    corr_kl, p_kl = pearsonr(df_nodes["kl_div"], df_nodes["mean_neighbor_similarity"])
    print(f"Correlation: node kl_div vs. mean neighbor similarity: {corr_kl:.3f} (p={p_kl:.3g})")
    
    corr_ent, p_ent = pearsonr(df_nodes["mean_entropy"], df_nodes["mean_neighbor_similarity"])
    print(f"Correlation: node mean_entropy vs. mean neighbor similarity: {corr_ent:.3f} (p={p_ent:.3g})")

    return df_nodes

# TODO: make this generally work; assumes that there are always two clusters per hypothesis
def analyze_hypothesis_scores_vs_cluster_metrics(
    G: nx.Graph,
    hypothesis_origin_clusters: List[Tuple[int, int, str]],
    all_validated_accuracy_scores: List[List[float]],
    model_label_for_node_1="1_",  # or just "1_"
    model_label_for_node_2="2_"
):
    """
    Build a mapping from each hypothesis to the node metrics of its cluster(s).
    Then correlate the cluster metrics with the average validated score.
    """
    # Suppose all_validated_accuracy_scores has shape (num_rephrases + 1, num_hypotheses)
    all_validated_accuracy_scores = np.array(all_validated_accuracy_scores)
    num_hypotheses = all_validated_accuracy_scores.shape[1]
    print("all_validated_accuracy_scores.shape:", all_validated_accuracy_scores.shape)
    
    rows = []
    for i in range(len(hypothesis_origin_clusters)):
        # mean validation score across rephrasings
        mean_val_score = np.mean(all_validated_accuracy_scores[:, i])
        
        # Which cluster pair generated hypothesis i?
        # e.g. hypothesis_origin_clusters[i] = (cluster_id_base, cluster_id_ft)
        try:
            base_cid, ft_cid, condition_str = hypothesis_origin_clusters[i]
            if type(base_cid) != int:
                base_cid = base_cid['cluster_id']
            if type(ft_cid) != int:
                ft_cid = ft_cid['cluster_id']
            if base_cid == -1:
                print(f"Hypothesis {i} has base cluster id -1. Skipping.")
                continue
            if ft_cid == -1:
                print(f"Hypothesis {i} has finetuned cluster id -1. Skipping.")
                continue
        except Exception as e:
            print(f"Hypothesis {i} lacked expected structure: {hypothesis_origin_clusters[i]}")
            print(f"Encountered error: {e}")
            continue
        
        # Convert them to node names
        node_base = f"{model_label_for_node_1}{base_cid}"
        node_ft   = f"{model_label_for_node_2}{ft_cid}"
        
        # Grab node-level metrics
        # (In some workflows you might have one node or both nodes. Up to you how you combine them.)
        if node_base in G.nodes:
            base_mauve = G.nodes[node_base].get("mauve", np.nan)
            base_kl = G.nodes[node_base].get("kl_div", np.nan)
            base_entropy = G.nodes[node_base].get("mean_entropy", np.nan)
            if np.isnan(base_mauve):
                print(f"Base node {node_base} has NaN mauve.")
            if np.isnan(base_kl):
                print(f"Base node {node_base} has NaN kl_div.")
            if np.isnan(base_entropy):
                print(f"Base node {node_base} has NaN mean_entropy.")
        else:
            print(f"Base node {node_base} not found in graph.")
            base_mauve = base_kl = base_entropy = np.nan
        
        if node_ft in G.nodes:
            ft_mauve = G.nodes[node_ft].get("mauve", np.nan)
            ft_kl = G.nodes[node_ft].get("kl_div", np.nan)
            ft_entropy = G.nodes[node_ft].get("mean_entropy", np.nan)
            if np.isnan(ft_mauve):
                print(f"Finetuned node {node_ft} has NaN mauve.")
            if np.isnan(ft_kl):
                print(f"Finetuned node {node_ft} has NaN kl_div.")
            if np.isnan(ft_entropy):
                print(f"Finetuned node {node_ft} has NaN mean_entropy.")
        else:
            print(f"Finetuned node {node_ft} not found in graph.")
            ft_mauve = ft_kl = ft_entropy = np.nan
        
        # Store them in a row. You can combine the base & finetuned metrics or keep them separate.
        # Some might want the difference, others might want average, etc.
        rows.append({
            "hypothesis_index": i,
            "mean_validation_score": mean_val_score,
            "base_mauve": base_mauve,
            "ft_mauve": ft_mauve,
            "base_kl": base_kl,
            "ft_kl": ft_kl,
            "base_entropy": base_entropy,
            "ft_entropy": ft_entropy
        })
    
    df = pd.DataFrame(rows)
    print("df.head():", df.head())
    print("G.nodes:", G.nodes)
    
    # We can do a correlation with base_mauve alone:
    corr_mauve, p_mauve = pearsonr(df["base_mauve"], df["mean_validation_score"])
    print(f"Base mauve vs. hypothesis validation score: {corr_mauve:.3f} (p={p_mauve:.3g})")

    # Do the same for ft_mauve
    corr_ft_mauve, p_ft_mauve = pearsonr(df["ft_mauve"], df["mean_validation_score"])
    print(f"Finetuned mauve vs. hypothesis validation score: {corr_ft_mauve:.3f} (p={p_ft_mauve:.3g})")
    
    # or maybe we look at the average mauve across the pair:
    df["avg_mauve"] = (df["base_mauve"] + df["ft_mauve"]) / 2
    corr_avg_mauve, p_avg_mauve = pearsonr(df["avg_mauve"], df["mean_validation_score"])
    print(f"Avg mauve vs. hypothesis validation score: {corr_avg_mauve:.3f} (p={p_avg_mauve:.3g})")
    
    df["diff_mauve"] = df["base_mauve"] - df["ft_mauve"]
    corr_diff_mauve, p_diff_mauve = pearsonr(df["diff_mauve"], df["mean_validation_score"])
    print(f"Diff mauve vs. hypothesis validation score: {corr_diff_mauve:.3f} (p={p_diff_mauve:.3g})")

    # Do the same for kl_div
    corr_base_kl, p_base_kl = pearsonr(df["base_kl"], df["mean_validation_score"])
    print(f"Base kl vs. hypothesis validation score: {corr_base_kl:.3f} (p={p_base_kl:.3g})")

    corr_ft_kl, p_ft_kl = pearsonr(df["ft_kl"], df["mean_validation_score"])
    print(f"Finetuned kl vs. hypothesis validation score: {corr_ft_kl:.3f} (p={p_ft_kl:.3g})")
    
    df["avg_kl"] = (df["base_kl"] + df["ft_kl"]) / 2
    corr_avg_kl, p_avg_kl = pearsonr(df["avg_kl"], df["mean_validation_score"])
    print(f"Avg kl vs. hypothesis validation score: {corr_avg_kl:.3f} (p={p_avg_kl:.3g})")
    
    df["diff_kl"] = df["base_kl"] - df["ft_kl"]
    corr_diff_kl, p_diff_kl = pearsonr(df["diff_kl"], df["mean_validation_score"])
    print(f"Diff kl vs. hypothesis validation score: {corr_diff_kl:.3f} (p={p_diff_kl:.3g})")

    # Do the same for entropy
    corr_base_entropy, p_base_entropy = pearsonr(df["base_entropy"], df["mean_validation_score"])
    print(f"Base entropy vs. hypothesis validation score: {corr_base_entropy:.3f} (p={p_base_entropy:.3g})")

    corr_ft_entropy, p_ft_entropy = pearsonr(df["ft_entropy"], df["mean_validation_score"])
    print(f"Finetuned entropy vs. hypothesis validation score: {corr_ft_entropy:.3f} (p={p_ft_entropy:.3g})")

    df["avg_entropy"] = (df["base_entropy"] + df["ft_entropy"]) / 2
    corr_avg_entropy, p_avg_entropy = pearsonr(df["avg_entropy"], df["mean_validation_score"])
    print(f"Avg entropy vs. hypothesis validation score: {corr_avg_entropy:.3f} (p={p_avg_entropy:.3g})")

    df["diff_entropy"] = df["base_entropy"] - df["ft_entropy"]
    corr_diff_entropy, p_diff_entropy = pearsonr(df["diff_entropy"], df["mean_validation_score"])
    print(f"Diff entropy vs. hypothesis validation score: {corr_diff_entropy:.3f} (p={p_diff_entropy:.3g})")

    return df

# Takes two sets of clusterings and returns the optimal pairwise matching of clusters between the two sets, seeking to minimize the sum of the squared Euclidean distances between the centroids of each pair of matching clusters.
def match_clusterings(
        clustering_assignments_1: List[int], 
        embeddings_list_1: List[List[float]], 
        clustering_assignments_2: List[int], 
        embeddings_list_2: List[List[float]]
        ) -> List[Tuple[int, int]]:
    """
    Match clusters between two different clusterings based on the similarity of their centroids.

    This function finds the optimal pairwise matching between clusters from two different
    clustering assignments. It minimizes the sum of squared Euclidean distances between
    the centroids of matched cluster pairs.

    Args:
        clustering_assignments_1 (List[int]): Cluster assignments for the first set of data points.
        embeddings_list_1 (List[List[float]]): Embeddings for the first set of data points.
        clustering_assignments_2 (List[int]): Cluster assignments for the second set of data points.
        embeddings_list_2 (List[List[float]]): Embeddings for the second set of data points.

    Returns:
        List[Tuple[int, int]]: A list of tuples, where each tuple contains the indices
        of matched clusters (cluster_index_1, cluster_index_2).

    Note:
        - The function assumes that the order and length of clustering_assignments match
          the order and length of the corresponding embeddings_list.
        - It uses the Hungarian algorithm to find the optimal matching between cluster centroids.
        - The number of clusters in the two sets may be different; the matching will be
          performed up to the number of clusters in the smaller set.
    """
    # Calculate centroids for each cluster in both sets
    centroids_1 = [np.mean([embeddings_list_1[i] for i in range(len(clustering_assignments_1)) if clustering_assignments_1[i] == cluster], axis=0) for cluster in range(max(clustering_assignments_1) + 1)]
    centroids_2 = [np.mean([embeddings_list_2[i] for i in range(len(clustering_assignments_2)) if clustering_assignments_2[i] == cluster], axis=0) for cluster in range(max(clustering_assignments_2) + 1)]

    # Compute the distance matrix between centroids of both sets
    distance_matrix = cdist(centroids_1, centroids_2, 'sqeuclidean')

    # Use the Hungarian algorithm (linear sum assignment) to find the optimal matching
    row_ind, col_ind = linear_sum_assignment(distance_matrix)

    # Return the matched cluster pairs
    return list(zip(row_ind, col_ind))

# Given a possible description of how the cluster 1 model and cluster 2 models differ, we want to see if the 
# assistant LLM can generate strings that are differentially more probable under one model as compared to the other.
def assistant_generative_compare(
        difference_descriptions: List[str], 
        local_model: AutoModel, 
        labeling_tokenizer: AutoTokenizer, 
        api_provider: str,
        api_model_str: str,
        auth_key: str,
        client: Optional[Union[Anthropic, OpenAI, Client]] = None,
        device: str = "cuda:0",
        starting_model_str: Optional[str] = None,
        comparison_model_str: Optional[str] = None,
        common_tokenizer_str: str = "meta-llama/Meta-Llama-3-8B",
        starting_model: Optional[AutoModel] = None,
        comparison_model: Optional[AutoModel] = None,
        use_correlation_coefficient: bool = True,
        num_generated_texts_per_description: int = 10,
        permute_labels: bool = False,
        bnb_config: BitsAndBytesConfig = None,
        api_interactions_save_loc: Optional[str] = None,
        logging_level: str = "INFO",
        logger: Optional[BoundLoggerLazyProxy] = None
        ) -> Tuple[List[float], List[List[str]], List[List[str]]]:
    """
    Generate texts based on descriptions of cluster differences and evaluate whether the language
    modeling scores of the texts follow the expected pattern (meaning higher log odds under the 
    model associated with the target cluster).

    This function uses an assistant model (either local or via API) to generate texts that are
    supposed to be differentially more probable for one cluster or the other, based on provided 
    descriptions of their differences. It then evaluates how well these generated texts can be 
    distinguished based on the language modeling scores of the texts under the two different 
    language models.

    The function performs the following steps:
    1. Generates prompts for text generation based on the cluster difference descriptions.
    2. Uses either a local model or an API to generate texts that are more likely to be attributed
       to one cluster or the other.
    3. Computes language model scores for the generated texts using models for both clusters.
    4. Calculates correlation coefficients between the difference in LM scores and the intended 
       target during generation to evaluate how well the language modeling scores correlate with
       the intended target.

    Args:
        difference_descriptions (List[str]): Descriptions of differences between clusters.
        local_model (AutoModel): Local model for text generation if not using API.
        labeling_tokenizer (AutoTokenizer): Tokenizer for the local model.
        api_provider (str): API provider for text generation (e.g., 'openai').
        api_model_str (str): Model string for API requests.
        auth_key (str): Authentication key for API requests.
        client (Optional[Union[Anthropic, OpenAI, Client]], optional): API client for text
            generation. Defaults to None.
        starting_model_str (Optional[str]): Model identifier for the first cluster's language model. 
            Defaults to None.
        comparison_model_str (Optional[str]): Model identifier for the second cluster's language model. 
            Defaults to None.
        common_tokenizer_str (str): Identifier for the tokenizer used by both language models.
            Defaults to "meta-llama/Meta-Llama-3-8B".
        starting_model (Optional[AutoModel]): Model that generates texts for the first cluster.
            Defaults to None.
        comparison_model (Optional[AutoModel]): Model that generates texts for the second cluster.
            Defaults to None.
        device (str, optional): Device to use for computations (e.g., 'cuda:0'). Defaults to "cuda:0".
        use_correlation_coefficient (bool, optional): If True, use the correlation coefficient to 
            evaluate the association between the difference in LM scores and the intended target.   
            If False, use the AUC of the difference in LM scores as a predictor of which model the 
            texts were generated for.
        num_generated_texts_per_description (int): Number of texts to generate per description.
        permute_labels (bool): If True, randomly permute the labels for null hypothesis testing.
        bnb_config (BitsAndBytesConfig): Configuration for quantization.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
    Returns:
        Tuple[List[float], List[List[str]], List[List[str]]]: A tuple containing:
            - per_label_scores: List of correlation coefficients for each description.
            - generated_texts_1: List of lists of texts generated for cluster 1.
            - generated_texts_2: List of lists of texts generated for cluster 2.

    Note:
        - The permute_labels option can be used for creating a null distribution for significance testing.
    """
    if starting_model is None and starting_model_str is None:
        raise ValueError("Either starting_model or starting_model_str must be provided.")
    if comparison_model is None and comparison_model_str is None:
        raise ValueError("Either comparison_model or comparison_model_str must be provided.")
    
    if ':' in api_model_str:
        dataset_api_str = api_model_str.split(':')[0]
    else:
        dataset_api_str = api_model_str

    # Compile the list of prompts that encourage the assistant to generate texts attributed to the cluster 1 model.
    prompts_1 = []
    for description in difference_descriptions:
        if api_provider is None:
            prompt = f"Given the following description of how the texts generated by Model 1 differ from those generated by Model 2: {description}, generate a new text that is closer to Model 1."
        else:
            prompt = f'Given the following description of how the texts generated by Model 1 differ from those generated by Model 2: {description}, generate two short texts that are closer to Model 1. Format your response as a JSON array of strings, where each string is a new text. Example response format: ["Text 1", "Text 2"]. Aim for about 100 words per text.'
        prompts_1.append(prompt)
    # Now, compile the list of prompts that encourage the assistant to generate texts attributed to the cluster 2 model.
    prompts_2 = []
    for description in difference_descriptions:
        if api_provider is None:
            prompt = f"Given the following description of how the texts generated by Model 1 differ from those generated by Model 2: {description}, generate a new text that is more likely to be generated by Model 2."
        else:
            prompt = f'Given the following description of how the texts generated by Model 1 differ from those generated by Model 2: {description}, generate two short texts that are more likely to be generated by Model 2. Format your response as a JSON array of strings, where each string is a new text. Example response format: ["Text 1", "Text 2"]. Aim for about 100 words per text.'
        prompts_2.append(prompt)

    # Generate texts for each prompt using the assistant model
    generated_texts_1 = []
    for prompt in tqdm(prompts_1, desc="Generating texts for Model 1"):
        if api_provider is None:
            inputs = labeling_tokenizer(prompt, return_tensors="pt", padding=True).to(device)
            outputs = local_model.generate(
                **inputs, 
                max_new_tokens=100, 
                num_return_sequences=num_generated_texts_per_description, 
                do_sample=True, 
                pad_token_id=labeling_tokenizer.pad_token_id
            )
            generated_texts_1.append([labeling_tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
        else:
            try:
                json_response = collect_dataset_from_api(
                    prompt, 
                    api_provider, 
                    dataset_api_str, 
                    auth_key,
                    client=client,
                    num_datapoints=num_generated_texts_per_description,
                    max_tokens=2048,
                    api_interactions_save_loc=api_interactions_save_loc,
                    logging_level=logging_level,
                    logger=logger,
                    request_info={"pipeline_stage": "generating model 1 attributed texts"}
                )
                generated_texts_1.append(json_response)
            except Exception as e:
                print(f"Error generating texts for Model 1: {e}")
                generated_texts_1.append(["Text generation failed."] * num_generated_texts_per_description)
    generated_texts_2 = []
    for prompt in tqdm(prompts_2, desc="Generating texts for Model 2"):
        if api_provider is None:
            inputs = labeling_tokenizer(prompt, return_tensors="pt", padding=True).to(device)
            outputs = local_model.generate(
                **inputs, 
                max_new_tokens=100, 
                num_return_sequences=num_generated_texts_per_description, 
                do_sample=True, 
                pad_token_id=labeling_tokenizer.pad_token_id
            )
            generated_texts_2.append([labeling_tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
        else:
            try:
                json_response = collect_dataset_from_api(
                    prompt, 
                    api_provider, 
                    dataset_api_str, 
                    auth_key,
                    client=client,
                    num_datapoints=num_generated_texts_per_description,
                    max_tokens=2048,
                    api_interactions_save_loc=api_interactions_save_loc,
                    logging_level=logging_level,
                    logger=logger,
                    request_info={"pipeline_stage": "generating model 2 attributed texts"}
                )
                generated_texts_2.append(json_response)
            except Exception as e:
                print(f"Error generating texts for Model 2: {e}")
                generated_texts_2.append(["Text generation failed."] * num_generated_texts_per_description)
    
    generated_texts = [gt_1 + gt_2 for gt_1, gt_2 in zip(generated_texts_1, generated_texts_2)]
    generated_text_labels = [[0] * len(gt_1) + [1] * len(gt_2) for gt_1, gt_2 in zip(generated_texts_1, generated_texts_2)]
    if permute_labels:
        generated_text_labels = [np.random.permutation(labels) for labels in generated_text_labels]

    # Compute probabilities of the generated texts under both models

    # First, load the tokenizer
    common_tokenizer = AutoTokenizer.from_pretrained(common_tokenizer_str)
    if common_tokenizer.pad_token is None:
        common_tokenizer.pad_token = common_tokenizer.eos_token
        common_tokenizer.pad_token_id = common_tokenizer.eos_token_id

    # then, load the cluster 1 model and compute scores
    current_model = starting_model if starting_model is not None else AutoModelForCausalLM.from_pretrained(starting_model_str, device_map={"": 0} if device == "cuda:0" else "auto", quantization_config=bnb_config, torch_dtype=torch.float16)
    generated_texts_scores = []
    for texts_for_label in tqdm(generated_texts, desc="Computing scores for generated texts (Model 1 attributed)"):
        generated_texts_scores_for_label = []
        for text in texts_for_label:
            inputs = common_tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                try:
                    model_loss = current_model(**inputs, labels=inputs["input_ids"]).loss
                except Exception as e:
                    print(f"Error computing scores for generated text: {e}")
                    print(f"Text: {text}")
                    model_loss = torch.tensor(0.0)
            generated_texts_scores_for_label.append(model_loss.item())
        generated_texts_scores.append(generated_texts_scores_for_label)
    
    # Next, load the cluster 2 model and compute scores, subtracting the cluster 2 scores from the cluster 1 scores
    current_model = comparison_model if comparison_model is not None else AutoModelForCausalLM.from_pretrained(comparison_model_str, device_map={"": 0} if device == "cuda:0" else "auto", quantization_config=bnb_config, torch_dtype=torch.float16)
    print("generated_texts has been generated")
    for i in tqdm(range(len(generated_texts)), desc="Computing scores for generated texts (Model 2 attributed)"):
        for j in range(len(generated_texts[i])):
            inputs = common_tokenizer(generated_texts[i][j], return_tensors="pt").to(device)
            with torch.no_grad():
                try:
                    model_loss = current_model(**inputs, labels=inputs["input_ids"]).loss
                except Exception as e:
                    print(f"Error computing scores for generated text: {e}")
                    print(f"Text: {generated_texts[i][j]}")
                    model_loss = torch.tensor(0.0)
            generated_texts_scores[i][j] -= model_loss.item()
    
    # target = 0 -> Model 1
    # target = 1 -> Model 2
    # score = Model 1 score - Model 2 score
    # score > 0 -> text is more associated with Model 2
    # score < 0 -> text is more associated with Model 1

    if use_correlation_coefficient:
        per_label_scores = []
        for true_labels_set, generated_texts_scores_for_label in zip(generated_text_labels, generated_texts_scores):
            # Convert labels to numeric values: 0 for Model 1, 1 for Model 2
            numeric_labels = np.array(true_labels_set)

            # Convert to numpy array for easier handling
            scores_array = np.array(generated_texts_scores_for_label)
            
            # Find nan locations
            nan_mask = np.isnan(scores_array)
            num_nans = np.sum(nan_mask)
            
            if num_nans > 0:
                print(f"Removing {num_nans} data points with NaN scores")
                # Remove nans from both arrays
                valid_mask = ~nan_mask
                numeric_labels = numeric_labels[valid_mask]
                scores_array = scores_array[valid_mask]
                generated_texts_scores_for_label = scores_array.tolist()

            # Calculate the correlation coefficient
            # Check if there's variance in both arrays
            if len(set(numeric_labels)) == 1 or len(set(generated_texts_scores_for_label)) == 1:
                # If either array is constant, correlation is undefined
                correlation_coeff = 0.0
            else:
                # Calculate the correlation coefficient
                correlation_coeff, _ = pearsonr(numeric_labels, generated_texts_scores_for_label)
            
            per_label_scores.append(correlation_coeff)
        
        print("Correlation coefficients:", per_label_scores)
    else:
        # Calculate the AUCs of the score differences as a way of looking at each text and determing which model it is intended 
        # to be more associated with.
        per_label_scores = [roc_auc_score(true_labels_set, generated_texts_scores_for_label) for true_labels_set, generated_texts_scores_for_label in zip(generated_text_labels, generated_texts_scores)]

        print("AUCs:", per_label_scores)

    return per_label_scores, generated_texts_1, generated_texts_2


def _compute_avg_kl_divergence(
    assistant_model: AutoModel,
    target_model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: str,
    batch_size: int = 8,
    prompt_template: Optional[str] = None,
    hypothesis: Optional[str] = None
) -> float:
    """
    Helper to compute the average KL divergence KL(target_model || assistant_model)
    over a list of texts, with optional prompting for the assistant model.
    """
    total_kl_div = 0
    num_texts = len(texts)
    if num_texts == 0:
        return 0.0

    assistant_model.eval()
    target_model.eval()
    kl_loss_fn = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    with torch.no_grad():
        for i in tqdm(range(0, num_texts, batch_size), desc="Computing KL Divergence", leave=False):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize for target model (no prompt)
            target_inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True)
            target_input_ids = target_inputs.input_ids.to(device)
            target_attention_mask = target_inputs.attention_mask.to(device)

            # Get target model logits
            target_logits = target_model(target_input_ids, attention_mask=target_attention_mask).logits

            # Get assistant model logits
            if prompt_template and hypothesis:
                prompt = prompt_template.format(hypothesis=hypothesis)
                prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
                prompt_len = prompt_ids.shape[1]

                # Combine prompt with each text in the batch
                full_texts = [prompt + text for text in batch_texts]
                assistant_inputs = tokenizer(full_texts, return_tensors='pt', padding=True, truncation=True)
                assistant_input_ids = assistant_inputs.input_ids.to(device)
                assistant_attention_mask = assistant_inputs.attention_mask.to(device)
                
                assistant_logits_full = assistant_model(assistant_input_ids, attention_mask=assistant_attention_mask).logits
                # Align logits: remove prompt part from assistant logits
                assistant_logits = assistant_logits_full[:, prompt_len:prompt_len + target_logits.shape[1], :]
            else:
                assistant_logits = assistant_model(target_input_ids, attention_mask=target_attention_mask).logits

            # Align sequence lengths if they differ after slicing
            min_seq_len = min(target_logits.shape[1], assistant_logits.shape[1])
            target_logits = target_logits[:, :min_seq_len, :]
            assistant_logits = assistant_logits[:, :min_seq_len, :]

            # Mask out padding tokens from KL calculation
            # We use the target mask, shifted to align with logits (predicting next token)
            mask = target_attention_mask[:, 1:min_seq_len + 1].unsqueeze(-1).expand_as(target_logits)
            
            # Log-softmax the logits for KLDivLoss
            log_p_target = torch.nn.functional.log_softmax(target_logits, dim=-1)
            log_p_assistant = torch.nn.functional.log_softmax(assistant_logits, dim=-1)

            # Apply mask
            masked_log_p_target = log_p_target * mask
            masked_log_p_assistant = log_p_assistant * mask

            kl_div = kl_loss_fn(masked_log_p_assistant, masked_log_p_target)
            total_kl_div += kl_div.item() * len(batch_texts)

    return total_kl_div / num_texts


def assistant_generative_compare_KL_div(
        difference_descriptions: List[str],
        models_generated_strs: List[Tuple[List[str], List[str]]],
        local_model: AutoModel,
        labeling_tokenizer: AutoTokenizer,
        starting_model: AutoModel,
        comparison_model: AutoModel,
        device: str = "cuda:0",
        batch_size: int = 8,
) -> List[float]:
    """
    Measures the information content of hypotheses by evaluating how much they help an assistant
    model imitate a starting and a comparison model.

    The score for each hypothesis is based on the change in KL divergence between the assistant
    and the target models when the assistant is prompted with the hypothesis.

    Args:
        difference_descriptions (List[str]): List of hypotheses to evaluate.
        models_generated_strs (List[Tuple[List[str], List[str]]]): For each hypothesis, a tuple
            containing a list of strings from the starting model and a list from the comparison model.
        local_model (AutoModel): The assistant model to be tested.
        labeling_tokenizer (AutoTokenizer): The tokenizer for all models.
        starting_model (AutoModel): The first model to be imitated.
        comparison_model (AutoModel): The second model to be imitated.
        device (str, optional): Device to run the models on. Defaults to "cuda:0".
        batch_size (int, optional): Batch size for loss computation. Defaults to 8.

    Returns:
        List[float]: A list of scores, one for each hypothesis.
    """
    if labeling_tokenizer.pad_token is None:
        labeling_tokenizer.pad_token = labeling_tokenizer.eos_token
        labeling_tokenizer.pad_token_id = labeling_tokenizer.eos_token_id

    all_scores = []

    for i, hypothesis in enumerate(tqdm(difference_descriptions, desc="Evaluating hypotheses with KL divergence")):
        starting_model_texts, comparison_model_texts = models_generated_strs[i]
        
        prompt_template_S = "Use the following hypothesis to imitate the starting model: {hypothesis}. Now complete the text: "
        prompt_template_C = "Use the following hypothesis to imitate the comparison model: {hypothesis}. Now complete the text: "

        # 1. KL(P_C || P_A) on D_C
        kl_C_no_h = _compute_avg_kl_divergence(local_model, comparison_model, labeling_tokenizer, comparison_model_texts, device, batch_size)
        
        # 2. KL(P_C || P_A_h_C) on D_C
        kl_C_with_h = _compute_avg_kl_divergence(local_model, comparison_model, labeling_tokenizer, comparison_model_texts, device, batch_size, prompt_template_C, hypothesis)
        
        # Improvement in imitating comparison model (lower KL is better)
        delta_C = kl_C_no_h - kl_C_with_h

        # 3. KL(P_S || P_A) on D_S
        kl_S_no_h = _compute_avg_kl_divergence(local_model, starting_model, labeling_tokenizer, starting_model_texts, device, batch_size)

        # 4. KL(P_S || P_A_h_S) on D_S
        kl_S_with_h = _compute_avg_kl_divergence(local_model, starting_model, labeling_tokenizer, starting_model_texts, device, batch_size, prompt_template_S, hypothesis)
        
        # Improvement in imitating starting model
        delta_S = kl_S_no_h - kl_S_with_h

        score = (delta_S + delta_C) / 2
        all_scores.append(score)

    return all_scores



# Uses assistant_generative_compare to generate the correlation coefficients / AUCs representing how well LM scores function to differentiate between the texts generated for one model and the other using the different descriptions. Optionally computes p-values for the descriptions scores.
def validated_assistant_generative_compare(
        difference_descriptions: List[str], 
        local_model: AutoModel, 
        labeling_tokenizer: AutoTokenizer, 
        api_provider: str,
        api_model_str: str,
        auth_key: str,
        api_stronger_model_str: str = None,
        client: Optional[Union[Anthropic, OpenAI, Client]] = None,
        starting_model_str: Optional[str] = None,
        comparison_model_str: Optional[str] = None,
        common_tokenizer_str: str = "meta-llama/Meta-Llama-3-8B",
        starting_model: Optional[AutoModel] = None,
        comparison_model: Optional[AutoModel] = None,
        device: str = "cuda:0",
        use_correlation_coefficient: bool = True,
        num_permutations: int = 1,
        use_normal_distribution_for_p_values: bool = False,
        num_generated_texts_per_description: int = 10,
        num_rephrases_for_validation: int = 0,
        return_generated_texts: bool = False,
        bnb_config: BitsAndBytesConfig = None,
        api_interactions_save_loc: Optional[str] = None,
        logging_level: str = "INFO",
        logger: Optional[BoundLoggerLazyProxy] = None
    ) -> Union[
        Tuple[List[List[float]], List[List[float]], List[List[str]]],
        Tuple[List[List[float]], List[List[float]], List[List[str]], List[List[List[str]]], List[List[List[str]]]]
        ]:
    """
    Validate the ability of an assistant model to generate texts that discriminate between two language models based on given descriptions.

    This function uses the assistant_generative_compare function to generate texts and compute correlation coefficients / AUC scores. It then either approximates the null distribution of scores using a normal distribution or performs a permutation test to assess the statistical significance of these scores.

    Optionally, if num_rephrases_for_validation > 0, the function will rephrase each of the descriptions in difference_descriptions num_rephrases_for_validation times and compute the correlation coefficients / AUCs and p-values for the rephrased descriptions.

    Args:
        difference_descriptions (List[str]): Descriptions of differences between the two language models.
        local_model (AutoModel): Local model for text generation if not using API.
        labeling_tokenizer (AutoTokenizer): Tokenizer for the local model.
        api_provider (str): API provider for text generation (e.g., 'openai').
        api_model_str (str): Model string for API requests.
        auth_key (str): Authentication key for API requests.
        api_stronger_model_str (str): Model string for an optional stronger API model for hypothesisrephrasing only. Defaults to None.
        client (Optional[Union[Anthropic, OpenAI, Client]], optional): API client for text
            generation. Defaults to None.
        starting_model_str (Optional[str]): Identifier for the first language model. Defaults to None.
        comparison_model_str (Optional[str]): Identifier for the second language model. Defaults to None.
        common_tokenizer_str (str): Identifier for the tokenizer used by both language models.
            Defaults to "meta-llama/Meta-Llama-3-8B".
        starting_model (Optional[AutoModel]): Model that generates texts for the first cluster.
            Defaults to None.
        comparison_model (Optional[AutoModel]): Model that generates texts for the second cluster.
            Defaults to None.
        device (str, optional): Device to use for computations (e.g., 'cuda:0'). Defaults to "cuda:0".
        use_correlation_coefficient (bool, optional): If True, use the correlation coefficient 
            between the difference in LM score and the intended target during generation as our
            indicator of label performance. If false, use the AUCs of the labels as predictors of
            which model the texts were generated for. Defaults to True.
        num_permutations (int, optional): Number of permutations for the null distribution. Defaults to 1.
        use_normal_distribution_for_p_values (bool, optional): If True, use normal distribution to 
            compute p-values and avoid having to compute the empirical distribution. If False, use 
            empirical distribution. Defaults to False.
        num_generated_texts_per_description (int, optional): Number of texts to generate per description. 
            Defaults to 10.
        num_rephrases_for_validation (int, optional): Number of rephrases of each generated hypothesis 
            to generate for validation. Defaults to 0.
        return_generated_texts (bool, optional): If True, return the generated texts. Defaults to False.
        bnb_config (BitsAndBytesConfig, optional): Configuration for quantization. Defaults to None.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
    Returns:
        If return_generated_texts is False:
            Tuple[List[List[float]], List[List[float]], List[List[str]]]: A tuple containing:
                - real_scores: List of lists of correlation coefficients / AUCs for each description (entry 0) and each rephrase (entries > 0).
                - p_values: List of lists of p-values corresponding to each correlation coefficient / AUC (entry 0) and each rephrase (entries > 0).
                - tested_descriptions: List of lists of descriptions that were tested. Original description at entry 0, rephrased descriptions at entries > 0.
        If return_generated_texts is True:
            Tuple[List[List[float]], List[List[float]], List[List[str]], List[List[List[str]]], List[List[List[str]]]]: A tuple containing:
                - real_scores: List of lists of correlation coefficients / AUCs for each description (entry 0) and each rephrase (entries > 0).
                - p_values: List of lists of p-values corresponding to each correlation coefficient / AUC (entry 0) and each rephrase (entries > 0).
                - tested_descriptions: List of lists of descriptions that were tested. Original description at entry 0, rephrased descriptions at entries > 0.
                - generated_texts_1: 3-layer nested list of texts generated for the first model. First layer is for each description, second layer is for each rephrase, third layer is for each text. Original texts at entry 0 of the second layer, rephrased texts at entries > 0 of the second layer.
                - generated_texts_2: 3-layer nested list of texts generated for the second model. First layer is for each description, second layer is for each rephrase, third layer is for each text. Original texts at entry 0 of the second layer, rephrased texts at entries > 0 of the second layer.
    """
    # First, generate any required rephrased descriptions
    if num_rephrases_for_validation > 0:
        total_descriptions = []
        rephrase_model_str = api_stronger_model_str if api_stronger_model_str is not None else api_model_str
        for i, description in enumerate(difference_descriptions):
            total_descriptions.append([description])
            total_descriptions[i].extend(
                rephrase_description(
                    description, 
                    api_provider, 
                    rephrase_model_str, 
                    auth_key,
                    client=client,
                    num_rephrases=num_rephrases_for_validation,
                    max_tokens=4096,
                    api_interactions_save_loc=api_interactions_save_loc,
                    logging_level=logging_level,
                    logger=logger
                )
            )
        difference_descriptions = total_descriptions
        print("difference_descriptions", difference_descriptions)
    else:
        difference_descriptions = [difference_descriptions]
    # First, generate the correlation coefficients / AUCs using the real labels
    all_real_scores = []
    all_p_values = []
    all_generated_texts_1 = []
    all_generated_texts_2 = []
    for description_set in difference_descriptions:
        real_scores, generated_texts_1, generated_texts_2 = assistant_generative_compare(
            description_set, 
            local_model, 
            labeling_tokenizer, 
            api_provider, 
            api_model_str, 
            auth_key, 
            client=client,
            starting_model_str=starting_model_str, 
            comparison_model_str=comparison_model_str, 
            common_tokenizer_str=common_tokenizer_str, 
            starting_model=starting_model,
            comparison_model=comparison_model,
            device=device, 
            use_correlation_coefficient=use_correlation_coefficient,
            num_generated_texts_per_description=num_generated_texts_per_description, 
            bnb_config=bnb_config,
            api_interactions_save_loc=api_interactions_save_loc,
            logging_level=logging_level,
            logger=logger
        )
        all_real_scores.append(real_scores)
        all_generated_texts_1.append(generated_texts_1)
        all_generated_texts_2.append(generated_texts_2)
    if not use_normal_distribution_for_p_values:
        # Now, perform the permutation test
        all_permuted_scores = []
        for i, description_set in enumerate(difference_descriptions):
            for _ in range(num_permutations):
                fake_scores, _, _ = assistant_generative_compare(
                    description_set, 
                    local_model, 
                    labeling_tokenizer, 
                    api_provider, 
                    api_model_str, 
                    auth_key, 
                    client=client,
                    starting_model_str=starting_model_str, 
                    comparison_model_str=comparison_model_str, 
                    common_tokenizer_str=common_tokenizer_str, 
                    starting_model=starting_model,
                    comparison_model=comparison_model,
                    device=device, 
                    use_correlation_coefficient=use_correlation_coefficient,
                    num_generated_texts_per_description=num_generated_texts_per_description, 
                    permute_labels=True, 
                    bnb_config=bnb_config,
                    api_interactions_save_loc=api_interactions_save_loc,
                    logging_level=logging_level,
                    logger=logger
                )
                all_permuted_scores.extend(fake_scores)
            all_p_values.append([np.sum(np.array(all_permuted_scores) > current_rephrase_real_score) / len(all_permuted_scores) for current_rephrase_real_score in all_real_scores[i]])
    # Now, compute the p-values using a normal distribution matched to the std of all the real correlation coefficients / AUCs
    else:
        null_mean = 0.0 if use_correlation_coefficient else 0.5
        flattened_real_scores = [item for sublist in all_real_scores for item in sublist]
        null_std = np.std(flattened_real_scores)

        for i, description_set in enumerate(difference_descriptions):
            p_values = []
            for current_rephrase_real_score in all_real_scores[i]:
                p_value = 1 - stats.norm.cdf(current_rephrase_real_score, loc=null_mean, scale=null_std)
                p_values.append(p_value)
            all_p_values.append(p_values)

    if return_generated_texts:
        return all_real_scores, all_p_values, difference_descriptions, all_generated_texts_1, all_generated_texts_2
    else:
        return all_real_scores, all_p_values, difference_descriptions


def generate_responses_in_batches(
    model: AutoModel,
    queries: List[str],
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    responses_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    only_return_generated_text: bool = True,
    join_str: str = "Response A"
) -> List[str]:
    """
    Generate responses for a list of queries in batches to manage memory usage.
    
    Args:
        model: The model to use for generation
        queries: List of input queries
        tokenizer: Tokenizer for the model
        batch_size: Number of queries to process at once
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        responses_per_prompt: Number of responses each model generates for each prompt.
        device: Device to run inference on (defaults to model's device)
        only_return_generated_text: Whether to only return the generated text, or to return the prompt and generated text.
        join_str: String to join the responses with in case of multiple responses per prompt.
    Returns:
        List[str]: List of generated responses corresponding to input queries
    """
    if device is None:
        device = next(model.parameters()).device
        
    all_responses = []
    
    try:
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            inputs = tokenizer(batch_queries, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    num_return_sequences=responses_per_prompt
                )
            outputs = outputs.cpu()

            if only_return_generated_text:
                # Only decode the newly generated tokens for each sequence
                input_lengths = [len(seq) for seq in inputs['input_ids']]
                # Expand input_lengths to match number of responses per prompt
                input_lengths = [length for length in input_lengths for _ in range(responses_per_prompt)]
                batch_responses = [
                    tokenizer.decode(output[input_len:], skip_special_tokens=True)
                    for output, input_len in zip(outputs, input_lengths)
                ]
            else:
                # Decode the entire sequence including the prompt
                batch_responses = [
                    tokenizer.decode(output, skip_special_tokens=True) 
                    for output in outputs
                ]
            del inputs, outputs
            # Reshape responses to group them by original prompt
            # Since we generate responses_per_prompt responses for each input
            reshaped_responses = []
            for j in range(0, len(batch_responses), responses_per_prompt):
                start_idx = j
                end_idx = min(j + responses_per_prompt, len(batch_responses))
                prompt_responses = batch_responses[start_idx:end_idx]
                # Join multiple responses with newlines
                combined_response = ''
                for k in range(len(prompt_responses)):
                    combined_response += f'<{join_str} {k+1}>\n{prompt_responses[k]}</{join_str} {k+1}>\n'
                reshaped_responses.append(combined_response)
            all_responses.extend(reshaped_responses)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

        return all_responses
        
    except Exception as e:
        raise RuntimeError(f"Error in batch generation: {str(e)}")


def validated_embeddings_discriminative_single_unknown_ICL(
    difference_descriptions: List[str],
    api_provider: str,
    api_model_str: str,
    auth_key: Optional[str] = None,
    client: Optional[Union[Anthropic, OpenAI, Client]] = None,
    common_tokenizer_str: str = "meta-llama/Meta-Llama-3-8B",
    starting_model: Optional[AutoModel] = None,
    comparison_model: Optional[AutoModel] = None,
    num_rounds: int = 3,
    num_validation_runs: int = 5,
    responses_per_prompt: int = 1,
    on_prompt_responses_to_show_with_model_names: int = 0,
    max_retries: int = 2,
    max_tokens: int = 1000,
    max_decoding_length: int = 96,
    api_interactions_save_loc: Optional[str] = None,
    logging_level: str = "INFO",
    logger: Optional["BoundLoggerLazyProxy"] = None,
    num_workers: int = 1,
    model_batch_size: int = 16,
    embedding_model_str: str = "nvidia/NV-Embed-v2",
    embedding_tokenizer_str: Optional[str] = None,
    bnb_config: BitsAndBytesConfig = None,
    distance_metric: str = "cosine",
    recompute_embeddings: bool = False,
    embedding_batch_size: int = 16,
    tqdm_disable: bool = True,
    device: str = "cuda:0"
) -> Dict[str, Any]:
    """
    This variation replaces the final assistant guess with an embedding-based similarity check:
    
    1) The external assistant is still used to produce a new query each round.
    2) We generate responses from Model 1 and Model 2, pick one as Model X, and
       show the assistant that we "did it", but we do NOT let the assistant guess.
    3) Instead, we parse the newly generated responses, compute embeddings for each
       set (M1, M2, X), and guess X's identity by whichever embedding centroid
       is closer.
    4) Repeat for all rounds, compute accuracy, etc.
    """
    import random
    import numpy as np
    from scipy.stats import binomtest
    import torch
    import pickle
    from tqdm import tqdm

    # ----------------------------------------------------------------
    # 0) Basic checks and setup
    # ----------------------------------------------------------------
    if starting_model is None or comparison_model is None:
        raise ValueError("Both starting_model and comparison_model must be provided.")
    if num_rounds < 1 or num_validation_runs < 1:
        raise ValueError("num_rounds and num_validation_runs must each be at least 1.")

    # Load tokenizer for local generation
    try:
        common_tokenizer = AutoTokenizer.from_pretrained(common_tokenizer_str, padding_side="left")
        if common_tokenizer.pad_token is None:
            common_tokenizer.pad_token = common_tokenizer.eos_token
            common_tokenizer.pad_token_id = common_tokenizer.eos_token_id
    except Exception as e:
        raise RuntimeError(f"Failed to load generation tokenizer: {e}")

    device = next(starting_model.parameters()).device

    # Select the discriminator API model to use
    if ':' in api_model_str:
        discriminator_api_model_str = api_model_str.split(':')[0]
    else:
        discriminator_api_model_str = api_model_str

    # Prepare test cases
    test_cases = []
    for desc_idx, description in enumerate(difference_descriptions):
        for run_idx in range(num_validation_runs):
            test_cases.append({
                "desc_idx": desc_idx,
                "description": description,
                "run_idx": run_idx,
                "conversation_history": [],
                "correctness_list": [],
                "status": "active"
            })

    # Track correctness for each hypothesis
    all_predictions_accurate = [[] for _ in difference_descriptions]

    # Track accuracies for each hypothesis across rounds
    all_round_accuracies = []

    # For convenience in text messages
    responses_str = "responses" if responses_per_prompt > 1 else "response"
    responses_str_on_prompt = (
        "responses" if on_prompt_responses_to_show_with_model_names > 1 else "response"
    )

    # ----------------------------------------------------------------
    # 1) Optionally load an embedding model (8-bit)
    # ----------------------------------------------------------------
    # Only load if we are actually going to embed. You can load once up front to save repeated overhead.
    # If you want to minimize memory usage, you could load/unload inside each round, but that's slower.
    embedding_model = None
    embedding_tokenizer = None

    # If using a local embedding model:
    if embedding_model_str:
        # You can override the tokenizer string if needed
        if embedding_tokenizer_str is None:
            embedding_tokenizer_str = embedding_model_str

    # A small helper to parse out the individual responses from the joined string
    # used by generate_responses_in_batches. By default, that function uses e.g.
    # `join_str="Model 1 Response REPLACE_WITH_INDICATOR"`.
    def split_joined_responses(
        joined_str: str,
        join_str: str = "REPLACE_WITH_INDICATOR"
    ) -> List[str]:
        """
        Split the concatenated string of multiple responses into a list of separate responses.
        The join_str might include some label text like 'Model 1 Response REPLACE_WITH_INDICATOR'.
        Here we just split on 'REPLACE_WITH_INDICATOR' to get each chunk.
        """
        # If the model generation included the literal join_str, we can split on it:
        parts = joined_str.split(join_str)
        # Clean up whitespace. Some parts might be empty if join_str was at the end.
        parts = [p.strip() for p in parts if p.strip()]
        return parts

    # ----------------------------------------------------------------
    # Main round loop
    # ----------------------------------------------------------------
    for round_idx in range(num_rounds):
        # Gather all active test cases
        active_test_cases = [tc for tc in test_cases if tc["status"] == "active"]
        if not active_test_cases:
            print("No active test cases left.")
            break

        # ------------------------------------------------------------
        # (1) Prompt the assistant to produce the next query
        # ------------------------------------------------------------
        plan_query_prompts = []
        for test_case in active_test_cases:
            interaction_text = "\n".join(test_case["conversation_history"])

            # Keep only the relevant conversation for the plan prompt if desired;
            # here we just pass it all through.
            filtered_history = []
            for entry in test_case["conversation_history"]:
                if (
                    entry.startswith("[Assistant planning+query")
                    or entry.startswith("[Model responses")
                    or entry.startswith("[Assistant guess")
                    or entry.startswith("[Feedback")
                ):
                    filtered_history.append(entry)
            interactions = "\n".join(filtered_history)

            # Construct instructions to the assistant
            planning_prompt_str = (
                "Please plan how you will continue investigating. Then write your next prompt "
                "prefixed by '**Prompt:**'. You may include reasoning above, but only the text "
                "after '**Prompt:**' will actually be shown to the two models.\n\n"
            )

            if on_prompt_responses_to_show_with_model_names > 0:
                how_we_show_responses_str = (
                    f"You will be shown your new prompt's responses from both models, labeled properly, "
                    f"({on_prompt_responses_to_show_with_model_names} {responses_str_on_prompt} each), then see "
                    f"{responses_per_prompt} additional {responses_str} from 'Model X' (identity hidden). "
                    "We will record which is correct automatically."
                )
            else:
                how_we_show_responses_str = (
                    f"You will be shown your new prompt's responses from both models, labeled, then see "
                    f"{responses_per_prompt} additional {responses_str} from 'Model X'."
                )

            prompt = (
                f"You are analyzing two models (Model 1 and Model 2) to see if they differ "
                f"according to this hypothesis:\n\n"
                f"<hypothesis>{test_case['description']}</hypothesis>\n\n"
                f"There will be {num_rounds} rounds of interaction in total. Each round:\n"
                " 1. You create a prompt.\n"
                f" 2. That prompt is shown to both models.\n"
                f" 3. {how_we_show_responses_str}\n"
                f"\nSo far, your full interaction history is:\n<history>{interactions}</history>\n\n"
                f"{planning_prompt_str}"
                "Your goal is to find prompts that reliably expose the difference (if any) between Model 1 "
                "and Model 2 as described by the hypothesis. Let's proceed."
            )
            plan_query_prompts.append(prompt)

        # Request the assistant's next prompt
        plan_query_responses = parallel_make_api_requests(
            prompts=plan_query_prompts,
            api_provider=api_provider,
            api_model_str=discriminator_api_model_str,
            auth_key=auth_key,
            client=client,
            api_interactions_save_loc=api_interactions_save_loc,
            logging_level=logging_level,
            logger=logger,
            num_workers=num_workers,
            request_info={
                "pipeline_stage": "discriminative_validation_ICL_single_unknown",
                "round": str(round_idx),
                "batch": "plan_query"
            },
            max_retries=max_retries,
            max_tokens=max_tokens
        )

        # Parse out the actual query from the assistant's response
        queries_for_local_model = []
        for test_case, plan_resp in zip(active_test_cases, plan_query_responses):
            test_case["conversation_history"].append(
                f"[Assistant planning+query, Round {round_idx}]:\n{plan_resp}"
            )
            if "**Prompt:**" in plan_resp:
                query_text = plan_resp.split("**Prompt:**", 1)[-1].strip()
                # If the assistant put extra lines, you can strip or keep them
                query_text = query_text.split("\n", 1)[0].strip()
            else:
                print(
                    f"Warning: No '**Prompt:**' found in plan response (round {round_idx})."
                )
                query_text = "ERROR: No query found."
                test_case["status"] = "failed"
            queries_for_local_model.append(query_text)

        # ------------------------------------------------------------
        # (2) Generate responses from both local models
        # ------------------------------------------------------------
        try:
            active_test_cases = [tc for tc in active_test_cases if tc["status"] == "active"]

            # --- Generate responses from Model 1 (hidden / "X" portion) ---
            model1_responses = generate_responses_in_batches(
                model=starting_model,
                queries=queries_for_local_model,
                tokenizer=common_tokenizer,
                batch_size=model_batch_size,
                device=device,
                max_new_tokens=max_decoding_length,
                responses_per_prompt=responses_per_prompt,
                join_str="Model 1 Response REPLACE_WITH_INDICATOR"  # For parsing later
            )

            # --- Generate responses from Model 2 (hidden / "X" portion) ---
            model2_responses = generate_responses_in_batches(
                model=comparison_model,
                queries=queries_for_local_model,
                tokenizer=common_tokenizer,
                batch_size=model_batch_size,
                device=device,
                max_new_tokens=max_decoding_length,
                responses_per_prompt=responses_per_prompt,
                join_str="Model 2 Response REPLACE_WITH_INDICATOR"
            )

            # Optionally: generate "on-prompt" revealed responses
            if on_prompt_responses_to_show_with_model_names > 0:
                model1_responses_on_prompt = generate_responses_in_batches(
                    model=starting_model,
                    queries=queries_for_local_model,
                    tokenizer=common_tokenizer,
                    batch_size=model_batch_size,
                    device=device,
                    max_new_tokens=max_decoding_length,
                    responses_per_prompt=on_prompt_responses_to_show_with_model_names,
                    join_str="Model 1 Response REPLACE_WITH_INDICATOR"
                )
                model2_responses_on_prompt = generate_responses_in_batches(
                    model=comparison_model,
                    queries=queries_for_local_model,
                    tokenizer=common_tokenizer,
                    batch_size=model_batch_size,
                    device=device,
                    max_new_tokens=max_decoding_length,
                    responses_per_prompt=on_prompt_responses_to_show_with_model_names,
                    join_str="Model 2 Response REPLACE_WITH_INDICATOR"
                )
            else:
                model1_responses_on_prompt = [""] * len(active_queries)
                model2_responses_on_prompt = [""] * len(active_queries)

        except Exception as e:
            print(f"Error generating responses from local models: {e}")
            for tc in active_test_cases:
                tc["status"] = "failed"
            # Provide placeholders
            model1_responses = ["ERROR"] * len(active_test_cases)
            model2_responses = ["ERROR"] * len(active_test_cases)
            model1_responses_on_prompt = ["" for _ in range(len(active_test_cases))]
            model2_responses_on_prompt = ["" for _ in range(len(active_test_cases))]

        # ------------------------------------------------------------
        # (3) Randomly pick one model's response as Model X
        #     Then embed & classify
        # ------------------------------------------------------------
        # Instead of prompting the assistant to guess, we do an embedding-based approach.
        print(f"Loading embedding model {embedding_model_str} in 8-bit (if possible)")
        embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_tokenizer_str)
        embedding_model = AutoModel.from_pretrained(
            embedding_model_str,
            quantization_config=bnb_config, 
            trust_remote_code=True,     # if needed for custom architecture
            device_map={"": 0} if device == "cuda:0" else "auto"
        )
        embedding_model.eval()
        random_choices = []
        for i, test_case in enumerate(active_test_cases):
            if test_case["status"] != "active":
                random_choices.append(None)
                continue

            resp1 = model1_responses[i]
            resp2 = model2_responses[i]
            resp1_on_prompt = (
                model1_responses_on_prompt[i] if on_prompt_responses_to_show_with_model_names > 0 else ""
            )
            resp2_on_prompt = (
                model2_responses_on_prompt[i] if on_prompt_responses_to_show_with_model_names > 0 else ""
            )

            # Save them in conversation with correct labels
            test_case["conversation_history"].append(
                f"[Model responses, Round {round_idx}]:\n"
                f"Model 1 {responses_str}:\n{resp1}\n\n"
                f"Model 2 {responses_str}:\n{resp2}"
            )

            # Randomly pick which model is X
            if random.random() < 0.5:
                x_model_number = 1
                x_resp_str = resp1_on_prompt
            else:
                x_model_number = 2
                x_resp_str = resp2_on_prompt
            random_choices.append(x_model_number)

            # (a) If you want to show "on_prompt" labeled responses in conversation:
            if on_prompt_responses_to_show_with_model_names > 0:
                test_case["conversation_history"].append(
                    f"On-prompt labeled responses:\n"
                    f"Model 1:\n{resp1_on_prompt}\n\n"
                    f"Model 2:\n{resp2_on_prompt}\n"
                )

            # (b) Now parse out the individual responses from each string so we can embed them
            # The string may contain multiple responses joined by "REPLACE_WITH_INDICATOR".
            # For example, if responses_per_prompt=3, model1_responses[i] is a single string
            # that includes all 3 Model 1 completions. We'll split them out:
            m1_parts = split_joined_responses(resp1)
            m2_parts = split_joined_responses(resp2)
            x_parts = split_joined_responses(x_resp_str)

            # (c) Embed each set of parts
            
            if m1_parts and m2_parts and x_parts and embedding_model is not None:
                m1_embs = read_past_embeddings_or_generate_new(
                    path=None, 
                    client=None, 
                    decoded_strs=m1_parts, 
                    local_embedding_model_str=embedding_model_str, 
                    local_embedding_model=embedding_model, 
                    tokenizer=embedding_tokenizer, 
                    device=device, 
                    recompute_embeddings=True, 
                    batch_size=embedding_batch_size, 
                    save_embeddings=False, 
                    tqdm_disable=True, 
                )
                m2_embs = read_past_embeddings_or_generate_new(
                    path=None, 
                    client=None, 
                    decoded_strs=m2_parts, 
                    local_embedding_model_str=embedding_model_str, 
                    local_embedding_model=embedding_model, 
                    tokenizer=embedding_tokenizer, 
                    device=device, 
                    recompute_embeddings=True, 
                    batch_size=embedding_batch_size, 
                    save_embeddings=False, 
                    tqdm_disable=True, 
                )
                x_embs  = read_past_embeddings_or_generate_new(
                    path=None, 
                    client=None, 
                    decoded_strs=x_parts, 
                    local_embedding_model_str=embedding_model_str, 
                    local_embedding_model=embedding_model, 
                    tokenizer=embedding_tokenizer, 
                    device=device, 
                    recompute_embeddings=True, 
                    batch_size=embedding_batch_size, 
                    save_embeddings=False, 
                    tqdm_disable=True, 
                )

                # average each group
                m1_mean = np.mean(m1_embs, axis=0)
                m2_mean = np.mean(m2_embs, axis=0)
                x_mean  = np.mean(x_embs,  axis=0)

                # (d) Compare distances
                # If distance_metric == 'cosine', we can do 1 - cos_sim
                # or for L2, do np.linalg.norm(...)
                if distance_metric == "cosine":
                    # cos_sim(a,b) = a·b / (||a||*||b||)
                    # We'll treat smaller distance => more similar
                    def cos_sim(a, b):
                        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

                    dist_m1 = 1.0 - cos_sim(x_mean, m1_mean)
                    dist_m2 = 1.0 - cos_sim(x_mean, m2_mean)
                else:  # "l2"
                    dist_m1 = np.linalg.norm(x_mean - m1_mean)
                    dist_m2 = np.linalg.norm(x_mean - m2_mean)

                if dist_m1 < dist_m2:
                    predicted_model = 1
                else:
                    predicted_model = 2
            else:
                # If we had no valid strings or an error, guess randomly or set None
                predicted_model = None

            # (e) Record correctness
            correct_option = x_model_number
            if predicted_model in [1, 2]:
                correct_bool = int(predicted_model == correct_option)
                test_case["correctness_list"].append(correct_bool)
                test_case["conversation_history"].append(
                    f"[Embedding-based guess, Round {round_idx}]: Predicted Model X = Model {predicted_model}; "
                    f"True identity = Model {correct_option}. "
                    f"{'CORRECT' if correct_bool else 'WRONG'}.\n"
                )
            else:
                test_case["conversation_history"].append(
                    f"[Embedding-based guess, Round {round_idx}]: No valid guess parsed.\n"
                )

        del embedding_model
        # clear memory
        torch.cuda.empty_cache()
        print(f"Round {round_idx} complete.")

        # ------------------------------------------------------------
        # (4) Compute per-hypothesis accuracy for this round
        # ------------------------------------------------------------
        round_accuracies = []
        for desc_idx in range(len(difference_descriptions)):
            # gather correctness for test cases belonging to this hypothesis
            round_predictions = []
            for test_case in test_cases:
                if (
                    test_case["desc_idx"] == desc_idx
                    and len(test_case["correctness_list"]) > round_idx
                ):
                    round_predictions.append(test_case["correctness_list"][round_idx])
            if round_predictions:
                accuracy = float(np.mean(round_predictions))
            else:
                accuracy = 0.0
            round_accuracies.append(accuracy)

        all_round_accuracies.append(round_accuracies)

    # ----------------------------------------------------------------
    # Aggregate across all rounds & compute stats
    # ----------------------------------------------------------------
    for test_case in test_cases:
        desc_idx = test_case["desc_idx"]
        all_predictions_accurate[desc_idx].extend(test_case["correctness_list"])

    # Compute accuracy and binomial p-values for each hypothesis
    hypothesis_accuracies = []
    hypothesis_p_values = []
    for predictions in all_predictions_accurate:
        if len(predictions) == 0:
            hypothesis_accuracies.append(0.0)
            hypothesis_p_values.append(1.0)
            continue
        accuracy = float(np.mean(predictions))
        num_correct = int(np.sum(predictions))
        n = len(predictions)
        result = binomtest(num_correct, n, p=0.5, alternative='greater')
        pval = result.pvalue
        hypothesis_accuracies.append(accuracy)
        hypothesis_p_values.append(pval)

    # ----------------------------------------------------------------
    # Print out interaction histories for debugging
    # ----------------------------------------------------------------
    for test_case in test_cases:
        print(f"Hypothesis: {test_case['description']} (Run {test_case['run_idx']})")
        print("Interaction History:")
        for entry in test_case["conversation_history"]:
            print(entry)
        print("=" * 80)

    return {
        "hypothesis_accuracies": hypothesis_accuracies,
        "hypothesis_p_values": hypothesis_p_values,
        "all_round_accuracies": all_round_accuracies
    }
