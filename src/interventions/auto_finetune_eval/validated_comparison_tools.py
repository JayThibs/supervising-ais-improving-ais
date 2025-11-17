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


def seed_everything(seed: int) -> None:
    """
    Seed Python, NumPy and (optionally) PyTorch. Call once, early.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Optional: stricter determinism (slower; may error on some ops)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False

def get_max_labeling_tokens(api_provider: str, api_model_str: str) -> int:
    """
    Get the maximum number of tokens allowed for labeling a single contrastive label. Weaker models will have a lower maximum number of tokens.
    """
    if api_provider == "gemini":
        if 'flash-lite' in api_model_str:
            return 2000
        elif 'flash' in api_model_str:
            return 3000
        elif 'pro' in api_model_str:
            return 10000
        else:
            return 10000
    elif api_provider == "anthropic":
        if 'haiku' in api_model_str:
            return 2000
        elif 'sonnet' in api_model_str:
            return 3000
        elif 'opus' in api_model_str:
            return 10000
        else:
            return 10000
    elif api_provider == "openai":
        if 'nano' in api_model_str:
            return 2000
        elif 'mini' in api_model_str:
            return 3000
        else:
            return 10000

def contrastive_label_double_cluster(
        decoded_strs_1: List[str], 
        clustering_assignments_1: List[int], 
        cluster_id_1: int, 
        decoded_strs_2: List[str], 
        clustering_assignments_2: List[int], 
        cluster_id_2: int, 
        local_model: Optional[AutoModel] = None, 
        labeling_tokenizer: Optional[AutoTokenizer] = None, 
        api_provider: Optional[str] = None,
        api_model_str: Optional[str] = None,
        auth_key: Optional[str] = None,
        client: Optional[Union[Anthropic, OpenAI, Client]] = None,
        device: str = "cuda:0", 
        sampled_texts_per_cluster: int = 10, 
        generated_labels_per_cluster: int = 3, 
        max_labeling_tokens: int = None,
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_1: Dict = None,
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_2: Dict = None,
        num_decodings_per_prompt: int = None,
        contrastive_cluster_label_instruction: Optional[str] = None,
        current_label_diversification_content_str: Optional[str] = None,
        verified_diversity_promoter_labels: List[str] = None,
        api_interactions_save_loc: Optional[str] = None,
        logging_level: str = "INFO",
        logger: Optional[BoundLoggerLazyProxy] = None,
        split_clusters_by_prompt: bool = False,
        frac_prompts_for_label_generation: float = 0.5,
        base_decoded_texts_prompt_ids: List[int] = None,
        finetuned_decoded_texts_prompt_ids: List[int] = None
        ) -> Tuple[List[str], List[int], List[int]]:
    """
    Generate contrastive labels for two clusters of texts using either a local model or an API.

    This function samples texts from two clusters, then prepares an input string that tells either
    a local model or an API to generate labels that describe the key differences between the two clusters.

    Args:
        decoded_strs_1 (List[str]): List of decoded strings to which the first cluster belongs.
        clustering_assignments_1 (List[int]): Cluster assignments for decoded_strs_1.
        cluster_id_1 (int): ID of the first cluster to compare in decoded_strs_1.
        decoded_strs_2 (List[str]): List of decoded strings to which the second cluster belongs.
        clustering_assignments_2 (List[int]): Cluster assignments for decoded_strs_2.
        cluster_id_2 (int): ID of the matched comparison cluster in decoded_strs_2.
        local_model (AutoModel, optional): Deprecated.
        labeling_tokenizer (AutoTokenizer, optional): Deprecated.
        max_labeling_tokens (int, optional): Maximum number of tokens allowed for labeling a single contrastive label. Defaults to None.
            If None, will be determined based on the api_provider and api_model_str.
        api_provider (str, optional): API provider for text generation. Defaults to None.
        api_model_str (str, optional): Model string for API requests. Defaults to None.
        auth_key (str, optional): Authentication key for API requests. Defaults to None.
        client (Optional[Union[Anthropic, OpenAI, Client]]): The client to use for the API request.
        device (str, optional): Device to use for local model. Defaults to "cuda:0".
        sampled_texts_per_cluster (int, optional): Number of texts to sample from each cluster. 
            Defaults to 10.
        generated_labels_per_cluster (int, optional): Number of labels to generate. Defaults to 3.
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_1 (Dict, optional): Nested dict. First dict is indexed by cluster id.
            Leads to a dict indexed by prompt id, which leads to a list of indices for where the decodings of that
            prompt can be found in decoded_strs. This dict is for the decodings of the base model.
            Can be provided to make the label generation select only num_decodings_per_prompt decodings per prompt 
            to base the labels off of.
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_2 (Dict, optional): Nested dict. First dict is indexed by cluster id.
            Leads to a dict indexed by prompt id, which leads to a list of indices for where the decodings of that
            prompt can be found in decoded_strs. This dict is for the decodings of the finetuned model.
            Can be provided to make the label generation select only num_decodings_per_prompt decodings per prompt 
            to base the labels off of.
        num_decodings_per_prompt (int, optional): The number of decodings per prompt we use to generate labels,
            assuming the cluster_ids_to_prompt_ids_to_decoding_ids_dict was provided.
        contrastive_cluster_label_instruction (Optional[str]): Instruction for label generation using API. 
            Defaults to a predefined string if None is provided.
        current_label_diversification_content_str (Optional[str]): Instructions for label diversification. Contains a 
            description of the common themes across the previously generated labels, and a request to generate a 
            new label that touches on new themes. Defaults to None.
        verified_diversity_promoter_labels (List[str], optional): Verified diversity promoter labels. Can optionally be used 
            to encourage the assistant to generate labels that are more diverse by avoiding generating labels that are 
            similar to the verified diversity promoter labels. Defaults to None.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        split_clusters_by_prompt (bool, optional): Whether to split the clusters by prompt during discriminative 
            evaluation of the labels. If True, we will ensure no overlap in prompts between the label generation 
            and evaluation splits of each cluster. Defaults to False.
        frac_prompts_for_label_generation (float, optional): The fraction of prompts to use for label generation.
            Defaults to 0.5.
        base_decoded_texts_prompt_ids (List[int], optional): List of prompt IDs for the base decoded texts.
            Defaults to None.
        finetuned_decoded_texts_prompt_ids (List[int], optional): List of prompt IDs for the finetuned decoded texts.
            Defaults to None.
    Returns:
        Tuple[List[str], List[int], List[int]]: A tuple containing:
            - List of generated contrastive labels
            - List of indices of selected texts from cluster 1 used to generate the contrastive labels
            - List of indices of selected texts from cluster 2 used to generate the contrastive labels
    """
    if contrastive_cluster_label_instruction is None:
        contrastive_cluster_label_instruction = "You will be given two sets of texts generated by different LLM models. Concisely describe the key themes that differentiate the texts generated by these two models, based on the texts provided."
    if max_labeling_tokens is None:
        max_labeling_tokens = get_max_labeling_tokens(api_provider, api_model_str)

    cluster_indices_1 = [i for i, x in enumerate(clustering_assignments_1) if x == cluster_id_1]
    cluster_indices_2 = [i for i, x in enumerate(clustering_assignments_2) if x == cluster_id_2]

    # Track all selected indices across different samples
    all_selected_indices_1 = []
    all_selected_indices_2 = []
    decoded_labels = []

    for i in range(generated_labels_per_cluster):
        # Adjust sampled_texts_per_cluster if necessary
        actual_samples = min(sampled_texts_per_cluster, len(cluster_indices_1), len(cluster_indices_2))
        if actual_samples < sampled_texts_per_cluster:
            print(f"Warning: Sampling {actual_samples} texts instead of {sampled_texts_per_cluster} due to cluster size limitations.")

        if not cluster_ids_to_prompt_ids_to_decoding_ids_dict_1 is None and not cluster_ids_to_prompt_ids_to_decoding_ids_dict_2 is None and not num_decodings_per_prompt is None:
            # TODO: Implement this with split_clusters_by_prompt
            if split_clusters_by_prompt:
                raise NotImplementedError("split_clusters_by_prompt is not implemented for contrastive label generation when using cluster_ids_to_prompt_ids_to_decoding_ids_dict.")
            # For cluster 1
            num_prompts_to_sample = int(sampled_texts_per_cluster / num_decodings_per_prompt)
            prompt_ids_to_decoding_ids_dict_1 = cluster_ids_to_prompt_ids_to_decoding_ids_dict_1[cluster_id_1]
            prompts_indices_1 = list(prompt_ids_to_decoding_ids_dict_1.keys())
            # There may not be enough prompts to sample from, so we check and adjust if necessary
            if len(prompts_indices_1) < num_prompts_to_sample:
                num_prompts_to_sample = len(prompts_indices_1)
                current_num_decodings_per_prompt = int(sampled_texts_per_cluster / num_prompts_to_sample)
                print(f"Warning: There are fewer prompts in base model than requested. Sampling with num_prompts_to_sample = {num_prompts_to_sample} and current_num_decodings_per_prompt = {current_num_decodings_per_prompt}.")
            else:
                current_num_decodings_per_prompt = num_decodings_per_prompt
            prompts_indices_1 = np.random.choice(prompts_indices_1, num_prompts_to_sample, replace=False)
            all_decoding_ids_lists_1 = []
            for idx in prompts_indices_1:
                if len(prompt_ids_to_decoding_ids_dict_1[idx]) < current_num_decodings_per_prompt:
                    print(f"Warning: Prompt {idx} in base model has fewer than {current_num_decodings_per_prompt} decodings. Sampling with replacement instead. Note: this should *not* happen.")
                    print(f"Prompt {idx} in base model has {len(prompt_ids_to_decoding_ids_dict_1[idx])} decodings.")
                    print(f"Prompt {idx} in base model has indices {prompt_ids_to_decoding_ids_dict_1[idx]}")
                    current_ids_list = np.random.choice(prompt_ids_to_decoding_ids_dict_1[idx], current_num_decodings_per_prompt, replace=True)
                else:
                    current_ids_list = np.random.choice(prompt_ids_to_decoding_ids_dict_1[idx], current_num_decodings_per_prompt, replace=False)
                all_decoding_ids_lists_1.append(current_ids_list)
            
            selected_text_indices_1 = [idx for current_ids_list in all_decoding_ids_lists_1 for idx in current_ids_list]
            all_selected_indices_1.append(selected_text_indices_1)

            # For cluster 2
            num_prompts_to_sample = int(sampled_texts_per_cluster / num_decodings_per_prompt)
            prompt_ids_to_decoding_ids_dict_2 = cluster_ids_to_prompt_ids_to_decoding_ids_dict_2[cluster_id_2]
            prompts_indices_2 = list(prompt_ids_to_decoding_ids_dict_2.keys())
            # There may not be enough prompts to sample from, so we check and adjust if necessary
            if len(prompts_indices_2) < num_prompts_to_sample:
                num_prompts_to_sample = len(prompts_indices_2)
                current_num_decodings_per_prompt = int(sampled_texts_per_cluster / num_prompts_to_sample)
                print(f"Warning: There are fewer prompts in finetuned model than requested. Sampling with num_prompts_to_sample = {num_prompts_to_sample} and current_num_decodings_per_prompt = {current_num_decodings_per_prompt}.")
            else:
                current_num_decodings_per_prompt = num_decodings_per_prompt
            prompts_indices_2 = np.random.choice(prompts_indices_2, num_prompts_to_sample, replace=False)
            all_decoding_ids_lists_2 = []
            for idx in prompts_indices_2:
                if len(prompt_ids_to_decoding_ids_dict_2[idx]) < current_num_decodings_per_prompt:
                    print(f"Warning: Prompt {idx} in finetuned model has fewer than {current_num_decodings_per_prompt} decodings. Sampling with replacement instead. Note: this should *not* happen.")
                    print(f"Prompt {idx} in finetuned model has {len(prompt_ids_to_decoding_ids_dict_2[idx])} decodings.")
                    print(f"Prompt {idx} in finetuned model has indices {prompt_ids_to_decoding_ids_dict_2[idx]}")
                    current_ids_list = np.random.choice(prompt_ids_to_decoding_ids_dict_2[idx], current_num_decodings_per_prompt, replace=True)
                else:
                    current_ids_list = np.random.choice(prompt_ids_to_decoding_ids_dict_2[idx], current_num_decodings_per_prompt, replace=False)
            selected_text_indices_2 = [idx for current_ids_list in all_decoding_ids_lists_2 for idx in current_ids_list]
            all_selected_indices_2.append(selected_text_indices_2)

        else:
            # Sample different texts for each label
            if split_clusters_by_prompt:
                # First get the set of prompt IDs for the current clusters
                print(f"len(base_decoded_texts_prompt_ids): {len(base_decoded_texts_prompt_ids)}, len(finetuned_decoded_texts_prompt_ids): {len(finetuned_decoded_texts_prompt_ids)}")
                print(f"len(cluster_indices_1): {len(cluster_indices_1)}, len(cluster_indices_2): {len(cluster_indices_2)}")
                prompt_ids_1 = set([base_decoded_texts_prompt_ids[i] for i in cluster_indices_1])
                prompt_ids_2 = set([finetuned_decoded_texts_prompt_ids[i] for i in cluster_indices_2])
                # Then, assign half the texts to label generation and half to label validation
                num_base_prompts_for_label_generation = int(len(prompt_ids_1) * frac_prompts_for_label_generation)
                num_finetuned_prompts_for_label_generation = int(len(prompt_ids_2) * frac_prompts_for_label_generation)
                selected_prompt_ids_1 = np.random.choice(list(prompt_ids_1), num_base_prompts_for_label_generation, replace=False)
                selected_prompt_ids_2 = np.random.choice(list(prompt_ids_2), num_finetuned_prompts_for_label_generation, replace=False)
                # Then, split the cluster indices into two lists based on the prompt IDs. 
                # We only want indices that correspond to the prompt IDs for the label generation.
                valid_cluster_indices_1 = [i for i in cluster_indices_1 if base_decoded_texts_prompt_ids[i] in selected_prompt_ids_1]
                valid_cluster_indices_2 = [i for i in cluster_indices_2 if finetuned_decoded_texts_prompt_ids[i] in selected_prompt_ids_2]
                print(f"cluster_id_1: {cluster_id_1}, cluster_id_2: {cluster_id_2}, num_base_prompts_for_label_generation: {num_base_prompts_for_label_generation}, num_finetuned_prompts_for_label_generation: {num_finetuned_prompts_for_label_generation}, len(cluster_indices_1): {len(cluster_indices_1)}, len(cluster_indices_2): {len(cluster_indices_2)}, len(valid_cluster_indices_1): {len(valid_cluster_indices_1)}, len(valid_cluster_indices_2): {len(valid_cluster_indices_2)}")
                try:
                    selected_text_indices_1 = np.random.choice(valid_cluster_indices_1, actual_samples, replace=False)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"valid_cluster_indices_1: {valid_cluster_indices_1}")
                    print(f"cluster_indices_1: {cluster_indices_1}")
                    print(f"selected_prompt_ids_1: {selected_prompt_ids_1}")
                try:
                    selected_text_indices_2 = np.random.choice(valid_cluster_indices_2, actual_samples, replace=False)
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"valid_cluster_indices_2: {valid_cluster_indices_2}")
                    print(f"cluster_indices_2: {cluster_indices_2}")
                    print(f"selected_prompt_ids_2: {selected_prompt_ids_2}")
                # We'll also add all the indices of selected_prompt_ids_1 and selected_prompt_ids_2 to the all_selected_indices lists
                # so that we don't use them for label validation
                all_selected_indices_1.append(np.array(valid_cluster_indices_1))
                all_selected_indices_2.append(np.array(valid_cluster_indices_2))
            else:
                selected_text_indices_1 = np.random.choice(cluster_indices_1, actual_samples, replace=False)
                selected_text_indices_2 = np.random.choice(cluster_indices_2, actual_samples, replace=False)
                all_selected_indices_1.append(selected_text_indices_1)
                all_selected_indices_2.append(selected_text_indices_2)

        selected_texts_1 = [decoded_strs_1[i] for i in selected_text_indices_1]
        selected_texts_2 = [decoded_strs_2[i] for i in selected_text_indices_2]

        for j, text in enumerate(selected_texts_1):
            selected_texts_1[j] = text.replace("<s>", "").replace('\n', '\\n')
            selected_texts_1[j] = f"Model 1 Text {j}: " + selected_texts_1[j]
        for j, text in enumerate(selected_texts_2):
            selected_texts_2[j] = text.replace("<s>", "").replace('\n', '\\n')
            selected_texts_2[j] = f"Model 2 Text {j}: " + selected_texts_2[j]

        str_instruction_to_assistant_model = contrastive_cluster_label_instruction + "\n" + "Model 1 selected texts:\n" + '\n'.join(selected_texts_1) + "\nModel 2 selected texts:\n" + '\n'.join(selected_texts_2)

        if current_label_diversification_content_str is not None:
            str_instruction_to_assistant_model = str_instruction_to_assistant_model + "\n\n" + current_label_diversification_content_str
        
        if verified_diversity_promoter_labels is not None and len(verified_diversity_promoter_labels) > 0:
            str_instruction_to_assistant_model = str_instruction_to_assistant_model + "\n\n" + "Avoid generating labels that are similar to the following labels in content:\n" + "\n".join(verified_diversity_promoter_labels)
            str_instruction_to_assistant_model = str_instruction_to_assistant_model + "\nThe above labels have already been generated, so look for novel ways to describe the differences between the two sets of texts."
        
        str_instruction_to_assistant_model = str_instruction_to_assistant_model + "\n\nKeep the answer short and concise."

        
        if i > 0:
            # Add previously generated labels to the prompt
            str_instruction_with_history = str_instruction_to_assistant_model + "\n\nPreviously generated labels describing the differences between these sets (based on different samples):\n"
            str_instruction_with_history += "\n".join([f"{j+1}. {label}" for j, label in enumerate(decoded_labels)])
            str_instruction_with_history += "\n\nPlease provide a different label that focuses on patterns or themes not covered by the previous labels."
            current_prompt = str_instruction_with_history
        else:
            current_prompt = str_instruction_to_assistant_model
        

        
        label = make_api_request(
            current_prompt, 
            api_provider, 
            api_model_str, 
            auth_key, 
            client=client,
            max_tokens=max_labeling_tokens,
            max_thinking_tokens=10000,
            api_interactions_save_loc=api_interactions_save_loc,
            logging_level=logging_level,
            logger=logger,
            request_info={
                "pipeline_stage": "contrastive cluster label generation", 
                "cluster_id_1": str(cluster_id_1),
                "cluster_id_2": str(cluster_id_2),
                "label_number": str(i+1)
            },
        )
        decoded_labels.append(label)

    for i, label in enumerate(decoded_labels):
        print(f"Label {i}: {label}")
        if label.startswith(" "):
            decoded_labels[i] = label[1:]
            if "<s>" in label:
                decoded_labels[i] = label[:label.index("<s>")]

    return decoded_labels, all_selected_indices_1, all_selected_indices_2


def label_single_cluster(
        decoded_strs: List[str], 
        clustering_assignments: List[int], 
        cluster_id: int, 
        local_model: AutoModel = None, 
        labeling_tokenizer: AutoTokenizer = None, 
        max_labeling_tokens: int = None,
        api_provider: str = None,
        api_model_str: str = None,
        auth_key: str = None,
        client: Optional[Union[Anthropic, OpenAI, Client]] = None,
        device: str = "cuda:0", 
        sampled_texts_per_cluster: int = 10, 
        generated_labels_per_cluster: int = 3, 
        cluster_ids_to_prompt_ids_to_decoding_ids_dict: Dict = None,
        num_decodings_per_prompt: int = None,
        single_cluster_label_instruction: Optional[str] = None,
        cluster_strs_list: Optional[List[str]] = None,
        api_interactions_save_loc: Optional[str] = None,
        logging_level: str = "INFO",
        logger: Optional[BoundLoggerLazyProxy] = None
        ) -> Tuple[List[str], List[int]]:
    """
    Generate labels for a single cluster of texts using either a local model or an API.

    This function samples texts from a given cluster, then uses either a local model or an API
    to generate labels that describe the key themes of the sampled texts.

    Args:
        decoded_strs (List[str]): List of all decoded strings.
        clustering_assignments (List[int]): Cluster assignments for texts in decoded_strs.
        cluster_id (int): ID of the cluster to label in decoded_strs.
        local_model (AutoModel, optional): Local model for text generation. Defaults to None.
        labeling_tokenizer (AutoTokenizer, optional): Tokenizer for the local model. Defaults to None.
        max_labeling_tokens (int, optional): Maximum number of tokens allowed for labeling a single cluster label. Defaults to None.
            If None, will be determined based on the api_provider and api_model_str.
        api_provider (str, optional): API provider for text generation. Defaults to None.
        api_model_str (str, optional): Model string for API requests. Defaults to None.
        auth_key (str, optional): Authentication key for API requests. Defaults to None.
        client (Optional[Union[Anthropic, OpenAI, Client]]): The client to use for the API request.
        device (str, optional): Device to use for local model. Defaults to "cuda:0".
        sampled_texts_per_cluster (int, optional): Number of texts to sample from the cluster. 
            Defaults to 10.
        generated_labels_per_cluster (int, optional): Number of labels to generate. Defaults to 3.
        cluster_ids_to_prompt_ids_to_decoding_ids_dict (Dict, optional): Nested dict. First dict is indexed by cluster id.
            Leads to a dict indexed by prompt id, which leads to a list of indices for where the decodings of that
            prompt can be found in decoded_strs.
            Can be provided to make the label generation select only num_decodings_per_prompt decodings per prompt 
            to base the labels off of.
        num_decodings_per_prompt (int, optional): The number of decodings per prompt we use to generate labels,
            assuming the cluster_ids_to_prompt_ids_to_decoding_ids_dict was provided.
        single_cluster_label_instruction (Optional[str]): Instruction for label generation. Defaults 
            to a predefined string if None is provided.
        cluster_strs_list (Optional[List[str]]): Predefined list of strings for the cluster. 
            Defaults to None.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.

    Returns:
        Tuple[List[str], List[int]]: A tuple containing:
            - List of generated labels for the cluster
            - List of indices of selected texts from the cluster used to generate the labels

    Note:
        If cluster_strs_list is provided, it will be used instead of sampling from decoded_strs.
        The function uses either a local model or an API for text generation, depending on the provided arguments.
    """
    if single_cluster_label_instruction is None:
        single_cluster_label_instruction = "Concisely summarize the common themes of the texts shown to you. We are interested in the common themes of the set of texts these are drawn from, not the specific details of the texts we're showing you. So, only highlight the common themes, not the specific details of the texts we're showing you."
    if max_labeling_tokens is None:
        max_labeling_tokens = get_max_labeling_tokens(api_provider, api_model_str)
    if cluster_strs_list is not None:
        within_cluster_indices = list(range(len(cluster_strs_list)))
        selected_text_indices = np.random.choice(within_cluster_indices, sampled_texts_per_cluster, replace=False)
        selected_texts = [cluster_strs_list[i] for i in selected_text_indices]
    else:
        if not cluster_ids_to_prompt_ids_to_decoding_ids_dict is None and not num_decodings_per_prompt is None:
            num_prompts_to_sample = int(sampled_texts_per_cluster / num_decodings_per_prompt)
            prompt_ids_to_decoding_ids_dict = cluster_ids_to_prompt_ids_to_decoding_ids_dict[cluster_id]
            prompts_indices = list(prompt_ids_to_decoding_ids_dict.keys())
            # There may not be enough prompts to sample from, so we check and adjust if necessary
            if len(prompts_indices) < num_prompts_to_sample:
                num_prompts_to_sample = len(prompts_indices)
                current_num_decodings_per_prompt = int(sampled_texts_per_cluster / num_prompts_to_sample)
                print(f"Warning: There are fewer prompts in single cluster label generation than requested. Sampling with num_prompts_to_sample = {num_prompts_to_sample} and current_num_decodings_per_prompt = {current_num_decodings_per_prompt}.")
            else:
                current_num_decodings_per_prompt = num_decodings_per_prompt
            prompts_indices = np.random.choice(prompts_indices, num_prompts_to_sample, replace=False)
            all_decoding_ids_lists = []
            for idx in prompts_indices:
                if len(prompt_ids_to_decoding_ids_dict[idx]) < current_num_decodings_per_prompt:
                    print(f"Warning: Prompt {idx} has fewer than {current_num_decodings_per_prompt} decodings. Sampling with replacement instead. Note: this should *not* happen.")
                    print(f"Prompt {idx} has {len(prompt_ids_to_decoding_ids_dict[idx])} decodings.")
                    print(f"Prompt {idx} has indices {prompt_ids_to_decoding_ids_dict[idx]}")
                    current_ids_list = np.random.choice(prompt_ids_to_decoding_ids_dict[idx], current_num_decodings_per_prompt, replace=True)
                else:
                    current_ids_list = np.random.choice(prompt_ids_to_decoding_ids_dict[idx], current_num_decodings_per_prompt, replace=False)
                all_decoding_ids_lists.append(current_ids_list)
            selected_text_indices = [idx for current_ids_list in all_decoding_ids_lists for idx in current_ids_list]
        else:
            cluster_indices = [i for i, x in enumerate(clustering_assignments) if x == cluster_id]
            selected_text_indices = np.random.choice(cluster_indices, sampled_texts_per_cluster, replace=False)
        selected_texts = [decoded_strs[i] for i in selected_text_indices]
    for i, text in enumerate(selected_texts):
        selected_texts[i] = text.replace("<s>", "").replace('\n', '\\n')
        selected_texts[i] = f"Text {i}: " + selected_texts[i]

    # Use the assistant model to generate a cluster label for the selected texts
    # Generate input string for assistant model
    str_instruction_to_assistant_model = single_cluster_label_instruction + "\n" + "Texts in current set:\n" + '\n'.join(selected_texts)
    if api_provider is None:
        str_instruction_to_assistant_model = str_instruction_to_assistant_model + "\n" + "Set description: the common theme of the above texts is that they are all about"
        # Prepare inputs for text generation
        inputs = labeling_tokenizer(str_instruction_to_assistant_model, return_tensors="pt").to(device)
        inputs_length = inputs.input_ids.shape[1]
        # Generate labels using the Hugging Face text generation API
        with torch.no_grad():
            outputs = [local_model.generate(**inputs, max_new_tokens=15, num_return_sequences=1, do_sample=True, pad_token_id=labeling_tokenizer.eos_token_id) for _ in range(generated_labels_per_cluster)]
        # Decode labels to strings
        decoded_labels = [labeling_tokenizer.decode(output[0][inputs_length:], skip_special_tokens=True) for output in outputs]
    else:
        if single_cluster_label_instruction is None:
            str_instruction_to_assistant_model = str_instruction_to_assistant_model + "\n\nKeep the answer short and concise."
        decoded_labels = [
            make_api_request(
                str_instruction_to_assistant_model, 
                api_provider, 
                api_model_str, 
                auth_key, 
                client=client,
                max_tokens=max_labeling_tokens,
                max_thinking_tokens=10000,
                api_interactions_save_loc=api_interactions_save_loc,
                logging_level=logging_level,
                logger=logger,
                request_info={
                    "pipeline_stage": "single cluster label generation", 
                    "cluster_id": str(cluster_id)
                },
            ) for _ in range(generated_labels_per_cluster)
        ]
    for i, label in enumerate(decoded_labels):
        if label.startswith(" "):
            decoded_labels[i] = label[1:]
            if "<s>" in label:
                decoded_labels[i] = label[:label.index("<s>")]
    return decoded_labels, selected_text_indices

def get_cluster_labels_random_subsets(
        decoded_strs: List[str], 
        clustering_assignments: List[int], 
        local_model: AutoModel = None, 
        labeling_tokenizer: AutoTokenizer = None, 
        api_provider: str = None, 
        api_model_str: str = None, 
        auth_key: str = None, 
        client: Optional[Union[Anthropic, OpenAI, Client]] = None,
        device: str = "cuda:0", 
        sampled_texts_per_cluster: int = 10, 
        sampled_comparison_texts_per_cluster: int = 10, 
        generated_labels_per_cluster: int = 3,
        cluster_ids_to_prompt_ids_to_decoding_ids_dict: Dict = None,
        num_decodings_per_prompt: int = None,
        single_cluster_label_instruction: Optional[str] = None,
        api_interactions_save_loc: Optional[str] = None,
        logging_level: str = "INFO",
        logger: Optional[BoundLoggerLazyProxy] = None
        ) -> Tuple[Dict[int, List[str]], Dict[int, List[int]]]:
    """
    Generate labels for each cluster using random subsets of texts from the clusters.

    This function iterates through each cluster, selects a random subset of texts,
    and generates labels for those texts using either a local model or an API.

    Args:
        decoded_strs (List[str]): List of all decoded strings.
        clustering_assignments (List[int]): Cluster assignments for each string in decoded_strs.
        local_model (AutoModel, optional): Local model for text generation. Defaults to None.
        labeling_tokenizer (AutoTokenizer, optional): Tokenizer for the local model. Defaults to 
            None.
        api_provider (str, optional): API provider for text generation. Defaults to None.
        api_model_str (str, optional): Model string for API requests. Defaults to None.
        auth_key (str, optional): Authentication key for API requests. Defaults to None.
        client (Optional[Union[Anthropic, OpenAI, Client]]): The client to use for the API request.
        device (str, optional): Device to use for local model. Defaults to "cuda:0".
        sampled_texts_per_cluster (int, optional): Number of texts to sample for label generation. 
            Defaults to 10.
        sampled_comparison_texts_per_cluster (int, optional): Minimum number of additional texts required in 
            cluster. Defaults to 10.
        generated_labels_per_cluster (int, optional): Number of labels to generate per cluster. Defaults to 3.
        cluster_ids_to_prompt_ids_to_decoding_ids_dict (Dict, optional): Nested dict. First dict is indexed 
            by cluster id. Leads to a dict indexed by prompt id, which leads to a list of indices for where the 
            decodings of that prompt can be found in decoded_strs.
            Can be provided to make the label generation select only num_decodings_per_prompt decodings per prompt 
            to base the labels off of.
        num_decodings_per_prompt (int, optional): The number of decodings per prompt we use to generate labels,
            assuming the cluster_ids_to_prompt_ids_to_decoding_ids_dict was provided.
        single_cluster_label_instruction (Optional[str]): Instructions for generating the single cluster labels.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
    Returns:
        Tuple[Dict[int, List[str]], Dict[int, List[int]]]: A tuple containing:
            - Dictionary mapping cluster IDs to lists of generated labels.
            - Dictionary mapping cluster IDs to lists of indices of texts used for label generation.

    Note:
        Clusters with fewer than sampled_texts_per_cluster + sampled_comparison_texts_per_cluster 
            texts are skipped.
        The function uses either a local model or an API for text generation, depending on the 
            provided arguments.
    """
    cluster_labels = {}
    all_cluster_texts_used_for_label_strs_ids = {}
    for cluster in set(clustering_assignments):
        # For each cluster, select sampled_texts_per_cluster random texts (skipping the cluster if it has less than sampled_comparison_texts_per_cluster + sampled_texts_per_cluster texts)
        cluster_indices = [i for i, x in enumerate(clustering_assignments) if x == cluster]
        if len(cluster_indices) < sampled_comparison_texts_per_cluster + sampled_texts_per_cluster:
            print("Skipping cluster", cluster, "because it has fewer than", sampled_comparison_texts_per_cluster + sampled_texts_per_cluster, "texts.")
            cluster_labels[cluster] = ["No labels generated for this cluster because it has fewer than " + str(sampled_comparison_texts_per_cluster + sampled_texts_per_cluster) + " texts."]
            all_cluster_texts_used_for_label_strs_ids[cluster] = []
            continue
        decoded_labels, selected_text_indices = label_single_cluster(
            decoded_strs, 
            clustering_assignments, 
            cluster, 
            local_model, 
            labeling_tokenizer, 
            api_provider, 
            api_model_str, 
            auth_key, 
            client,
            device, 
            sampled_texts_per_cluster, 
            generated_labels_per_cluster,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict=cluster_ids_to_prompt_ids_to_decoding_ids_dict,
            num_decodings_per_prompt=num_decodings_per_prompt,
            single_cluster_label_instruction=single_cluster_label_instruction,
            api_interactions_save_loc=api_interactions_save_loc,
            logging_level=logging_level,
            logger=logger
        )
        all_cluster_texts_used_for_label_strs_ids[cluster] = selected_text_indices
        cluster_labels[cluster] = decoded_labels
        
        print(f"Cluster {cluster} labels: {cluster_labels[cluster]}")

    return cluster_labels, all_cluster_texts_used_for_label_strs_ids

def api_based_label_text_matching(
        label: str, 
        text_A: str, 
        text_B: str, 
        api_provider: str, 
        model_str: str, 
        api_key: Optional[str] = None,
        client: Optional[Union[Anthropic, OpenAI, Client]] = None,
        mode: str = "single_cluster",
        api_interactions_save_loc: Optional[str] = None,
        logging_level: str = "INFO",
        logger: Optional[BoundLoggerLazyProxy] = None,
        cluster_id_1: Optional[int] = None,
        cluster_id_2: Optional[int] = None
        ) -> Tuple[float, float]:
    """
    Use an API to determine how well two texts match a given label.

    This function sends a request to a specified API (e.g., OpenAI, Anthropic) to rate
    how well each of two given texts matches a provided label. The API is asked to
    provide scores on a scale of 0 to 10 for each text. These scores are then normalized
    to probabilities.

    Args:
        label (str): The label against which to compare the texts.
        text_A (str): The first text to be compared.
        text_B (str): The second text to be compared.
        api_provider (str): The API provider to use (e.g., 'openai', 'anthropic').
        model_str (str): The specific model to use within the chosen API.
        api_key (Optional[str]): The authentication key for the API.
        client (Optional[Union[Anthropic, OpenAI, Client]]): The client to use for the API request.
        mode (str): The mode to use for label evaluation. Defaults to "single_cluster". Set to 
            "double_cluster" or "contrastive" to evaluate a contrastive label between two clusters.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        cluster_id_1 (Optional[int]): The ID of the first cluster, if using a contrastive label.
        cluster_id_2 (Optional[int]): The ID of the second cluster, if using a contrastive label.
    Returns:
        Tuple[float, float]: A tuple containing two floats:
            - prob_A: The normalized probability (0 to 1) that text_A matches the label.
            - prob_B: The normalized probability (0 to 1) that text_B matches the label.

    Note:
        - The function uses a single API call to get scores for both texts.
        - If the API response cannot be parsed, default scores of 5 are used for both texts.
        - The scores are normalized to probabilities that sum to 1.
    """
    if mode == "single_cluster":
        prompt = f"""
        Given the label '{label}', rate how well each of the following texts matches this label on a scale from 0 to 100:

        Text A: {text_A}

        Text B: {text_B}

        Provide your response in the following JSON format:
        {{
            "text_A_score": <score for Text A>,
            "text_B_score": <score for Text B>
        }}
        """
    elif mode == "double_cluster" or mode == "contrastive":
        prompt = f"""
        The following label describes the difference between two language models: '{label}'

        Given this description, rate how well each of the following texts matches Model 1 (as opposed to Model 2) on a scale from 0 to 100:

        Text A: {text_A}

        Text B: {text_B}

        Provide your response in the following JSON format:
        {{
            "text_A_score": <score for Text A>,
            "text_B_score": <score for Text B>
        }}

        A higher score indicates the text is more likely to belong to Model 1. Respond only with the JSON. Do not explain your decision.
        """
    
    response = make_api_request(
        prompt, 
        api_provider, 
        model_str, 
        api_key, 
        client,
        api_interactions_save_loc=api_interactions_save_loc,
        logging_level=logging_level,
        logger=logger,
        request_info={
            "pipeline_stage": "double_cluster text matching",
            "cluster_id_1": str(cluster_id_1),
            "cluster_id_2": str(cluster_id_2)
        }
    )
    json_data = extract_json_from_string(response)
    
    if json_data and isinstance(json_data, list) and len(json_data) > 0:
        print(f"JSON data: {json_data}")
        scores = json_data[0]
        if isinstance(scores, dict):
            if "text_A_score" in scores and "text_B_score" in scores:
                score_A = float(scores.get("text_A_score", 50))  # Default to 50 if missing
                score_B = float(scores.get("text_B_score", 50))  # Default to 50 if missing
            else:
                print(f"Unexpected response format for api scores: {scores}")
                score_A = score_B = 50
        else:
            print(f"Unexpected response format for api scores: {scores}")
            score_A = score_B = 50
    elif json_data and isinstance(json_data, dict):
        if "text_A_score" in json_data and "text_B_score" in json_data:
            score_A = float(json_data.get("text_A_score", 50))
            score_B = float(json_data.get("text_B_score", 50))
        else:
            print(f"Unexpected response format for api scores: {json_data}")
            score_A = score_B = 50
    else:
        # If we couldn't parse the JSON or it's empty, default to equal scores
        print(f"Error parsing JSON api scores response: {response}")
        score_A = score_B = 50
    
    # Normalize scores to probabilities
    total = score_A + score_B
    prob_A = score_A / total if total > 0 else 0.5
    prob_B = score_B / total if total > 0 else 0.5
    
    return prob_A, prob_B


def evaluate_label_discrimination(
        label: str,
        sampled_texts_1: List[int],
        sampled_texts_2: List[int],
        decoded_strs_1: List[str],
        decoded_strs_2: List[str],
        local_model: Optional[AutoModel],
        labeling_tokenizer: Optional[AutoTokenizer],
        api_provider: Optional[str],
        api_model_str: Optional[str],
        auth_key: Optional[str],
        client: Optional[Union[Anthropic, OpenAI, Client]] = None,
        device: str = "cuda:0",
        mode: str = "single_cluster",
        n_head_to_head_comparisons_per_text: Optional[int] = None,
        use_unitary_comparisons: bool = False,
        max_unitary_comparisons_per_label: int = 100,
        api_interactions_save_loc: Optional[str] = None,
        logging_level: str = "INFO",
        scores_logging_mode: str = "validation",
        logger: Optional[BoundLoggerLazyProxy] = None,
        cluster_id_1: Optional[int] = None,
        cluster_id_2: Optional[int] = None,
        n_permutations: int = 0
        ) -> Tuple[float, float, float]:
    """
    Evaluate the discrimination power of a given label between two sets of texts.

    This function compares pairs of texts from two matched clusters and determines
    how well the given label discriminates between them. It uses either a local model
    or an API to perform the evaluation.

    Args:
        label (str): The label being evaluated.
        sampled_texts_1 (List[int]): Indices of sampled texts from the first cluster. Acts as indices into 
            decoded_strs_1 to specify which texts are in the cluster.
        sampled_texts_2 (List[int]): Indices of sampled texts from the second cluster. Acts as indices into 
            decoded_strs_2 to specify which texts are in the cluster.
        decoded_strs_1 (List[str]): List of decoded strings for the first clustering.
        decoded_strs_2 (List[str]): List of decoded strings for the second (comparison) clustering.
        local_model (Optional[AutoModel]): The local model to use, if not using an API.
        labeling_tokenizer (Optional[AutoTokenizer]): The tokenizer for the local model.
        api_provider (Optional[str]): The API provider to use, if any.
        api_model_str (Optional[str]): The API model string to use, if any.
        auth_key (Optional[str]): The authentication key for the API, if any.
        client (Optional[Union[Anthropic, OpenAI, Client]]): The client to use for the API request.
        device (str): The device to use for computations. Defaults to "cuda:0".
        mode (str): The mode to use for label evaluation. Defaults to "single_cluster". Set to 
            "double_cluster" or "contrastive" to evaluate a contrastive label between two clusters.
        n_head_to_head_comparisons_per_text (Optional[int]): For each comparison text, how many of 
            the other comparison texts should it be tested against. Defaults to None, which means 
            all other comparison texts are used.
        use_unitary_comparisons (bool): Whether to use unitary comparisons, i.e. test labels
            by their ability to let the assistant determine which cluster a given text belongs to, 
            without considering another text for comparison. Defaults to False.
        max_unitary_comparisons_per_label (int): The maximum number of unitary comparisons to perform 
            for each label. Defaults to 100.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        scores_logging_mode (str): The mode to use for logging scores. Defaults to "validation", which means 
            logging the scores for the validation set. Set to "cross-validation" to indicate that the scores
            are for cross-validation.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        cluster_id_1 (Optional[int]): The ID of the first cluster, if using a contrastive label.
        cluster_id_2 (Optional[int]): The ID of the second cluster, if using a contrastive label.
        n_permutations (int): The number of permutations to perform to evaluate the p-value of the label's
            discrimination power via permutation test. Defaults to 0.
    Returns:
        float: The accuracy or AUC score representing the label's discrimination power.
        float: The p-value of the label's discrimination power via permutation test.
        float: The AUC score representing the label's discrimination power.
    """
    scores = []
    alternate_scores = [] # For if we use additional Discriminator models for comparison
    true_labels = []

    if ':' in api_model_str:
        main_discriminator_api_model_str = api_model_str.split(':')[0]
        alternate_discriminator_api_model_str_list = api_model_str.split(':')[1:]
    else:
        main_discriminator_api_model_str = api_model_str
        alternate_discriminator_api_model_str_list = []
    for alternate_discriminator_api_model_str in alternate_discriminator_api_model_str_list:
        alternate_scores.append([])
    print(f"main_discriminator_api_model_str: {main_discriminator_api_model_str}")
    print(f"alternate_discriminator_api_model_str_list: {alternate_discriminator_api_model_str_list}")
    if use_unitary_comparisons:
        all_texts = [(text_id, decoded_strs_1[text_id], 1) for text_id in sampled_texts_1] + \
                    [(text_id, decoded_strs_2[text_id], 0) for text_id in sampled_texts_2]
        random.shuffle(all_texts)
        print(f"max_unitary_comparisons_per_label: {max_unitary_comparisons_per_label}")
        print(f"len(all_texts): {len(all_texts)}")
        if len(all_texts) > max_unitary_comparisons_per_label:
            all_texts = all_texts[:max_unitary_comparisons_per_label]
        
        prompts = []
        for text_id, text, true_label in tqdm(all_texts, desc="Performing unitary comparisons", disable=(not api_provider is None)):
            if api_provider is None:
                if mode == "single_cluster":
                    input_str = f"Does the following text fit the label '{label}'? Text: {text}\nAnswer (Yes or No):"
                elif mode == "double_cluster" or mode == "contrastive":
                    input_str = f"The following string describes the difference between two clusters of texts: '{label}'. Does the following text fit better with cluster 1 or cluster 2?\nText: {text}\nAnswer (1 or 2):"
                input_ids = labeling_tokenizer(input_str, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = local_model(**input_ids).logits
                    if mode == "single_cluster":
                        prob_yes = torch.nn.functional.softmax(logits, dim=-1)[0, -1, labeling_tokenizer.encode("Yes", add_special_tokens=False)[0]].item()
                        prob_no = torch.nn.functional.softmax(logits, dim=-1)[0, -1, labeling_tokenizer.encode("No", add_special_tokens=False)[0]].item()
                        score = prob_yes / max(prob_yes + prob_no, 1e-10)
                    else:
                        prob_1 = torch.nn.functional.softmax(logits, dim=-1)[0, -1, labeling_tokenizer.encode("1", add_special_tokens=False)[0]].item()
                        prob_2 = torch.nn.functional.softmax(logits, dim=-1)[0, -1, labeling_tokenizer.encode("2", add_special_tokens=False)[0]].item()
                        score = prob_1 / max(prob_1 + prob_2, 1e-10)
                scores.append(score)
            else:
                if mode == "single_cluster":
                    prompt = f"""
                    Given the label '{label}', determine if the following text matches this label:

                    Text: {text}

                    Provide your response as a single number between 0 and 100, where 0 means the text doesn't match the label at all, and 100 means it matches perfectly. Provide only the number, and nothing else.
                    """
                elif mode == "double_cluster" or mode == "contrastive":
                    prompt = f"""The following label describes the difference between two clusters of texts: '{label}'
                    
                    Given this description, rate how well the following text matches Model 1 (as opposed to Model 2) on a scale from 0 to 100:

                    Text: {text}

                    Provide your response as a single number between 0 and 100, where 0 means the text definitely belongs to Model 2, and 100 means it definitely belongs to Model 1. Provide only the number, and nothing else.
                    """
                prompt = prompt.replace("                    ", "")
                prompts.append(prompt)
            true_labels.append(true_label)
        if api_provider is not None:
            request_info = {
                "pipeline_stage": "unitary comparisons of contrastive label discrimination",
                "cluster_id_1": str(cluster_id_1),
                "cluster_id_2": str(cluster_id_2)
            }
            # Can provide multiple API models for comparison, separated by a colon in the api_model_str
            for i,alternate_discriminator_api_model_str in enumerate(alternate_discriminator_api_model_str_list):
                # Make parallel API requests for each alternate discriminator
                alternate_scores_texts = parallel_make_api_requests(
                    prompts=prompts,
                    api_provider=api_provider,
                    api_model_str=alternate_discriminator_api_model_str,
                    auth_key=auth_key,
                    client=client,
                    api_interactions_save_loc=api_interactions_save_loc,
                    logging_level=logging_level,
                    logger=logger,
                    request_info=request_info,
                    max_tokens=512,
                    max_thinking_tokens=0
                )
                for score_text in alternate_scores_texts:
                    try:
                        score = float(score_text.strip()) / 100  # Normalize to [0, 1]
                    except ValueError:
                        print(f"Error parsing API response: {score_text}")
                        score = 0.5  # Default to neutral score if parsing fails
                    # check for nan or invalid value
                    if np.isnan(score) or not np.isfinite(score) or score < 0 or score > 1:
                        print(f"Invalid Alternate Discriminator score: {score}")
                        score = 0.5
                    alternate_scores[i].append(score)

            main_discriminator_scores_texts = parallel_make_api_requests(
                prompts=prompts,
                api_provider=api_provider,
                api_model_str=main_discriminator_api_model_str,
                auth_key=auth_key,
                client=client,
                api_interactions_save_loc=api_interactions_save_loc,
                logging_level=logging_level,
                logger=logger,
                request_info=request_info,
                max_tokens=512,
                max_thinking_tokens=0
            )
            for score_text in main_discriminator_scores_texts:
                try:
                    score = float(score_text.strip()) / 100  # Normalize to [0, 1]
                except ValueError:
                    print(f"Error parsing API response: {score_text}")
                    score = 0.5  # Default to neutral score if parsing fails
                # check for nan or invalid values
                if np.isnan(score) or not np.isfinite(score) or score < 0 or score > 1:
                    print(f"Invalid Discriminator score: {score}")
                    score = 0.5
                scores.append(score)
    else:
        for text_id_1 in sampled_texts_1:
            n_comparisons_performed = 0
            for text_id_2 in sampled_texts_2:
                if text_id_1 == text_id_2:
                    continue
                if n_head_to_head_comparisons_per_text is not None and n_comparisons_performed >= n_head_to_head_comparisons_per_text:
                    break
                n_comparisons_performed += 1
                text_1 = decoded_strs_1[text_id_1]
                text_2 = decoded_strs_2[text_id_2]
                if np.random.rand() > 0.5:
                    text_A, text_B = text_1, text_2
                    true_label = 1
                else:
                    text_A, text_B = text_2, text_1
                    true_label = 0

                if api_provider is None:
                    if mode == "single_cluster":
                        input_str = f"Which text fits the label '{label}' better? Text A: {text_A} or Text B: {text_B}\nAnswer: Text"
                    elif mode == "double_cluster" or mode == "contrastive":
                        input_str = f"The following string describes the difference between two language models: '{label}'. Which of the following texts better fits with Model 1?\nText A: {text_A}\nText B: {text_B}\nAnswer: Text"
                    input_ids = labeling_tokenizer(input_str, return_tensors="pt").to(device)
                    with torch.no_grad():
                        logits = local_model(**input_ids).logits
                        prob_A = torch.nn.functional.softmax(logits, dim=-1)[0, -1, labeling_tokenizer.encode("A", add_special_tokens=False)[0]].item()
                        prob_B = torch.nn.functional.softmax(logits, dim=-1)[0, -1, labeling_tokenizer.encode("B", add_special_tokens=False)[0]].item()
                        normalized_prob_A = prob_A / max(prob_A + prob_B, 1e-10)
                else:
                    normalized_prob_A, _ = api_based_label_text_matching(
                        label, 
                        text_A, 
                        text_B, 
                        api_provider, 
                        main_discriminator_api_model_str, 
                        auth_key, 
                        client=client,
                        mode=mode,
                        api_interactions_save_loc=api_interactions_save_loc,
                        logging_level=logging_level,
                        logger=logger,
                        cluster_id_1=cluster_id_1,
                        cluster_id_2=cluster_id_2
                    )

                scores.append(normalized_prob_A)
                true_labels.append(true_label)
            # Permute sampled_texts_2
            sampled_texts_2 = random.sample(sampled_texts_2, len(sampled_texts_2))
    
    try:
        auc = roc_auc_score(true_labels, scores)
    except ValueError:
        auc = float('nan') 
    accuracy = sum([1 for i, score in enumerate(scores) if score > 0.5 and true_labels[i] == 1 or score < 0.5 and true_labels[i] == 0]) / len([score for score in scores if score is not None and score != 0.5])
    if n_permutations > 0:
        p_value = permutation_test_auc(scores, true_labels, n_permutations, seed=42)
    else:
        p_value = None
    if logging_level in ["DEBUG", "SCORES"]:
        logger.info(f"SCORES Logging {scores_logging_mode} Accuracy: {accuracy}, P-value: {p_value}, AUC: {auc}")
        logger.info(f"SCORES Logging {scores_logging_mode} True labels: {true_labels}")
        logger.info(f"SCORES Logging {scores_logging_mode} Scores: {scores}")
        logger.info(f"SCORES Logging {scores_logging_mode} Alternate scores: {alternate_scores}")
    return accuracy, p_value, auc

# For each possible contrastive label of each cluster pair, we select sampled_comparison_texts_per_cluster random texts from
# each cluster (which must not be in the associated cluster's all_cluster_texts_used_for_label_strs_ids, skipping the cluster 
# pair and printing a warning if there aren't enough texts in both clusters).
# We then perform pairwise comparisons between each group of sampled_comparison_texts_per_cluster texts in the cluster pair, 
# asking the assistant LLM to identify which of the two texts comes from which cluster.
# We score potential contrastive labels based on their ability to let the assistant LLM distinguish between texts from the two
# clusters, quantified with either AUC or accuracy. We then return the scores for each cluster pair.
def validate_cluster_label_comparative_discrimination_power(
        decoded_strs_1: List[str], 
        clustering_assignments_1: List[int], 
        all_cluster_texts_used_for_label_strs_ids_1: Dict[Tuple[int, int], List[List[int]]], 
        decoded_strs_2: List[str], 
        clustering_assignments_2: List[int], 
        all_cluster_texts_used_for_label_strs_ids_2: Dict[Tuple[int, int], List[List[int]]], 
        cluster_label_strs: Dict[Tuple[int, int], List[str]], # dict of lists of contrastive labels for each cluster pair, so only one set of labels per cluster pair
        local_model: AutoModel = None, 
        labeling_tokenizer: AutoTokenizer = None, 
        api_provider: str = None,
        api_model_str: str = None,
        auth_key: str = None,
        client: Optional[Union[Anthropic, OpenAI, Client]] = None,
        device: str = "cuda:0", 
        sampled_comparison_texts_per_cluster: int = 10, 
        n_head_to_head_comparisons_per_text: Optional[int] = None,
        use_unitary_comparisons: bool = False,
        max_unitary_comparisons_per_label: int = 100,
        api_interactions_save_loc: Optional[str] = None,
        logging_level: str = "INFO",
        logger: Optional[BoundLoggerLazyProxy] = None,
        n_permutations: int = 0,
        split_clusters_by_prompt: bool = False,
        use_baseline_discrimination: bool = False,
        base_decoded_texts_prompt_ids: List[int] = None,
        finetuned_decoded_texts_prompt_ids: List[int] = None,
        random_seed: int = 0,
        ) -> Tuple[
            Dict[Tuple[int, int], Dict[str, float]], # (cluster_1_id, cluster_2_id), label, accuracy/AUC
            Dict[Tuple[int, int], Tuple[List[int], List[int]]], # (cluster_1_id, cluster_2_id), list of text ids used for generating labels, list of text ids used for validating labels
            Dict[Tuple[int, int], Dict[str, float]], # (cluster_1_id, cluster_2_id), label, p-value
            Dict[Tuple[int, int], Dict[str, float]], # (cluster_1_id, cluster_2_id), label, AUC scores
            Optional[Dict[Tuple[int, int], Dict[str, float]]], # (cluster_1_id, cluster_2_id), label, baseline accuracy scores
            Optional[Dict[Tuple[int, int], Dict[str, float]]]]: # (cluster_1_id, cluster_2_id), label, baseline AUC scores
    """
    Validate the discrimination power of contrastive labels for pairs of clusters.

    This function evaluates how well the generated contrastive labels can distinguish between
    texts from two different clusters. It samples texts from each cluster pair, excluding those
    previously used for label generation, and uses either a local model or an API to differentiate
    between texts from the two clusters using the labels provided. It quantifies the disciminative 
    power of the labels using the Area Under the Curve (AUC) score of the model's predictions when
    using the labels to identify which text comes from which cluster.

    Args:
        decoded_strs_1 (List[str]): List of decoded strings which contains the texts in the first clustering.
        clustering_assignments_1 (List[int]): Cluster assignments for the first clustering.
        all_cluster_texts_used_for_label_strs_ids_1 (Dict[Tuple[int, int], List[List[int]]]): Text IDs previously
            used for each label generation in the first clustering. Will be excluded from the sampled texts used to validate the labels.
        decoded_strs_2 (List[str]): List of decoded strings which contains the texts in the second clustering.
        clustering_assignments_2 (List[int]): Cluster assignments for the second clustering.
        all_cluster_texts_used_for_label_strs_ids_2 (Dict[Tuple[int, int], List[List[int]]]): Text IDs previously
            used for each label generation in the second clustering. Will be excluded from the sampled texts used to validate the labels.
        cluster_label_strs (Dict[Tuple[int, int], List[str]]): Contrastive labels for each cluster pair. 
            Formatted as {(clustering_1_id, clustering_2_id): [list of contrastive labels]}
        local_model (AutoModel, optional): Local model for text generation. Defaults to None.
        labeling_tokenizer (AutoTokenizer, optional): Tokenizer for the local model. Defaults to None.
        api_provider (str, optional): API provider for text generation. Defaults to None.
        api_model_str (str, optional): Model string for API requests. Defaults to None.
        auth_key (str, optional): Authentication key for API requests. Defaults to None.
        client (Optional[Union[Anthropic, OpenAI, Client]], optional): API client for text generation. Defaults to None.
        device (str, optional): Device to use for local model. Defaults to "cuda:0".
        sampled_comparison_texts_per_cluster (int, optional): Number of texts to sample from each cluster. 
            Defaults to 10.
        n_head_to_head_comparisons_per_text (int, optional): For each comparison text, how many of the other 
            comparison texts should it be tested against. Defaults to None, which means all other comparison 
            texts are used.
        use_unitary_comparisons (bool, optional): Whether to use unitary comparisons, i.e. test labels
            by their ability to let the assistant determine which cluster a given text belongs to, 
            without considering another text for comparison. Defaults to False.
        max_unitary_comparisons_per_label (int, optional): Maximum number of unitary comparisons to perform 
            per label. Defaults to 100.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        n_permutations (int, optional): Number of permutations to perform for the permutation test. Defaults to 0.
        split_clusters_by_prompt (bool, optional): Whether to split the clusters by prompt during discriminative
            evaluation of the labels. If True, we will ensure no overlap in prompts between the label generation
            and evaluation splits of each cluster. Defaults to False.
        base_decoded_texts_prompt_ids (List[int], optional): List of prompt IDs for the base decoded texts.
            Defaults to None.
        finetuned_decoded_texts_prompt_ids (List[int], optional): List of prompt IDs for the finetuned decoded texts.
            Defaults to None.
        random_seed (int, optional): Random seed to use for reproducibility. Defaults to 0. Used for the baseline 
            discriminator.
    Returns:
        Tuple[
            Dict[Tuple[int, int], Dict[str, float]], 
            Dict[Tuple[int, int], Tuple[List[int], List[int]]]
            Dict[Tuple[int, int], Dict[str, float]]
            Dict[Tuple[int, int], Dict[str, float]]
            Dict[Tuple[int, int], Dict[str, float]]
            Dict[Tuple[int, int], Dict[str, LogisticBoWDiscriminator]]
            ]:
            - A dictionary mapping cluster pairs to a dictionary of contrastive label strings and their associated accuracy or AUC scores.
            - A dictionary mapping cluster pairs to the indices of texts used for validation.
            - A dictionary mapping cluster pairs to a dictionary of contrastive label strings and their associated AUC scores.
            - A dictionary mapping cluster pairs to a dictionary of contrastive label strings and their associated baseline accuracy scores.
            - A dictionary mapping cluster pairs to a dictionary of contrastive label strings and their associated baseline AUC scores.
            - A dictionary mapping cluster pairs to a dictionary of contrastive label strings and their associated baseline discriminators.
    Note:
        This function skips cluster pairs that don't have enough available texts for sampling.
        It uses the evaluate_label_discrimination function to compute accuracy or AUC scores for each label.
    """
    cluster_pair_scores = {}
    cluster_pair_auc_scores = {}
    all_cluster_texts_used_for_validating_label_strs_ids = {}
    auc_permutation_p_values = {}
    baseline_cluster_pair_scores = {}
    baseline_cluster_pair_auc_scores = {}
    baseline_cluster_pair_discriminators = {}
    for (cluster_id_1, cluster_id_2), cluster_label_candidates in tqdm(cluster_label_strs.items(), desc="Processing cluster pairs"):
        cluster_pair_scores[(cluster_id_1, cluster_id_2)] = {}
        cluster_pair_auc_scores[(cluster_id_1, cluster_id_2)] = {}
        auc_permutation_p_values[(cluster_id_1, cluster_id_2)] = {}
        all_cluster_texts_used_for_validating_label_strs_ids[(cluster_id_1, cluster_id_2)] = []
        baseline_cluster_pair_scores[(cluster_id_1, cluster_id_2)] = {}
        baseline_cluster_pair_auc_scores[(cluster_id_1, cluster_id_2)] = {}
        baseline_cluster_pair_discriminators[(cluster_id_1, cluster_id_2)] = {}
        # Get the lists of texts used for each label in this cluster pair
        label_texts_1 = all_cluster_texts_used_for_label_strs_ids_1.get((cluster_id_1, cluster_id_2), [])
        label_texts_2 = all_cluster_texts_used_for_label_strs_ids_2.get((cluster_id_1, cluster_id_2), [])

        # Evaluate each label using its corresponding texts
        for label_idx, label in enumerate(cluster_label_candidates):
            # Get the texts used to generate this specific label
            texts_for_this_label_1 = label_texts_1[label_idx] if label_idx < len(label_texts_1) else []
            texts_for_this_label_2 = label_texts_2[label_idx] if label_idx < len(label_texts_2) else []

            # Sample validation texts, excluding those used for this specific label
            cluster_1_indices = [i for i, x in enumerate(clustering_assignments_1) if x == cluster_id_1 and i not in texts_for_this_label_1]
            cluster_2_indices = [i for i, x in enumerate(clustering_assignments_2) if x == cluster_id_2 and i not in texts_for_this_label_2]

            # Ensure there are enough texts remaining
            cluster_1_avail_len = len(cluster_1_indices)
            cluster_2_avail_len = len(cluster_2_indices)
            if cluster_1_avail_len < sampled_comparison_texts_per_cluster or cluster_2_avail_len < sampled_comparison_texts_per_cluster:
                print(f"Warning: Not enough texts for cluster pair {cluster_id_1} ({cluster_1_avail_len}), {cluster_id_2} ({cluster_2_avail_len}). Setting accuracy/AUC to -1.0 to indicate invalid result.")
                sampled_texts_1, sampled_texts_2 = [], []
            else:
                sampled_texts_1 = random.sample(cluster_1_indices, sampled_comparison_texts_per_cluster)
                sampled_texts_2 = random.sample(cluster_2_indices, sampled_comparison_texts_per_cluster)

            all_cluster_texts_used_for_validating_label_strs_ids[(cluster_id_1, cluster_id_2)].append((sampled_texts_1, sampled_texts_2))

            if len(sampled_texts_1) == 0 or len(sampled_texts_2) == 0:
                accuracy_score = -1.0
                auc_score = -1.0
                p_value = None
                baseline_accuracy_score = -1.0
                baseline_p_value = None
                baseline_auc_score = -1.0
            else:
                accuracy_score, p_value, auc_score = evaluate_label_discrimination(
                    label,
                    sampled_texts_1,
                    sampled_texts_2,
                    decoded_strs_1,
                    decoded_strs_2,
                    local_model,
                    labeling_tokenizer,
                    api_provider,
                    api_model_str,
                    auth_key,
                    client=client,
                    device=device,
                    mode="contrastive",
                    n_head_to_head_comparisons_per_text=n_head_to_head_comparisons_per_text,
                    use_unitary_comparisons=use_unitary_comparisons,
                    max_unitary_comparisons_per_label=max_unitary_comparisons_per_label,
                    api_interactions_save_loc=api_interactions_save_loc,
                    logging_level=logging_level,
                    logger=logger,
                    scores_logging_mode="validation",
                    cluster_id_1=cluster_id_1,
                    cluster_id_2=cluster_id_2,
                    n_permutations=n_permutations
                )
                if use_baseline_discrimination:
                    baseline_accuracy_score, baseline_auc_score, baseline_discriminator = baseline_discrimination(
                        [decoded_strs_1[text_id] for text_id in sampled_texts_1], 
                        [decoded_strs_2[text_id] for text_id in sampled_texts_2],
                        top_k_features_to_show=10,
                        label_names=("M1", "M2"),
                        random_state=random_seed
                    )
                    baseline_cluster_pair_scores[(cluster_id_1, cluster_id_2)][label] = baseline_accuracy_score
                    baseline_cluster_pair_auc_scores[(cluster_id_1, cluster_id_2)][label] = baseline_auc_score
                    baseline_cluster_pair_discriminators[(cluster_id_1, cluster_id_2)][label] = baseline_discriminator
                else:
                    baseline_accuracy_score = None
                    baseline_auc_score = None
            cluster_pair_scores[(cluster_id_1, cluster_id_2)][label] = accuracy_score
            cluster_pair_auc_scores[(cluster_id_1, cluster_id_2)][label] = auc_score
            auc_permutation_p_values[(cluster_id_1, cluster_id_2)][label] = p_value

    return cluster_pair_scores, all_cluster_texts_used_for_validating_label_strs_ids, auc_permutation_p_values, cluster_pair_auc_scores, baseline_cluster_pair_scores, baseline_cluster_pair_auc_scores, baseline_cluster_pair_discriminators


# For each possible label of each cluster, we select sampled_comparison_texts_per_cluster random texts (which must not
# be in all_cluster_texts_used_for_label_strs_ids, skipping the cluster and printing a warning if there aren't enough texts in the cluster).
# We then perform pairwise comparisons between each of the sampled_comparison_texts_per_cluster texts in the cluster and 
# non_cluster_comparison_texts random texts, asking the assistant LLM which of the two texts better fits the given label.
# We score potential cluster labels based on their ability to let the assistant LLM distinguish between texts from the cluster
# and texts not from the cluster, quantified with either AUC or accuracy. We then return the scores for each cluster.
def validate_cluster_label_discrimination_power(
        decoded_strs, 
        clustering_assignments, 
        cluster_label_strs, 
        all_cluster_texts_used_for_label_strs_ids, 
        local_model = None, 
        labeling_tokenizer = None, 
        api_provider = None,
        api_model_str = None,
        auth_key = None,
        client: Optional[Union[Anthropic, OpenAI, Client]] = None,
        device = "cuda:0", 
        sampled_comparison_texts_per_cluster = 10, 
        non_cluster_comparison_texts = 10,
        api_interactions_save_loc: Optional[str] = None,
        logging_level: str = "INFO",
        logger: Optional[BoundLoggerLazyProxy] = None,
        n_permutations: int = 0
        ) -> Tuple[Dict[int, Dict[str, float]], Dict[int, List[int]]]:
    """
    Validate the discrimination power of cluster labels by testing if an assistant LLM can use the 
    labels to distinguish between texts within the cluster and texts outside the cluster. 
    
    It uses either a local model or an API to perform text comparisons and calculates either an AUC 
    or accuracy score for each label's discriminative power.

    Args:
        decoded_strs (List[str]): List of all decoded strings.
        clustering_assignments (List[int]): Cluster assignments for each string in decoded_strs.
        cluster_label_strs (Dict[int, List[str]]): Dictionary mapping cluster IDs to lists of candidate 
            labels.
        all_cluster_texts_used_for_label_strs_ids (Dict[int, List[int]]): Dictionary mapping cluster IDs 
            to lists of text indices previously used for generating labels. Will be excluded from the sampled 
            texts used to validate the labels.
        local_model (Optional[AutoModel]): Local model for text generation. Defaults to None.
        labeling_tokenizer (Optional[AutoTokenizer]): Tokenizer for the local model. Defaults to None.
        api_provider (Optional[str]): API provider for text generation. Defaults to None.
        api_model_str (Optional[str]): Model string for API requests. Defaults to None.
        auth_key (Optional[str]): Authentication key for API requests. Defaults to None.
        client (Optional[Union[Anthropic, OpenAI, Client]], optional): API client for text
            generation. Defaults to None.
        device (str): Device to use for local model. Defaults to "cuda:0".
        sampled_comparison_texts_per_cluster (int): Number of texts to sample from each cluster for 
            comparison. Defaults to 10.
        non_cluster_comparison_texts (int): Number of texts to sample from outside each cluster for 
            comparison. Defaults to 10.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        n_permutations (int, optional): Number of permutations to perform for the permutation test. Defaults to 0.
    Returns:
        Tuple[Dict[int, Dict[str, float]], Dict[int, List[int]]]: A tuple containing:
            - Dictionary mapping cluster IDs to a dictionary of label strings and their associated accuracy or AUC scores.
            - Dictionary mapping cluster IDs to a list of text indices used for validation.

    Note:
        This function skips clusters that don't have enough texts for sampling.
        It uses the evaluate_label_discrimination function to compute accuracy or AUC scores for each label.
    """
    cluster_label_scores = {}
    cluster_label_auc_scores = {}
    auc_permutation_p_values = {}
    all_cluster_texts_used_for_validating_label_strs_ids = {}
    for cluster_id, cluster_label_candidates in tqdm(cluster_label_strs.items(), desc="Processing clusters"):
        cluster_label_scores[cluster_id] = {}
        cluster_label_auc_scores[cluster_id] = {}
        auc_permutation_p_values[cluster_id] = {}

        cluster_indices = [i for i, x in enumerate(clustering_assignments) if x == cluster_id]
        texts_used_for_current_label_ids = all_cluster_texts_used_for_label_strs_ids[cluster_id]
        texts_NOT_used_for_current_label_ids = [i for i in range(len(decoded_strs)) if i not in texts_used_for_current_label_ids]
       
        # Sample texts from the cluster but outside texts_used_for_current_label_ids
        cluster_ids_outside_selected = [i for i in cluster_indices if i not in texts_used_for_current_label_ids]
        if len(cluster_ids_outside_selected) < sampled_comparison_texts_per_cluster:
            print(f"Warning: Not enough texts for cluster {cluster_id} outside those selected for labeling. Skipping.")
            continue
        sampled_cluster_texts = np.random.choice(cluster_ids_outside_selected, sampled_comparison_texts_per_cluster, replace=False)
        # Sample non-cluster texts
        sampled_non_cluster_texts = np.random.choice(texts_NOT_used_for_current_label_ids, non_cluster_comparison_texts, replace=False)
        all_cluster_texts_used_for_validating_label_strs_ids[cluster_id] = sampled_cluster_texts

        for label in cluster_label_candidates:
            accuracy_score, p_value, auc_score = evaluate_label_discrimination(
                label,
                sampled_cluster_texts,
                sampled_non_cluster_texts,
                decoded_strs,
                decoded_strs, # We use the same decoded_strs, since this is done within a single clustering.
                local_model,
                labeling_tokenizer,
                api_provider,
                api_model_str,
                auth_key,
                client=client,
                device=device,
                api_interactions_save_loc=api_interactions_save_loc,
                logging_level=logging_level,
                scores_logging_mode="validation",
                logger=logger,
                cluster_id_1=cluster_id,
                cluster_id_2=None,
                n_permutations=n_permutations
            )
            cluster_label_scores[cluster_id][label] = accuracy_score
            cluster_label_auc_scores[cluster_id][label] = auc_score
            auc_permutation_p_values[cluster_id][label] = p_value

    return cluster_label_scores, cluster_label_auc_scores, all_cluster_texts_used_for_validating_label_strs_ids, auc_permutation_p_values

# From https://huggingface.co/intfloat/multilingual-e5-large-instruct
def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

# To save costs / time with embeddings, we will save a copy of whatever embeddings we generate and attempt to load past embeddings
# from file if they exist. We will only generate new embeddings if we can't find past embeddings on file. This function thus first
# checks if there is a previously saved embeddings file to load, and if not, generates new embeddings and saves them to file.
def read_past_embeddings_or_generate_new(
        path: str, 
        client: object, 
        decoded_strs: List[str], 
        local_embedding_model_str: str = "intfloat/multilingual-e5-large-instruct", 
        local_embedding_model: AutoModel = None, 
        tokenizer: AutoTokenizer = None, 
        device: str = "cuda:0", 
        recompute_embeddings: bool = False, 
        batch_size: int = 16, 
        save_embeddings: bool = True, 
        tqdm_disable: bool = False, 
        clustering_instructions: str = "Identify the topic or theme of the given text", 
        max_length: int = 512, 
        bnb_config: BitsAndBytesConfig = None
        ) -> List[List[float]]:
    """
    Load existing embeddings from a file or generate new ones if necessary.

    This function first attempts to load previously saved embeddings from a file. If the file doesn't exist,
    the embeddings have the wrong dimensions, or recomputation is forced, it generates new embeddings using
    either an API client or a local embedding model.

    Args:
        path (str): Base path for saving/loading embeddings file.
        client (object): API client for generating embeddings (e.g., OpenAI client).
        decoded_strs (List[str]): List of strings to embed.
        local_embedding_model_str (str, optional): HuggingFace model identifier for local embedding. Defaults to "thenlper/gte-large".
        local_embedding_model (AutoModel, optional): Pre-loaded local embedding model. Defaults to None.
        tokenizer (AutoTokenizer, optional): Tokenizer for the local model. Required if local_embedding_model is provided. Defaults to None.
        device (str, optional): Device to use for computations. Defaults to "cuda:0".
        recompute_embeddings (bool, optional): Force recomputation of embeddings. Defaults to False.
        batch_size (int, optional): Batch size for embedding generation. Defaults to 8.
        save_embeddings (bool, optional): Whether to save newly generated embeddings. Defaults to True.
        tqdm_disable (bool, optional): Disable tqdm progress bars. Defaults to False.
        clustering_instructions (str, optional): Instructions for embedding model. Defaults to "Identify the topic or theme of the given texts".
        max_length (int, optional): Maximum sequence length for tokenization. Defaults to 512.
        bnb_config (BitsAndBytesConfig, optional): Quantization configuration. Defaults to None.

    Returns:
        List[List[float]]: List of embeddings, where each embedding is a list of floats.

    Raises:
        ValueError: If the loaded embeddings have incorrect dimensions or if tokenizer is not provided with local_embedding_model.

    Note:
        - The function saves embeddings to '{path}_embeddings.pkl' if save_embeddings is True.
        - It uses either an API client (if provided) or a local model for generating embeddings.
        - For the 'nvidia/NV-Embed-v1' model, it uses a specific encoding method.
    """
    # First, try to load past embeddings from file:
    if not recompute_embeddings:
        try:
            print(f"Loading past embeddings from {path + '_embeddings.pkl'}")
            with open(path + "_embeddings.pkl", "rb") as f:
                embeddings_list = pickle.load(f)
                # Check that embeddings_list has the expected dimensions
                if len(embeddings_list) != len(decoded_strs):
                    raise ValueError("The loaded embeddings have the wrong dimensions.")
                return embeddings_list
        except:
            print("Could not load past embeddings. Generating new embeddings.")

    # Then, check if we have a local embedding model or API client:
    if client is None and local_embedding_model is None and local_embedding_model_str is None:
        raise ValueError("Either a local embedding model or an API client must be provided.")
    
    embeddings_list = []
    if client is not None:
        for i in tqdm(range(0, len(decoded_strs), batch_size), desc="Generating embeddings", disable=tqdm_disable):
            batch = decoded_strs[i:i+batch_size]
            embeddings = client.embeddings.create(input = batch, model = "text-embedding-ada-002").data
            embeddings_list.extend([e.embedding for e in embeddings])
    else:
        if local_embedding_model is None:
            # Load local embedding model from HuggingFace
            tokenizer = AutoTokenizer.from_pretrained(local_embedding_model_str)

            if local_embedding_model_str in ["nvidia/NV-Embed-v1", "nvidia/NV-Embed-v2"]:
                local_embedding_model = AutoModel.from_pretrained(local_embedding_model_str, trust_remote_code = True, quantization_config=bnb_config, torch_dtype=torch.float16, device_map={"": 0} if device == "cuda:0" else "auto")
            else:
                local_embedding_model = AutoModel.from_pretrained(local_embedding_model_str).to(device)
            keep_embedding_model = False
        else:
            if tokenizer is None:
                raise ValueError("The tokenizer must be provided if the local embedding model is provided.")
            keep_embedding_model = True
        
        pad_token_id = tokenizer.pad_token_id
        with torch.no_grad():
            for i in tqdm(range(0, len(decoded_strs), batch_size), desc="Generating embeddings", disable=tqdm_disable):
                batch = decoded_strs[i:i+batch_size]
                if local_embedding_model_str == "nvidia/NV-Embed-v1" or local_embedding_model_str == "nvidia/NV-Embed-v2":
                    embeddings = local_embedding_model.encode(batch, instruction = clustering_instructions, max_length = max_length)
                else:
                    batch = [get_detailed_instruct(clustering_instructions, query) for query in batch]
                    batch_ids = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
                    output = local_embedding_model(**batch_ids, output_hidden_states=True)
                    token_embeddings = output.hidden_states[-1]
                    # Note, must avoid averaging over padding tokens.
                    mask = batch_ids['input_ids'] != pad_token_id
                    # Compute the average token embedding, excluding padding tokens
                    embeddings = torch.sum(token_embeddings * mask.unsqueeze(-1), dim=1) / mask.sum(dim=1, keepdim=True)
                    del output
                embeddings_list.extend([e.numpy() for e in embeddings.detach().cpu()])
        if not keep_embedding_model:
            del local_embedding_model
            del tokenizer
    # Also, save the new embeddings to file
    if save_embeddings and path is not None:
        with open(path + "_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings_list, f)
    return embeddings_list

def update_diversification_instructions(
    label_diversification_str_instructions: str,
    prior_labels_for_diversification: List[str],
    local_embedding_model_str: str = "intfloat/multilingual-e5-large-instruct",
    num_cluster_centers_to_use_for_label_diversification: int = 5,
    tsne_save_path: Optional[str] = None,
    run_prefix: str = "",
    api_provider: str = "openai",
    api_model_str: str = "gpt-4.1",
    api_stronger_model_str: Optional[str] = None,
    max_labeling_tokens: int = None,
    auth_key: str = None,
    client: Optional[Union[Anthropic, OpenAI, Client]] = None,
    api_interactions_save_loc: Optional[str] = None,
    logging_level: str = "INFO",
    logger: Optional[BoundLoggerLazyProxy] = None,
    random_seed: int = 0,
) -> str:
    """
    Update label diversification instructions by analyzing prior labels and generating theme summaries.
    
    This function:
    1. Embeds the prior labels
    2. Clusters the embeddings using k-means
    3. Finds representative labels closest to cluster centers
    4. Optionally creates t-SNE visualization
    5. Asks the assistant to summarize common themes
    6. Returns updated diversification instructions
    
    Args:
        label_diversification_str_instructions: Instructions for diversifying the contrastive labels.
            Especailly about how to summarize the common themes across the labels closest to the cluster centers.
            If None, will use the default instructions.
        prior_labels_for_diversification: List of previously generated labels
        local_embedding_model_str: Name of local embedding model to use
        num_cluster_centers_to_use_for_label_diversification: Number of cluster centers for diversification
        tsne_save_path: Path to save t-SNE plot (None to skip plotting)
        run_prefix: Prefix for t-SNE plot filename
        api_provider: API provider for theme summarization
        api_model_str: API model string
        api_stronger_model_str: Optional stronger API model string
        max_labeling_tokens: Maximum number of tokens allowed for labeling a single cluster label
        auth_key: API authentication key
        client: API client object
        api_interactions_save_loc: File to save API interactions
        logging_level: Logging level
        logger: Logger for API requests
        random_seed: Random seed to use for reproducibility. Defaults to 0. Used for sklearn's random state initialization.
    Returns:
        Updated label diversification instructions string
    """
    if max_labeling_tokens is None:
        max_labeling_tokens = get_max_labeling_tokens(api_provider, api_model_str if api_stronger_model_str is None else api_stronger_model_str)

    # 1. Embed the prior labels
    label_embeddings = read_past_embeddings_or_generate_new(
        path=None,  # No persistence needed
        client=None,  # Use local embedding model
        decoded_strs=prior_labels_for_diversification,
        local_embedding_model_str=local_embedding_model_str,
        save_embeddings=False,  # Don't save
        recompute_embeddings=True,  # Always recompute
        tqdm_disable=True  # Don't show progress bar for this small task
    )
    label_embeddings = np.array(label_embeddings)
    
    # 2. Cluster the embeddings using k-means
    n_clusters = min(num_cluster_centers_to_use_for_label_diversification, len(prior_labels_for_diversification))
    
    if n_clusters > 1 and len(prior_labels_for_diversification) >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(label_embeddings)
        cluster_centers = kmeans.cluster_centers_
        
        # 3. Find the labels closest to each cluster center
        distances = cdist(cluster_centers, label_embeddings)
        closest_label_indices = np.argmin(distances, axis=1)
        representative_labels = [prior_labels_for_diversification[idx] for idx in closest_label_indices]
    else:
        # If we don't have enough labels to cluster meaningfully, use all of them
        representative_labels = prior_labels_for_diversification
    
    # Plot the t-SNE of the label embeddings
    if tsne_save_path is not None and len(label_embeddings) > 1:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(tsne_save_path), exist_ok=True)
        
        # Compute t-SNE (use perplexity that's appropriate for the data size)
        perplexity = min(30, max(4, len(label_embeddings) - 1))
        tsne = TSNE(n_components=2, random_state=random_seed, perplexity=perplexity)
        tsne_embeddings = tsne.fit_transform(label_embeddings)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                            c=cluster_labels, cmap='tab10', alpha=0.7, s=50)
        
        # Add labels for representative points (closest to cluster centers)
        if n_clusters > 1:
            for i, idx in enumerate(closest_label_indices):
                plt.annotate(f'C{i}', 
                        (tsne_embeddings[idx, 0], tsne_embeddings[idx, 1]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.title(f't-SNE Visualization of {len(prior_labels_for_diversification)} Generated Labels\n'
                f'({n_clusters} clusters identified)')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        
        # Add colorbar if we have multiple clusters
        if n_clusters > 1:
            plt.colorbar(scatter, label='Cluster')
        
        plt.tight_layout()
        save_path = tsne_save_path + f"{run_prefix}_tsne_plot.pdf"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()  # Close to free memory
        
        print(f"Saved t-SNE plot of label embeddings to {tsne_save_path}")
    
    # 4. Ask the assistant to summarize common themes
    if label_diversification_str_instructions is None:
        theme_summary_prompt = (
            "Please analyze the following labels that describe differences between two language models and "
            "summarize the common themes or patterns they focus on:\n\n"
            + "\n".join([f"- {label}" for label in representative_labels]) +
            "\n\nProvide a concise summary of the main themes these labels cover."
        )
    else:
        pre_label_instructions = label_diversification_str_instructions.split("<labels>")[0]
        post_label_instructions = label_diversification_str_instructions.split("<labels>")[1]
        theme_summary_prompt = (
            pre_label_instructions + "\n\n"
            + "\n".join([f"- {label}" for label in representative_labels]) +
            "\n\n" + post_label_instructions
        )

    theme_summary = make_api_request(
        theme_summary_prompt,
        api_provider,
        api_model_str if api_stronger_model_str is None else api_stronger_model_str,
        auth_key,
        client=client,
        max_tokens=max_labeling_tokens,
        max_thinking_tokens=10000,
        api_interactions_save_loc=api_interactions_save_loc,
        logging_level=logging_level,
        logger=logger,
        request_info={
            "pipeline_stage": "label diversification theme summary",
            "num_prior_labels": len(prior_labels_for_diversification)
        },
    )
    
    # 5. Update current_label_diversification_content_str
    current_label_diversification_content_str = (
        f"Prior labels have already covered the following themes as distinguishing features between the two models, so your proposed label should focus on different features from the following: {theme_summary.strip()}\n"
        "To maintain diversity, please focus on different features to distinguish the current sets of texts."
    )
    
    print(f"Updated label diversification instructions after {len(prior_labels_for_diversification)} SAFFRON-validated labels: \n\n{current_label_diversification_content_str}")
    
    return current_label_diversification_content_str
 

# For each matching pairs of clusters, generate validated cluster labels based on asking the assistant LLM for the 
# difference between the texts of the two clusters. 
def get_validated_contrastive_cluster_labels(
        decoded_strs_1: List[str], 
        clustering_assignments_1: List[int], 
        decoded_strs_2: List[str], 
        clustering_assignments_2: List[int], 
        cluster_matches: List[Tuple[int, int]],
        local_model: AutoModel, 
        labeling_tokenizer: AutoTokenizer, 
        api_provider: str,
        api_model_str: str,
        api_stronger_model_str: Optional[str] = None,
        auth_key: str = None,
        client: Optional[Union[Anthropic, OpenAI, Client]] = None,
        device: str = "cuda:0",
        compute_p_values: bool = True,
        sampled_comparison_texts_per_cluster: int = 10,
        sampled_texts_per_cluster: int = 10,
        generated_labels_per_cluster: int = 3,
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_1: Dict = None,
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_2: Dict = None,
        num_decodings_per_prompt: int = None,
        contrastive_cluster_label_instruction: Optional[str] = None,
        current_label_diversification_content_str: Optional[str] = None,
        verified_diversity_promoter_labels: List[str] = None,
        pick_top_n_labels: Optional[int] = None,
        n_head_to_head_comparisons_per_text: Optional[int] = None,
        use_unitary_comparisons: bool = False,
        max_unitary_comparisons_per_label: int = 100,
        api_interactions_save_loc: Optional[str] = None,
        logging_level: str = "INFO",
        logger: Optional[BoundLoggerLazyProxy] = None,
        n_permutations: int = 0,
        split_clusters_by_prompt: bool = False,
        frac_prompts_for_label_generation: float = 0.5,
        use_baseline_discrimination: bool = False,
        base_decoded_texts_prompt_ids: List[int] = None,
        finetuned_decoded_texts_prompt_ids: List[int] = None,
        random_seed: int = 0,
        ) ->  Dict[str, Union[
                Dict[Tuple[int, int], Dict[str, float]], # (cluster_1_id, cluster_2_id), label, accuracy
                Dict[Tuple[int, int], Tuple[List[int], List[int]]], # (cluster_1_id, cluster_2_id), list of text ids used for generating labels, list of text ids used for validating labels
                Dict[Tuple[int, int], Dict[str, float]], # (cluster_1_id, cluster_2_id), label, AUC
                Optional[Dict[Tuple[int, int], Dict[str, float]]], # (cluster_1_id, cluster_2_id), label, p-value
                Optional[Dict[Tuple[int, int], Dict[str, float]]], # (cluster_1_id, cluster_2_id), label, baseline accuracy
                Optional[Dict[Tuple[int, int], Dict[str, float]]], # (cluster_1_id, cluster_2_id), label, baseline AUC
                Optional[Dict[Tuple[int, int], Dict[str, LogisticBoWDiscriminator]]], # (cluster_1_id, cluster_2_id), label, baseline discriminator
            ]
        ]:
    """
    Generate and validate contrastive cluster labels for matched pairs of clusters from two different clusterings.

    This function performs the following steps:
    1. Generates contrastive labels for each pair of matched clusters.
    2. Validates the discrimination power of these labels.
    3. Optionally computes p-values for the label scores.
    4. Optionally selects top-performing labels.

    Args:
        decoded_strs_1 (List[str]): List of decoded strings for the first clustering.
        clustering_assignments_1 (List[int]): Cluster assignments of decoded_strs_1.
        decoded_strs_2 (List[str]): List of decoded strings for the second clustering.
        clustering_assignments_2 (List[int]): Cluster assignments of decoded_strs_2.
        cluster_matches (List[Tuple[int, int]]): List of matched cluster pairs between the two clusterings.
            Formatted as (cluster_id_1, cluster_id_2).
        local_model (AutoModel): Local model for text generation if not using API.
        labeling_tokenizer (AutoTokenizer): Tokenizer for the local model.
        api_provider (str): API provider for text generation (e.g., 'openai').
        api_model_str (str): Model string for API requests.
        api_stronger_model_str (str): Model string for API requests for the stronger model. Can be used to generate labels.
        auth_key (str): Authentication key for API requests.
        client (Optional[Union[Anthropic, OpenAI, Client]], optional): API client for text
            generation. Defaults to None.
        device (str, optional): Device to use for computations (e.g., 'cuda:0'). Defaults to "cuda:0".
        compute_p_values (bool, optional): Whether to compute p-values. Defaults to True.
        sampled_comparison_texts_per_cluster (int, optional): Number of texts to sample per cluster for 
            comparison. Defaults to 10.
        sampled_texts_per_cluster (int, optional): Number of texts to sample per cluster for label generation. 
            Defaults to 10.
        generated_labels_per_cluster (int, optional): Number of labels to generate per cluster pair. 
            Defaults to 3.
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_1 (Dict, optional): Nested dict. First dict is indexed by cluster id.
            Leads to a dict indexed by prompt id, which leads to a list of indices for where the decodings of that
            prompt can be found in decoded_strs, for the base model. Can be provided to make the label generation select only num_decodings_per_prompt decodings per prompt to base the labels off of.
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_2 (Dict, optional): Same as 
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_1, but for finetuned model.
        num_decodings_per_prompt (int, optional): The number of decodings per prompt we use to generate labels,
            assuming the cluster_ids_to_prompt_ids_to_decoding_ids_dict was provided.
        contrastive_cluster_label_instruction (Optional[str]): Instruction for label generation.
        diversify_contrastive_labels (bool, optional): Whether to automatically diversify the contrastive labels 
            by keeping a running summary of the common themes from previous labels, and then generating new labels 
            that touch on new themes. Defaults to True. Running summary is based on a clustering of the previously 
            generated labels, then using the assistant to summarize the common themes across the labels closest
            to the cluster centers.
        current_label_diversification_content_str (Optional[str]): Label diversification instructions. Defaults to None.
        verified_diversity_promoter_labels (List[str], optional): Verified diversity promoter labels. Can optionally be used 
            to encourage the assistant to generate labels that are more diverse by avoiding generating labels that are 
            similar to the verified diversity promoter labels. Defaults to None.
        pick_top_n_labels (Optional[int], optional): Number of top labels to select per cluster pair. 
            Defaults to None.
        n_head_to_head_comparisons_per_text (Optional[int], optional): For each comparison text, how 
            many of the other comparison texts should it be tested against. Defaults to None, which 
            means all other comparison texts are used.
        use_unitary_comparisons (bool, optional): Whether to use unitary comparisons, i.e. test labels
            by their ability to let the assistant determine which cluster a given text belongs to, 
            without considering another text for comparison. Defaults to False.
        max_unitary_comparisons_per_label (int, optional): Maximum number of unitary comparisons to perform 
            per label. Defaults to 100.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        n_permutations (int, optional): Number of permutations to perform for p-value computation. Defaults to 0.
        split_clusters_by_prompt (bool, optional): Whether to split the clusters by prompt during discriminative 
            evaluation of the labels. If True, we will ensure no overlap in prompts between the label generation 
            and evaluation splits of each cluster. Defaults to False.
        frac_prompts_for_label_generation (float, optional): The fraction of prompts to use for label generation.
            Defaults to 0.5.
        base_decoded_texts_prompt_ids (List[int], optional): List of prompt IDs for the base decoded texts.
            Defaults to None.
        finetuned_decoded_texts_prompt_ids (List[int], optional): List of prompt IDs for the finetuned decoded texts.
            Defaults to None.
        random_seed (int, optional): Random seed to use for reproducibility. Defaults to 0. Used for the baseline 
            discriminator.
    Returns:
        dict: A dictionary containing the following key / value pairs:
            - 'cluster_pair_scores': (Dict[Tuple[int, int], Dict[str, float]]) accuracy scores for each label 
                in each cluster pair.
            - 'all_cluster_texts_used_for_validating_label_strs_ids': 
                (Dict[Tuple[int, int], Tuple[List[int], List[int]]]) Indices of texts used for validating labels.
            - 'cluster_pair_auc_scores': (Dict[Tuple[int, int], Dict[str, float]]) AUC scores for each label in each
                cluster pair.            
            - 'p_values': (Dict[Tuple[int, int], Dict[str, float]]) P-values for each label, based on the accuracy scores
                and binomial test (if compute_p_values is True).
            - 'auc_permutation_p_values': (Dict[Tuple[int, int], Dict[str, float]]) Permutation p-values for each label, based 
                on the AUC scores and permutation test (if n_permutations > 0).
            - 'baseline_cluster_pair_scores': (Optional[Dict[Tuple[int, int], Dict[str, float]]]) Accuracy scores for 
                each label in each cluster pair, based on the random forest bag of words baseline discrimination model
                (if use_baseline_discrimination is True).
            - 'baseline_cluster_pair_auc_scores': (Optional[Dict[Tuple[int, int], Dict[str, float]]]) AUC scores for 
                each label in each cluster pair, based on the random forest bag of words baseline discrimination model
                (if use_baseline_discrimination is True).
            - 'baseline_cluster_pair_discriminators': (Optional[Dict[Tuple[int, int], Dict[str, LogisticBoWDiscriminator]]]) 
                Baseline discriminators for each label in each cluster pair, based on the random forest bag of words baseline 
                discrimination model (if use_baseline_discrimination is True).
    """
    # Initialize dictionaries to store cluster labels and text indices used for generating labels
    all_cluster_texts_used_for_label_strs_ids_1 = {}
    all_cluster_texts_used_for_label_strs_ids_2 = {}
    cluster_label_strs = {}

    # Generate contrastive labels for each pair of matched clusters
    for cluster_id_1, cluster_id_2 in cluster_matches:
        # Generate labels describing the differences between the two clusters
        decoded_labels, selected_text_indices_1, selected_text_indices_2 = contrastive_label_double_cluster(
            decoded_strs_1=decoded_strs_1, 
            clustering_assignments_1=clustering_assignments_1, 
            cluster_id_1=cluster_id_1,
            decoded_strs_2=decoded_strs_2, 
            clustering_assignments_2=clustering_assignments_2, 
            cluster_id_2=cluster_id_2, 
            local_model=local_model, 
            labeling_tokenizer=labeling_tokenizer, 
            api_provider=api_provider,
            api_model_str=api_model_str if api_stronger_model_str is None else api_stronger_model_str,
            auth_key=auth_key,
            client=client,
            device=device, 
            sampled_texts_per_cluster=sampled_texts_per_cluster, 
            generated_labels_per_cluster=generated_labels_per_cluster,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_1=cluster_ids_to_prompt_ids_to_decoding_ids_dict_1,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_2=cluster_ids_to_prompt_ids_to_decoding_ids_dict_2,
            num_decodings_per_prompt=num_decodings_per_prompt,
            contrastive_cluster_label_instruction=contrastive_cluster_label_instruction,
            current_label_diversification_content_str=current_label_diversification_content_str,
            verified_diversity_promoter_labels=verified_diversity_promoter_labels,
            api_interactions_save_loc=api_interactions_save_loc,
            logging_level=logging_level,
            logger=logger,
            split_clusters_by_prompt=split_clusters_by_prompt,
            frac_prompts_for_label_generation=frac_prompts_for_label_generation,
            base_decoded_texts_prompt_ids=base_decoded_texts_prompt_ids,
            finetuned_decoded_texts_prompt_ids=finetuned_decoded_texts_prompt_ids
        )
        
        # Store the generated labels and the indices of texts used for label generation
        cluster_label_strs[(cluster_id_1, cluster_id_2)] = decoded_labels
        all_cluster_texts_used_for_label_strs_ids_1[(cluster_id_1, cluster_id_2)] = selected_text_indices_1
        all_cluster_texts_used_for_label_strs_ids_2[(cluster_id_1, cluster_id_2)] = selected_text_indices_2

    # Validate the discrimination power of the generated labels
    cluster_pair_scores, all_cluster_texts_used_for_validating_label_strs_ids, auc_permutation_p_values, cluster_pair_auc_scores, baseline_cluster_pair_scores, baseline_cluster_pair_auc_scores, baseline_cluster_pair_discriminators = validate_cluster_label_comparative_discrimination_power(
        decoded_strs_1, 
        clustering_assignments_1, 
        all_cluster_texts_used_for_label_strs_ids_1,
        decoded_strs_2, 
        clustering_assignments_2, 
        all_cluster_texts_used_for_label_strs_ids_2,
        cluster_label_strs, 
        local_model, 
        labeling_tokenizer, 
        api_provider, 
        api_model_str, 
        auth_key, 
        client=client,
        device=device, 
        sampled_comparison_texts_per_cluster=sampled_comparison_texts_per_cluster,
        n_head_to_head_comparisons_per_text=n_head_to_head_comparisons_per_text,
        use_unitary_comparisons=use_unitary_comparisons,
        max_unitary_comparisons_per_label=max_unitary_comparisons_per_label,
        api_interactions_save_loc=api_interactions_save_loc,
        logging_level=logging_level,
        logger=logger,
        n_permutations=n_permutations,
        split_clusters_by_prompt=split_clusters_by_prompt,
        base_decoded_texts_prompt_ids=base_decoded_texts_prompt_ids,
        finetuned_decoded_texts_prompt_ids=finetuned_decoded_texts_prompt_ids,
        use_baseline_discrimination=use_baseline_discrimination,
        random_seed=random_seed
    )

    if pick_top_n_labels is not None:
        # For each cluster, pick the top n labels based on AUC score
        top_n_labels = {}
        for cluster_pair, label_scores in cluster_pair_auc_scores.items():
            # Sort labels by their AUC scores in descending order
            sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Pick the top n labels
            top_n = sorted_labels[:pick_top_n_labels]
            
            # Store the top n labels and their scores
            top_n_labels[cluster_pair] = {label: score for label, score in top_n}
        # Update cluster_labels and auc_permutation_p_values with only the top n labels
        cluster_pair_scores = {cluster_pair: top_n_labels[cluster_pair] for cluster_pair in cluster_pair_scores.keys()}
        cluster_pair_auc_scores = {cluster_pair: top_n_labels[cluster_pair] for cluster_pair in cluster_pair_auc_scores.keys()}
        auc_permutation_p_values = {cluster_pair: auc_permutation_p_values[cluster_pair] for cluster_pair in auc_permutation_p_values.keys()}
        baseline_cluster_pair_scores = {cluster_pair: baseline_cluster_pair_scores[cluster_pair] for cluster_pair in baseline_cluster_pair_scores.keys()}
        baseline_cluster_pair_auc_scores = {cluster_pair: baseline_cluster_pair_auc_scores[cluster_pair] for cluster_pair in baseline_cluster_pair_auc_scores.keys()}
        baseline_cluster_pair_discriminators = {cluster_pair: baseline_cluster_pair_discriminators[cluster_pair] for cluster_pair in baseline_cluster_pair_discriminators.keys()}
    # Optionally compute p-values if required
    if compute_p_values:
        # Use exact binomial test for p-values (separate from the permutation-based p-values already computed)
        # Under null hypothesis, each score is the average of max_unitary_comparisons_per_label 
        # IID coin flips (50/50 between 0 and 1)
        p_values = {}
        for cluster_pair, label_scores in cluster_pair_scores.items():
            p_values[cluster_pair] = {}
            for label, accuracy_score in label_scores.items():
                if accuracy_score > 0:
                    # Convert accuracy back to number of correct classifications
                    num_correct = round(accuracy_score * max_unitary_comparisons_per_label)
                    
                    # Use one-sided exact binomial test
                    # H0: p = 0.5 (no discriminative power)
                    # H1: p > 0.5 (discriminative power)
                    result = binomtest(num_correct, max_unitary_comparisons_per_label, p=0.5, alternative='greater')
                    p_value = result.pvalue
                    p_values[cluster_pair][label] = p_value
                else:
                    p_values[cluster_pair][label] = 1.0
    
        # Print the aucs and p-values for each label in each cluster
        for cluster_pair, labels_p_values in p_values.items():
            print(f"Clusters {cluster_pair} accuracy scores and p-values:")
            for label, p_value in labels_p_values.items():
                accuracy_score = cluster_pair_scores[cluster_pair][label]
                permutation_p_value = auc_permutation_p_values[cluster_pair][label]
                baseline_accuracy_score = baseline_cluster_pair_scores[cluster_pair][label]
                baseline_auc_score = baseline_cluster_pair_auc_scores[cluster_pair][label]
                outstr_label = label.replace("\n", "\\n")
                binomial_p_value_str = "N/A" if p_value is None else f"{p_value:.5f}"
                permutation_p_value_str = "N/A" if permutation_p_value is None else f"{permutation_p_value:.5f}"
                print(f"accuracy: {accuracy_score:.5f}, Binomial P-value: {binomial_p_value_str}, Permutation P-value: {permutation_p_value_str}, Label: {outstr_label}")
                print(f"Label AUC score: {cluster_pair_auc_scores[cluster_pair][label]:.5f}")
                print(f"Baseline accuracy score: {baseline_accuracy_score:.5f}, Baseline AUC score: {baseline_auc_score:.5f}")

    return_dict = {
        "cluster_pair_scores": cluster_pair_scores,
        "all_cluster_texts_used_for_validating_label_strs_ids": all_cluster_texts_used_for_validating_label_strs_ids,
        "cluster_pair_auc_scores": cluster_pair_auc_scores,
    }
    if use_baseline_discrimination:
        return_dict["baseline_cluster_pair_scores"] = baseline_cluster_pair_scores
        return_dict["baseline_cluster_pair_auc_scores"] = baseline_cluster_pair_auc_scores
        return_dict["baseline_cluster_pair_discriminators"] = baseline_cluster_pair_discriminators
    if compute_p_values:
        return_dict["p_values"] = p_values
    if n_permutations > 0:
        return_dict["auc_permutation_p_values"] = auc_permutation_p_values
    return return_dict


def build_contrastive_K_neighbor_similarity_graph(
    decoded_strs_1: List[str],
    clustering_assignments_1: List[int],
    embeddings_1: List[List[float]],
    cluster_labels_1: List[str],
    decoded_strs_2: List[str],
    clustering_assignments_2: List[int],
    embeddings_2: List[List[float]],
    cluster_labels_2: List[str],
    K: int,
    match_by_ids: bool = False,
    local_model: AutoModel = None,
    labeling_tokenizer: AutoTokenizer = None,
    api_provider: str = None,
    api_model_str: str = None,
    api_stronger_model_str: Optional[str] = None,
    max_labeling_tokens: int = None,
    auth_key: str = None,
    client: Optional[Union[Anthropic, OpenAI, Client]] = None,
    local_embedding_model_str: str = "intfloat/multilingual-e5-large-instruct",
    device: str = "cuda:0",
    sampled_comparison_texts_per_cluster: int = 50,
    cross_validate_contrastive_labels: bool = False,
    cross_validate_on_all_clusters: bool = True,
    sampled_texts_per_cluster: int = 10,
    generated_labels_per_cluster: int = 3,
    cluster_ids_to_prompt_ids_to_decoding_ids_dict_1: Dict = None,
    cluster_ids_to_prompt_ids_to_decoding_ids_dict_2: Dict = None,
    num_decodings_per_prompt: int = None,
    contrastive_cluster_label_instruction: Optional[str] = None,
    label_diversification_str_instructions: Optional[str] = None,
    diversify_contrastive_labels: bool = False,
    verified_diversity_promoter: bool = False,
    max_verified_diversity_promoter_labels: int = 10,
    n_head_to_head_comparisons_per_text: Optional[int] = None,
    use_unitary_comparisons: bool = False,
    max_unitary_comparisons_per_label: int = 20,
    api_interactions_save_loc: Optional[str] = None,
    logging_level: str = "INFO",
    logger: Optional[BoundLoggerLazyProxy] = None,
    tsne_save_path: str = "diversity_labels_tsne/",
    run_prefix: str = "",
    n_permutations: int = 0,
    split_clusters_by_prompt: bool = False,
    frac_prompts_for_label_generation: float = 0.5,
    base_decoded_texts_prompt_ids: List[int] = None,
    finetuned_decoded_texts_prompt_ids: List[int] = None,
    random_seed: int = 0,
) -> nx.Graph:
    """
    Build a K-nearest neighbor similarity graph between two sets of clusters based on contrastive labels.

    Args:
        decoded_strs_1 (List[str]): List of decoded strings for the first set.
        clustering_assignments_1 (List[int]): Cluster assignments for the first set.
        embeddings_1 (List[List[float]]): Embeddings for the first set.
        cluster_labels_1 (List[str]): Cluster labels for the first set.
        decoded_strs_2 (List[str]): List of decoded strings for the second set.
        clustering_assignments_2 (List[int]): Cluster assignments for the second set.
        embeddings_2 (List[List[float]]): Embeddings for the second set.
        cluster_labels_2 (List[str]): Cluster labels for the second set.
        K (int): Number of nearest neighbors to consider for each cluster.
        match_by_ids (bool): Whether to match clusters by their IDs, not embedding distances.
        local_model (AutoModel): Local model for text generation.
        labeling_tokenizer (AutoTokenizer): Tokenizer for the local model.
        api_provider (str): API provider for text generation.
        api_model_str (str): Model string for API requests.
        api_stronger_model_str (str): Model string for API requests for the stronger model. Can be used to generate labels.
            If None, will use the same model as api_model_str.
        max_labeling_tokens (int): Maximum number of tokens allowed for labeling a single cluster label.
            If None, will be determined based on the api_provider and api_model_str.
        auth_key (str): Authentication key for API requests.
        client (Optional[Union[Anthropic, OpenAI, Client]], optional): API client for text
            generation. Defaults to None.
        local_embedding_model_str (str): Model string for local embedding model. Defaults to 
            "intfloat/ multilingual-e5-large-instruct".
        device (str, optional): Device to use for computations (e.g., 'cuda:0'). Defaults to "cuda:0".
        sampled_comparison_texts_per_cluster (int): Number of texts to sample per cluster for comparison.
        cross_validate_contrastive_labels (bool): Whether to cross-validate the contrastive labels by testing the 
            discriminative score of the labels on different clusters from which they were generated.
        cross_validate_on_all_clusters (bool): Whether to cross-validate the contrastive labels on texts sampled from all other clusters.
        sampled_texts_per_cluster (int): Number of texts to sample per cluster for label generation.
        generated_labels_per_cluster (int): Number of labels to generate per cluster pair.
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_1 (Dict, optional): Nested dict. First dict is indexed by cluster id.
            Leads to a dict indexed by prompt id, which leads to a list of indices for where the decodings of that
            prompt can be found in decoded_strs. Can be provided to make the label generation select only num_decodings_per_prompt to base the labels off of.
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_2 (Dict, optional): Same as 
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_1, but for finetuned model.
        num_decodings_per_prompt (int, optional): Number of decodings per prompt to use for label generation.
        contrastive_cluster_label_instruction (Optional[str]): Instruction for label generation.
        label_diversification_str_instructions (Optional[str]): Instructions for diversifying the contrastive labels.
        diversify_contrastive_labels (bool): Whether to diversify the contrastive labels by clustering the previously 
            generated labels, and then using the assistant to summarize the common themes across the labels closest to 
            the cluster centers. We then provide those summaries to the assistant to generate new labels that are different 
            from the previous ones.
        verified_diversity_promoter (bool): Whether to promote diversity in the contrastive labels by recording any 
            hypotheses that are verified discriminatively, providing them to the assistant, and asking the assistant to 
            look for other hypotheses that are different.
        max_verified_diversity_promoter_labels (int): Maximum number of verified diversity promoter labels to use.
        n_head_to_head_comparisons_per_text (Optional[int]): Number of head-to-head comparisons per text.
        use_unitary_comparisons (bool): Whether to use unitary comparisons.
        max_unitary_comparisons_per_label (int): Maximum number of unitary comparisons to perform per label.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        tsne_save_path (str): Path to save the t-SNE plot of the label embeddings.
        run_prefix (str): Prefix to use for the t-SNE plot file name. Defaults to \"\".
        n_permutations (int, optional): Number of permutations to perform for the permutation test. Defaults to 0.
        split_clusters_by_prompt (bool): Whether to split the clusters by prompt during discriminative evaluation of the labels.
            If True, we will ensure no overlap in prompts between the label generation and evaluation splits of each cluster.
            Defaults to False.
        frac_prompts_for_label_generation (float, optional): The fraction of prompts to use for label generation.
            Defaults to 0.5.
        base_decoded_texts_prompt_ids (List[int], optional): List of prompt IDs for the base decoded texts.
            Defaults to None.
        finetuned_decoded_texts_prompt_ids (List[int], optional): List of prompt IDs for the finetuned decoded texts.
            Defaults to None.
        random_seed (int, optional): Random seed to use for reproducibility. Defaults to 0. Used for random state initialization 
            of all RNGs used in the code, and for the baseline discriminator.
    Returns:
        nx.Graph: A graph where nodes are clusters from both sets and edge attributes include
            similarity scores, labels, and per-label accuracy / AUC scores.
    """

    if random_seed is not None:
        seed_everything(random_seed)
    # Create a graph
    G = nx.Graph()
    # Calculate cluster centroids for both sets
    unique_clusters_1 = np.unique(clustering_assignments_1)
    unique_clusters_2 = np.unique(clustering_assignments_2)

     # Add nodes (clusters) to the graph
    for cluster in unique_clusters_1:
        G.add_node(f"1_{cluster}", cluster_id=cluster, cluster_label_str=cluster_labels_1[cluster])
    for cluster in unique_clusters_2:
        G.add_node(f"2_{cluster}", cluster_id=cluster, cluster_label_str=cluster_labels_2[cluster])

    prior_labels_for_diversification = []
    verified_diversity_promoter_labels = []
    current_label_diversification_content_str = ""

    hypotheses = []
    
    accuracy_cross_validation_scores = []
    accuracy_validation_scores = []
    accuracy_binomial_p_values = []

    auc_validation_scores = []
    auc_cross_validation_scores = []
    auc_permutation_p_values = []

    baseline_accuracy_validation_scores = []
    baseline_auc_validation_scores = []

    baseline_accuracy_cross_validation_scores = []
    baseline_auc_cross_validation_scores = []

    # Initialize SAFFRON for multiple comparison correction
    saffron = SAFFRON(alpha=0.05, lambda_param=0.5, gamma_param=0.5)
    
    # Function to compute similarity and add edge
    def compute_similarity_and_add_edge(cluster1, cluster2, set1, set2, prior_labels_for_diversification, diversify_contrastive_labels, current_label_diversification_content_str, verified_diversity_promoter_labels, saffron, accuracy_cross_validation_scores, accuracy_validation_scores, auc_validation_scores, auc_cross_validation_scores, auc_permutation_p_values, random_seed: int = 0):
        print(f"Computing similarity and adding edge for clusters {cluster1} and {cluster2} from set {set1} and {set2}")
        cluster_matches = [(cluster1, cluster2)]

        if diversify_contrastive_labels and len(prior_labels_for_diversification) % 10 == 0 and len(prior_labels_for_diversification) >= 10:
            print(f"Length of prior labels for diversification: {len(prior_labels_for_diversification)}")
            current_label_diversification_content_str = update_diversification_instructions(
                label_diversification_str_instructions=label_diversification_str_instructions,
                prior_labels_for_diversification=prior_labels_for_diversification,
                local_embedding_model_str=local_embedding_model_str,
                num_cluster_centers_to_use_for_label_diversification=5,
                tsne_save_path=tsne_save_path,
                run_prefix=run_prefix,
                api_provider=api_provider,
                api_model_str=api_model_str,
                api_stronger_model_str=api_stronger_model_str if api_stronger_model_str is not None else api_model_str,
                max_labeling_tokens=max_labeling_tokens,
                auth_key=auth_key,
                client=client,
                api_interactions_save_loc=api_interactions_save_loc,
                logging_level=logging_level,
                logger=logger,
                random_seed=random_seed
            )
        result = get_validated_contrastive_cluster_labels(
            decoded_strs_1 if set1 == 1 else decoded_strs_2,
            clustering_assignments_1 if set1 == 1 else clustering_assignments_2,
            decoded_strs_2 if set2 == 2 else decoded_strs_1,
            clustering_assignments_2 if set2 == 2 else clustering_assignments_1,
            cluster_matches,
            local_model, 
            labeling_tokenizer,
            api_provider, 
            api_model_str, 
            api_stronger_model_str if api_stronger_model_str is not None else api_model_str,
            auth_key, 
            client=client,
            device=device,
            compute_p_values=True,
            sampled_comparison_texts_per_cluster=sampled_comparison_texts_per_cluster,
            sampled_texts_per_cluster=sampled_texts_per_cluster,
            generated_labels_per_cluster=generated_labels_per_cluster,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_1=cluster_ids_to_prompt_ids_to_decoding_ids_dict_1 if set1 == 1 else cluster_ids_to_prompt_ids_to_decoding_ids_dict_2,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_2=cluster_ids_to_prompt_ids_to_decoding_ids_dict_2 if set1 == 1 else cluster_ids_to_prompt_ids_to_decoding_ids_dict_1,
            num_decodings_per_prompt=num_decodings_per_prompt,
            contrastive_cluster_label_instruction=contrastive_cluster_label_instruction,
            n_head_to_head_comparisons_per_text=n_head_to_head_comparisons_per_text,
            use_unitary_comparisons=use_unitary_comparisons,
            max_unitary_comparisons_per_label=max_unitary_comparisons_per_label,
            current_label_diversification_content_str=current_label_diversification_content_str,
            verified_diversity_promoter_labels=verified_diversity_promoter_labels,
            api_interactions_save_loc=api_interactions_save_loc,
            logging_level=logging_level,
            logger=logger,
            n_permutations=n_permutations,
            split_clusters_by_prompt=split_clusters_by_prompt,
            frac_prompts_for_label_generation=frac_prompts_for_label_generation,
            use_baseline_discrimination=True,
            base_decoded_texts_prompt_ids=base_decoded_texts_prompt_ids if set1 == 1 else finetuned_decoded_texts_prompt_ids,
            finetuned_decoded_texts_prompt_ids=finetuned_decoded_texts_prompt_ids if set1 == 1 else base_decoded_texts_prompt_ids,
            random_seed=random_seed
        )
        p_values = result['p_values']
        auc_p_values = result['auc_permutation_p_values']
        
        labels_and_accuracy_scores = result['cluster_pair_scores'][(cluster1, cluster2)]
        labels_and_auc_scores = result['cluster_pair_auc_scores'][(cluster1, cluster2)]
        similarity_score = np.mean([1 - accuracy_score for accuracy_score in labels_and_accuracy_scores.values()])

        baseline_labels_and_accuracy_scores = result['baseline_cluster_pair_scores'][(cluster1, cluster2)]
        baseline_labels_and_auc_scores = result['baseline_cluster_pair_auc_scores'][(cluster1, cluster2)]
        baseline_labels_and_discriminators = result['baseline_cluster_pair_discriminators'][(cluster1, cluster2)]
        # Add basic edge information
        edge_data = {
            'weight': similarity_score,
            'labels': list(labels_and_accuracy_scores.keys()),
            'label_accuracy_scores': labels_and_accuracy_scores,
            'label_p_values': p_values[(cluster1, cluster2)],
            'significant_labels': []  # Will be populated by SAFFRON
        }
        
        # Test each label with SAFFRON
        round_labels = list(labels_and_accuracy_scores.keys())
        round_accuracy_scores = list(labels_and_accuracy_scores.values())
        round_auc_scores = list(labels_and_auc_scores.values())
        round_accuracy_binomial_p_values = list(p_values[(cluster1, cluster2)].values())
        round_auc_permutation_p_values = list(auc_p_values[(cluster1, cluster2)].values())
        round_baseline_accuracy_scores = list(baseline_labels_and_accuracy_scores.values())
        round_baseline_auc_scores = list(baseline_labels_and_auc_scores.values())
        round_baseline_discriminators = list(baseline_labels_and_discriminators.values())

        any_rejected = False
        for i, (label, accuracy_score, auc_score, baseline_accuracy_score, baseline_auc_score) in enumerate(zip(round_labels, round_accuracy_scores, round_auc_scores, round_baseline_accuracy_scores, round_baseline_auc_scores)):
            label_p_value = p_values[(cluster1, cluster2)][label]
            permutation_p_value = auc_p_values[(cluster1, cluster2)][label]
            
            # Test with SAFFRON
            hypothesis_info = {
                'label': label,
                'accuracy_score': accuracy_score,
                'auc_score': auc_score,
                'baseline_accuracy_score': baseline_accuracy_score,
                'baseline_auc_score': baseline_auc_score,
                'clusters': (cluster1, cluster2),
                'set1': set1,
                'set2': set2
            }
            
            is_rejected, alpha_threshold = saffron.test_hypothesis(permutation_p_value, hypothesis_info)
            
            if is_rejected:
                any_rejected = True
                edge_data['significant_labels'].append({
                    'label': label,
                    'metric_score': accuracy_score,
                    'auc_score': auc_score,
                    'baseline_accuracy_score': baseline_accuracy_score,
                    'baseline_auc_score': baseline_auc_score,
                    'p_value': label_p_value,
                    'permutation_p_value': permutation_p_value,
                    'alpha_threshold': alpha_threshold
                })
                
                # Track for diversity promotion if enabled
                if verified_diversity_promoter:
                    verified_diversity_promoter_labels.append(label)
                    num_verified_diversity_promoter_labels = len(verified_diversity_promoter_labels)
                    if num_verified_diversity_promoter_labels > max_verified_diversity_promoter_labels:
                        verified_diversity_promoter_labels = verified_diversity_promoter_labels[num_verified_diversity_promoter_labels - max_verified_diversity_promoter_labels:]
        
        G.add_edge(f"{set1}_{cluster1}", f"{set2}_{cluster2}", **edge_data)
        
        # Select the label with the highest AUC score to represent the cluster pair for diversification purposes
        max_auc_score_idx = np.argmax(round_auc_scores)
        new_label = round_labels[max_auc_score_idx]
        new_baseline_discriminator = round_baseline_discriminators[max_auc_score_idx]
        if any_rejected:
            prior_labels_for_diversification.append(new_label)
        accuracy_validation_scores.append(round_accuracy_scores[max_auc_score_idx])
        auc_validation_scores.append(round_auc_scores[max_auc_score_idx])
        accuracy_binomial_p_values.append(round_accuracy_binomial_p_values[max_auc_score_idx])
        auc_permutation_p_values.append(round_auc_permutation_p_values[max_auc_score_idx])
        baseline_accuracy_validation_scores.append(round_baseline_accuracy_scores[max_auc_score_idx])
        baseline_auc_validation_scores.append(round_baseline_auc_scores[max_auc_score_idx])
        hypotheses.append(new_label)
        if cross_validate_contrastive_labels:
            if cross_validate_on_all_clusters:
                # Cross-validate the contrastive label by testing the discriminative score of the label on texts sampled from all other clusters.
                print(f"Cross-validating label '{new_label.replace(chr(10), ' ')}' on all clusters")
                # First, determine how many texts we're going to sample from each cluster.
                # In total, we sample max(max_unitary_comparisons_per_label, sampled_comparison_texts_per_cluster) text pairs, so we need to split this up across the clusters.
                total_text_pairs_to_sample = max(max_unitary_comparisons_per_label, sampled_comparison_texts_per_cluster)
                texts_to_sample_per_cluster = total_text_pairs_to_sample // len(unique_clusters_1) + 1
                if match_by_ids:
                    valid_cluster_pairs = [(unique_clusters_1[i], unique_clusters_2[i]) for i in range(len(unique_clusters_1))]
                else:
                    valid_cluster_pairs = [(c1, c2) for c1 in unique_clusters_1 for c2 in unique_clusters_2 if (c1 != cluster1 or c2 != cluster2)]
                if not valid_cluster_pairs:
                    print("Warning: No other cluster pairs available for cross-validation.")
                    accuracy_cross_validation_scores.append(-1.0)
                    auc_cross_validation_scores.append(-1.0)
                else:
                    # Now, we iterate over the cluster pairs and sample texts_to_sample_per_cluster text pairs from each cluster, before shuffling the list and taking the first total_text_pairs_to_sample text pairs.
                    sampled_texts_1 = []
                    sampled_texts_2 = []
                    for val_cluster1, val_cluster2 in valid_cluster_pairs:
                        val_cluster_1_indices = [i for i, x in enumerate(clustering_assignments_1) if x == val_cluster1]
                        val_cluster_2_indices = [i for i, x in enumerate(clustering_assignments_2) if x == val_cluster2]
                        cluster_texts_1 = random.sample(val_cluster_1_indices, texts_to_sample_per_cluster)
                        cluster_texts_2 = random.sample(val_cluster_2_indices, texts_to_sample_per_cluster)
                        sampled_texts_1.extend(cluster_texts_1)
                        sampled_texts_2.extend(cluster_texts_2)
                    random_suffle_indices = list(range(len(sampled_texts_1)))
                    random.shuffle(random_suffle_indices)
                    sampled_texts_1 = [sampled_texts_1[i] for i in random_suffle_indices]
                    sampled_texts_2 = [sampled_texts_2[i] for i in random_suffle_indices]
                    sampled_texts_1 = sampled_texts_1[:total_text_pairs_to_sample]
                    sampled_texts_2 = sampled_texts_2[:total_text_pairs_to_sample]
            else:
                # Legacy approach kept for comparison and reference
                print("Alert: Cross-validation on single randomly sampled cluster is deprecated. Using legacy approach.")
                # Cross-validate the contrastive labels by testing the discriminative score of the labels on different clusters from which they were generated. Choose a random cluster, then test new_label on that cluster. Print the result.
                possible_pairs = [(c1, c2) for c1 in unique_clusters_1 for c2 in unique_clusters_2 if (c1 != cluster1 or c2 != cluster2)]
                if not possible_pairs:
                    print("Warning: No other cluster pairs available for cross-validation.")
                    accuracy_cross_validation_scores.append(-1.0)
                    auc_cross_validation_scores.append(-1.0)
                    baseline_accuracy_cross_validation_scores.append(-1.0)
                    baseline_auc_cross_validation_scores.append(-1.0)
                else:
                    val_cluster1, val_cluster2 = random.choice(possible_pairs)
                    print(f"Cross-validating label on cluster pair ({val_cluster1}, {val_cluster2})")
                    val_cluster_1_indices = [i for i, x in enumerate(clustering_assignments_1) if x == val_cluster1]
                    val_cluster_2_indices = [i for i, x in enumerate(clustering_assignments_2) if x == val_cluster2]
                    if len(val_cluster_1_indices) < sampled_comparison_texts_per_cluster or len(val_cluster_2_indices) < sampled_comparison_texts_per_cluster:
                        print(f"Warning: Not enough texts for cross-validation on cluster pair ({val_cluster1}, {val_cluster2}). Required {sampled_comparison_texts_per_cluster} texts per cluster. Got {len(val_cluster_1_indices)} and {len(val_cluster_2_indices)}. Skipping.")
                        accuracy_cross_validation_scores.append(-1.0)
                        auc_cross_validation_scores.append(-1.0)
                        baseline_accuracy_cross_validation_scores.append(-1.0)
                        baseline_auc_cross_validation_scores.append(-1.0)
                    else:
                        sampled_texts_1 = random.sample(val_cluster_1_indices, sampled_comparison_texts_per_cluster)
                        sampled_texts_2 = random.sample(val_cluster_2_indices, sampled_comparison_texts_per_cluster)
            accuracy_score, p_value, auc_score = evaluate_label_discrimination(
                new_label,
                sampled_texts_1,
                sampled_texts_2,
                decoded_strs_1,
                decoded_strs_2,
                local_model,
                labeling_tokenizer,
                api_provider,
                api_model_str,
                auth_key,
                client=client,
                device=device,
                mode="contrastive",
                n_head_to_head_comparisons_per_text=n_head_to_head_comparisons_per_text,
                use_unitary_comparisons=use_unitary_comparisons,
                max_unitary_comparisons_per_label=max_unitary_comparisons_per_label,
                api_interactions_save_loc=api_interactions_save_loc,
                logging_level=logging_level,
                scores_logging_mode="cross-validation",
                logger=logger,
                cluster_id_1=val_cluster1,
                cluster_id_2=val_cluster2,
                n_permutations=n_permutations
            )
            baseline_accuracy_cross_validation_score, baseline_auc_cross_validation_score = new_baseline_discriminator(
                [decoded_strs_1[text_id] for text_id in sampled_texts_1], 
                [decoded_strs_2[text_id] for text_id in sampled_texts_2],
            )
            print(f"Cross-validation accuracy for label on pair ({val_cluster1}, {val_cluster2}): {accuracy_score:.5f}")
            print(f"Cross-validation AUC for label: {auc_score:.5f}")
            print(f"Cross-validation baseline accuracy for label: {baseline_accuracy_cross_validation_score:.5f}")
            print(f"Cross-validation baseline AUC for label: {baseline_auc_cross_validation_score:.5f}")
            p_value_str = f"{p_value:.5f}" if n_permutations > 0 else "None"
            print(f"Permutation p-value for label: {p_value_str}")
            accuracy_cross_validation_scores.append(accuracy_score)
            auc_cross_validation_scores.append(auc_score)
            baseline_accuracy_cross_validation_scores.append(baseline_accuracy_cross_validation_score)
            baseline_auc_cross_validation_scores.append(baseline_auc_cross_validation_score)
        return prior_labels_for_diversification, verified_diversity_promoter_labels, current_label_diversification_content_str

    if match_by_ids:
        # Match cluster i in set 1 to cluster i in set 2
        for i in range(len(unique_clusters_1)):
            prior_labels_for_diversification, verified_diversity_promoter_labels, current_label_diversification_content_str = compute_similarity_and_add_edge(unique_clusters_1[i], unique_clusters_2[i], 1, 2, prior_labels_for_diversification, diversify_contrastive_labels, current_label_diversification_content_str, verified_diversity_promoter_labels, saffron, accuracy_cross_validation_scores, accuracy_validation_scores, auc_validation_scores, auc_cross_validation_scores, auc_permutation_p_values, random_seed=random_seed)
    else:
        # Find the K nearest neighbors for each cluster in set 1 from set 2
        centroids_1 = []
        centroids_2 = []

        for cluster in unique_clusters_1:
            cluster_embeddings = [embeddings_1[i] for i in range(len(clustering_assignments_1)) if clustering_assignments_1[i] == cluster]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids_1.append(centroid)

        for cluster in unique_clusters_2:
            cluster_embeddings = [embeddings_2[i] for i in range(len(clustering_assignments_2)) if clustering_assignments_2[i] == cluster]
            centroid = np.mean(cluster_embeddings, axis=0)
            centroids_2.append(centroid)

        # Find K nearest neighbors for each cluster in set 1 from set 2
        nn = NearestNeighbors(n_neighbors=K, metric='euclidean')
        nn.fit(centroids_2)
        distances_1, indices_1 = nn.kneighbors(centroids_1)

        # Find K nearest neighbors for each cluster in set 2 from set 1
        nn.fit(centroids_1)
        distances_2, indices_2 = nn.kneighbors(centroids_2)


        # Compute similarities and add edges for set 1 to set 2
        for i, cluster1 in enumerate(unique_clusters_1):
            for j in range(K):
                cluster2 = unique_clusters_2[indices_1[i][j]]
                if not G.has_edge(f"1_{cluster1}", f"2_{cluster2}"):
                    prior_labels_for_diversification, verified_diversity_promoter_labels, current_label_diversification_content_str = compute_similarity_and_add_edge(cluster1, cluster2, 1, 2, prior_labels_for_diversification, diversify_contrastive_labels, current_label_diversification_content_str, verified_diversity_promoter_labels, saffron, accuracy_cross_validation_scores, accuracy_validation_scores, auc_validation_scores, auc_cross_validation_scores, auc_permutation_p_values, random_seed=random_seed)

        # Compute similarities and add edges for set 2 to set 1 (if not already computed)
        for i, cluster2 in enumerate(unique_clusters_2):
            for j in range(K):
                cluster1 = unique_clusters_1[indices_2[i][j]]
                if not G.has_edge(f"2_{cluster2}", f"1_{cluster1}"):
                    prior_labels_for_diversification, verified_diversity_promoter_labels, current_label_diversification_content_str = compute_similarity_and_add_edge(cluster1, cluster2, 1, 2, prior_labels_for_diversification, diversify_contrastive_labels, current_label_diversification_content_str, verified_diversity_promoter_labels, saffron, accuracy_cross_validation_scores, accuracy_validation_scores, auc_validation_scores, auc_cross_validation_scores, auc_permutation_p_values, random_seed=random_seed)
        if cross_validate_contrastive_labels:
            print(f"Validation accuracy scores: {accuracy_validation_scores}")
            print(f"Cross-validation accuracy scores: {accuracy_cross_validation_scores}")
            print(f"Validation AUC scores: {auc_validation_scores}")
            print(f"Cross-validation AUC scores: {auc_cross_validation_scores}")
            print("\n------------------------------------------------\n")
            print(f"Mean validation accuracy score: {np.mean(accuracy_validation_scores)}")
            print(f"Mean cross-validation accuracy score: {np.mean(accuracy_cross_validation_scores)}")
            print(f"Mean validation AUC score: {np.mean(auc_validation_scores)}")
            print(f"Mean cross-validation AUC score: {np.mean(auc_cross_validation_scores)}")
            print("\n------------------------------------------------\n")
            print(f"Standard deviation of validation accuracy scores: {np.std(accuracy_validation_scores)}")
            print(f"Standard deviation of cross-validation accuracy scores: {np.std(accuracy_cross_validation_scores)}")
            print(f"Standard deviation of validation AUC scores: {np.std(auc_validation_scores)}")
            print(f"Standard deviation of cross-validation AUC scores: {np.std(auc_cross_validation_scores)}")
            print(f"Max cross-validation accuracy score: {np.max(accuracy_cross_validation_scores)}")
            print(f"Max validation accuracy score: {np.max(accuracy_validation_scores)}")
            print(f"Min cross-validation accuracy score: {np.min(accuracy_cross_validation_scores)}")

    # Add SAFFRON summary to graph attributes
    saffron_summary = saffron.get_summary()
    G.graph['saffron_summary'] = saffron_summary
    G.graph['verified_diversity_promoter_labels'] = verified_diversity_promoter_labels
    
    # Print SAFFRON summary
    print("\n=== SAFFRON Multiple Comparison Correction Summary ===")
    print(f"Total hypothesis tests: {saffron_summary['total_tests']}")
    print(f"Total rejections (significant labels): {saffron_summary['total_rejections']}")
    print(f"Rejection rate: {saffron_summary['rejection_rate']:.5f}")
    print(f"Current wealth: {saffron_summary['current_wealth']:.5f}")
    print(f"Active candidates: {saffron_summary['active_candidates']}")

    results_dict = {
        "accuracy_validation_scores": accuracy_validation_scores,
        "accuracy_cross_validation_scores": accuracy_cross_validation_scores,
        "auc_validation_scores": auc_validation_scores,
        "auc_cross_validation_scores": auc_cross_validation_scores,
        "accuracy_binomial_p_values": accuracy_binomial_p_values,
        "auc_permutation_p_values": auc_permutation_p_values,
        "baseline_accuracy_validation_scores": baseline_accuracy_validation_scores,
        "baseline_auc_validation_scores": baseline_auc_validation_scores,
        "baseline_accuracy_cross_validation_scores": baseline_accuracy_cross_validation_scores,
        "baseline_auc_cross_validation_scores": baseline_auc_cross_validation_scores,
        "hypotheses": hypotheses,
    }

    return G, results_dict