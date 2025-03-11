from sklearn.metrics import roc_auc_score
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
from google.generativeai import GenerativeModel
import random
import sys
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import traceback
import pandas as pd

sys.path.append("..")
sys.path.append("../interventions/auto_finetune_eval")
from auto_finetuning_helpers import make_api_request, extract_json_from_string, collect_dataset_from_api, rephrase_description, parallel_make_api_requests

from typing import List, Tuple, Dict, Optional, Union, Any

import warnings
warnings.filterwarnings('ignore', message='You have modified the pretrained model configuration to control generation.*')

from structlog._config import BoundLoggerLazyProxy


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
        client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
        device: str = "cuda:0", 
        sampled_texts_per_cluster: int = 10, 
        generated_labels_per_cluster: int = 3, 
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_1: Dict = None,
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_2: Dict = None,
        num_decodings_per_prompt: int = None,
        contrastive_cluster_label_instruction: Optional[str] = None,
        api_interactions_save_loc: Optional[str] = None,
        logger: Optional[BoundLoggerLazyProxy] = None
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
        api_provider (str, optional): API provider for text generation. Defaults to None.
        api_model_str (str, optional): Model string for API requests. Defaults to None.
        auth_key (str, optional): Authentication key for API requests. Defaults to None.
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]]): The client to use for the API request.
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
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
    Returns:
        Tuple[List[str], List[int], List[int]]: A tuple containing:
            - List of generated contrastive labels
            - List of indices of selected texts from cluster 1 used to generate the contrastive labels
            - List of indices of selected texts from cluster 2 used to generate the contrastive labels
    """
    if contrastive_cluster_label_instruction is None:
        contrastive_cluster_label_instruction = "You will be given two sets of texts generated by different LLM models. Concisely describe the key themes that differentiate the texts generated by these two models, based on the texts provided."
    if local_model is not None or labeling_tokenizer is not None:
        print("Warning: Local model and labeling tokenizer are deprecated for contrastive label generation.")

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
        str_instruction_to_assistant_model = str_instruction_to_assistant_model + "\n\nKeep the answer short and concise, under 100 words."
        
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
            max_tokens=250,
            api_interactions_save_loc=api_interactions_save_loc,
            logger=logger,
            request_info={
                "pipeline_stage": "contrastive cluster label generation", 
                "cluster_id_1": str(cluster_id_1),
                "cluster_id_2": str(cluster_id_2),
                "label_number": str(i+1)
            }
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
        api_provider: str = None,
        api_model_str: str = None,
        auth_key: str = None,
        client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
        device: str = "cuda:0", 
        sampled_texts_per_cluster: int = 10, 
        generated_labels_per_cluster: int = 3, 
        cluster_ids_to_prompt_ids_to_decoding_ids_dict: Dict = None,
        num_decodings_per_prompt: int = None,
        single_cluster_label_instruction: Optional[str] = None,
        cluster_strs_list: Optional[List[str]] = None,
        api_interactions_save_loc: Optional[str] = None,
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
        api_provider (str, optional): API provider for text generation. Defaults to None.
        api_model_str (str, optional): Model string for API requests. Defaults to None.
        auth_key (str, optional): Authentication key for API requests. Defaults to None.
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]]): The client to use for the API request.
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
            str_instruction_to_assistant_model = str_instruction_to_assistant_model + "\n\nKeep the answer short and concise, under 100 words."
        decoded_labels = [
            make_api_request(
                str_instruction_to_assistant_model, 
                api_provider, 
                api_model_str, 
                auth_key, 
                client=client,
                max_tokens=250,
                api_interactions_save_loc=api_interactions_save_loc,
                logger=logger,
                request_info={
                    "pipeline_stage": "single cluster label generation", 
                    "cluster_id": str(cluster_id)
                }
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
        client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
        device: str = "cuda:0", 
        sampled_texts_per_cluster: int = 10, 
        sampled_comparison_texts_per_cluster: int = 10, 
        generated_labels_per_cluster: int = 3,
        cluster_ids_to_prompt_ids_to_decoding_ids_dict: Dict = None,
        num_decodings_per_prompt: int = None,
        single_cluster_label_instruction: Optional[str] = None,
        api_interactions_save_loc: Optional[str] = None,
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
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]]): The client to use for the API request.
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
        client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
        mode: str = "single_cluster",
        api_interactions_save_loc: Optional[str] = None,
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
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]]): The client to use for the API request.
        mode (str): The mode to use for label evaluation. Defaults to "single_cluster". Set to 
            "double_cluster" or "contrastive" to evaluate a contrastive label between two clusters.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
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
        client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
        device: str = "cuda:0",
        mode: str = "single_cluster",
        n_head_to_head_comparisons_per_text: Optional[int] = None,
        use_unitary_comparisons: bool = False,
        max_unitary_comparisons_per_label: int = 100,
        api_interactions_save_loc: Optional[str] = None,
        logger: Optional[BoundLoggerLazyProxy] = None,
        cluster_id_1: Optional[int] = None,
        cluster_id_2: Optional[int] = None,
        metric: str = "acc"
        ) -> float:
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
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]]): The client to use for the API request.
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
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        cluster_id_1 (Optional[int]): The ID of the first cluster, if using a contrastive label.
        cluster_id_2 (Optional[int]): The ID of the second cluster, if using a contrastive label.
        metric (str): The metric to use for evaluation. Defaults to "acc". Set to "auc" to use AUC.
    Returns:
        float: The accuracy or AUC score representing the label's discrimination power.
    """
    scores = []
    true_labels = []
    if use_unitary_comparisons:
        all_texts = [(text_id, decoded_strs_1[text_id], 1) for text_id in sampled_texts_1] + \
                    [(text_id, decoded_strs_2[text_id], 0) for text_id in sampled_texts_2]
        random.shuffle(all_texts)
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
                    prompt = f"""
                    The following label describes the difference between two clusters of texts: '{label}'

                    Given this description, rate how well the following text matches Model 1 (as opposed to Model 2) on a scale from 0 to 100:

                    Text: {text}

                    Provide your response as a single number between 0 and 100, where 0 means the text definitely belongs to Model 2, and 100 means it definitely belongs to Model 1. Provide only the number, and nothing else.
                    """
                prompts.append(prompt)
            true_labels.append(true_label)
        if api_provider is not None:
            request_info = {
                "pipeline_stage": "unitary comparisons of contrastive label discrimination",
                "cluster_id_1": str(cluster_id_1),
                "cluster_id_2": str(cluster_id_2)
            }
            # Make parallel API requests
            scores_texts = parallel_make_api_requests(
                prompts=prompts,
                api_provider=api_provider,
                api_model_str=api_model_str,
                auth_key=auth_key,
                client=client,
                api_interactions_save_loc=api_interactions_save_loc,
                logger=logger,
                request_info=request_info,
                max_tokens=150
            )
            for score_text in scores_texts:
                try:
                    score = float(score_text.strip()) / 100  # Normalize to [0, 1]
                except ValueError:
                    print(f"Error parsing API response: {score_text}")
                    score = 0.5  # Default to neutral score if parsing fails
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
                        api_model_str, 
                        auth_key, 
                        client=client,
                        mode=mode,
                        api_interactions_save_loc=api_interactions_save_loc,
                        logger=logger,
                        cluster_id_1=cluster_id_1,
                        cluster_id_2=cluster_id_2
                    )

                scores.append(normalized_prob_A)
                true_labels.append(true_label)
            # Permute sampled_texts_2
            sampled_texts_2 = random.sample(sampled_texts_2, len(sampled_texts_2))
    
    if metric == "auc":
        try:
            auc = roc_auc_score(true_labels, scores)
        except ValueError:
            auc = float('nan') 
    elif metric == "acc":
        accuracy = sum([1 for i, score in enumerate(scores) if score > 0.5 and true_labels[i] == 1 or score < 0.5 and true_labels[i] == 0]) / len(scores)
    else:
        raise ValueError(f"Invalid metric: {metric}. Must be 'auc' or 'acc'.")
    return auc if metric == "auc" else accuracy

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
        client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
        device: str = "cuda:0", 
        sampled_comparison_texts_per_cluster: int = 10, 
        n_head_to_head_comparisons_per_text: Optional[int] = None,
        use_unitary_comparisons: bool = False,
        max_unitary_comparisons_per_label: int = 100,
        api_interactions_save_loc: Optional[str] = None,
        logger: Optional[BoundLoggerLazyProxy] = None,
        metric: str = "acc"
        ) -> Tuple[Dict[Tuple[int, int], Dict[str, float]], Dict[Tuple[int, int], Tuple[List[int], List[int]]]]:
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
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]], optional): API client for text generation. Defaults to None.
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
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        metric (str): The metric to use for evaluation. Defaults to "acc". Set to "auc" to use AUC.
    Returns:
        Tuple[
            Dict[Tuple[int, int], Dict[str, float]], 
            Dict[Tuple[int, int], Tuple[List[int], List[int]]]
            ]:
            - A dictionary mapping cluster pairs to a dictionary of contrastive label strings and their associated AUC scores.
            - A dictionary mapping cluster pairs to the indices of texts used for validation.

    Note:
        This function skips cluster pairs that don't have enough available texts for sampling.
        It uses the evaluate_label_discrimination function to compute accuracy or AUC scores for each label.
    """
    cluster_pair_scores = {}
    all_cluster_texts_used_for_validating_label_strs_ids = {}
    for (cluster_id_1, cluster_id_2), cluster_label_candidates in tqdm(cluster_label_strs.items(), desc="Processing cluster pairs"):
        cluster_pair_scores[(cluster_id_1, cluster_id_2)] = {}
        all_cluster_texts_used_for_validating_label_strs_ids[(cluster_id_1, cluster_id_2)] = []

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
                print(f"Warning: Not enough texts for cluster pair {cluster_id_1} ({cluster_1_avail_len}), {cluster_id_2} ({cluster_2_avail_len}). Setting {metric} to -1.0 to indicate invalid result.")
                sampled_texts_1, sampled_texts_2 = [], []
            else:
                sampled_texts_1 = random.sample(cluster_1_indices, sampled_comparison_texts_per_cluster)
                sampled_texts_2 = random.sample(cluster_2_indices, sampled_comparison_texts_per_cluster)

            all_cluster_texts_used_for_validating_label_strs_ids[(cluster_id_1, cluster_id_2)].append((sampled_texts_1, sampled_texts_2))

            if len(sampled_texts_1) == 0 or len(sampled_texts_2) == 0:
                metric_score = -1.0
            else:
                metric_score = evaluate_label_discrimination(
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
                    logger=logger,
                    cluster_id_1=cluster_id_1,
                    cluster_id_2=cluster_id_2,
                    metric=metric
                )
            cluster_pair_scores[(cluster_id_1, cluster_id_2)][label] = metric_score

    return cluster_pair_scores, all_cluster_texts_used_for_validating_label_strs_ids


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
        client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
        device = "cuda:0", 
        sampled_comparison_texts_per_cluster = 10, 
        non_cluster_comparison_texts = 10,
        api_interactions_save_loc: Optional[str] = None,
        logger: Optional[BoundLoggerLazyProxy] = None,
        metric: str = "acc"
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
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]], optional): API client for text
            generation. Defaults to None.
        device (str): Device to use for local model. Defaults to "cuda:0".
        sampled_comparison_texts_per_cluster (int): Number of texts to sample from each cluster for 
            comparison. Defaults to 10.
        non_cluster_comparison_texts (int): Number of texts to sample from outside each cluster for 
            comparison. Defaults to 10.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        metric (str): The metric to use for evaluation. Defaults to "acc". Set to "auc" to use AUC.
    Returns:
        Tuple[Dict[int, Dict[str, float]], Dict[int, List[int]]]: A tuple containing:
            - Dictionary mapping cluster IDs to a dictionary of label strings and their associated accuracy or AUC scores.
            - Dictionary mapping cluster IDs to a list of text indices used for validation.

    Note:
        This function skips clusters that don't have enough texts for sampling.
        It uses the evaluate_label_discrimination function to compute accuracy or AUC scores for each label.
    """
    cluster_label_scores = {}
    all_cluster_texts_used_for_validating_label_strs_ids = {}
    for cluster_id, cluster_label_candidates in tqdm(cluster_label_strs.items(), desc="Processing clusters"):
        cluster_label_scores[cluster_id] = {}

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
            metric_score = evaluate_label_discrimination(
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
                logger=logger,
                cluster_id_1=cluster_id,
                cluster_id_2=None
            )
            cluster_label_scores[cluster_id][label] = metric_score
                
    return cluster_label_scores, all_cluster_texts_used_for_validating_label_strs_ids

# To save costs / time with embeddings, we will save a copy of whatever embeddings we generate and attempt to load past embeddings
# from file if they exist. We will only generate new embeddings if we can't find past embeddings on file. This function thus first
# checks if there is a previously saved embeddings file to load, and if not, generates new embeddings and saves them to file.
def read_past_embeddings_or_generate_new(
        path: str, 
        client: object, 
        decoded_strs: List[str], 
        local_embedding_model_str: str = "thenlper/gte-large", 
        local_embedding_model: AutoModel = None, 
        tokenizer: AutoTokenizer = None, 
        device: str = "cuda:0", 
        recompute_embeddings: bool = False, 
        batch_size: int = 16, 
        save_embeddings: bool = True, 
        tqdm_disable: bool = False, 
        clustering_instructions: str = "Identify the topic or theme of the given texts", 
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
        client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
        device: str = "cuda:0",
        compute_p_values: bool = True,
        num_permutations: int = 3,
        use_normal_distribution_for_p_values: bool = False,
        sampled_comparison_texts_per_cluster: int = 10,
        generated_labels_per_cluster: int = 3,
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_1: Dict = None,
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_2: Dict = None,
        num_decodings_per_prompt: int = None,
        contrastive_cluster_label_instruction: Optional[str] = None,
        pick_top_n_labels: Optional[int] = None,
        n_head_to_head_comparisons_per_text: Optional[int] = None,
        use_unitary_comparisons: bool = False,
        max_unitary_comparisons_per_label: int = 100,
        api_interactions_save_loc: Optional[str] = None,
        logger: Optional[BoundLoggerLazyProxy] = None,
        metric: str = "acc"
        ) ->  Dict[str, Union[
                Dict[Tuple[int, int], Dict[str, float]], # (cluster_1_id, cluster_2_id), label, accuracy/AUC
                Dict[Tuple[int, int], Tuple[List[int], List[int]]], # (cluster_1_id, cluster_2_id), list of text ids used for generating labels, list of text ids used for validating labels
                Optional[Dict[Tuple[int, int], Dict[str, float]]] # (cluster_1_id, cluster_2_id), label, p-value
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
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]], optional): API client for text
            generation. Defaults to None.
        device (str, optional): Device to use for computations (e.g., 'cuda:0'). Defaults to "cuda:0".
        compute_p_values (bool, optional): Whether to compute p-values. Defaults to True.
        num_permutations (int, optional): Number of permutations for p-value computation. Defaults to 3.
        use_normal_distribution_for_p_values (bool, optional): Use normal distribution for p-values, 
            as opposed to the empirical distribution, to avoid having to compute the null distribution. 
            Defaults to False.
        sampled_comparison_texts_per_cluster (int, optional): Number of texts to sample per cluster for 
            comparison. Defaults to 10.
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
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        metric (str, optional): Metric to use for label validation. Defaults to "acc".
    Returns:
        dict: A dictionary containing the following key / value pairs:
            - 'cluster_pair_scores': (Dict[Tuple[int, int], Dict[str, float]]) accuracy / AUC scores for each label 
                in each cluster pair.
            - 'all_cluster_texts_used_for_validating_label_strs_ids': 
                (Dict[Tuple[int, int], Tuple[List[int], List[int]]]) Indices of texts used for validating labels.
            - 'p_values': (Dict[Tuple[int, int], Dict[str, float]]) P-values for each label 
                (if compute_p_values is True).
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
            sampled_texts_per_cluster=sampled_comparison_texts_per_cluster, 
            generated_labels_per_cluster=generated_labels_per_cluster,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_1=cluster_ids_to_prompt_ids_to_decoding_ids_dict_1,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_2=cluster_ids_to_prompt_ids_to_decoding_ids_dict_2,
            num_decodings_per_prompt=num_decodings_per_prompt,
            contrastive_cluster_label_instruction=contrastive_cluster_label_instruction,
            api_interactions_save_loc=api_interactions_save_loc,
            logger=logger
        )
        
        # Store the generated labels and the indices of texts used for label generation
        cluster_label_strs[(cluster_id_1, cluster_id_2)] = decoded_labels
        all_cluster_texts_used_for_label_strs_ids_1[(cluster_id_1, cluster_id_2)] = selected_text_indices_1
        all_cluster_texts_used_for_label_strs_ids_2[(cluster_id_1, cluster_id_2)] = selected_text_indices_2

    # Validate the discrimination power of the generated labels
    cluster_pair_scores, all_cluster_texts_used_for_validating_label_strs_ids = validate_cluster_label_comparative_discrimination_power(
        decoded_strs_1, clustering_assignments_1, 
        all_cluster_texts_used_for_label_strs_ids_1,
        decoded_strs_2, clustering_assignments_2, 
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
        logger=logger,
        metric=metric
    )

    if compute_p_values and use_normal_distribution_for_p_values:
        # Collect all accuracy / AUC scores to compute the standard deviation (before picking top n labels, so we're 
        # using as much data as possible to estimate the null distribution and not biased towards higher-accuracy/
        # AUC labels)
        all_metric_scores = [score for scores in cluster_pair_scores.values() for score in scores.values()]
        
        # Compute the standard deviation of the accuracy / AUC scores
        null_mean = 0.5  # accuracy / AUC of 0.5 represents random chance
        if metric == "acc":
            # Use the standard deviation of the binomial distribution to estimate the null distribution
            null_std = np.sqrt(null_mean * (1 - null_mean) / len(all_metric_scores))
        else:
            # Otherwise, use the standard deviation of the AUC scores
            null_std = np.std(all_metric_scores)
        
        print(f"Using normal distribution for p-values with mean {null_mean} and std {null_std}")

    if pick_top_n_labels is not None:
        # For each cluster, pick the top n labels based on accuracy / AUC score
        top_n_labels = {}
        for cluster_pair, label_scores in cluster_pair_scores.items():
            # Sort labels by their accuracy / AUC scores in descending order
            sorted_labels = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Pick the top n labels
            top_n = sorted_labels[:pick_top_n_labels]
            
            # Store the top n labels and their scores
            top_n_labels[cluster_pair] = {label: score for label, score in top_n}
        # Update cluster_labels with only the top n labels
        cluster_pair_scores = {cluster_pair: top_n_labels[cluster_pair] for cluster_pair in cluster_pair_scores.keys()}

    # Optionally compute p-values if required
    if compute_p_values:
        if not use_normal_distribution_for_p_values:
            # If not using normal distribution for p-values, compute the empirical null distribution for label accuracy / AUCs 
            # by permuting the cluster labels (so the label strings no longer match the cluster ids) and recomputing 
            # the accuracy / AUCs.
            null_distribution_metric_scores = []
            for _ in range(num_permutations):
                # Permute the cluster assignments
                permuted_clustering_assignments_1 = np.random.permutation(clustering_assignments_1)
                permuted_clustering_assignments_2 = np.random.permutation(clustering_assignments_2)

                # Recompute the accuracy / AUCs for the permuted labels
                permuted_scores, _ = validate_cluster_label_comparative_discrimination_power(
                    decoded_strs_1, 
                    permuted_clustering_assignments_1, 
                    all_cluster_texts_used_for_label_strs_ids_1,
                    decoded_strs_2, 
                    permuted_clustering_assignments_2, 
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
                    logger=logger,
                    metric=metric
                )

                # Collect the AUC scores from the permuted data
                for score_dict in permuted_scores.values():
                    for label, score in score_dict.items():
                        null_distribution_metric_scores.append(score)

        # Calculate p-values based on the null distribution
        p_values = {}
        #print("null_distribution_metric_scores", null_distribution_metric_scores)
        for cluster_pair, label_scores in cluster_pair_scores.items():
            p_values[cluster_pair] = {}
            #print("label_scores", label_scores, "label_scores.items()", label_scores.items(), "cluster_pair", cluster_pair)
            for label, metric_score in label_scores.items():
                # Calculate the p-value using the cumulative distribution function (CDF) of the normal distribution
                if use_normal_distribution_for_p_values:
                    p_value = 1 - stats.norm.cdf(metric_score, loc=null_mean, scale=null_std)
                else:
                    p_value = sum(1 for score in null_distribution_metric_scores if score > metric_score) / len(null_distribution_metric_scores)
                p_values[cluster_pair][label] = p_value
    
    metric_str = "accuracy" if metric == "acc" else "AUC"
    # Print the aucs and p-values for each label in each cluster
    for cluster_pair, labels_p_values in p_values.items():
        print(f"Clusters {cluster_pair} {metric_str} scores and p-values:")
        for label, p_value in labels_p_values.items():
            metric_score = cluster_pair_scores[cluster_pair][label]
            outstr_label = label.replace("\n", "\\n")
            print(f"{metric_str}: {metric_score:.4f}, P-value: {p_value:.4f}, Label: {outstr_label}")

    return_dict = {
        "cluster_pair_scores": cluster_pair_scores,
        "all_cluster_texts_used_for_validating_label_strs_ids": all_cluster_texts_used_for_validating_label_strs_ids
    }
    if compute_p_values:
        return_dict["p_values"] = p_values
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
    auth_key: str = None,
    client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
    device: str = "cuda:0",
    sampled_comparison_texts_per_cluster: int = 10,
    generated_labels_per_cluster: int = 3,
    cluster_ids_to_prompt_ids_to_decoding_ids_dict_1: Dict = None,
    cluster_ids_to_prompt_ids_to_decoding_ids_dict_2: Dict = None,
    num_decodings_per_prompt: int = None,
    contrastive_cluster_label_instruction: Optional[str] = None,
    n_head_to_head_comparisons_per_text: Optional[int] = None,
    use_unitary_comparisons: bool = False,
    max_unitary_comparisons_per_label: int = 20,
    api_interactions_save_loc: Optional[str] = None,
    logger: Optional[BoundLoggerLazyProxy] = None,
    metric: str = "acc"
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
        auth_key (str): Authentication key for API requests.
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]], optional): API client for text
            generation. Defaults to None.
        device (str, optional): Device to use for computations (e.g., 'cuda:0'). Defaults to "cuda:0".
        sampled_comparison_texts_per_cluster (int): Number of texts to sample per cluster for comparison.
        generated_labels_per_cluster (int): Number of labels to generate per cluster pair.
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_1 (Dict, optional): Nested dict. First dict is indexed by cluster id.
            Leads to a dict indexed by prompt id, which leads to a list of indices for where the decodings of that
            prompt can be found in decoded_strs. Can be provided to make the label generation select only num_decodings_per_prompt to base the labels off of.
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_2 (Dict, optional): Same as 
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_1, but for finetuned model.
        num_decodings_per_prompt (int, optional): Number of decodings per prompt to use for label generation.
        contrastive_cluster_label_instruction (Optional[str]): Instruction for label generation.
        n_head_to_head_comparisons_per_text (Optional[int]): Number of head-to-head comparisons per text.
        use_unitary_comparisons (bool): Whether to use unitary comparisons.
        max_unitary_comparisons_per_label (int): Maximum number of unitary comparisons to perform per label.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        metric (str): Metric to use for label validation. Defaults to "acc".
    Returns:
        nx.Graph: A graph where nodes are clusters from both sets and edge attributes include
            similarity scores, labels, and per-label accuracy / AUC scores.
    """
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
    
    # Function to compute similarity and add edge
    def compute_similarity_and_add_edge(cluster1, cluster2, set1, set2):
        print(f"Computing similarity and adding edge for clusters {cluster1} and {cluster2} from set {set1} and {set2}")
        cluster_matches = [(cluster1, cluster2)]
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
            api_stronger_model_str,
            auth_key, 
            client=client,
            device=device,
            compute_p_values=True,
            use_normal_distribution_for_p_values=True,
            sampled_comparison_texts_per_cluster=sampled_comparison_texts_per_cluster,
            generated_labels_per_cluster=generated_labels_per_cluster,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_1=cluster_ids_to_prompt_ids_to_decoding_ids_dict_1 if set1 == 1 else cluster_ids_to_prompt_ids_to_decoding_ids_dict_2,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_2=cluster_ids_to_prompt_ids_to_decoding_ids_dict_2 if set1 == 1 else cluster_ids_to_prompt_ids_to_decoding_ids_dict_1,
            num_decodings_per_prompt=num_decodings_per_prompt,
            contrastive_cluster_label_instruction=contrastive_cluster_label_instruction,
            n_head_to_head_comparisons_per_text=n_head_to_head_comparisons_per_text,
            use_unitary_comparisons=use_unitary_comparisons,
            max_unitary_comparisons_per_label=max_unitary_comparisons_per_label,
            api_interactions_save_loc=api_interactions_save_loc,
            logger=logger,
            metric=metric
        )
        
        labels_and_metric_scores = result['cluster_pair_scores'][(cluster1, cluster2)]
        similarity_score = np.mean([1 - metric_score for metric_score in labels_and_metric_scores.values()])
        
        G.add_edge(f"{set1}_{cluster1}", f"{set2}_{cluster2}", 
                   weight=similarity_score,
                   labels=list(labels_and_metric_scores.keys()),
                   label_metric_scores=labels_and_metric_scores)

    if match_by_ids:
        # Match cluster i in set 1 to cluster i in set 2
        for i in range(len(unique_clusters_1)):
            compute_similarity_and_add_edge(unique_clusters_1[i], unique_clusters_2[i], 1, 2)
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
                    compute_similarity_and_add_edge(cluster1, cluster2, 1, 2)

        # Compute similarities and add edges for set 2 to set 1 (if not already computed)
        for i, cluster2 in enumerate(unique_clusters_2):
            for j in range(K):
                cluster1 = unique_clusters_1[indices_2[i][j]]
                if not G.has_edge(f"2_{cluster2}", f"1_{cluster1}"):
                    compute_similarity_and_add_edge(cluster1, cluster2, 1, 2)

    return G


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
            # ... etc. for each raw metric if you want
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
        # Get this node’s metric
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
    all_validated_metric_scores: List[List[float]],
    model_label_for_node_1="1_",  # or just "1_"
    model_label_for_node_2="2_"
):
    """
    Build a mapping from each hypothesis to the node metrics of its cluster(s).
    Then correlate the cluster metrics with the average validated score.
    """
    # Suppose all_validated_metric_scores has shape (num_rephrases + 1, num_hypotheses)
    all_validated_metric_scores = np.array(all_validated_metric_scores)
    num_hypotheses = all_validated_metric_scores.shape[1]
    print("all_validated_metric_scores.shape:", all_validated_metric_scores.shape)
    
    rows = []
    for i in range(len(hypothesis_origin_clusters)):
        # mean validation score across rephrasings
        mean_val_score = np.mean(all_validated_metric_scores[:, i])
        
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
        client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
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
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]], optional): API client for text
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
    # Compile the list of prompts that encourage the assistant to generate texts attributed to the cluster 1 model.
    prompts_1 = []
    for description in difference_descriptions:
        if api_provider is None:
            prompt = f"Given the following description of how the texts generated by Model 1 differ from those generated by Model 2: {description}, generate a new text that is closer to Model 1."
        else:
            prompt = f'Given the following description of how the texts generated by Model 1 differ from those generated by Model 2: {description}, generate 5 short texts that are closer to Model 1. Format your response as a JSON array of strings, where each string is a new text. Example response format: ["Text 1", "Text 2", ..., "Text 3"]. Aim for about 100 words per text.'
        prompts_1.append(prompt)
    # Now, compile the list of prompts that encourage the assistant to generate texts attributed to the cluster 2 model.
    prompts_2 = []
    for description in difference_descriptions:
        if api_provider is None:
            prompt = f"Given the following description of how the texts generated by Model 1 differ from those generated by Model 2: {description}, generate a new text that is more likely to be generated by Model 2."
        else:
            prompt = f'Given the following description of how the texts generated by Model 1 differ from those generated by Model 2: {description}, generate 5 short texts that are more likely to be generated by Model 2. Format your response as a JSON array of strings, where each string is a new text. Example response format: ["Text 1", "Text 2", ..., "Text 3"]. Aim for about 100 words per text.'
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
                    api_model_str, 
                    auth_key,
                    client=client,
                    num_datapoints=num_generated_texts_per_description,
                    max_tokens=2048,
                    api_interactions_save_loc=api_interactions_save_loc,
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
                    api_model_str, 
                    auth_key,
                    client=client,
                    num_datapoints=num_generated_texts_per_description,
                    max_tokens=2048,
                    api_interactions_save_loc=api_interactions_save_loc,
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

# Uses assistant_generative_compare to generate the correlation coefficients / AUCs representing how well LM scores function to differentiate between the texts generated for one model and the other using the different descriptions. Optionally computes p-values for the descriptions scores.
def validated_assistant_generative_compare(
        difference_descriptions: List[str], 
        local_model: AutoModel, 
        labeling_tokenizer: AutoTokenizer, 
        api_provider: str,
        api_model_str: str,
        auth_key: str,
        api_stronger_model_str: str = None,
        client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
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
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]], optional): API client for text
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
    client: Optional[Union["Anthropic", "OpenAI", "GenerativeModel"]] = None,
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
    # If you want to minimize memory usage, you could load/unload inside each round, but that’s slower.
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

        # Request the assistant’s next prompt
        plan_query_responses = parallel_make_api_requests(
            prompts=plan_query_prompts,
            api_provider=api_provider,
            api_model_str=api_model_str,
            auth_key=auth_key,
            client=client,
            api_interactions_save_loc=api_interactions_save_loc,
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

        # Parse out the actual query from the assistant’s response
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
        # (3) Randomly pick one model’s response as Model X
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
