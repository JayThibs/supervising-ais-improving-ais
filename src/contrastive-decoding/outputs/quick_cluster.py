import pandas as pd
import argparse
from openai import OpenAI
from sklearn.cluster import SpectralClustering, HDBSCAN, KMeans
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import pickle
import scipy.stats as stats
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, BitsAndBytesConfig
from copy import deepcopy
import random
import sys
sys.path.append("..")
from model_comparison_helpers import string_with_token_colors
from typing import List, Tuple, Dict
from analysis_helpers import literal_eval_fallback

import warnings
warnings.filterwarnings('ignore', message='You have modified the pretrained model configuration to control generation.*')


def contrastive_label_double_cluster(decoded_strs_1: List[str], 
                                     clustering_assignments_1: List[int], 
                                     cluster_id_1: int, 
                                     decoded_strs_2: List[str], 
                                     clustering_assignments_2: List[int], 
                                     cluster_id_2: int, 
                                     local_model: AutoModel = None, 
                                     labeling_tokenizer: AutoTokenizer = None, 
                                     device: str = "cuda:0", 
                                     sampled_texts_per_cluster: int = 10, 
                                     generated_labels_per_cluster: int = 3, 
                                     cluster_label_instruction: str = "You are an expert at describing the differences between clusters of texts. When given a list of texts belonging to two clusters, you immediately respond with a short description of the key themes that separate the two clusters.",
                                     cluster_strs_list_1: List[str] = None,
                                     cluster_strs_list_2: List[str] = None
                                    ):
    if cluster_strs_list_1 is not None and cluster_strs_list_2 is not None:
        within_cluster_indices_1 = list(range(len(cluster_strs_list_1)))
        within_cluster_indices_2 = list(range(len(cluster_strs_list_2)))
        selected_text_indices_1 = np.random.choice(within_cluster_indices_1, sampled_texts_per_cluster, replace=False)
        selected_text_indices_2 = np.random.choice(within_cluster_indices_2, sampled_texts_per_cluster, replace=False)
        selected_texts_1 = [cluster_strs_list_1[i] for i in selected_text_indices_1]
        selected_texts_2 = [cluster_strs_list_2[i] for i in selected_text_indices_2]
    else:
        #print(f"clustering_assignments_1: {clustering_assignments_1}, cluster_id_1: {cluster_id_1}")

        cluster_indices_1 = [i for i, x in enumerate(clustering_assignments_1) if x == cluster_id_1]
        #print(f"cluster_indices_1: {cluster_indices_1}")
        cluster_indices_2 = [i for i, x in enumerate(clustering_assignments_2) if x == cluster_id_2]

        selected_text_indices_1 = np.random.choice(cluster_indices_1, sampled_texts_per_cluster, replace=False)
        selected_text_indices_2 = np.random.choice(cluster_indices_2, sampled_texts_per_cluster, replace=False)

        selected_texts_1 = [decoded_strs_1[i] for i in selected_text_indices_1]
        selected_texts_2 = [decoded_strs_2[i] for i in selected_text_indices_2]
    for i, text in enumerate(selected_texts_1):
        selected_texts_1[i] = text.replace("<s>", "").replace('\n', '\\n')
        selected_texts_1[i] = f"Cluster 1 Text {i}: " + selected_texts_1[i]
    for i, text in enumerate(selected_texts_2):
        selected_texts_2[i] = text.replace("<s>", "").replace('\n', '\\n')
        selected_texts_2[i] = f"Cluster 2 Text {i}: " + selected_texts_2[i]
    # Use the local model to generate a label that describes the difference between the two clusters
    # Generate input string for local model
    str_instruction_to_local_model = cluster_label_instruction + "\n" + "Cluster 1 selected texts:\n" + '\n'.join(selected_texts_1) + "\nCluster 2 selected texts:\n" + '\n'.join(selected_texts_2)
    str_instruction_to_local_model = str_instruction_to_local_model + "\n" + "Concisely describe the key themes that differentiate these two clusters.\nAnswer:"
    # Prepare inputs for text generation
    inputs = labeling_tokenizer(str_instruction_to_local_model, return_tensors="pt").to(device)
    inputs_length = inputs.input_ids.shape[1]
    # Generate labels using the Hugging Face text generation API
    with torch.no_grad():
        outputs = [local_model.generate(**inputs, max_new_tokens=100, num_return_sequences=generated_labels_per_cluster, do_sample=True, pad_token_id=labeling_tokenizer.eos_token_id) for _ in range(generated_labels_per_cluster)]
    # Decode labels to strings
    decoded_labels = [labeling_tokenizer.decode(output[0][inputs_length:], skip_special_tokens=True) for output in outputs]
    for i, label in enumerate(decoded_labels):
        if label.startswith(" "):
            decoded_labels[i] = label[1:]
            if "<s>" in label:
                decoded_labels[i] = label[:label.index("<s>")]
    return decoded_labels, selected_text_indices_1, selected_text_indices_2

def label_single_cluster(decoded_strs: List[str], 
                         clustering_assignments: List[int], 
                         cluster_id: int, 
                         local_model: AutoModel = None, 
                         labeling_tokenizer: AutoTokenizer = None, 
                         device: str = "cuda:0", 
                         sampled_texts_per_cluster: int = 10, 
                         generated_labels_per_cluster: int = 3, 
                         cluster_label_instruction: str = "You are an expert at describing clusters of texts. When given a list of texts belonging to a cluster, you immediately respond with a short description of the key themes of the texts shown to you.",
                         cluster_strs_list: List[str] = None
                         ):
    if cluster_strs_list is not None:
        within_cluster_indices = list(range(len(selected_texts)))
        selected_text_indices = np.random.choice(within_cluster_indices, sampled_texts_per_cluster, replace=False)
        selected_texts = [cluster_strs_list[i] for i in selected_text_indices]
    else:
        cluster_indices = [i for i, x in enumerate(clustering_assignments) if x == cluster_id]
        selected_text_indices = np.random.choice(cluster_indices, sampled_texts_per_cluster, replace=False)
        selected_texts = [decoded_strs[i] for i in selected_text_indices]
    for i, text in enumerate(selected_texts):
        selected_texts[i] = text.replace("<s>", "").replace('\n', '\\n')
        selected_texts[i] = f"Text {i}: " + selected_texts[i]

    # Use the local model to generate a cluster label for the selected texts
    # Generate input string for local model
    str_instruction_to_local_model = cluster_label_instruction + "\n" + "Texts in current cluster:\n" + '\n'.join(selected_texts)
    str_instruction_to_local_model = str_instruction_to_local_model + "\n" + "Cluster description: the common theme of the above texts is that they are all about"
    # Prepare inputs for text generation
    inputs = labeling_tokenizer(str_instruction_to_local_model, return_tensors="pt").to(device)
    inputs_length = inputs.input_ids.shape[1]
    # Generate labels using the Hugging Face text generation API
    with torch.no_grad():
        outputs = [local_model.generate(**inputs, max_new_tokens=15, num_return_sequences=1, do_sample=True, pad_token_id=labeling_tokenizer.eos_token_id) for _ in range(generated_labels_per_cluster)]
    # Decode labels to strings
    #decoded_interactions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    #print(decoded_interactions)
    decoded_labels = [labeling_tokenizer.decode(output[0][inputs_length:], skip_special_tokens=True) for output in outputs]
    for i, label in enumerate(decoded_labels):
        if label.startswith(" "):
            decoded_labels[i] = label[1:]
            if "<s>" in label:
                decoded_labels[i] = label[:label.index("<s>")]
    return decoded_labels, selected_text_indices

def get_cluster_labels_random_subsets(decoded_strs, clustering_assignments, local_model = None, labeling_tokenizer = None, device = "cuda:0", sampled_texts_per_cluster = 10, sampled_comparison_texts_per_cluster = 10, generated_labels_per_cluster = 3):
    cluster_labels = {}
    all_cluster_texts_used_for_label_strs_ids = {}
    for cluster in set(clustering_assignments):
        # For each cluster, select sampled_texts_per_cluster random texts (skipping the cluster if it has less than 2x sampled_texts_per_cluster texts)
        cluster_indices = [i for i, x in enumerate(clustering_assignments) if x == cluster]
        if len(cluster_indices) < sampled_comparison_texts_per_cluster + sampled_texts_per_cluster:
            continue
        decoded_labels, selected_text_indices = label_single_cluster(decoded_strs, clustering_assignments, cluster, local_model, labeling_tokenizer, device, sampled_texts_per_cluster, generated_labels_per_cluster)
        all_cluster_texts_used_for_label_strs_ids[cluster] = selected_text_indices
        cluster_labels[cluster] = decoded_labels
        
        print(f"Cluster {cluster} labels: {cluster_labels[cluster]}")

    return cluster_labels, all_cluster_texts_used_for_label_strs_ids

# For each possible contrastive label of each cluster pair, we select sampled_comparison_texts_per_cluster random texts from
# each cluster (which must not be in the associated cluster's all_cluster_texts_used_for_label_strs_ids, skipping the cluster 
# pair and printing a warning if there aren't enough texts in both clusters).
# We then perform pairwise comparisons between each group of sampled_comparison_texts_per_cluster texts in the cluster pair, 
# asking the local_model LLM to identify which of the two texts comes from which cluster.
# We score potential contrastive labels based on their ability to let the assistant LLM distinguish between texts from the two
# clusters, quantified with AUC. We then return the scores for each cluster pair.
def validate_cluster_label_comparative_discrimination_power(
        decoded_strs_1: List[str], 
        clustering_assignments_1: List[int], 
        all_cluster_texts_used_for_label_strs_ids_1: Dict[int, List[int]], 
        decoded_strs_2: List[str], 
        clustering_assignments_2: List[int], 
        all_cluster_texts_used_for_label_strs_ids_2: Dict[int, List[int]], 
        cluster_label_strs: Dict[Tuple[int, int], List[str]], # dict of lists of contrastive labels for each cluster pair, so only one set of labels per cluster pair
        local_model: AutoModel = None, 
        labeling_tokenizer: AutoTokenizer = None, 
        device: str = "cuda:0", 
        sampled_comparison_texts_per_cluster: int = 10, 
        ):
    cluster_pair_scores = {}
    all_cluster_texts_used_for_validating_label_strs_ids = {}
    for (cluster_id_1, cluster_id_2), cluster_label_candidates in tqdm(cluster_label_strs.items(), desc="Processing cluster pairs"):
        cluster_pair_scores[(cluster_id_1, cluster_id_2)] = {}

        # Sample texts from each cluster, excluding those previously used to generate contrastive labels for this cluster pair
        # Exclude texts used in generating labels for this cluster pair
        cluster_1_indices = [i for i, x in enumerate(clustering_assignments_1) if x == cluster_id_1]
        cluster_2_indices = [i for i, x in enumerate(clustering_assignments_2) if x == cluster_id_2]
        previously_used_texts_1 = all_cluster_texts_used_for_label_strs_ids_1.get(cluster_id_1, [])
        previously_used_texts_2 = all_cluster_texts_used_for_label_strs_ids_2.get(cluster_id_2, [])
        
        # Available texts are those not previously used
        available_texts_1 = [i for i in cluster_1_indices if i not in previously_used_texts_1]
        available_texts_2 = [i for i in cluster_2_indices if i not in previously_used_texts_2]

        # Ensure there are enough texts remaining in both clusters
        cluster_1_avail_len = len(available_texts_1)
        cluster_2_avail_len = len(available_texts_2)
        if cluster_1_avail_len < sampled_comparison_texts_per_cluster or cluster_2_avail_len < sampled_comparison_texts_per_cluster:
            print(f"Warning: Not enough texts for cluster pair {cluster_id_1} ({cluster_1_avail_len}), {cluster_id_2} ({cluster_2_avail_len}). Skipping.")
            continue
        sampled_texts_1 = random.sample(available_texts_1, sampled_comparison_texts_per_cluster)
        sampled_texts_2 = random.sample(available_texts_2, sampled_comparison_texts_per_cluster)

        all_cluster_texts_used_for_validating_label_strs_ids[(cluster_id_1, cluster_id_2)] = (sampled_texts_1, sampled_texts_2)

        for label in cluster_label_candidates:
            # Initialize scores for this label
            scores = []
            true_labels = []

            for text_id_1 in sampled_texts_1:
                for text_id_2 in sampled_texts_2:
                    # Prepare input for local model
                    text_1 = decoded_strs_1[text_id_1]
                    text_2 = decoded_strs_2[text_id_2]
                    # Randomize the order of presentation to the model
                    if np.random.rand() > 0.5:
                        text_A = text_1
                        text_B = text_2
                        true_label = 1
                    else:
                        text_A = text_2
                        text_B = text_1
                        true_label = 0

                    input_str = f"Which text fits the label '{label}' better? Text A: {text_A} or Text B: {text_B}\nAnswer: Text"
                    input_ids = labeling_tokenizer(input_str, return_tensors="pt").to(device)
                    with torch.no_grad():
                        logits = local_model(**input_ids).logits
                        prob_A = torch.nn.functional.softmax(logits, dim=-1)[0, -1, labeling_tokenizer.encode("A", add_special_tokens=False)[0]].item()
                        prob_B = torch.nn.functional.softmax(logits, dim=-1)[0, -1, labeling_tokenizer.encode("B", add_special_tokens=False)[0]].item()
                        normalized_prob_A = prob_A / max(prob_A + prob_B, 1e-10)

                    scores.append(normalized_prob_A)
                    true_labels.append(true_label)

            # Calculate AUC for this label
            try:
                auc = roc_auc_score(true_labels, scores)
            except ValueError:
                auc = float('nan')  # Not defined when only one class is present
            cluster_pair_scores[(cluster_id_1, cluster_id_2)][label] = auc

    return cluster_pair_scores, all_cluster_texts_used_for_validating_label_strs_ids

# For each possible label of each cluster, we select sampled_comparison_texts_per_cluster random texts (which must not
# be in all_cluster_texts_used_for_label_strs_ids, skipping the cluster and printing a warning if there aren't enough texts in the cluster).
# We then perform pairwise comparisons between each of the sampled_comparison_texts_per_cluster texts in the cluster and 
# non_cluster_comparison_texts random texts, asking the local_model LLM which of the two texts better fits the given label.
# We score potential cluster labels based on their ability to let the assistant LLM distinguish between texts from the cluster
# and texts not from the cluster, quantified with AUC. We then return the scores for each cluster.
def validate_cluster_label_discrimination_power(
        decoded_strs, 
        clustering_assignments, 
        cluster_label_strs, 
        all_cluster_texts_used_for_label_strs_ids, 
        local_model = None, 
        labeling_tokenizer = None, 
        device = "cuda:0", 
        sampled_comparison_texts_per_cluster = 10, 
        non_cluster_comparison_texts = 10
        ):
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
            scores = []
            true_labels = []
            
            for cluster_text_id in sampled_cluster_texts:
                for non_cluster_text_id in sampled_non_cluster_texts:
                    # Prepare input for local model
                    cluster_text = decoded_strs[cluster_text_id]
                    non_cluster_text = decoded_strs[non_cluster_text_id]
                    
                    # Randomize the order of presentation
                    if np.random.rand() > 0.5:
                        text_A = cluster_text
                        text_B = non_cluster_text
                        prob_A_target = 1  # Correct choice is A if cluster text is A
                    else:
                        text_A = non_cluster_text
                        text_B = cluster_text
                        prob_A_target = 0  # Correct choice is B if cluster text is B
                    
                    input_str = f"Which text fits the label '{label}' better? Text A: {text_A} or Text B: {text_B}\nAnswer: Text"
                    input_ids = labeling_tokenizer(input_str, return_tensors="pt").to(device)
                    #print("input_str:", input_str)
                    
                    # Query the local model
                    with torch.no_grad():
                        output = local_model(**input_ids)
                        logits = output.logits
                        # Compare the probabilities for "A" and "B" as continuations
                        prob_A = torch.nn.functional.softmax(logits, dim=-1)[0, -1, labeling_tokenizer.encode("A", add_special_tokens=False)[0]].item()
                        prob_B = torch.nn.functional.softmax(logits, dim=-1)[0, -1, labeling_tokenizer.encode("B", add_special_tokens=False)[0]].item()
                        normalized_prob_A = prob_A / max(prob_A + prob_B, 1e-10)
            
                    scores.append(normalized_prob_A)
                    true_labels.append(prob_A_target)
            # Compute AUC for the current label
            try:
                auc = roc_auc_score(true_labels, scores)
            except ValueError:
                auc = float('nan')  # Not defined when only one class is present
            cluster_label_scores[cluster_id][label] = auc
                
    return cluster_label_scores, all_cluster_texts_used_for_validating_label_strs_ids

# Given a list with the cluster label of each text and embeddings for each text, this function creates a list for every cluster,
# which contains the id of each text in that cluster, sorted by distance from the center of the cluster, in increasing order.
# The function returns a list of such lists, one for every cluster in the original list of cluster labels.
def sort_by_distance(clustering_assignments, embedding_list):
    # Initialize an empty list to store the sorted clusters
    sorted_clusters = []

    # Iterate over each unique label in the clustering_assignments list
    for label in set(clustering_assignments):
        # Get the indices of the texts in this cluster
        cluster_indices = [i for i, x in enumerate(clustering_assignments) if x == label]

        # Get the embeddings for this cluster
        cluster_embeddings = [embedding_list[i] for i in cluster_indices]

        # Calculate the center of this cluster
        cluster_center = np.mean(cluster_embeddings, axis=0)

        # Calculate the distances from the center for each text in this cluster
        distances = [np.linalg.norm(embedding - cluster_center) for embedding in cluster_embeddings]

        # Sort the indices of the texts in this cluster by their distances from the center
        sorted_indices = [i for _, i in sorted(zip(distances, cluster_indices))]

        # Add the sorted indices to the list of sorted clusters
        sorted_clusters.append(sorted_indices)

    # Return the list of sorted clusters
    return sorted_clusters

# To save costs / time with embeddings, we will save a copy of whatever embeddings we generate and attempt to load past embeddings
# from file if they exist. We will only generate new embeddings if we can't find past embeddings on file. This function thus first
# checks if there is a previously saved embeddings file to load, and if not, generates new embeddings and saves them to file.
def read_past_embeddings_or_generate_new(path, client, decoded_strs, local_embedding_model_str = "thenlper/gte-large", local_embedding_model = None, tokenizer = None, device = "cuda:0", recompute_embeddings = False, batch_size = 8, save_embeddings = True, tqdm_disable = False, clustering_instructions = "Identify the topic or theme of the given texts", max_length = 512, bnb_config = None):
    # First, try to load past embeddings from file:
    if not recompute_embeddings:
        try:
            with open(path + "_embeddings.pkl", "rb") as f:
                embeddings_list = pickle.load(f)
                # Check that embeddings_list has the expected dimensions
                if len(embeddings_list) != len(decoded_strs):
                    raise ValueError("The loaded embeddings have the wrong dimensions.")
                return embeddings_list
        except:
            print("Could not load past embeddings. Generating new embeddings.")
    
    # Otherwise, compute new embeddings
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

            if local_embedding_model_str == "nvidia/NV-Embed-v1":
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
                if local_embedding_model_str == "nvidia/NV-Embed-v1":
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
    if save_embeddings:
        with open(path + "_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings_list, f)
        
    return embeddings_list

def extract_df_info(df, original_tokenizer_str, path, skip_every_n_decodings = 0, color_by_divergence = False, local_embedding_model_str = "thenlper/gte-large", device = "cuda:0", recompute_embeddings = True, save_embeddings = True, tqdm_disable = False, clustering_instructions = None, bnb_config = None):
    divergence_values = df['divergence'].values
    loaded_strs = df['decoding'].values
    
    if color_by_divergence:
        all_token_divergences = df['all_token_divergences'].values
        divs_0 = literal_eval_fallback(all_token_divergences[0], None)
        if divs_0 is None:
            raise ValueError("The first set of token divergence values is not readable.")
        all_token_divergences = [literal_eval_fallback(s, len(divs_0) * [0]) for s in all_token_divergences]
        original_tokenizer = AutoTokenizer.from_pretrained(original_tokenizer_str)

        max_token_divergence = max([max(token_divergences) for token_divergences in all_token_divergences])
        min_token_divergence = min([min(token_divergences) for token_divergences in all_token_divergences])
    else:
        all_token_divergences = None
        original_tokenizer = None
        max_token_divergence = None
        min_token_divergence = None


    if skip_every_n_decodings > 0:
        loaded_strs = loaded_strs[::skip_every_n_decodings]
        divergence_values = divergence_values[::skip_every_n_decodings]
        if color_by_divergence:
            all_token_divergences = all_token_divergences[::skip_every_n_decodings]

    #decoded_strs = [s.split("|")[1] for s in loaded_strs]
    decoded_strs = [s.replace("|", "").replace("<begin_of_text>", "") for s in loaded_strs]

    for i in range(len(decoded_strs)):
        if decoded_strs[i][0] == ' ':
            decoded_strs[i] = decoded_strs[i][1:]
    #print(decoded_strs)

    # Generate embeddings for the past results.
    # embeddings_list is a n_datapoints x embedding_dim list of floats
    embeddings_list = read_past_embeddings_or_generate_new(path, 
                                                           client, 
                                                           decoded_strs, 
                                                           local_embedding_model_str=local_embedding_model_str, 
                                                           device=device,
                                                           recompute_embeddings=recompute_embeddings,
                                                           save_embeddings=save_embeddings,
                                                           clustering_instructions=clustering_instructions,
                                                           tqdm_disable=tqdm_disable,
                                                           bnb_config=bnb_config
                                                        )
    return divergence_values, decoded_strs, embeddings_list, all_token_divergences, original_tokenizer, max_token_divergence, min_token_divergence


# First, assign possible labels to each cluster using get_cluster_labels_random_subsets. Then, use
# validate_cluster_label_discrimination_power to get the AUC for each cluster label. Then, for each
# cluster, print the randomly selected texts used to generate the cluster labels, as well as every
# cluster label and its AUC.
def get_validated_cluster_labels(decoded_strs: List[str], 
                                 clustering_assignments: List[int], 
                                 local_model: AutoModel, 
                                 labeling_tokenizer: AutoTokenizer, 
                                 device: str,
                                 compute_p_values: bool = True,
                                 num_permutations: int = 3,
                                 use_normal_distribution_for_p_values: bool = False,
                                 sampled_texts_per_cluster: int = 10,
                                 sampled_comparison_texts_per_cluster: int = 10,
                                 non_cluster_comparison_texts: int = 10,
                                 generated_labels_per_cluster: int = 3
                                 ):
    cluster_labels, all_cluster_texts_used_for_label_strs_ids = get_cluster_labels_random_subsets(
            decoded_strs, 
            clustering_assignments, 
            local_model, 
            labeling_tokenizer, 
            device=device, 
            sampled_texts_per_cluster=sampled_texts_per_cluster,
            sampled_comparison_texts_per_cluster=sampled_comparison_texts_per_cluster, 
            generated_labels_per_cluster=generated_labels_per_cluster
        )
    cluster_label_scores, all_cluster_texts_used_for_validating_label_strs_ids = validate_cluster_label_discrimination_power(
        decoded_strs, 
        clustering_assignments, 
        cluster_labels, 
        all_cluster_texts_used_for_label_strs_ids, 
        local_model=local_model, 
        labeling_tokenizer=labeling_tokenizer, 
        device=device,
        sampled_comparison_texts_per_cluster=sampled_comparison_texts_per_cluster,
        non_cluster_comparison_texts=non_cluster_comparison_texts
        )

    for cluster_id, label_info in cluster_labels.items():
        print(f"Cluster {cluster_id} labels and their AUC scores:")
        for label in label_info:
            auc_score = cluster_label_scores.get(cluster_id, {}).get(label, "N/A")
            if auc_score == "N/A":
                print(f"AUC: {auc_score} Label: {label}")
            else:
                print(f"AUC: {auc_score:.3f} Label: {label}")
        print("#######################################################################")
        print("\nSelected texts for generating labels:")
        print("#######")
        selected_text_indices = all_cluster_texts_used_for_label_strs_ids.get(cluster_id, [])
        for idx in selected_text_indices:
            outstr = decoded_strs[idx].replace("\n", "\\n")
            print(f"- {outstr}")
        print("#######")
        print("\nAdditional texts for validating labels:")
        print("#######")
        additional_text_indices = all_cluster_texts_used_for_validating_label_strs_ids.get(cluster_id, [])
        for idx in additional_text_indices:
            outstr = decoded_strs[idx].replace("\n", "\\n")
            print(f"- {outstr}")
        print("#######")
        print("\n\n")
    
    if compute_p_values:
        # Now, compute the null distribution for label AUCs by permuting the cluster labels (so the label strings
        # no longer match the cluster ids) and recomputing the AUCs.
        null_distribution_auc_scores = []
        for _ in range(num_permutations):
            permuted_cluster_labels = {}
            permuted_cluster_ids = random.sample(list(cluster_labels.keys()), len(cluster_labels))
            for i, cluster_id in enumerate(permuted_cluster_ids):
                ith_cluster_label_key = list(cluster_labels.keys())[i]
                permuted_cluster_labels[ith_cluster_label_key] = cluster_labels[cluster_id]
            null_cluster_label_scores_nested_dict, _ = validate_cluster_label_discrimination_power(
                decoded_strs, 
                clustering_assignments, 
                permuted_cluster_labels, 
                all_cluster_texts_used_for_label_strs_ids, 
                local_model=local_model, 
                labeling_tokenizer=labeling_tokenizer, 
                device=device,
                sampled_comparison_texts_per_cluster=sampled_comparison_texts_per_cluster,
                non_cluster_comparison_texts=non_cluster_comparison_texts
                )
            null_cluster_label_scores_list = []
            for cluster_id, scores_dict in null_cluster_label_scores_nested_dict.items():
                for label, score in scores_dict.items():
                    null_cluster_label_scores_list.append(score)
            null_distribution_auc_scores.extend(null_cluster_label_scores_list)

        # Calculate p-values for each cluster label's AUC score against the null distribution
        p_values = {}
        # Calculate mean and standard deviation of the null distribution
        null_mean = np.mean(null_distribution_auc_scores)
        null_std = np.std(null_distribution_auc_scores)

        for cluster_id, label_scores in cluster_label_scores.items():
            p_values[cluster_id] = {}
            for label, auc_score in label_scores.items():
                # Calculate the p-value using the cumulative distribution function (CDF) of the normal distribution
                if use_normal_distribution_for_p_values:
                    p_value = 1 - stats.norm.cdf(auc_score, loc=null_mean, scale=null_std)
                else:
                    p_value = sum(1 for score in null_distribution_auc_scores if score > auc_score) / len(null_distribution_auc_scores)
                p_values[cluster_id][label] = p_value

        # Print the p-values for each label in each cluster
        for cluster_id, labels_p_values in p_values.items():
            print(f"Cluster {cluster_id} p-values:")
            for label, p_value in labels_p_values.items():
                outstr_label = label.replace("\n", "\\n")
                print(f"P-value: {p_value:.4f}, Label: {outstr_label}")
    return_dict = {
        "cluster_label_scores": cluster_label_scores,
        "cluster_labels": cluster_labels
    }
    if compute_p_values:
        return_dict["p_values"] = p_values
    return return_dict

# For each matching pairs of clusters, generate validated cluster labels based on asking the assistant LLM for the 
# difference between the texts of the two clusters. 
def get_validated_contrastive_cluster_labels(decoded_strs_1: List[str], 
                                             clustering_assignments_1: List[int], 
                                             decoded_strs_2: List[str], 
                                             clustering_assignments_2: List[int], 
                                             cluster_matches: List[Tuple[int, int]],
                                             local_model: AutoModel, 
                                             labeling_tokenizer: AutoTokenizer, 
                                             device: str,
                                             compute_p_values: bool = True,
                                             num_permutations: int = 3,
                                             use_normal_distribution_for_p_values: bool = False,
                                             sampled_comparison_texts_per_cluster: int = 10,
                                             generated_labels_per_cluster: int = 3,

                                             ):
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
            device=device, 
            sampled_texts_per_cluster=sampled_comparison_texts_per_cluster, 
            generated_labels_per_cluster=generated_labels_per_cluster
        )
        
        # Store the generated labels and the indices of texts used for label generation
        cluster_label_strs[(cluster_id_1, cluster_id_2)] = decoded_labels
        all_cluster_texts_used_for_label_strs_ids_1[cluster_id_1] = selected_text_indices_1
        all_cluster_texts_used_for_label_strs_ids_2[cluster_id_2] = selected_text_indices_2

    # Validate the discrimination power of the generated labels
    cluster_pair_scores, all_cluster_texts_used_for_validating_label_strs_ids = validate_cluster_label_comparative_discrimination_power(
        decoded_strs_1, clustering_assignments_1, all_cluster_texts_used_for_label_strs_ids_1,
        decoded_strs_2, clustering_assignments_2, all_cluster_texts_used_for_label_strs_ids_2,
        cluster_label_strs, local_model, labeling_tokenizer, device, sampled_comparison_texts_per_cluster
    )

    # Optionally compute p-values if required
    if compute_p_values:
        # Now, compute the null distribution for label AUCs by permuting the cluster labels (so the label strings
        # no longer match the cluster ids) and recomputing the AUCs.
        null_distribution_auc_scores = []
        for _ in range(num_permutations):
            # Permute the cluster assignments
            permuted_clustering_assignments_1 = np.random.permutation(clustering_assignments_1)
            permuted_clustering_assignments_2 = np.random.permutation(clustering_assignments_2)

            # Recompute the AUCs for the permuted labels
            permuted_scores, _ = validate_cluster_label_comparative_discrimination_power(
                decoded_strs_1, permuted_clustering_assignments_1, all_cluster_texts_used_for_label_strs_ids_1,
                decoded_strs_2, permuted_clustering_assignments_2, all_cluster_texts_used_for_label_strs_ids_2,
                cluster_label_strs, local_model, labeling_tokenizer, device, sampled_comparison_texts_per_cluster
            )

            # Collect the AUC scores from the permuted data
            for score_dict in permuted_scores.values():
                for label, score in score_dict.items():
                    null_distribution_auc_scores.append(score)

        # Calculate p-values based on the null distribution
        p_values = {}
        null_mean = np.mean(null_distribution_auc_scores)
        null_std = np.std(null_distribution_auc_scores)
        #print("null_distribution_auc_scores", null_distribution_auc_scores)
        for cluster_pair, label_scores in cluster_pair_scores.items():
            p_values[cluster_pair] = {}
            #print("label_scores", label_scores, "label_scores.items()", label_scores.items(), "cluster_pair", cluster_pair)
            for label, auc_score in label_scores.items():
                # Calculate the p-value using the cumulative distribution function (CDF) of the normal distribution
                if use_normal_distribution_for_p_values:
                    p_value = 1 - stats.norm.cdf(auc_score, loc=null_mean, scale=null_std)
                else:
                    p_value = sum(1 for score in null_distribution_auc_scores if score > auc_score) / len(null_distribution_auc_scores)
                p_values[cluster_pair][label] = p_value

    return_dict = {
        "cluster_pair_scores": cluster_pair_scores,
        "all_cluster_texts_used_for_validating_label_strs_ids": all_cluster_texts_used_for_validating_label_strs_ids
    }
    if compute_p_values:
        return_dict["p_values"] = p_values
    return return_dict
        
    



# Takes two sets of clusterings and returns the optimal pairwise matching of clusters between the two sets, seeking to minimize the sum of the squared Euclidean distances between the centroids of each pair of matching clusters.
def match_clusterings(clustering_assignments_1: List[int], embeddings_list_1: List[List[float]], clustering_assignments_2: List[int], embeddings_list_2: List[List[float]]) -> List[Tuple[int, int]]:
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
def assistant_generative_compare(difference_descriptions: List[str], 
                                 local_model: AutoModel, 
                                 labeling_tokenizer: AutoTokenizer, 
                                 starting_model_str: str,
                                 comparison_model_str: str,
                                 common_tokenizer_str: str,
                                 device: str,
                                 num_generated_texts_per_description: int = 10,
                                 permute_labels: bool = False,
                                 bnb_config: BitsAndBytesConfig = None
                                ):
    # Compile the list of prompts that encourage the assistant to generate texts attributed to the cluster 1 model.
    prompts_1 = []
    for description in difference_descriptions:
        prompt = f"Given the following description of how the texts in cluster 1 differ from those in cluster 2: {description}, generate a new text that is closer to cluster 1."
        prompts_1.append(prompt)
    # Now, compile the list of prompts that encourage the assistant to generate texts attributed to the cluster 2 model.
    prompts_2 = []
    for description in difference_descriptions:
        prompt = f"Given the following description of how the texts in cluster 1 differ from those in cluster 2: {description}, generate a new text that is closer to cluster 2."
        prompts_2.append(prompt)

    # Generate texts for each prompt using the assistant model
    generated_texts_1 = []
    for prompt in tqdm(prompts_1, desc="Generating texts for cluster 1"):
        inputs = labeling_tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        outputs = local_model.generate(**inputs, max_new_tokens=70, num_return_sequences=num_generated_texts_per_description, do_sample=True, pad_token_id=labeling_tokenizer.pad_token_id)
        generated_texts_1.append([labeling_tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
    generated_texts_2 = []
    for prompt in tqdm(prompts_2, desc="Generating texts for cluster 2"):
        inputs = labeling_tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        outputs = local_model.generate(**inputs, max_new_tokens=70, num_return_sequences=num_generated_texts_per_description, do_sample=True, pad_token_id=labeling_tokenizer.pad_token_id)
        generated_texts_2.append([labeling_tokenizer.decode(output, skip_special_tokens=True) for output in outputs])
    
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

    # First, load the cluster 1 model and compute scores
    current_model = AutoModelForCausalLM.from_pretrained(starting_model_str, device_map={"": 0} if device == "cuda:0" else "auto", quantization_config=bnb_config, torch_dtype=torch.float16)
    generated_texts_scores = []
    for texts_for_label in tqdm(generated_texts, desc="Computing scores for generated texts (cluster 1 attributed)"):
        generated_texts_scores_for_label = []
        for text in texts_for_label:
            inputs = common_tokenizer(text, return_tensors="pt").to(device)
            with torch.no_grad():
                model_loss = current_model(**inputs, labels=inputs["input_ids"]).loss
            generated_texts_scores_for_label.append(model_loss.item())
        generated_texts_scores.append(generated_texts_scores_for_label)
    
    # Next, load the cluster 2 model and compute scores, subtracting the cluster 2 scores from the cluster 1 scores
    current_model = AutoModelForCausalLM.from_pretrained(comparison_model_str, device_map={"": 0} if device == "cuda:0" else "auto", quantization_config=bnb_config, torch_dtype=torch.float16)
    for i in tqdm(range(len(generated_texts)), desc="Computing scores for generated texts (cluster 2 attributed)"):
        for j in range(len(generated_texts[i])):
            inputs = common_tokenizer(generated_texts[i][j], return_tensors="pt").to(device)
            with torch.no_grad():
                model_loss = current_model(**inputs, labels=inputs["input_ids"]).loss
            generated_texts_scores[i][j] -= model_loss.item()
    
    # target = 0 -> cluster 1
    # target = 1 -> cluster 2
    # score = cluster 1 score - cluster 2 score
    # score > 0 -> text is more associated with cluster 2
    # score < 0 -> text is more associated with cluster 1

    # Calculate the AUCs of the score differences as a way of looking at each text and determing which model it is intended to be 
    # closer to.
    per_label_aucs = [roc_auc_score(true_labels_set, generated_texts_scores_for_label) for true_labels_set, generated_texts_scores_for_label in zip(generated_text_labels, generated_texts_scores)]

    print("roc_auc_score", per_label_aucs)

    return per_label_aucs, generated_texts_1, generated_texts_2

# Uses assistant_generative_compare to generate the AUCs representing how well LM scores function to differentiate between the texts generated for one model and the other using the different descriptions.
# Also computes p-values for the descriptions AUCs.
def validated_assistant_generative_compare(difference_descriptions: List[str], 
                                           local_model: AutoModel, 
                                           labeling_tokenizer: AutoTokenizer, 
                                           starting_model_str: str,
                                           comparison_model_str: str,
                                           common_tokenizer_str: str,
                                           device: str,
                                           num_permutations: int = 1,
                                           use_normal_distribution_for_p_values: bool = False,
                                           num_generated_texts_per_description: int = 10,
                                           return_generated_texts: bool = False,
                                           bnb_config: BitsAndBytesConfig = None
                                        ):
    # First, generate the aucs using the real labels
    real_aucs, generated_texts_1, generated_texts_2 = assistant_generative_compare(difference_descriptions, local_model, labeling_tokenizer, starting_model_str, comparison_model_str, common_tokenizer_str, device, num_generated_texts_per_description, bnb_config=bnb_config)
    # Now, perform the permutation test
    permuted_aucs = []
    for _ in range(num_permutations):
        fake_aucs, _, _ = assistant_generative_compare(difference_descriptions, local_model, labeling_tokenizer, starting_model_str, comparison_model_str, common_tokenizer_str, device, num_generated_texts_per_description, permute_labels=True, bnb_config=bnb_config)
        permuted_aucs.extend(fake_aucs)
    # Now, compute the p-values
    if use_normal_distribution_for_p_values:
        null_mean = np.mean(permuted_aucs)
        null_std = np.std(permuted_aucs)
        p_values = [1 - stats.norm.cdf(auc_score, loc=null_mean, scale=null_std) for auc_score in real_aucs]
    else:
        p_values = [np.sum(np.array(permuted_aucs) > real_auc) / len(permuted_aucs) for real_auc in real_aucs]
    if return_generated_texts:
        return real_aucs, p_values, generated_texts_1, generated_texts_2
    else:
        return real_aucs, p_values





if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--compare_to_path", type=str, default=None)
    parser.add_argument("--compare_to_self", action="store_true")
    parser.add_argument("--api_key_path", type=str, default=None)
    parser.add_argument("--n_clusters", type=int, default=30)
    parser.add_argument("--n_clusters_compare", type=int, default=None)
    parser.add_argument("--min_cluster_size", type=int, default=7)
    parser.add_argument("--max_cluster_size", type=int, default=2000)
    parser.add_argument("--cluster_method", type=str, default="kmeans")
    parser.add_argument("--clustering_instructions", type=str, default="Identify the topic or theme of the given texts")

    parser.add_argument("--n_strs_show", type=int, default=0)
    parser.add_argument("--tqdm_disable", action="store_true")

    parser.add_argument("--skip_p_value_computation", action="store_true")
    parser.add_argument("--use_normal_distribution_for_p_values", action="store_true")
    parser.add_argument("--sampled_texts_per_cluster", type=int, default=10)
    parser.add_argument("--sampled_comparison_texts_per_cluster", type=int, default=10)
    parser.add_argument("--non_cluster_comparison_texts", type=int, default=10)
    parser.add_argument("--generated_labels_per_cluster", type=int, default=3)
    parser.add_argument("--permutations_for_null", type=int, default=1)

    parser.add_argument("--divergence_and_clustering_labels", action="store_true")
    parser.add_argument("--skip_every_n_decodings", type=int, default=0)
    parser.add_argument("--skip_every_n_decodings_compare", type=int, default=None)
    #parser.add_argument("--local_embedding_model_str", default="thenlper/gte-large")
    parser.add_argument("--local_embedding_model_str", default="nvidia/NV-Embed-v1")
    parser.add_argument("--local_labelings_model_str", default="NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recompute_embeddings", action="store_true")
    parser.add_argument("--color_by_divergence", action="store_true")
    parser.add_argument("--original_tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")

    parser.add_argument("--starting_model_str", type=str, default="NousResearch/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--comparison_model_str", type=str, default="NousResearch/Meta-Llama-3-8B")
    parser.add_argument("--common_tokenizer_str", type=str, default="NousResearch/Meta-Llama-3-8B")
    parser.add_argument("--num_generated_texts_per_description", type=int, default=10)

    args = parser.parse_args()
    path = args.path
    compare_to_path = args.compare_to_path
    n_clusters = int(args.n_clusters)
    n_clusters_compare = int(args.n_clusters_compare) if args.n_clusters_compare is not None else n_clusters
    min_cluster_size = int(args.min_cluster_size)
    max_cluster_size = int(args.max_cluster_size)
    n_strs_show = args.n_strs_show
    skip_every_n_decodings = args.skip_every_n_decodings
    skip_every_n_decodings_compare = args.skip_every_n_decodings_compare if args.skip_every_n_decodings_compare is not None else skip_every_n_decodings
    color_by_divergence = args.color_by_divergence
    compute_p_values = not args.skip_p_value_computation

    # Read OpenAI auth key from a file
    if args.api_key_path is not None:
        with open(args.api_key_path, 'r') as file:
            openai_auth_key = file.read().strip()
        # Authenticate with the OpenAI API
        client = OpenAI(api_key=openai_auth_key)
    else:
        client = None
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # First, load past results into arrays:
    # decoded_strs is an array of strings
    # divergence_values stores a single float for each entry in decoded_strs
    df = pd.read_csv(path)
    divergence_values, decoded_strs, embeddings_list, all_token_divergences, original_tokenizer, max_token_divergence, min_token_divergence = extract_df_info(df, args.original_tokenizer, path=path, skip_every_n_decodings=skip_every_n_decodings, color_by_divergence=color_by_divergence, local_embedding_model_str=args.local_embedding_model_str, device=args.device, recompute_embeddings=args.recompute_embeddings, save_embeddings=True, clustering_instructions=args.clustering_instructions, tqdm_disable=args.tqdm_disable, bnb_config=bnb_config)
    n_datapoints = len(decoded_strs)
    if compare_to_path is not None:
        df_compare = pd.read_csv(compare_to_path)
        divergence_values_compare, decoded_strs_compare, embeddings_list_compare, all_token_divergences_compare, original_tokenizer_compare, max_token_divergence_compare, min_token_divergence_compare = extract_df_info(df_compare, args.original_tokenizer, compare_to_path, skip_every_n_decodings=skip_every_n_decodings_compare, color_by_divergence=color_by_divergence, local_embedding_model_str=args.local_embedding_model_str, device=args.device, recompute_embeddings=args.recompute_embeddings, save_embeddings=True, clustering_instructions=args.clustering_instructions, tqdm_disable=args.tqdm_disable, bnb_config=bnb_config)
        n_datapoints_compare = len(decoded_strs_compare)
    elif args.compare_to_self:
        df_compare = deepcopy(df)
        # Intervene on the contents of df_compare 'decoding' column.
        # Replace 5% of all words with "aardvark"
        for i in range(len(df_compare)):
            decoding = df_compare.iloc[i]["decoding"]
            decoding = decoding.split()
            decoding = [word if (random.random() < 0.95 or i < 2) else "aardvark" for i,word in enumerate(decoding)]
            df_compare.at[i, "decoding"] = " ".join(decoding)
        divergence_values_compare, decoded_strs_compare, embeddings_list_compare, all_token_divergences_compare, original_tokenizer_compare, max_token_divergence_compare, min_token_divergence_compare = extract_df_info(df_compare, args.original_tokenizer, path, skip_every_n_decodings=skip_every_n_decodings_compare, color_by_divergence=color_by_divergence, local_embedding_model_str=args.local_embedding_model_str, device=args.device, recompute_embeddings=args.recompute_embeddings, save_embeddings=True, clustering_instructions=args.clustering_instructions, tqdm_disable=args.tqdm_disable, bnb_config=bnb_config)

    # Load the local assistant model (assuming we are not using the API):
    if args.api_key_path is None:
        local_model = AutoModelForCausalLM.from_pretrained(args.local_labelings_model_str,
                                                quantization_config=bnb_config,
                                                torch_dtype=torch.float16,
                                                device_map={"": 0} if args.device == "cuda:0" else "auto")
        
        labeling_tokenizer = AutoTokenizer.from_pretrained(args.local_labelings_model_str)
        if labeling_tokenizer.pad_token is None:
            labeling_tokenizer.pad_token = labeling_tokenizer.eos_token
            labeling_tokenizer.pad_token_id = labeling_tokenizer.eos_token_id
            local_model.generation_config.pad_token_id = labeling_tokenizer.pad_token_id
            print("PAD TOKEN ID: ", labeling_tokenizer.pad_token_id)
    
    # Cluster the entries of embeddings_list
    #clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0).fit(embeddings_list)
    print(f"Clustering with {args.cluster_method}")
    if args.cluster_method == "kmeans":
        clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(embeddings_list)
    elif args.cluster_method == "hdbscan":
        clustering = HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size).fit(embeddings_list)
    clustering_assignments = clustering.labels_
    # Print cluster sizes
    print("Cluster sizes:")
    for cluster in range(max(clustering_assignments) + 1):
        print(f"Cluster {cluster}: {len([i for i in range(n_datapoints) if clustering_assignments[i] == cluster])}")

    if args.divergence_and_clustering_labels:
        # Calculate the average divergence value of all the texts in each cluster.
        average_divergence_values = {}
        n_texts_in_cluster = {}
        for i in range(n_datapoints):
            cluster = clustering_assignments[i]
            if cluster in average_divergence_values:
                average_divergence_values[cluster] += divergence_values[i]
                n_texts_in_cluster[cluster] += 1
            else:
                average_divergence_values[cluster] = divergence_values[i]
                n_texts_in_cluster[cluster] = 1

        for k in average_divergence_values.keys():
            average_divergence_values[k] /= n_texts_in_cluster[k]

        # First, sort the clusters by average divergence values of the texts in the clusters.
        # Then for each cluster, print the n_strs_show lowest divergence texts, as well as the n_strs_show highest divergence texts.
        # Show their per-text divergence values and color the printed tokens by their per-token divergences.
        sorted_clusters = sorted(average_divergence_values.items(), key=lambda x: x[1])
        for cluster, avg_divergence in sorted_clusters:
            print(f"Cluster {cluster} average divergence: {avg_divergence}")
            cluster_indices = [i for i, x in enumerate(clustering_assignments) if x == cluster]
            sorted_indices = sorted(cluster_indices, key=lambda x: divergence_values[x])
            print("Lowest divergence texts:")
            for i in sorted_indices[:n_strs_show]:
                if color_by_divergence:
                    outstr = string_with_token_colors(decoded_strs[i], 
                                                    all_token_divergences[i], 
                                                    original_tokenizer,
                                                    min_score=min_token_divergence,
                                                    max_score=max_token_divergence)
                    print(f"Div: {divergence_values[i]:.3f}: {outstr}")
            print("Highest divergence texts:")
            for i in sorted_indices[-n_strs_show:]:
                if color_by_divergence:
                    outstr = string_with_token_colors(decoded_strs[i], 
                                                    all_token_divergences[i], 
                                                    original_tokenizer,
                                                    min_score=min_token_divergence,
                                                    max_score=max_token_divergence)
                    print(f"Div: {divergence_values[i]:.3f}: {outstr}")
    else:
        validated_clustering = get_validated_cluster_labels(decoded_strs, 
                                                            clustering_assignments, 
                                                            local_model, 
                                                            labeling_tokenizer, 
                                                            args.device,
                                                            compute_p_values=compute_p_values,
                                                            use_normal_distribution_for_p_values=args.use_normal_distribution_for_p_values,
                                                            sampled_texts_per_cluster=args.sampled_texts_per_cluster,
                                                            sampled_comparison_texts_per_cluster=args.sampled_comparison_texts_per_cluster,
                                                            non_cluster_comparison_texts=args.non_cluster_comparison_texts,
                                                            generated_labels_per_cluster=args.generated_labels_per_cluster,
                                                            num_permutations=args.permutations_for_null
                                                        )
        if compare_to_path is not None or args.compare_to_self:
            print(f"Clustering comparison texts with {args.cluster_method}")
            if args.cluster_method == "kmeans":
                clustering_compare =  KMeans(n_clusters=n_clusters_compare, random_state=0, n_init=10).fit(embeddings_list_compare)
            elif args.cluster_method == "hdbscan":
                clustering_compare =  HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size).fit(embeddings_list_compare)
            clustering_assignments_compare = clustering_compare.labels_
            print("Cluster sizes:")
            for cluster in range(max(clustering_assignments_compare) + 1):
                print(f"Cluster {cluster}: {len([i for i in range(n_datapoints_compare) if clustering_assignments_compare[i] == cluster])}")

            validated_clustering_compare = get_validated_cluster_labels(decoded_strs_compare, 
                                                                        clustering_assignments_compare, 
                                                                        local_model, 
                                                                        labeling_tokenizer, 
                                                                        args.device,
                                                                        compute_p_values=compute_p_values,
                                                                        use_normal_distribution_for_p_values=args.use_normal_distribution_for_p_values,
                                                                        sampled_texts_per_cluster=args.sampled_texts_per_cluster,
                                                                        sampled_comparison_texts_per_cluster=args.sampled_comparison_texts_per_cluster,
                                                                        non_cluster_comparison_texts=args.non_cluster_comparison_texts,
                                                                        generated_labels_per_cluster=args.generated_labels_per_cluster,
                                                                        num_permutations=args.permutations_for_null
                                                                        )
            cluster_matches = match_clusterings(clustering_assignments, embeddings_list, clustering_assignments_compare, embeddings_list_compare)
            base_cluster_scores = validated_clustering['cluster_label_scores']
            compare_cluster_scores = validated_clustering_compare['cluster_label_scores']
            base_cluster_p_values = validated_clustering['p_values']
            compare_cluster_p_values = validated_clustering_compare['p_values']
            print("Matching clusters:")
            for base_cluster, compare_cluster in cluster_matches:
                if not base_cluster in base_cluster_scores or not compare_cluster in compare_cluster_scores or not base_cluster in base_cluster_p_values or not compare_cluster in compare_cluster_p_values:
                    continue
                print(f"\n\nMost central texts in Cluster {base_cluster} (base) and Cluster {compare_cluster} (compare):")
                base_label_scores = base_cluster_scores[base_cluster]
                top_auc_base_label = max(base_label_scores, key=base_label_scores.get)
                compare_label_scores = compare_cluster_scores[compare_cluster]
                top_auc_compare_label = max(compare_label_scores, key=compare_label_scores.get)
                try:
                    print(f"Top-AUC base label (AUC: {base_label_scores[top_auc_base_label]:.3f}, P-value: {base_cluster_p_values[base_cluster][top_auc_base_label]:.3f}): {top_auc_base_label}")
                except KeyError:
                    print(f"Top-AUC base label (AUCs: {base_label_scores}, P-values: {base_cluster_p_values}): {top_auc_base_label}")
                try:
                    print(f"Top-AUC compare label (AUC: {compare_label_scores[top_auc_compare_label]:.3f}, P-value: {compare_cluster_p_values[compare_cluster][top_auc_compare_label]:.3f}): {top_auc_compare_label}")
                except KeyError:
                    print(f"Top-AUC compare label (AUCs: {compare_label_scores}, P-values: {compare_cluster_p_values}): {top_auc_compare_label}")
                # Base cluster central texts
                base_cluster_indices = [i for i, x in enumerate(clustering_assignments) if x == base_cluster]
                base_cluster_center = np.mean([embeddings_list[i] for i in base_cluster_indices], axis=0)
                distances_base = [np.linalg.norm(embeddings_list[i] - base_cluster_center) for i in base_cluster_indices]
                sorted_base_indices = np.argsort(distances_base)
                print("\nBase dataset central texts:")
                for i in sorted_base_indices[:n_strs_show]:
                    printstr = decoded_strs[base_cluster_indices[i]].replace("\n", "\\n")
                    print(f"Div: {divergence_values[base_cluster_indices[i]]:.3f}: {printstr}")

                # Compare cluster central texts
                compare_cluster_indices = [i for i, x in enumerate(clustering_assignments_compare) if x == compare_cluster]
                compare_cluster_center = np.mean([embeddings_list_compare[i] for i in compare_cluster_indices], axis=0)
                distances_compare = [np.linalg.norm(embeddings_list_compare[i] - compare_cluster_center) for i in compare_cluster_indices]
                sorted_compare_indices = np.argsort(distances_compare)
                print("\nCompare dataset central texts:")
                for i in sorted_compare_indices[:n_strs_show]:
                    printstr = decoded_strs_compare[compare_cluster_indices[i]].replace("\n", "\\n")
                    print(f"Div: {divergence_values_compare[compare_cluster_indices[i]]:.3f}: {printstr}")
            
            # Now generate contrastive LLM labels for the matching clusters pairs
            # Generate contrastive labels for each pair of matching clusters
            contrastive_labels = {}
            texts_used_for_contrastive_labels_1 = {}
            texts_used_for_contrastive_labels_2 = {}

            for base_cluster, compare_cluster in cluster_matches:
                # Generate contrastive labels for this pair of clusters
                decoded_labels, selected_text_indices_1, selected_text_indices_2 = contrastive_label_double_cluster(
                    decoded_strs_1=decoded_strs, 
                    clustering_assignments_1=clustering_assignments, 
                    cluster_id_1=base_cluster,
                    decoded_strs_2=decoded_strs_compare, 
                    clustering_assignments_2=clustering_assignments_compare, 
                    cluster_id_2=compare_cluster,
                    local_model=local_model, 
                    labeling_tokenizer=labeling_tokenizer, 
                    device=args.device, 
                    sampled_texts_per_cluster=args.sampled_comparison_texts_per_cluster,
                    generated_labels_per_cluster=args.generated_labels_per_cluster
                )
                contrastive_labels[(base_cluster, compare_cluster)] = decoded_labels
                texts_used_for_contrastive_labels_1[base_cluster] = selected_text_indices_1
                texts_used_for_contrastive_labels_2[compare_cluster] = selected_text_indices_2

            # Validate the discrimination power of the generated contrastive labels
            cluster_pair_scores, all_cluster_texts_used_for_validating_label_strs_ids = validate_cluster_label_comparative_discrimination_power(
                decoded_strs_1=decoded_strs, 
                clustering_assignments_1=clustering_assignments, 
                all_cluster_texts_used_for_label_strs_ids_1=texts_used_for_contrastive_labels_1, 
                decoded_strs_2=decoded_strs_compare, 
                clustering_assignments_2=clustering_assignments_compare, 
                all_cluster_texts_used_for_label_strs_ids_2=texts_used_for_contrastive_labels_2, 
                cluster_label_strs=contrastive_labels, 
                local_model=local_model, 
                labeling_tokenizer=labeling_tokenizer, 
                device=args.device, 
                sampled_comparison_texts_per_cluster=args.sampled_comparison_texts_per_cluster
            )

            # Get and print validated contrastive labels with their scores
            cluster_labeling_results_dict = get_validated_contrastive_cluster_labels(
                decoded_strs_1=decoded_strs, 
                clustering_assignments_1=clustering_assignments, 
                decoded_strs_2=decoded_strs_compare, 
                clustering_assignments_2=clustering_assignments_compare, 
                cluster_matches=cluster_matches,
                local_model=local_model, 
                labeling_tokenizer=labeling_tokenizer, 
                device=args.device,
                compute_p_values=True,
                num_permutations=args.permutations_for_null,
                use_normal_distribution_for_p_values=False,
                sampled_comparison_texts_per_cluster=args.sampled_comparison_texts_per_cluster,
                generated_labels_per_cluster=args.generated_labels_per_cluster
            )
            cluster_pair_scores = cluster_labeling_results_dict["cluster_pair_scores"]
            #all_cluster_texts_used_for_validating_label_strs_ids = cluster_labeling_results_dict["all_cluster_texts_used_for_validating_label_strs_ids"]
            p_values = cluster_labeling_results_dict["p_values"]
            result_stats = []

            # Print the contrastive labels for each pair of matching clusters
            print("Contrastive labels for each pair of matching clusters:")
            for cluster_pair, label_scores in cluster_pair_scores.items():
                print(f"Contrastive labels for Cluster {cluster_pair[0]} (base) and Cluster {cluster_pair[1]} (compare):")
                for label, score in label_scores.items():
                    print(f"Score: {score:.3f}, P-value: {p_values[cluster_pair][label]:.3f}, Label: {label}\n\n")
                    result_stats.append([label, cluster_pair, score, p_values[cluster_pair][label]])
            
            # Finally, let's see if the contrastive labels allow the assistant model to generate texts that are more similar to the cluster 1 / cluster 2 models.
            validated_assistant_aucs, validated_assistant_p_values, generated_validation_texts_1, generated_validation_texts_2 = validated_assistant_generative_compare([r[0] for r in result_stats], local_model, labeling_tokenizer, starting_model_str=args.starting_model_str, comparison_model_str=args.comparison_model_str, common_tokenizer_str=args.common_tokenizer_str, device=args.device, num_generated_texts_per_description=args.num_generated_texts_per_description, return_generated_texts=True, bnb_config=bnb_config)

            # Print the validated assistant labels AUCs and p-values
            for i, (description, cluster_pair, auc_score, p_value) in enumerate(result_stats):
                print(f"Description {i+1} (clusters {cluster_pair[0]} and {cluster_pair[1]}): {description}")
                print(f"Intra-cluster AUC: {auc_score:.3f}, Intra-cluster P-value: {p_value:.3f}")
                print(f"Generative model AUC: {validated_assistant_aucs[i]:.3f}, Generative model P-value: {validated_assistant_p_values[i]:.3f}")
                print("\n")
                print("Generated validation texts for cluster 1:")
                for j, text in enumerate(generated_validation_texts_1[i][:n_strs_show]):
                    print(f"Text {j+1}: {text}")
                print("\n\n")
                print("Generated validation texts for cluster 2:")
                for j, text in enumerate(generated_validation_texts_2[i][:n_strs_show]):
                    print(f"Text {j+1}: {text}")
                print("\n\n")
    