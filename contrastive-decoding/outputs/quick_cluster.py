
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from openai import OpenAI
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import pickle
import scipy.stats as stats
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, BitsAndBytesConfig
from ast import literal_eval
import random
import sys
sys.path.append("..")
from model_comparison_helpers import string_with_token_colors

import warnings
warnings.filterwarnings('ignore', message='You have modified the pretrained model configuration to control generation.*')

def LLM_cluster_labeling(decoded_strs, divergence_values, clustering_assignments, client, local_model = None, tokenizer = None, device = "cuda:0"):
    # For each cluster, we print out the average divergence as well as the 5 top divergence texts and the 5 bottom divergence texts assigned to that cluster.
    
    cluster_label_instruction = "You are an expert at labeling clusters. You produce a single susinct label immediately in response to every query."

    for cluster in set(clustering_assignments):
        print("===================================================================")
        print(f"Cluster {cluster}:")
        print(f"Average divergence: {average_divergence_values[cluster]}")
        
        cluster_indices = [i for i, x in enumerate(clustering_assignments) if x == cluster]
        cluster_divergences = [divergence_values[i] for i in cluster_indices]
        cluster_texts = [decoded_strs[i] for i in cluster_indices]
        
        sorted_indices = sorted(range(len(cluster_divergences)), key=lambda k: cluster_divergences[k])

        top_5_texts = [cluster_texts[i] for i in sorted_indices[:5]]
        bottom_5_texts = [cluster_texts[i] for i in sorted_indices[-5:]]

        if client is not None:
            # Use OpenAI chat completions to generate a cluster label based on the top 5 texts
            top_5_texts_label = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": cluster_label_instruction},
                    {"role": "user", "content": "Texts in current cluster: " + ', '.join(top_5_texts)}
                ],
                max_tokens=30)
            bottom_5_texts_label = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": cluster_label_instruction},
                    {"role": "user", "content": "Texts in current cluster: " + ', '.join(bottom_5_texts)}
                ],
                max_tokens=30)
            top_label = top_5_texts_label.choices[0].message.content
            bottom_label = bottom_5_texts_label.choices[0].message.content
        else:
            # Use a local language model to generate cluster labels
            # Still a WIP
            with torch.no_grad():
                # Generate input strings for local model
                str_instruction_to_local_model_label_top_5 = cluster_label_instruction + "\n" + "Texts in current cluster: " + ', '.join(top_5_texts)
                str_instruction_to_local_model_label_bottom_5 = cluster_label_instruction + "\n" + "Texts in current cluster: " + ', '.join(bottom_5_texts)
                # Convert to ids
                ids_instruction_to_local_model_label_top_5 = tokenizer(str_instruction_to_local_model_label_top_5, return_tensors="pt").to(device)
                ids_instruction_to_local_model_label_bottom_5 = tokenizer(str_instruction_to_local_model_label_bottom_5, return_tensors="pt").to(device)
                # Generate labels
                output_top_5 = local_model(**ids_instruction_to_local_model_label_top_5)
                output_bottom_5 = local_model(**ids_instruction_to_local_model_label_bottom_5)

                # Decode labels to strings
                print(output_top_5.logits.argmax(dim=2))
                top_label = tokenizer.decode(output_top_5.logits.argmax(dim=2)[0])
                bottom_label = tokenizer.decode(output_bottom_5.logits.argmax(dim=2)[0])

        print("Top 5 divergence texts:")
        for i in sorted_indices[:5]:
            print(f"Divergence: {round(cluster_divergences[i], 4)} Text: {cluster_texts[i]}")
        print("Cluster label (top 5):")
        print(top_label)
        print("\n")
        
        print("Bottom 5 divergence texts:")
        for i in sorted_indices[-5:]:
            print(f"Divergence: {round(cluster_divergences[i], 4)} Text: {cluster_texts[i]}")
        print("Cluster label (bottom 5):")
        print(bottom_label)
        print("\n\n")

def get_cluster_labels_random_subsets(decoded_strs, clustering_assignments, local_model = None, tokenizer = None, device = "cuda:0", sampled_texts_per_cluster = 7, generated_labels_per_cluster = 3):
    cluster_label_instruction = "You are an expert at describing clusters of texts. When given a list of texts belonging to a cluster, you immediately respond with a short description of the key themes of the texts shown to you."
    cluster_labels = {}
    all_cluster_texts_used_for_label_strs_ids = {}
    for cluster in set(clustering_assignments):
        # For each cluster, select sampled_texts_per_cluster random texts (skipping the cluster if it has less than 2x sampled_texts_per_cluster texts)
        cluster_indices = [i for i, x in enumerate(clustering_assignments) if x == cluster]
        if len(cluster_indices) < 2 * sampled_texts_per_cluster:
            continue
        selected_text_indices = np.random.choice(cluster_indices, sampled_texts_per_cluster, replace=False)
        selected_texts = [decoded_strs[i].replace("\n", "\\n") for i in selected_text_indices]
        for i, text in enumerate(selected_texts):
            selected_texts[i] = text.replace("<s>", "")
            selected_texts[i] = f"Text {i}: " + selected_texts[i]
        all_cluster_texts_used_for_label_strs_ids[cluster] = selected_text_indices

        # Use the local model to generate a cluster label for the selected texts
        # Generate input string for local model
        str_instruction_to_local_model = cluster_label_instruction + "\n" + "Texts in current cluster:\n" + '\n'.join(selected_texts)
        str_instruction_to_local_model = str_instruction_to_local_model + "\n" + "Cluster description: the common theme of the above texts is that they are all about"
        # Prepare inputs for text generation
        inputs = tokenizer(str_instruction_to_local_model, return_tensors="pt").to(device)
        inputs_length = inputs.input_ids.shape[1]
        # Generate labels using the Hugging Face text generation API
        with torch.no_grad():
            outputs = [local_model.generate(**inputs, max_new_tokens=15, num_return_sequences=1, do_sample=True, pad_token_id=tokenizer.eos_token_id) for _ in range(generated_labels_per_cluster)]
        # Decode labels to strings
        #decoded_interactions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        #print(decoded_interactions)
        decoded_labels = [tokenizer.decode(output[0][inputs_length:], skip_special_tokens=True) for output in outputs]
        for i, label in enumerate(decoded_labels):
            if label.startswith(" "):
                decoded_labels[i] = label[1:]
                if "<s>" in label:
                    decoded_labels[i] = label[:label.index("<s>")]
        cluster_labels[cluster] = decoded_labels
        
        print(f"Cluster {cluster} labels: {cluster_labels[cluster]}")

    return cluster_labels, all_cluster_texts_used_for_label_strs_ids

def validate_cluster_label_discrimination_power(decoded_strs, clustering_assignments, cluster_label_strs, all_cluster_texts_used_for_label_strs_ids, local_model = None, tokenizer = None, device = "cuda:0", sampled_comparison_texts_per_cluster = 5, non_cluster_comparison_texts = 5):
    # For each possible label of each cluster, we select sampled_comparison_texts_per_cluster random texts (which must not
    # be in all_cluster_texts_used_for_label_strs_ids, skipping the cluster and printing a warning if there aren't enough texts in the cluster).
    # We then perform pairwise comparisons between each of the sampled_comparison_texts_per_cluster texts in the cluster and 
    # non_cluster_comparison_texts random texts, asking the local_model LLM which of the two texts better fits the given label.
    # We score potential cluster labels based on their ability to let the assistant LLM distinguish between texts from the cluster
    # and texts not from the cluster, quantified with AUC. We then return the scores for each cluster.
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
                    input_ids = tokenizer(input_str, return_tensors="pt").to(device)
                    #print("input_str:", input_str)
                    
                    # Query the local model
                    with torch.no_grad():
                        output = local_model(**input_ids)
                        logits = output.logits
                        # Compare the probabilities for "A" and "B" as continuations
                        prob_A = torch.nn.functional.softmax(logits, dim=-1)[0, -1, tokenizer.encode("A", add_special_tokens=False)[0]].item()
                        prob_B = torch.nn.functional.softmax(logits, dim=-1)[0, -1, tokenizer.encode("B", add_special_tokens=False)[0]].item()
                        normalized_prob_A = prob_A / max(prob_A + prob_B, 1e-10)
            
                    scores.append(normalized_prob_A)
                    true_labels.append(prob_A_target)
            # Compute AUC for the current label
            auc = roc_auc_score(true_labels, scores)
            cluster_label_scores[cluster_id][label] = auc
                
    return cluster_label_scores, all_cluster_texts_used_for_validating_label_strs_ids


def get_hierarchical_descendants(root_id, clustering):
    descendants = []
    stack = [root_id + len(clustering.labels_)]
    while stack:
        child = stack.pop()
        if child < len(clustering.labels_):
            descendants.append(child)
        else:
            stack.extend(clustering.children_[child - len(clustering.labels_)])
    return descendants

# This function produces a hierarchical clustering of the text, then searches for locations in that hierarchy where a single
# cluster splits into two subclusters, where one subcluster has high average divergence and the other has low average divergence
# (specifically meaning that the ratio of the subcluster average divergences is at least min_divergence_ratio, and both 
# subclusters have at least min_points_in_subcluster points).
# Returns a list of (cluster, low divergence subcluster, high divergence subcluster) tuples and the clustering.
def hierarchical_clustering_identify_divergence_split(divergence_values, embeddings_list, min_divergence_ratio = 1.2, min_points_in_subcluster = 10):
    # Create a hierarchical clustering of the embeddings
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0).fit(embeddings_list)

    # Initialize an empty list to store the clusters that meet the criteria
    divergence_split_clusters = []

    # Documentation of clustering.children_
    # array-like of shape (n_samples-1, 2)
    # The children of each non-leaf node. Values less than n_samples correspond to leaves of the tree which are the original samples. A node i greater than or equal to n_samples is a non-leaf node and has children children_[i - n_samples]. Alternatively at the i-th iteration, children[i][0] and children[i][1] are merged to form node n_samples + i.

    # We need to find the datapoint ids of each node (meaning join of two or more datapoints) in the clustering.
    # First, we calculate the number of nodes:
    n_nodes = max(np.concatenate(clustering.children_)) - len(clustering.labels_) + 1

    # Next, for each node, we find the ids of every datapoint that is ultimately a descendant of that node, as well as the ids
    # of its immediate children.
    node_descendants = []
    node_children_ids = []
    for i in range(n_nodes):
        # negative values indicate children are leaf nodes
        node_children_ids.append([-1, -1])
        if clustering.children_[i][0] >= len(clustering.labels_):
            node_children_ids[i][0] = clustering.children_[i][0] - len(clustering.labels_)
        if clustering.children_[i][1] >= len(clustering.labels_):
            node_children_ids[i][1] = clustering.children_[i][1] - len(clustering.labels_)
        
        descendants = get_hierarchical_descendants(i, clustering)
        node_descendants.append(descendants)
        #print(f"Node {i} has {len(descendants)} descendants. Its children are {node_children_ids[i]}")
    
    # Now, we iterate through each node and identify any nodes whose children:
    #    - are not leafs
    #    - each have at least min_points_in_subcluster total descendants
    #    - have a ratio of the average divergence of the two children that is at least min_divergence_ratio
    for i in range(n_nodes):
        child_node_0_id = node_children_ids[i][0]
        child_node_1_id = node_children_ids[i][1]
        # Check for leaf children
        if child_node_0_id == -1 or child_node_1_id == -1:
            continue
        # Check for minimum number of points in each subcluster
        if len(node_descendants[child_node_0_id]) < min_points_in_subcluster or len(node_descendants[child_node_1_id]) < min_points_in_subcluster:
            continue
        # Compute average divergences of each child node
        avg_divergence_0 = np.mean([divergence_values[j] for j in node_descendants[child_node_0_id]])
        avg_divergence_1 = np.mean([divergence_values[j] for j in node_descendants[child_node_1_id]])
        # Check for minimum divergence ratio
        if avg_divergence_0 / avg_divergence_1 >= min_divergence_ratio or avg_divergence_1 / avg_divergence_0 >= min_divergence_ratio:
            if avg_divergence_0 < avg_divergence_1:
                divergence_split_clusters.append((i, child_node_0_id, child_node_1_id))
            else:
                divergence_split_clusters.append((i, child_node_1_id, child_node_0_id))
    
    # Return the list of clusters that meet the criteria
    return divergence_split_clusters, clustering

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
def read_past_embeddings_or_generate_new(path, client, decoded_strs, local_embedding_model = "thenlper/gte-large", device = "cuda:0", recompute_embeddings = False):
    # First, try to load past embeddings from file:
    try:
        if recompute_embeddings:
            raise ValueError("Recomputing embeddings.")
        with open(path + "_embeddings.pkl", "rb") as f:
            embeddings_list = pickle.load(f)
            # Check that embeddings_list has the expected dimensions
            if len(embeddings_list) != len(decoded_strs):
                raise ValueError("The loaded embeddings have the wrong dimensions.")
    except:
        # Otherwise, compute new embeddings
        embeddings_list = []
        batch_size = 100
        if client is not None:
            for i in tqdm(range(0, len(decoded_strs), batch_size)):
                batch = decoded_strs[i:i+batch_size]
                embeddings = client.embeddings.create(input = batch, model = "text-embedding-ada-002").data
                embeddings_list.extend([e.embedding for e in embeddings])
        else:
            # Use local embedding model thenlper/gte-large from HuggingFace
            model = AutoModel.from_pretrained(local_embedding_model).to(device)
            tokenizer = AutoTokenizer.from_pretrained(local_embedding_model)
            with torch.no_grad():
                batch_size = 1
                # TODO: group texts by length so they can be batched without having to average over padding tokens.
                for i in range(0, len(decoded_strs), batch_size):
                    batch = decoded_strs[i:i+batch_size]
                    batch_ids = tokenizer(batch, return_tensors="pt", padding=False, truncation=False).to(device)
                    output = model(**batch_ids, output_hidden_states=True)
                    embeddings = output.hidden_states
                    embeddings_list.extend([e for e in embeddings[0].mean(dim=1).detach().cpu().numpy()])
                    del output
            del model
            del tokenizer
        # Also, save the new embeddings to file
        with open(path + "_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings_list, f)
        
    return embeddings_list

        
parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--api_key_path", type=str, default=None)
parser.add_argument("--n_clusters", type=int, default=30)
parser.add_argument("--n_strs_show", type=int, default=0)
parser.add_argument("--no_LM_desc", action="store_true")
parser.add_argument("--hierarchical", action="store_true")
parser.add_argument("--divergence_and_clustering_labels", action="store_true")
parser.add_argument("--skip_every_n_decodings", type=int, default=0)
parser.add_argument("--local_embedding_model", default="thenlper/gte-large")
parser.add_argument("--local_labelings_model", default="NousResearch/Meta-Llama-3-8B-Instruct")
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--recompute_embeddings", action="store_true")
parser.add_argument("--min_divergence_ratio", type=float, default=1.2)
parser.add_argument("--min_points_in_subcluster", type=int, default=10)
parser.add_argument("--color_by_divergence", action="store_true")
parser.add_argument("--llm_label_hierarchical_results", action="store_true")
parser.add_argument("--original_tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")

args = parser.parse_args()
path = args.path
n_clusters = int(args.n_clusters)
n_strs_show = args.n_strs_show
skip_every_n_decodings = args.skip_every_n_decodings
color_by_divergence = args.color_by_divergence

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
                bnb_8bit_use_double_quant=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=torch.bfloat16
            )
    local_model = AutoModelForCausalLM.from_pretrained(args.local_labelings_model,
                                            load_in_8bit=True, 
                                            device_map={"": 0} if args.device == "cuda:0" else "auto",
                                            quantization_config=bnb_config)
    labeling_tokenizer = AutoTokenizer.from_pretrained(args.local_labelings_model)
    if labeling_tokenizer.pad_token is None:
        labeling_tokenizer.pad_token = labeling_tokenizer.eos_token
        labeling_tokenizer.pad_token_id = labeling_tokenizer.eos_token_id

df = pd.read_csv(path)

# First, load past results into arrays:
# decoded_strs is an array of strings
# divergence_values stores a single float for each entry in decoded_strs
divergence_values = df['divergence'].values
loaded_strs = df['decoding'].values
if color_by_divergence:
    all_token_divergences = df['all_token_divergences'].values
    all_token_divergences = [literal_eval(s) for s in all_token_divergences]
    original_tokenizer = AutoTokenizer.from_pretrained(args.original_tokenizer)

    max_token_divergence = max([max(token_divergences) for token_divergences in all_token_divergences])
    min_token_divergence = min([min(token_divergences) for token_divergences in all_token_divergences])


if skip_every_n_decodings > 0:
    loaded_strs = loaded_strs[::skip_every_n_decodings]
    divergence_values = divergence_values[::skip_every_n_decodings]
    if color_by_divergence:
        all_token_divergences = all_token_divergences[::skip_every_n_decodings]

n_datapoints = len(loaded_strs)
#decoded_strs = [s.split("|")[1] for s in loaded_strs]
decoded_strs = [s.replace("|", "") for s in loaded_strs]

for i in range(len(decoded_strs)):
    if decoded_strs[i][0] == ' ':
        decoded_strs[i] = decoded_strs[i][1:]
#print(decoded_strs)

# Generate embeddings for the past results.
# embeddings_list is a n_datapoints x embedding_dim list of floats
embeddings_list = read_past_embeddings_or_generate_new(path, 
                                                       client, 
                                                       decoded_strs, 
                                                       local_embedding_model=args.local_embedding_model, 
                                                       device=args.device,
                                                       recompute_embeddings=args.recompute_embeddings
                                                       )
# Now, cluster the entries of embeddings_list
if not args.hierarchical:
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0).fit(embeddings_list)
    clustering_assignments = clustering.labels_

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
        # First, assign possible labels to each cluster using get_cluster_labels_random_subsets. Then, use
        # validate_cluster_label_discrimination_power to get the AUC for each cluster label. Then, for each
        # cluster, print the randomly selected texts used to generate the cluster labels, as well as every
        # cluster label and its AUC.
        cluster_labels, all_cluster_texts_used_for_label_strs_ids = get_cluster_labels_random_subsets(decoded_strs, clustering_assignments, local_model, labeling_tokenizer, device=args.device)
        cluster_label_scores, all_cluster_texts_used_for_validating_label_strs_ids = validate_cluster_label_discrimination_power(decoded_strs, clustering_assignments, cluster_labels, all_cluster_texts_used_for_label_strs_ids, local_model, labeling_tokenizer, device=args.device)

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
        
        # Now, compute the null distribution for label AUCs by permuting the cluster labels (so the label strings
        # no longer match the cluster ids) and recomputing the AUCs.
        num_permutations = 3
        null_distribution_auc_scores = []
        for _ in range(num_permutations):
            permuted_cluster_labels = {}
            permuted_cluster_ids = random.sample(list(cluster_labels.keys()), len(cluster_labels))
            for i, cluster_id in enumerate(permuted_cluster_ids):
                ith_cluster_label_key = list(cluster_labels.keys())[i]
                permuted_cluster_labels[ith_cluster_label_key] = cluster_labels[cluster_id]
            null_cluster_label_scores_nested_dict, _ = validate_cluster_label_discrimination_power(decoded_strs, clustering_assignments, permuted_cluster_labels, all_cluster_texts_used_for_label_strs_ids, local_model, labeling_tokenizer, device=args.device)
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
                p_value = 1 - stats.norm.cdf(auc_score, loc=null_mean, scale=null_std)
                p_values[cluster_id][label] = p_value

        # Print the p-values for each label in each cluster
        for cluster_id, labels_p_values in p_values.items():
            print(f"Cluster {cluster_id} p-values:")
            for label, p_value in labels_p_values.items():
                outstr_label = label.replace("\n", "\\n")
                print(f"Label: {outstr_label}, p-value: {p_value:.4f}")
        


# if not args.no_LM_desc:
#     LLM_cluster_labeling(decoded_strs, divergence_values, clustering_assignments, client, local_model=local_model, tokenizer=labeling_tokenizer, device=args.device)

if args.hierarchical:
    divergence_split_clusters, clustering = hierarchical_clustering_identify_divergence_split(divergence_values, 
                                                                                              embeddings_list, 
                                                                                              min_divergence_ratio=args.min_divergence_ratio, 
                                                                                              min_points_in_subcluster=args.min_points_in_subcluster
                                                                                              )
    print("divergence_split_clusters:", divergence_split_clusters)
    print("clustering.children_.shape:", clustering.children_.shape)
    print("clustering.labels_.shape:", clustering.labels_.shape)
    for cluster, low_divergence_subcluster, high_divergence_subcluster in divergence_split_clusters:
        print(f"Cluster {cluster}:")
        # Find all descendants of the cluster
        cluster_descendants = get_hierarchical_descendants(cluster, clustering)
        # Calculate and print average cluster divergence
        average_divergence = np.mean([divergence_values[i] for i in cluster_descendants])
        print(f"Cluster {cluster} size: {len(cluster_descendants)} average divergence: {average_divergence}")

        # Find all descendants of the low divergence subcluster
        low_divergence_subcluster_descendants = get_hierarchical_descendants(low_divergence_subcluster, clustering)
        # Calculate and print average divergence of low divergence subcluster
        average_divergence_low = np.mean([divergence_values[i] for i in low_divergence_subcluster_descendants])
        print(f"Low divergence subcluster size: {len(low_divergence_subcluster_descendants)} average divergence: {average_divergence_low}")
        # Find all descendants of the high divergence subcluster
        high_divergence_subcluster_descendants = get_hierarchical_descendants(high_divergence_subcluster, clustering)
        # Calculate and print average divergence of high divergence subcluster
        average_divergence_high = np.mean([divergence_values[i] for i in high_divergence_subcluster_descendants])
        print(f"High divergence subcluster size: {len(high_divergence_subcluster_descendants)} average divergence: {average_divergence_high}")
        
        subclusters_str = ''
        subclusters_str_for_llm_comparison = ''

        print("")
        subclusters_str += "\nLow divergence subcluster center texts:\n"
        subclusters_str_for_llm_comparison += "Low divergence subcluster texts:\n"
        low_divergence_subcluster_center_points = [embeddings_list[i] for i in low_divergence_subcluster_descendants]
        low_divergence_subcluster_center = np.mean(low_divergence_subcluster_center_points, axis=0)
        distances = [np.linalg.norm(embeddings_list[i] - low_divergence_subcluster_center) for i in low_divergence_subcluster_descendants]
        sorted_indices = np.argsort(distances)
        for i, idx in enumerate(sorted_indices[:n_strs_show]):
            if color_by_divergence:
                outstr = string_with_token_colors(decoded_strs[low_divergence_subcluster_descendants[idx]], 
                                                  all_token_divergences[low_divergence_subcluster_descendants[idx]], 
                                                  original_tokenizer,
                                                  min_score=min_token_divergence,
                                                  max_score=max_token_divergence)
            else:
                outstr = decoded_strs[low_divergence_subcluster_descendants[i]]
            subclusters_str += f"Div {divergence_values[low_divergence_subcluster_descendants[idx]]:.3f}: " + outstr + "\n"
            subclusters_str_for_llm_comparison += f"Text {i+1}: " + decoded_strs[low_divergence_subcluster_descendants[idx]] + "\n"
        
        subclusters_str += "\nHigh divergence subcluster center texts:\n"
        subclusters_str_for_llm_comparison += "High divergence subcluster texts:\n"
        high_divergence_subcluster_center_points = [embeddings_list[i] for i in high_divergence_subcluster_descendants]
        high_divergence_subcluster_center = np.mean(high_divergence_subcluster_center_points, axis=0)
        distances = [np.linalg.norm(embeddings_list[i] - high_divergence_subcluster_center) for i in high_divergence_subcluster_descendants]
        sorted_indices = np.argsort(distances)
        for i, idx in enumerate(sorted_indices[:n_strs_show]):
            if color_by_divergence:
                outstr = string_with_token_colors(decoded_strs[high_divergence_subcluster_descendants[idx]], 
                                                  all_token_divergences[high_divergence_subcluster_descendants[idx]], 
                                                  original_tokenizer,
                                                  min_score=min_token_divergence,
                                                  max_score=max_token_divergence)
            else:
                outstr = decoded_strs[high_divergence_subcluster_descendants[idx]]
            subclusters_str += f"Div {divergence_values[high_divergence_subcluster_descendants[idx]]:.3f}: " + outstr + "\n"
            subclusters_str_for_llm_comparison += f"Text {i+1}: " + decoded_strs[high_divergence_subcluster_descendants[idx]] + "\n"
        print(subclusters_str + '\n')

        if args.llm_label_hierarchical_results:
            answer_preface = "Answer: The low-divergence subcluster"
            llm_instruction_str = "Very concisely describe the key differences between the texts in the following two subclusters. Focus on overall themes, not individual texts:\n" + subclusters_str_for_llm_comparison + "\n\n" + answer_preface
            if client is not None:
                llm_answer = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": llm_instruction_str}
                    ],
                    max_tokens=500)
                print(llm_answer.choices[0].message.content)
            else:
                llm_instruction_ids = labeling_tokenizer(llm_instruction_str, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    output = local_model.generate(llm_instruction_ids['input_ids'], do_sample=True, top_p = 0.95, max_new_tokens = 220, return_dict_in_generate=True)
                output = output.sequences[0, llm_instruction_ids['input_ids'].shape[1]:]
                #print("output.logits.argmax(dim=2):", output.logits.argmax(dim=2))
                llm_answer = labeling_tokenizer.decode(output.tolist())
                print(answer_preface + " " + llm_answer)



