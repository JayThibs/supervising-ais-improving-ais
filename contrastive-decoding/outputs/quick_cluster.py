
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from openai import OpenAI
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from tqdm import tqdm
import numpy as np
import pickle
import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, BitsAndBytesConfig
from ast import literal_eval

import sys
sys.path.append("..")
from model_comparison_helpers import string_with_token_colors

import warnings
warnings.filterwarnings('ignore', message='You have modified the pretrained model configuration to control generation.*')

def LLM_cluster_labeling(decoded_strs, divergence_values, labels, client, use_openai = True, local_labelings_model = "teknium/OpenHermes-2.5-Mistral-7B", device = "cuda:0"):
    # For each cluster, we print out the average divergence as well as the 5 top divergence texts and the 5 bottom divergence texts assigned to that cluster.
    if not use_openai:
        local_model = AutoModelForCausalLM.from_pretrained(local_labelings_model).to(device)
        tokenizer = AutoTokenizer.from_pretrained(local_labelings_model)
    
    cluster_label_instruction = "You are an expert at labeling clusters. You produce a single susinct label immediately in response to every query."

    for cluster in set(labels):
        print("===================================================================")
        print(f"Cluster {cluster}:")
        print(f"Average divergence: {average_divergence_values[cluster]}")
        
        cluster_indices = [i for i, x in enumerate(labels) if x == cluster]
        cluster_divergences = [divergence_values[i] for i in cluster_indices]
        cluster_texts = [decoded_strs[i] for i in cluster_indices]
        
        sorted_indices = sorted(range(len(cluster_divergences)), key=lambda k: cluster_divergences[k])

        top_5_texts = [cluster_texts[i] for i in sorted_indices[:5]]
        bottom_5_texts = [cluster_texts[i] for i in sorted_indices[-5:]]

        if use_openai:
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
def sort_by_distance(labels, embedding_list):
    # Initialize an empty list to store the sorted clusters
    sorted_clusters = []

    # Iterate over each unique label in the labels list
    for label in set(labels):
        # Get the indices of the texts in this cluster
        cluster_indices = [i for i, x in enumerate(labels) if x == label]

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
def read_past_embeddings_or_generate_new(path, client, decoded_strs, use_openai = True, local_embedding_model = "thenlper/gte-large", device = "cuda:0", recompute_embeddings = False):
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
        if use_openai:
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
parser.add_argument("--path")
parser.add_argument("--api_key_path", default="../../../key.txt")
parser.add_argument("--n_clusters", default=30)
parser.add_argument("--n_strs_show", type=int, default=0)
parser.add_argument("--no_LM_desc", action="store_true")
parser.add_argument("--hierarchical", action="store_true")
parser.add_argument("--skip_every_n_decodings", type=int, default=0)
parser.add_argument("--local_embedding_model", default="thenlper/gte-large")
parser.add_argument("--local_labelings_model", default="Upstage/SOLAR-10.7B-Instruct-v1.0")
parser.add_argument("--use_openai", action="store_true")
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
with open(args.api_key_path, 'r') as file:
    openai_auth_key = file.read().strip()

# Authenticate with the OpenAI API
client = OpenAI(api_key=openai_auth_key)

df = pd.read_csv(path)

# First, load past results into arrays:
# decoded_strs is an array of strings
# divergence_values stores a single float for each entry in decoded_strs
divergence_values = df['divergence'].values
loaded_strs = df['decoding'].values
if color_by_divergence:
    all_token_divergences = df['all_token_divergences'].values
    all_token_divergences = [literal_eval(s) for s in all_token_divergences]
    tokenizer = AutoTokenizer.from_pretrained(args.original_tokenizer)

    max_token_divergence = max([max(token_divergences) for token_divergences in all_token_divergences])
    min_token_divergence = min([min(token_divergences) for token_divergences in all_token_divergences])

if args.llm_label_hierarchical_results:
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
                                                       use_openai=args.use_openai, 
                                                       local_embedding_model=args.local_embedding_model, 
                                                       device=args.device,
                                                       recompute_embeddings=args.recompute_embeddings
                                                       )
# Now, cluster the entries of embeddings_list
if not args.hierarchical:
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0).fit(embeddings_list)
    labels = clustering.labels_

    # Calculate the average divergence value of all the texts in each cluster.
    average_divergence_values = {}
    n_texts_in_cluster = {}
    for i in range(n_datapoints):
        cluster = labels[i]
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
        cluster_indices = [i for i, x in enumerate(labels) if x == cluster]
        sorted_indices = sorted(cluster_indices, key=lambda x: divergence_values[x])
        print("Lowest divergence texts:")
        for i in sorted_indices[:n_strs_show]:
            if color_by_divergence:
                outstr = string_with_token_colors(decoded_strs[i], 
                                                  all_token_divergences[i], 
                                                  tokenizer,
                                                  min_score=min_token_divergence,
                                                  max_score=max_token_divergence)
                print(f"Div: {divergence_values[i]:.3f}: {outstr}")
        print("Highest divergence texts:")
        for i in sorted_indices[-n_strs_show:]:
            if color_by_divergence:
                outstr = string_with_token_colors(decoded_strs[i], 
                                                  all_token_divergences[i], 
                                                  tokenizer,
                                                  min_score=min_token_divergence,
                                                  max_score=max_token_divergence)
                print(f"Div: {divergence_values[i]:.3f}: {outstr}")


if not args.no_LM_desc:
    LLM_cluster_labeling(decoded_strs, divergence_values, labels, client, use_openai=args.use_openai, local_labelings_model=args.local_labelings_model, device=args.device)

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
                                                  tokenizer,
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
                                                  tokenizer,
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
            if args.use_openai:
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
                    output = local_model.generate(llm_instruction_ids['input_ids'], do_sample=True, top_p = 0.95, max_new_tokens = 220)
                output = output[0, llm_instruction_ids['input_ids'].shape[1]:]
                #print("output.logits.argmax(dim=2):", output.logits.argmax(dim=2))
                llm_answer = labeling_tokenizer.decode(output.tolist())
                print(answer_preface + " " + llm_answer)



