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
sys.path.append("../interventions/auto_finetune_eval")
from model_comparison_helpers import string_with_token_colors
from auto_finetuning_helpers import load_api_key, extract_df_info

from validated_analysis import get_validated_cluster_labels, match_clusterings, get_validated_contrastive_cluster_labels, validate_cluster_label_comparative_discrimination_power, contrastive_label_double_cluster, validated_assistant_generative_compare


import warnings
warnings.filterwarnings('ignore', message='You have modified the pretrained model configuration to control generation.*')

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str)
    parser.add_argument("--compare_to_path", type=str, default=None)
    parser.add_argument("--compare_to_self", action="store_true")
    parser.add_argument("--api_key_path", type=str, default=None)
    parser.add_argument("--api_provider", type=str, default=None)
    parser.add_argument("--api_model_str", type=str, default="claude-3-haiku-20240307")    
    parser.add_argument("--n_clusters", type=int, default=30)
    parser.add_argument("--n_clusters_compare", type=int, default=None)
    parser.add_argument("--min_cluster_size", type=int, default=7)
    parser.add_argument("--max_cluster_size", type=int, default=2000)
    parser.add_argument("--cluster_method", type=str, default="kmeans")
    parser.add_argument("--clustering_instructions", type=str, default="Identify the topic or theme of the given texts")
    parser.add_argument("--init_clustering_from_base_model", action="store_true")

    parser.add_argument("--n_strs_show", type=int, default=0)
    parser.add_argument("--tqdm_disable", action="store_true")

    parser.add_argument("--skip_p_value_computation", action="store_true")
    parser.add_argument("--use_normal_distribution_for_p_values", action="store_true")
    parser.add_argument("--sampled_texts_per_cluster", type=int, default=10)
    parser.add_argument("--sampled_comparison_texts_per_cluster", type=int, default=10)
    parser.add_argument("--non_cluster_comparison_texts", type=int, default=10)
    parser.add_argument("--generated_labels_per_cluster", type=int, default=3)
    parser.add_argument("--pick_top_n_labels", type=int, default=None)
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
        auth_key = load_api_key(args.api_key_path)
        if args.api_provider == "openai":
            # Authenticate with the OpenAI API
            client = OpenAI(api_key=auth_key)
        else:
            client = None
    else:
        client = None
        auth_key = None
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    # First, load past results into arrays:
    # decoded_strs is an array of strings
    # divergence_values stores a single float for each entry in decoded_strs
    df = pd.read_csv(path)
    divergence_values, decoded_strs, embeddings_list, all_token_divergences, original_tokenizer, max_token_divergence, min_token_divergence = extract_df_info(
        df, 
        args.original_tokenizer, 
        path=path, 
        skip_every_n_decodings=skip_every_n_decodings, 
        color_by_divergence=color_by_divergence, 
        local_embedding_model_str=args.local_embedding_model_str, 
        device=args.device, 
        recompute_embeddings=args.recompute_embeddings, 
        save_embeddings=True, 
        clustering_instructions=args.clustering_instructions, 
        tqdm_disable=args.tqdm_disable, 
        bnb_config=bnb_config
    )
    n_datapoints = len(decoded_strs)
    if compare_to_path is not None:
        df_compare = pd.read_csv(compare_to_path)
        divergence_values_compare, decoded_strs_compare, embeddings_list_compare, all_token_divergences_compare, original_tokenizer_compare, max_token_divergence_compare, min_token_divergence_compare = extract_df_info(
            df_compare, 
            args.original_tokenizer, 
            compare_to_path, 
            skip_every_n_decodings=skip_every_n_decodings_compare, 
            color_by_divergence=color_by_divergence, 
            local_embedding_model_str=args.local_embedding_model_str, 
            device=args.device, 
            recompute_embeddings=args.recompute_embeddings, 
            save_embeddings=True, 
            clustering_instructions=args.clustering_instructions, 
            tqdm_disable=args.tqdm_disable,
            bnb_config=bnb_config
        )
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
        divergence_values_compare, decoded_strs_compare, embeddings_list_compare, all_token_divergences_compare, original_tokenizer_compare, max_token_divergence_compare, min_token_divergence_compare = extract_df_info(
            df_compare, 
            args.original_tokenizer, 
            path, 
            skip_every_n_decodings=skip_every_n_decodings_compare,
            color_by_divergence=color_by_divergence, 
            local_embedding_model_str=args.local_embedding_model_str, 
            device=args.device, 
            recompute_embeddings=args.recompute_embeddings, 
            save_embeddings=True, 
            clustering_instructions=args.clustering_instructions, 
            tqdm_disable=args.tqdm_disable, 
            bnb_config=bnb_config
        )

    # Load the local assistant model (assuming we are not using the API):
    if args.api_key_path is None:
        local_model = AutoModelForCausalLM.from_pretrained(
            args.local_labelings_model_str,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map={"": 0} if args.device == "cuda:0" else "auto"
        )
        labeling_tokenizer = AutoTokenizer.from_pretrained(args.local_labelings_model_str)
        if labeling_tokenizer.pad_token is None:
            labeling_tokenizer.pad_token = labeling_tokenizer.eos_token
            labeling_tokenizer.pad_token_id = labeling_tokenizer.eos_token_id
            local_model.generation_config.pad_token_id = labeling_tokenizer.pad_token_id
            print("PAD TOKEN ID: ", labeling_tokenizer.pad_token_id)
    else:
        local_model = None
        labeling_tokenizer = None
        print("Using API for labeling.")
    
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
                    outstr = string_with_token_colors(
                        decoded_strs[i], 
                        all_token_divergences[i], 
                        original_tokenizer,
                        min_score=min_token_divergence,
                        max_score=max_token_divergence
                    )
                    print(f"Div: {divergence_values[i]:.3f}: {outstr}")
            print("Highest divergence texts:")
            for i in sorted_indices[-n_strs_show:]:
                if color_by_divergence:
                    outstr = string_with_token_colors(
                        decoded_strs[i], 
                        all_token_divergences[i], 
                        original_tokenizer,
                        min_score=min_token_divergence,
                        max_score=max_token_divergence
                    )
                    print(f"Div: {divergence_values[i]:.3f}: {outstr}")
    else:
        validated_clustering = get_validated_cluster_labels(
            decoded_strs, 
            clustering_assignments, 
            local_model, 
            args.api_provider,
            args.api_model_str,
            auth_key,
            labeling_tokenizer, 
            args.device,
            compute_p_values=compute_p_values,
            use_normal_distribution_for_p_values=args.use_normal_distribution_for_p_values,
            sampled_texts_per_cluster=args.sampled_texts_per_cluster,
            sampled_comparison_texts_per_cluster=args.sampled_comparison_texts_per_cluster,
            non_cluster_comparison_texts=args.non_cluster_comparison_texts,
            generated_labels_per_cluster=args.generated_labels_per_cluster,
            num_permutations=args.permutations_for_null,
            pick_top_n_labels=args.pick_top_n_labels
        )
        if compare_to_path is not None or args.compare_to_self:
            print(f"Clustering comparison texts with {args.cluster_method}")
            if args.cluster_method == "kmeans":
                if args.init_clustering_from_base_model:
                    clustering_compare =  KMeans(n_clusters=n_clusters_compare, random_state=0, n_init=10, init=clustering.cluster_centers_).fit(embeddings_list_compare)
                else:
                    clustering_compare =  KMeans(n_clusters=n_clusters_compare, random_state=0, n_init=10).fit(embeddings_list_compare)
            elif args.cluster_method == "hdbscan":
                clustering_compare =  HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size).fit(embeddings_list_compare)
            clustering_assignments_compare = clustering_compare.labels_
            print("Cluster sizes:")
            for cluster in range(max(clustering_assignments_compare) + 1):
                print(f"Cluster {cluster}: {len([i for i in range(n_datapoints_compare) if clustering_assignments_compare[i] == cluster])}")

            validated_clustering_compare = get_validated_cluster_labels(
                decoded_strs_compare, 
                clustering_assignments_compare, 
                local_model, 
                args.api_provider,
                args.api_model_str,
                auth_key,
                labeling_tokenizer, 
                args.device,
                compute_p_values=compute_p_values,
                use_normal_distribution_for_p_values=args.use_normal_distribution_for_p_values,
                sampled_texts_per_cluster=args.sampled_texts_per_cluster,
                sampled_comparison_texts_per_cluster=args.sampled_comparison_texts_per_cluster,
                non_cluster_comparison_texts=args.non_cluster_comparison_texts,
                generated_labels_per_cluster=args.generated_labels_per_cluster,
                num_permutations=args.permutations_for_null,
                pick_top_n_labels=args.pick_top_n_labels
            )
            cluster_matches = match_clusterings(
                clustering_assignments, 
                embeddings_list, 
                clustering_assignments_compare, 
                embeddings_list_compare
            )
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
                    api_provider=args.api_provider,
                    api_model_str=args.api_model_str,
                    auth_key=auth_key,
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
                api_provider=args.api_provider,
                api_model_str=args.api_model_str,
                auth_key=auth_key,
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
                api_provider=args.api_provider,
                api_model_str=args.api_model_str,
                auth_key=auth_key,
                device=args.device,
                compute_p_values=True,
                num_permutations=args.permutations_for_null,
                use_normal_distribution_for_p_values=False,
                sampled_comparison_texts_per_cluster=args.sampled_comparison_texts_per_cluster,
                generated_labels_per_cluster=args.generated_labels_per_cluster,
                pick_top_n_labels=args.pick_top_n_labels
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
            validated_assistant_aucs, validated_assistant_p_values, generated_validation_texts_1, generated_validation_texts_2 = validated_assistant_generative_compare(
                [r[0] for r in result_stats], 
                local_model, 
                labeling_tokenizer, 
                api_provider=args.api_provider, 
                api_model_str=args.api_model_str, 
                auth_key=auth_key, 
                starting_model_str=args.starting_model_str, 
                comparison_model_str=args.comparison_model_str, 
                common_tokenizer_str=args.common_tokenizer_str, 
                device=args.device, 
                num_generated_texts_per_description=args.num_generated_texts_per_description, 
                return_generated_texts=True, 
                bnb_config=bnb_config
            )

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
    