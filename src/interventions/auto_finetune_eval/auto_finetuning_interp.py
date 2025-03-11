"""
This module contains code for applying an interpretability method to compare two models, using the functionality provided by validated_comparison_tools.py
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig, GPTNeoXForCausalLM, Qwen2ForCausalLM
from transformers.utils import logging
from anthropic import Anthropic
from openai import OpenAI
from google.generativeai import GenerativeModel
import random
from sklearn.cluster import KMeans, HDBSCAN
from statsmodels.stats.proportion import proportions_ztest
import numpy as np
import pandas as pd
from terminaltables import AsciiTable
from tqdm import tqdm
import re
import pickle
from auto_finetuning_helpers import plot_comparison_tsne, batch_decode_texts, load_statements_from_MWE_repo
from os import path

from validated_comparison_tools import read_past_embeddings_or_generate_new, match_clusterings, get_validated_contrastive_cluster_labels, validated_assistant_generative_compare, build_contrastive_K_neighbor_similarity_graph, get_cluster_labels_random_subsets, evaluate_label_discrimination, validated_embeddings_discriminative_single_unknown_ICL, attach_cluster_metrics_to_graph, analyze_metric_differences_vs_similarity, analyze_node_metric_vs_neighbor_similarity, analyze_hypothesis_scores_vs_cluster_metrics
from structlog._config import BoundLoggerLazyProxy


def setup_interpretability_method(
        base_model: PreTrainedModel, 
        finetuned_model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer,
        n_decoded_texts: int = 2000, 
        decoding_prefix_file: Optional[str] = None, 
        use_decoding_prefixes_as_cluster_labels: bool = False,
        auth_key: Optional[str] = None,
        client: Optional[Any] = None,
        local_embedding_model_str: Optional[str] = None, 
        local_embedding_api_key: Optional[str] = None,
        init_clustering_from_base_model: bool = False,
        num_decodings_per_prompt: int = None,
        include_prompts_in_decoded_texts: bool = False,
        clustering_instructions: str = "Identify the topic or theme of the given texts",
        n_clustering_inits: int = 10,
        use_prompts_as_clusters: bool = False,
        cluster_on_prompts: bool = False,
        device: str = "cuda:0",
        cluster_method: str = "kmeans",
        n_clusters: int = 30,
        min_cluster_size: int = 7,
        max_cluster_size: int = 2000,
        max_length: int = 32,
        decoding_batch_size: int = 32,
        decoded_texts_save_path: Optional[str] = None,
        decoded_texts_load_path: Optional[str] = None,
        loaded_texts_subsample: Optional[int] = None,
        path_to_MWE_repo: Optional[str] = None,
        num_statements_per_behavior: Optional[int] = None,
        num_responses_per_statement: Optional[int] = None,
        tsne_save_path: Optional[str] = None,
        tsne_title: Optional[str] = None,
        tsne_perplexity: int = 30,
        run_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
    """
    Set up the interpretability method to compare two models.

    This function prepares the necessary components for comparing two models, including
    decoding texts, generating embeddings, and performing clustering. It returns a dictionary
    containing the setup results for further analysis.

    Args:
        base_model (PreTrainedModel): The original, pre-finetuned model.
        finetuned_model (PreTrainedModel): The model after finetuning.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for text generation.
        n_decoded_texts (int): The number of texts to decode with each model.
        decoding_prefix_file (Optional[str]): The path to a file containing a set of prefixes to 
            prepend to the texts to be decoded.
        use_decoding_prefixes_as_cluster_labels (bool): Whether to use the decoding prefixes as cluster labels.
        auth_key (Optional[str]): The API key to use for clustering and analysis.
        client (Optional[Any]): The client object for API calls.
        local_embedding_model_str (Optional[str]): The name of the local embedding model to use.
        local_embedding_api_key (Optional[str]): The API key for the local embedding model.
        init_clustering_from_base_model (bool): Whether to initialize the clustering of the finetuned 
            model from the cluster centers of the base model. Only possible for kmeans clustering.
        num_decodings_per_prompt (int, optional): The number of decodings per prompt we use to generate labels,
            assuming the cluster_ids_to_prompt_ids_to_decoding_ids_dict was provided.
        include_prompts_in_decoded_texts (bool): Whether to include the prompts in the decoded texts. False by default.
        clustering_instructions (str): The instructions to use for clustering.
        n_clustering_inits (int): The number of clustering initializations to use. 10 by default.
        use_prompts_as_clusters (bool): Whether to use the prompts as the clusters. False by default.
        cluster_on_prompts (bool): Whether to cluster on the prompts or the decoded texts. False by default.
        device (str): The device to use for clustering. "cuda:0" by default.
        cluster_method (str): The method to use for clustering. "kmeans" or "hdbscan".
        n_clusters (int): The number of clusters to use. 30 by default.
        min_cluster_size (int): The minimum size of a cluster. 7 by default.
        max_cluster_size (int): The maximum size of a cluster. 2000 by default.
        max_length (int): The maximum length of the decoded texts. 32 by default.
        decoding_batch_size (int): The batch size to use for decoding. 32 by default.
        decoded_texts_save_path (Optional[str]): The path to save the decoded texts to. None by default.
        decoded_texts_load_path (Optional[str]): The path to load the decoded texts from. None by default.
        loaded_texts_subsample (Optional[int]): If specified, will randomly subsample the loaded decoded 
            texts to this number. None by default.
        path_to_MWE_repo (Optional[str]): Path to the Anthropic evals repository.
        num_statements_per_behavior (Optional[int]): Number of statements per behavior to read from the evals 
            repository and then generate responses from.
        num_responses_per_statement (Optional[int]): Number of responses per statement to generate from the statements 
            in the evals repository.
        tsne_save_path (Optional[str]): The path to save the t-SNE plot to. None by default.
        tsne_title (Optional[str]): The title of the t-SNE plot. None by default.
        tsne_perplexity (int): The perplexity of the t-SNE plot. 30 by default.
        run_prefix (Optional[str]): The prefix to use for the run. None by default.

    Returns:
        Dict[str, Any]: A dictionary containing the setup results, including clusterings,
        embeddings, and decoded texts for both models.
    """
    if local_embedding_model_str is None and local_embedding_api_key is None:
        raise ValueError("Either local_embedding_model_str or local_embedding_api_key must be provided.")
    if auth_key is None and client is None:
        raise ValueError("Either auth_key or client must be provided.")
    if path_to_MWE_repo is not None and (num_statements_per_behavior is None or num_responses_per_statement is None):
        raise ValueError("If path_to_MWE_repo is provided, num_statements_per_behavior and num_responses_per_statement must also be provided.")
    if path_to_MWE_repo is not None:
        print("Loading statements from MWE persona repository")
        statements = load_statements_from_MWE_repo(path_to_MWE_repo, num_statements_per_behavior)
        print("Number of statements loaded:", len(statements))
        print("Example statement:", statements[0])
        # Now format the statements into a list of prefixes for decoding
        prefixes = [f"Prompt: {statement}.\nResponse:" for statement in statements]
        texts_decoded_per_prefix = num_responses_per_statement
        n_decoded_texts = None
    else:
        texts_decoded_per_prefix = None
    
            # Initialize all cluster-related variables to None
    (base_loaded_cluster_indices, finetuned_loaded_cluster_indices,
         base_loaded_mauve_cluster_scores, finetuned_loaded_mauve_cluster_scores,
         base_loaded_kl_divergence_cluster, finetuned_loaded_kl_divergence_cluster,
         base_loaded_mean_entropy_cluster, finetuned_loaded_mean_entropy_cluster,
         deduplicated_prompts, deduplicated_prompt_index_to_cluster_indices) = (None,) * 10

    # Load the decoded texts from a prior run
    if decoded_texts_load_path is not None:
        if path_to_MWE_repo is not None:
            # Provided both a prior result from decodings and a repository from another source, raise an error
            raise ValueError("Provided both a prior result from decodings and a repository from another source, please provide only one, as it is unclear which one to use.")
        print("Loading decoded texts from: ", decoded_texts_load_path)
        try:
            decoded_texts = pd.read_csv(decoded_texts_load_path, escapechar='\\')
            print("decoded_texts.shape:", decoded_texts.shape)
            if loaded_texts_subsample is not None:
                decoded_texts = decoded_texts.sample(n=loaded_texts_subsample, random_state=0)
            if include_prompts_in_decoded_texts:
                # Get the prompts and texts from the loaded data
                base_prompts = decoded_texts[decoded_texts["model"] == "base"]["prompt"].tolist()
                finetuned_prompts = decoded_texts[decoded_texts["model"] == "finetuned"]["prompt"].tolist()
                base_texts = decoded_texts[decoded_texts["model"] == "base"]["text"].tolist()
                finetuned_texts = decoded_texts[decoded_texts["model"] == "finetuned"]["text"].tolist()
                
                # Combine prompts and texts with HTML-style tags
                base_decoded_texts = [f"<prompt>{prompt}</prompt><response>{text}</response>" 
                                    for prompt, text in zip(base_prompts, base_texts)]
                finetuned_decoded_texts = [f"<prompt>{prompt}</prompt><response>{text}</response>" 
                                        for prompt, text in zip(finetuned_prompts, finetuned_texts)]
            else:
                base_decoded_texts = decoded_texts[decoded_texts["model"] == "base"]["text"].tolist()
                finetuned_decoded_texts = decoded_texts[decoded_texts["model"] == "finetuned"]["text"].tolist()
            if "cluster_index" in decoded_texts.columns:
                print("Using cluster indices from loaded data")
                # columns will be "model", "text", "cluster_index", "mauve_cluster_score", "kl_divergence_cluster", "mean_entropy_cluster"
                base_loaded_cluster_indices = decoded_texts[decoded_texts["model"] == "base"]["cluster_index"].tolist()
                finetuned_loaded_cluster_indices = decoded_texts[decoded_texts["model"] == "finetuned"]["cluster_index"].tolist()
                base_loaded_mauve_cluster_scores = decoded_texts[decoded_texts["model"] == "base"]["mauve_cluster_score"].tolist()
                finetuned_loaded_mauve_cluster_scores = decoded_texts[decoded_texts["model"] == "finetuned"]["mauve_cluster_score"].tolist()
                base_loaded_kl_divergence_cluster = decoded_texts[decoded_texts["model"] == "base"]["kl_divergence_cluster"].tolist()
                finetuned_loaded_kl_divergence_cluster = decoded_texts[decoded_texts["model"] == "finetuned"]["kl_divergence_cluster"].tolist()
                base_loaded_mean_entropy_cluster = decoded_texts[decoded_texts["model"] == "base"]["mean_entropy_cluster"].tolist()
                finetuned_loaded_mean_entropy_cluster = decoded_texts[decoded_texts["model"] == "finetuned"]["mean_entropy_cluster"].tolist()
                base_loaded_prompts = decoded_texts[decoded_texts["model"] == "base"]["prompt"].tolist()
                finetuned_loaded_prompts = decoded_texts[decoded_texts["model"] == "finetuned"]["prompt"].tolist()

                n_unique_prompts = len(set(base_loaded_prompts))
                n_repeats = int(len(base_loaded_cluster_indices) / n_unique_prompts)
                print("n_unique_prompts:", n_unique_prompts)
                print("n_repeats:", n_repeats)
                deduplicated_prompts = [base_loaded_prompts[i] for i in range(len(base_loaded_prompts)) if i % n_repeats == 0]
                deduplicated_prompt_index_to_cluster_indices = {i: [i * n_repeats + j for j in range(n_repeats)] for i in range(n_unique_prompts)}
            else:
                print("No cluster indices found in loaded data")
        except Exception as e:
            print(f"Error loading decoded texts from {decoded_texts_load_path}: {e}")
            raise e
    else:
        # Load decoding prefixes
        if decoding_prefix_file and path.exists(decoding_prefix_file):
            if path_to_MWE_repo is not None:
                # Provided both a decoding prefix file and a repository from another source, raise an error
                raise ValueError("Provided both a decoding prefix file and a repository from another source, please provide only one, as it is unclear which one to use.")
            with open(decoding_prefix_file, 'r') as f:
                prefixes = [line.strip() for line in f.readlines()]
        elif path_to_MWE_repo is None:
            # Didn't load any prefixes, so we will be decoding from the empty string
            prefixes = None
            print("Decoding from empty string.")

        if isinstance(base_model, GPTNeoXForCausalLM) or isinstance(base_model, Qwen2ForCausalLM):
            logging.set_verbosity_error()

        # Decode texts with both models
        base_decoded_texts = batch_decode_texts(
            base_model, 
            tokenizer, 
            prefixes, 
            n_decoded_texts=n_decoded_texts,
            texts_decoded_per_prefix=texts_decoded_per_prefix,
            max_length=max_length,
            batch_size=decoding_batch_size
        )
        finetuned_decoded_texts = batch_decode_texts(
            finetuned_model, 
            tokenizer, 
            prefixes, 
            n_decoded_texts=n_decoded_texts,
            texts_decoded_per_prefix=texts_decoded_per_prefix,
            max_length=max_length,
            batch_size=decoding_batch_size
        )

    if decoded_texts_save_path is not None:
        # Create a list of tuples containing the model indicator and the decoded text
        combined_texts = [('base', text) for text in base_decoded_texts] + [('finetuned', text) for text in finetuned_decoded_texts]
        
        # Create a DataFrame
        df = pd.DataFrame(combined_texts, columns=['model', 'text'])
        
        # Save the DataFrame as a CSV file without an index
        df.to_csv(decoded_texts_save_path, index=False, escapechar='\\')
        
        print(f"Decoded texts saved to: {decoded_texts_save_path}")

    # Print out 50 randomly sampled decoded texts
    print("Base decoded texts:")
    for i, t in enumerate(random.sample(base_decoded_texts, k=min(50, len(base_decoded_texts)))):
        print(f"- {i}: {t}")
    print("Finetuned decoded texts:")
    for i, t in enumerate(random.sample(finetuned_decoded_texts, k=min(50, len(finetuned_decoded_texts)))):
        print(f"- {i}: {t}")

    # Generate embeddings for both sets of decoded texts
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    if run_prefix is not None:
        embeddings_save_str = run_prefix + "_model"
    else:
        embeddings_save_str = "model"

    if cluster_on_prompts:
        prompt_embeddings = read_past_embeddings_or_generate_new(
            "pkl_embeddings/prompt_" + embeddings_save_str,
            None,
            deduplicated_prompts,
            local_embedding_model_str=local_embedding_model_str,
            device=device,
            recompute_embeddings=False,
            save_embeddings=True,
            clustering_instructions=clustering_instructions,
            bnb_config=bnb_config
        )
        prompt_embeddings = np.array(prompt_embeddings)
        base_embeddings = prompt_embeddings
        finetuned_embeddings = prompt_embeddings
    else:
        base_embeddings = read_past_embeddings_or_generate_new(
            "pkl_embeddings/base_" + embeddings_save_str,
            None,
            base_decoded_texts,
            local_embedding_model_str=local_embedding_model_str,
            device=device,
            recompute_embeddings=False,
            save_embeddings=True,
            clustering_instructions=clustering_instructions,
            bnb_config=bnb_config
        )
        finetuned_embeddings = read_past_embeddings_or_generate_new(
            "pkl_embeddings/finetuned_" + embeddings_save_str,
            None,
            finetuned_decoded_texts,
            local_embedding_model_str=local_embedding_model_str,
            device=device,
            recompute_embeddings=False,
            save_embeddings=True,
            clustering_instructions=clustering_instructions,
            bnb_config=bnb_config
        )

        # If we loaded cluster indices from a previous run, use them and compute the cluster centers as the mean of the embeddings for each cluster
        base_embeddings = np.array(base_embeddings)
        finetuned_embeddings = np.array(finetuned_embeddings)
    if decoded_texts_load_path is not None and base_loaded_cluster_indices is not None and finetuned_loaded_cluster_indices is not None and not cluster_on_prompts:
        base_clustering_assignments = np.array(base_loaded_cluster_indices)
        finetuned_clustering_assignments = np.array(finetuned_loaded_cluster_indices)

        print("base_clustering_assignments:", base_clustering_assignments)
        print("type(base_clustering_assignments):", type(base_clustering_assignments))
        print("base_clustering_assignments == 0:", base_clustering_assignments == 0)
        print("base_embeddings.shape:", base_embeddings.shape)
        
        n_base_clusters = len(set(base_clustering_assignments))
        n_finetuned_clusters = len(set(finetuned_clustering_assignments))
        
        cluster_centers_base = np.array([
            np.mean(base_embeddings[base_clustering_assignments == i], axis=0)
            for i in range(n_base_clusters)
        ])
        cluster_centers_finetuned = np.array([
            np.mean(finetuned_embeddings[finetuned_clustering_assignments == i], axis=0)
            for i in range(n_finetuned_clusters)
        ])
        # Create mock clustering objects to match the expected format
        base_clustering = type('MockClustering', (), {
            'labels_': base_clustering_assignments,
            'cluster_centers_': cluster_centers_base
        })
        finetuned_clustering = type('MockClustering', (), {
            'labels_': finetuned_clustering_assignments,
            'cluster_centers_': cluster_centers_finetuned
        })
    elif use_prompts_as_clusters:
        # Each prompt represents a cluster, and the decoded texts are assigned to the cluster of their prompt
        # Cluster centroids are computed as the mean of the embeddings of the decoded texts assigned to each prompt
        if decoded_texts_load_path is None and path_to_MWE_repo is None:
            raise ValueError("use_prompts_as_clusters requires either decoded_texts_load_path or path_to_MWE_repo to be set")
        
        if path_to_MWE_repo is not None:
            # We know how many statements/prompts we have and how many responses per statement
            n_clusters = len(prefixes)
            
            # Create cluster assignments where all decodings from the same prompt go to the same cluster
            base_clustering_assignments = np.array([i // num_responses_per_statement for i in range(len(base_decoded_texts))])
            finetuned_clustering_assignments = np.array([i // num_responses_per_statement for i in range(len(finetuned_decoded_texts))])
            
            # Track prompt-to-cluster mapping
            deduplicated_prompts = prefixes
            deduplicated_prompt_index_to_cluster_indices = {
                i: [i * num_responses_per_statement + j for j in range(num_responses_per_statement)]
                for i in range(n_clusters)
            }
        else:
            # Working with loaded data that has prompt information
            if "prompt" not in decoded_texts.columns:
                raise ValueError("use_prompts_as_clusters requires prompt column in loaded data")
            
            # Get unique prompts
            base_prompts = decoded_texts[decoded_texts["model"] == "base"]["prompt"].tolist()
            finetuned_prompts = decoded_texts[decoded_texts["model"] == "finetuned"]["prompt"].tolist()
            
            # Create mapping from unique prompts to indices
            unique_prompts = list(set(base_prompts))
            prompt_to_id = {prompt: i for i, prompt in enumerate(unique_prompts)}
            n_clusters = len(unique_prompts)
            
            # Create cluster assignments
            base_clustering_assignments = np.array([prompt_to_id[prompt] for prompt in base_prompts])
            finetuned_clustering_assignments = np.array([prompt_to_id[prompt] for prompt in finetuned_prompts])
            
            # Track prompt-to-cluster mapping
            deduplicated_prompts = unique_prompts
            deduplicated_prompt_index_to_cluster_indices = {}
            for i, prompt in enumerate(unique_prompts):
                deduplicated_prompt_index_to_cluster_indices[i] = [
                    j for j, p in enumerate(base_prompts) if p == prompt
                ]
        
        # Calculate cluster centers as the mean of embeddings for each cluster
        cluster_centers_base = np.array([
            np.mean(base_embeddings[base_clustering_assignments == i], axis=0)
            for i in range(n_clusters)
        ])
        cluster_centers_finetuned = np.array([
            np.mean(finetuned_embeddings[finetuned_clustering_assignments == i], axis=0)
            for i in range(n_clusters)
        ])
        
        # Create mock clustering objects
        base_clustering = type('MockClustering', (), {
            'labels_': base_clustering_assignments,
            'cluster_centers_': cluster_centers_base
        })
        finetuned_clustering = type('MockClustering', (), {
            'labels_': finetuned_clustering_assignments,
            'cluster_centers_': cluster_centers_finetuned
        })
        
        print(f"Created {n_clusters} clusters based on prompts")
        print(f"Base cluster sizes: {[np.sum(base_clustering_assignments == i) for i in range(n_clusters)]}")
        print(f"Finetuned cluster sizes: {[np.sum(finetuned_clustering_assignments == i) for i in range(n_clusters)]}")
    
    elif cluster_on_prompts:
        # Perform clustering on the prompt embeddings, then derive the cluster assignments for the decoded texts from the prompt clustering
        # base and finetuned texts are assigned to the same cluster if their prompts are assigned to the same cluster
        prompt_clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init=n_clustering_inits).fit(prompt_embeddings)
        
        # Now we derive the cluster assignments for the decoded texts from the prompt clustering
        # Then create mock clustering objects to match the expected format
        # First, derive the cluster assignments: a list that maps each decoding to the cluster of its prompt
        base_decoding_to_prompt_cluster = []
        finetuned_decoding_to_prompt_cluster = []
        for i in range(n_unique_prompts):
            current_prompt_cluster = prompt_clustering.labels_[i]
            for _ in deduplicated_prompt_index_to_cluster_indices[i]:
                base_decoding_to_prompt_cluster.append(current_prompt_cluster)
                finetuned_decoding_to_prompt_cluster.append(current_prompt_cluster)
        base_clustering_assignments = np.array(base_decoding_to_prompt_cluster)
        finetuned_clustering_assignments = np.array(finetuned_decoding_to_prompt_cluster)

        # We also set each decoded text's embedding to be the embedding of its prompt
        base_embeddings = prompt_embeddings[base_clustering_assignments]
        finetuned_embeddings = prompt_embeddings[finetuned_clustering_assignments]

        # And compute the cluster centers as the mean of the embeddings for each cluster
        cluster_centers_base = np.array([
            np.mean(base_embeddings[base_clustering_assignments == i], axis=0)
            for i in range(n_clusters)
        ])
        cluster_centers_finetuned = np.array([
            np.mean(finetuned_embeddings[finetuned_clustering_assignments == i], axis=0)
            for i in range(n_clusters)
        ])
        
        base_clustering = type('MockClustering', (), {
            'labels_': base_clustering_assignments,
            'cluster_centers_': cluster_centers_base
        })
        finetuned_clustering = type('MockClustering', (), {
            'labels_': finetuned_clustering_assignments,
            'cluster_centers_': cluster_centers_finetuned
        })
        
    else:
        # Perform clustering on both sets of embeddings (if not using decoding prefixes as cluster labels)
        if not use_decoding_prefixes_as_cluster_labels:
            if cluster_method == "kmeans":
                base_clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init=n_clustering_inits).fit(base_embeddings)
                if init_clustering_from_base_model:
                    initial_centroids = base_clustering.cluster_centers_
                    finetuned_clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init=1, init=initial_centroids).fit(finetuned_embeddings)
                else:
                    finetuned_clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init=n_clustering_inits).fit(finetuned_embeddings)
            elif cluster_method == "hdbscan":
                base_clustering = HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size).fit(base_embeddings)
                finetuned_clustering = HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size).fit(finetuned_embeddings)
        
            print("Found", len(set(base_clustering.labels_)), "clusters for base model")
            print("Found", len(set(finetuned_clustering.labels_)), "clusters for finetuned model")
        
        else:
            # Create cluster assignments from decoding prefixes, matching the expected format
            texts_per_prefix = n_decoded_texts // len(prefixes)
            
            # Create cluster assignments based on prefix order
            base_clustering_assignments = np.array([i // texts_per_prefix for i in range(n_decoded_texts)])
            finetuned_clustering_assignments = np.array([i // texts_per_prefix for i in range(n_decoded_texts)])
            
            # Calculate cluster centers as the mean of embeddings for each cluster
            cluster_centers_base = np.array([
                np.mean(base_embeddings[base_clustering_assignments == i], axis=0)
                for i in range(len(prefixes))
            ])
            cluster_centers_finetuned = np.array([
                np.mean(finetuned_embeddings[finetuned_clustering_assignments == i], axis=0)
                for i in range(len(prefixes))
            ])
            
            # Create mock clustering objects to match the expected format
            base_clustering = type('MockClustering', (), {
                'labels_': base_clustering_assignments,
                'cluster_centers_': cluster_centers_base
            })
            finetuned_clustering = type('MockClustering', (), {
                'labels_': finetuned_clustering_assignments,
                'cluster_centers_': cluster_centers_finetuned
            })
    
        base_clustering_assignments = base_clustering.labels_
        finetuned_clustering_assignments = finetuned_clustering.labels_
        print("base_clustering_assignments:", base_clustering_assignments)
        print("finetuned_clustering_assignments:", finetuned_clustering_assignments)
        cluster_centers_base = base_clustering.cluster_centers_ if cluster_method == "kmeans" or init_clustering_from_base_model else None
        cluster_centers_finetuned = finetuned_clustering.cluster_centers_ if cluster_method == "kmeans" or init_clustering_from_base_model else None

    # (Optional) Perform t-SNE dimensionality reduction on the combined embeddings and color by model. Save the plot as a PDF.
    if tsne_save_path is not None: 
        plot_comparison_tsne(
            base_embeddings, 
            finetuned_embeddings, 
            tsne_save_path, 
            tsne_title, 
            tsne_perplexity,
            base_cluster_centers=cluster_centers_base,
            finetuned_cluster_centers=cluster_centers_finetuned
        )
        
    
    # Now we construct the cluster_ids_to_prompt_ids_to_decoding_ids_dict 1 and 2 to keep track of which decodings belong to which prompts and which prompts belong to which clusters, for the base and finetuned models respectively.
    # cluster_ids_to_prompt_ids_to_decoding_ids_dict_1 and 2 (Dict, optional): Nested dict. First dict is indexed by cluster id.
    # Leads to a dict indexed by prompt id, which leads to a list of indices for where the decodings of that prompt can be found in decoded_strs.
    if num_decodings_per_prompt is not None:
        # Initialize the nested dictionaries
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_1 = {}
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_2 = {}
        
        # Check if we have data loaded from a file with prompt information
        if decoded_texts_load_path is not None and "cluster_index" in decoded_texts.columns and "prompt" in decoded_texts.columns:
            # Process base model data
            for cluster_id in set(base_clustering_assignments):
                cluster_ids_to_prompt_ids_to_decoding_ids_dict_1[cluster_id] = {}
                
                # Get all indices that belong to this cluster
                cluster_indices = [i for i, c in enumerate(base_clustering_assignments) if c == cluster_id]
                
                # Group by prompt
                prompt_to_indices = {}
                for idx in cluster_indices:
                    prompt = base_loaded_prompts[idx]
                    if prompt not in prompt_to_indices:
                        prompt_to_indices[prompt] = []
                    prompt_to_indices[prompt].append(idx)
                
                # For each prompt, limit to num_decodings_per_prompt
                for prompt_id, prompt in enumerate(prompt_to_indices.keys()):
                    if prompt_to_indices[prompt]:
                        # Randomly select up to num_decodings_per_prompt indices
                        selected_indices = prompt_to_indices[prompt]
                        if len(selected_indices) > num_decodings_per_prompt:
                            selected_indices = random.sample(selected_indices, num_decodings_per_prompt)
                        cluster_ids_to_prompt_ids_to_decoding_ids_dict_1[cluster_id][prompt_id] = selected_indices
            
            # Process finetuned model data - similar to base model
            for cluster_id in set(finetuned_clustering_assignments):
                cluster_ids_to_prompt_ids_to_decoding_ids_dict_2[cluster_id] = {}
                
                # Get all indices that belong to this cluster
                cluster_indices = [i for i, c in enumerate(finetuned_clustering_assignments) if c == cluster_id]
                
                # Group by prompt
                prompt_to_indices = {}
                for idx in cluster_indices:
                    prompt = finetuned_loaded_prompts[idx]
                    if prompt not in prompt_to_indices:
                        prompt_to_indices[prompt] = []
                    prompt_to_indices[prompt].append(idx)
                
                # For each prompt, limit to num_decodings_per_prompt
                for prompt_id, prompt in enumerate(prompt_to_indices.keys()):
                    if prompt_to_indices[prompt]:
                        # Randomly select up to num_decodings_per_prompt indices
                        selected_indices = prompt_to_indices[prompt]
                        if len(selected_indices) > num_decodings_per_prompt:
                            selected_indices = random.sample(selected_indices, num_decodings_per_prompt)
                        cluster_ids_to_prompt_ids_to_decoding_ids_dict_2[cluster_id][prompt_id] = selected_indices
        
        # Handle case where we have deduplicated prompts data structure (from cluster_on_prompts)
        elif cluster_on_prompts and deduplicated_prompts is not None and deduplicated_prompt_index_to_cluster_indices is not None:
            # Process base model data using the deduplicated prompt information
            for cluster_id in set(base_clustering_assignments):
                cluster_ids_to_prompt_ids_to_decoding_ids_dict_1[cluster_id] = {}
                
                # Find all prompts that belong to this cluster
                prompt_indices_in_cluster = [i for i in range(len(deduplicated_prompts)) 
                                            if any(base_clustering_assignments[idx] == cluster_id 
                                                  for idx in deduplicated_prompt_index_to_cluster_indices[i])]
                
                # For each prompt, store its decodings that fall in this cluster
                for prompt_idx in prompt_indices_in_cluster:
                    decoding_indices = [idx for idx in deduplicated_prompt_index_to_cluster_indices[prompt_idx] 
                                       if base_clustering_assignments[idx] == cluster_id]
                    
                    if decoding_indices:
                        # Limit to num_decodings_per_prompt
                        if len(decoding_indices) > num_decodings_per_prompt:
                            decoding_indices = random.sample(decoding_indices, num_decodings_per_prompt)
                        cluster_ids_to_prompt_ids_to_decoding_ids_dict_1[cluster_id][prompt_idx] = decoding_indices
            
            # Process finetuned model data similarly
            for cluster_id in set(finetuned_clustering_assignments):
                cluster_ids_to_prompt_ids_to_decoding_ids_dict_2[cluster_id] = {}
                
                # Find all prompts that belong to this cluster
                prompt_indices_in_cluster = [i for i in range(len(deduplicated_prompts)) 
                                            if any(finetuned_clustering_assignments[idx] == cluster_id 
                                                  for idx in deduplicated_prompt_index_to_cluster_indices[i])]
                
                # For each prompt, store its decodings that fall in this cluster
                for prompt_idx in prompt_indices_in_cluster:
                    decoding_indices = [idx for idx in deduplicated_prompt_index_to_cluster_indices[prompt_idx] 
                                       if finetuned_clustering_assignments[idx] == cluster_id]
                    
                    if decoding_indices:
                        # Limit to num_decodings_per_prompt
                        if len(decoding_indices) > num_decodings_per_prompt:
                            decoding_indices = random.sample(decoding_indices, num_decodings_per_prompt)
                        cluster_ids_to_prompt_ids_to_decoding_ids_dict_2[cluster_id][prompt_idx] = decoding_indices
        
        # Case where we don't have prompt information (give a warning)
        else:
            print("Warning: num_decodings_per_prompt was specified but no prompt information is available.")
            print("Either load decoded texts with prompt information or use cluster_on_prompts=True.")
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_1 = None
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_2 = None
    else:
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_1 = None
        cluster_ids_to_prompt_ids_to_decoding_ids_dict_2 = None

    setup = {
        "base_clustering": base_clustering,
        "finetuned_clustering": finetuned_clustering,
        "base_clustering_assignments": base_clustering_assignments,
        "finetuned_clustering_assignments": finetuned_clustering_assignments,
        "base_embeddings": base_embeddings,
        "finetuned_embeddings": finetuned_embeddings,
        "base_decoded_texts": base_decoded_texts,
        "finetuned_decoded_texts": finetuned_decoded_texts,
        "bnb_config": bnb_config,
        "base_loaded_mauve_cluster_scores": base_loaded_mauve_cluster_scores,
        "finetuned_loaded_mauve_cluster_scores": finetuned_loaded_mauve_cluster_scores,
        "base_loaded_kl_divergence_cluster": base_loaded_kl_divergence_cluster,
        "finetuned_loaded_kl_divergence_cluster": finetuned_loaded_kl_divergence_cluster,
        "base_loaded_mean_entropy_cluster": base_loaded_mean_entropy_cluster,
        "finetuned_loaded_mean_entropy_cluster": finetuned_loaded_mean_entropy_cluster,
        "cluster_ids_to_prompt_ids_to_decoding_ids_dict_1": cluster_ids_to_prompt_ids_to_decoding_ids_dict_1,
        "cluster_ids_to_prompt_ids_to_decoding_ids_dict_2": cluster_ids_to_prompt_ids_to_decoding_ids_dict_2
    }
    return setup

def apply_interpretability_method(
        base_model: PreTrainedModel, 
        finetuned_model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer,
        n_decoded_texts: int = 2000, 
        decoding_prefix_file: Optional[str] = None, 
        api_provider: str = "anthropic",
        api_model_str: str = "claude-3-haiku-20240307",
        auth_key: Optional[str] = None,
        client: Optional[Any] = None,
        local_embedding_model_str: Optional[str] = None, 
        local_embedding_api_key: Optional[str] = None,
        init_clustering_from_base_model: bool = False,
        num_decodings_per_prompt: int = None,
        clustering_instructions: str = "Identify the topic or theme of the given texts",
        n_clustering_inits: int = 10,
        cluster_on_prompts: bool = False,
        device: str = "cuda:0",
        cluster_method: str = "kmeans",
        n_clusters: int = 30,
        min_cluster_size: int = 7,
        max_cluster_size: int = 2000,
        max_length: int = 32,
        decoding_batch_size: int = 32,
        decoded_texts_save_path: Optional[str] = None,
        decoded_texts_load_path: Optional[str] = None,
        loaded_texts_subsample: Optional[int] = None,
        num_rephrases_for_validation: int = 0,
        use_unitary_comparisons: bool = False,
        tsne_save_path: Optional[str] = None,
        tsne_title: Optional[str] = None,
        tsne_perplexity: int = 30,
        api_interactions_save_loc: Optional[str] = None,
        logger: Optional[BoundLoggerLazyProxy] = None,
        run_prefix: Optional[str] = None,
        metric: str = "acc"
    ) -> List[str]:
    """
    Real implementation of applying an interpretability method to compare two models.

    This function first decodes a text corpus with both models, then clusters the decoded outputs, and then performs pairwise matching of the two sets of clusters. It then feeds an assistant LLM texts from matched cluster pairs to generate hypotheses about how the models differ, and validates those hypotheses automatically by testing that the LLM assistant can use the hypotheses to differentiate between texts from the two models. It returns the validated hypotheses as a list of strings.

    Args:
        base_model (PreTrainedModel): The original, pre-finetuned model.
        finetuned_model (PreTrainedModel): The model after finetuning.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for text generation.
        n_decoded_texts (int): The number of texts to decode with each model.
        decoding_prefix_file (Optional[str]): The path to a file containing a set of prefixes to 
            prepend to the texts to be decoded.
        api_provider (str): The API provider to use for clustering and analysis.
        api_model_str (str): The API model to use for clustering and analysis.
        auth_key (Optional[str]): The API key to use for clustering and analysis.
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]]): The client to use for the API request.
        local_embedding_model_str (Optional[str]): The name of the local embedding model to use.
        local_embedding_api_key (Optional[str]): The API key for the local embedding model.
        init_clustering_from_base_model (bool): Whether to initialize the clustering of the finetuned 
            model from the cluster centers of the base model. Only possible for kmeans clustering.
        num_decodings_per_prompt (int, optional): The number of decodings per prompt we use to generate labels,
            assuming the cluster_ids_to_prompt_ids_to_decoding_ids_dict was provided.
        clustering_instructions (str): The instructions to use for clustering.
        n_clustering_inits (int): The number of clustering initializations to use. 10 by default.
        cluster_on_prompts (bool): Whether to cluster on the prompts or the decoded texts. False by default.
        device (str): The device to use for clustering. "cuda:0" by default.
        cluster_method (str): The method to use for clustering. "kmeans" or "hdbscan".
        n_clusters (int): The number of clusters to use. 30 by default.
        min_cluster_size (int): The minimum size of a cluster. 7 by default.
        max_cluster_size (int): The maximum size of a cluster. 2000 by default.
        max_length (int): The maximum length of the decoded texts. 32 by default.
        decoding_batch_size (int): The batch size to use for decoding. 32 by default.
        decoded_texts_save_path (Optional[str]): The path to save the decoded texts to. None by default.
        decoded_texts_load_path (Optional[str]): The path to load the decoded texts from. None by default.
        loaded_texts_subsample (Optional[int]): If specified, will randomly subsample the loaded decoded 
            texts to this number. None by default.
        num_rephrases_for_validation (int): The number of rephrases of each generated hypothesis to 
            generate for validation. 0 by default.
        use_unitary_comparisons (bool): Whether to use unitary comparisons, i.e. test labels
            by their ability to let the assistant determine which cluster a given text belongs to, 
            without considering another text for comparison. Defaults to False.        
        tsne_save_path (Optional[str]): The path to save the t-SNE plot to. None by default.
        tsne_title (Optional[str]): The title of the t-SNE plot. None by default.
        tsne_perplexity (int): The perplexity of the t-SNE plot. 30 by default.
        api_interactions_save_loc (Optional): Where to store interations with the API model, if anywhere.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        run_prefix (Optional[str]): The prefix to use for the run. None by default.
        metric (str): The metric to use for label validation. Defaults to "acc".
    Returns:
        List[str]: A list of validated hypotheses about how the models differ.
    """
    setup = setup_interpretability_method(
        base_model=base_model, 
        finetuned_model=finetuned_model, 
        tokenizer=tokenizer,
        n_decoded_texts=n_decoded_texts, 
        decoding_prefix_file=decoding_prefix_file, 
        auth_key=auth_key,
        client=client,
        local_embedding_model_str=local_embedding_model_str, 
        local_embedding_api_key=local_embedding_api_key,
        init_clustering_from_base_model=init_clustering_from_base_model,
        num_decodings_per_prompt=num_decodings_per_prompt,
        clustering_instructions=clustering_instructions,
        n_clustering_inits=n_clustering_inits,
        cluster_on_prompts=cluster_on_prompts,
        device=device,
        cluster_method=cluster_method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        max_length=max_length,
        decoding_batch_size=decoding_batch_size,
        decoded_texts_save_path=decoded_texts_save_path,
        decoded_texts_load_path=decoded_texts_load_path,
        loaded_texts_subsample=loaded_texts_subsample,
        tsne_save_path=tsne_save_path,
        tsne_title=tsne_title,
        tsne_perplexity=tsne_perplexity,
        run_prefix=run_prefix
    )

    base_embeddings = setup["base_embeddings"]
    finetuned_embeddings = setup["finetuned_embeddings"]
    base_clustering_assignments = setup["base_clustering_assignments"]
    finetuned_clustering_assignments = setup["finetuned_clustering_assignments"]
    base_decoded_texts = setup["base_decoded_texts"]
    finetuned_decoded_texts = setup["finetuned_decoded_texts"]
    bnb_config = setup["bnb_config"]

    # Match clusters between base and finetuned models
    cluster_matches = match_clusterings(base_clustering_assignments, base_embeddings, finetuned_clustering_assignments, finetuned_embeddings)

    # Generate and validate contrastive labels
    contrastive_labels_results = get_validated_contrastive_cluster_labels(
        decoded_strs_1=base_decoded_texts,
        clustering_assignments_1=base_clustering_assignments,
        decoded_strs_2=finetuned_decoded_texts,
        clustering_assignments_2=finetuned_clustering_assignments,
        cluster_matches=cluster_matches,
        local_model=None,
        labeling_tokenizer=None,
        api_provider=api_provider,
        api_model_str=api_model_str,
        auth_key=auth_key,
        client=client,
        device=device,
        compute_p_values=True,
        use_normal_distribution_for_p_values=True,
        sampled_comparison_texts_per_cluster=10,
        n_head_to_head_comparisons_per_text=3,
        generated_labels_per_cluster=2,
        pick_top_n_labels=1,
        use_unitary_comparisons=use_unitary_comparisons,
        api_interactions_save_loc=api_interactions_save_loc,
        logger=logger,
        metric=metric
    )

    cluster_pair_scores = contrastive_labels_results["cluster_pair_scores"]
    p_values = contrastive_labels_results["p_values"]
    metric_str = "accuracy" if metric == "acc" else "AUC"

    # Generate texts based on contrastive labels and validate
    hypotheses = []
    for cluster_pair, label_scores in cluster_pair_scores.items():
        for label, score in label_scores.items():
            p_value = p_values[cluster_pair][label]
            hypothesis_description = f"\n\nCluster pair: {cluster_pair}\n{metric_str}: {score:.3f}\nP-value: {p_value:.5f}\nLabel: {label}"
            print(hypothesis_description)
            hypotheses.append(label)
        cluster_id_1, cluster_id_2 = cluster_pair
        
        # Print 5 random texts from each cluster in the pair
        print(f"\n\nCluster pair: {cluster_pair}")
        print("5 random texts from cluster 1:")
        cluster_1_texts = [text for text, cluster in zip(base_decoded_texts, base_clustering_assignments) if cluster == cluster_id_1]
        for text in random.sample(cluster_1_texts, min(5, len(cluster_1_texts))):
            print(f"- {text}")
        
        print("\n5 random texts from cluster 2:")
        cluster_2_texts = [text for text, cluster in zip(finetuned_decoded_texts, finetuned_clustering_assignments) if cluster == cluster_id_2]
        for text in random.sample(cluster_2_texts, min(5, len(cluster_2_texts))):
            print(f"- {text}")
        
        print("*" * 50)
        

    # Validate hypotheses using generated texts
    validated_results = validated_assistant_generative_compare(
        hypotheses,
        None,
        None,
        api_provider=api_provider,
        api_model_str=api_model_str,
        auth_key=auth_key,
        api_stronger_model_str=None,
        client=client,
        starting_model_str=None,
        comparison_model_str=None,
        common_tokenizer_str=base_model.name_or_path,
        starting_model=base_model,
        comparison_model=finetuned_model,
        device=device,
        use_normal_distribution_for_p_values=True,
        num_generated_texts_per_description=10,
        num_rephrases_for_validation=num_rephrases_for_validation,
        bnb_config=bnb_config,
        api_interactions_save_loc=api_interactions_save_loc,
        logger=logger
    )

    all_validated_metric_scores, all_validated_p_values, all_validated_hypotheses = validated_results

    # Filter and format final hypotheses
    final_hypotheses = []
    print("all_validated_hypotheses", all_validated_hypotheses)
    print("all_validated_metric_scores", all_validated_metric_scores)
    print("all_validated_p_values", all_validated_p_values)
    for i, hypothesis_and_rephrases in enumerate(all_validated_hypotheses):
        if True: #all_validated_metric_scores[i][0] > 0.6 and all_validated_p_values[i][0] < 0.1:
            final_hypothesis = f"{hypothesis_and_rephrases[0]} (Validated {metric_str}: {all_validated_metric_scores[i][0]:.4f}, Validated P-value: {all_validated_p_values[i][0]:.4f})"
            final_hypotheses.append(final_hypothesis)
            # Print out the hypothesis, its rephrases, and the AUCs and P-values for each
            for j, rephrase in enumerate(hypothesis_and_rephrases):
                print("*" * 50)
                print(f"Hypothesis {i}, Rephrase {j}: {rephrase} \n\n(Validated {metric_str}: {all_validated_metric_scores[i][j]:.4f}\n Validated P-value: {all_validated_p_values[i][j]:.4f})")
            
            print("*" * 50)
            print("*" * 50)
            print("\n\n")
    return final_hypotheses


def get_individual_labels(
    decoded_strs: List[str],
    clustering_assignments: List[int],
    local_model: PreTrainedModel,
    labeling_tokenizer: PreTrainedTokenizer,
    api_provider: str,
    api_model_str: str,
    api_stronger_model_str: Optional[str] = None,
    auth_key: Optional[str] = None,
    client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
    device: str = "cuda:0",
    sampled_texts_per_cluster: int = 10,
    generated_labels_per_cluster: int = 3,
    cluster_ids_to_prompt_ids_to_decoding_ids_dict: Dict = None,
    num_decodings_per_prompt: int = None,
    single_cluster_label_instruction: Optional[str] = None,
    max_unitary_comparisons_per_label: int = 50,
    api_interactions_save_loc: Optional[str] = None,
    logger: Optional[BoundLoggerLazyProxy] = None,
    metric: str = "acc"
) -> Dict[int, Tuple[str, float]]:
    """
    Generate and evaluate individual labels for each cluster.

    This function generates labels for each cluster using a subset of texts,
    then evaluates the ability of those labels to discriminate between the 
    cluster they are generated for and other clusters using accuracy / AUC scores.
    It returns the best-performing label for each cluster along with its accuracy / AUC score.

    Args:
        decoded_strs (List[str]): List of decoded texts.
        clustering_assignments (List[int]): Cluster assignments for each text.
        local_model (PreTrainedModel): The model used for labeling.
        labeling_tokenizer (PreTrainedTokenizer): The tokenizer for the labeling model.
        api_provider (str): The API provider for label generation.
        api_model_str (str): The specific API model to use.
        api_stronger_model_str (Optional[str]): The specific API model to use for the stronger model. Can be used to generate labels.
        auth_key (Optional[str]): Authentication key for the API.
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]]): The client to use for the API request.
        device (str): The device to use for computations.
        sampled_texts_per_cluster (int): Number of texts to sample per cluster for label generation.
        generated_labels_per_cluster (int): Number of labels to generate per cluster.
        cluster_ids_to_prompt_ids_to_decoding_ids_dict (Dict, optional): Nested dict. First dict is indexed by cluster id.
            Leads to a dict indexed by prompt id, which leads to a list of indices for where the decodings of that
            prompt can be found in decoded_strs. Can be provided to make the label generation select only num_decodings_per_prompt
            decodings per prompt to base the labels off of.
        num_decodings_per_prompt (int): Number of decodings per prompt to use for label generation.
        single_cluster_label_instruction (Optional[str]): Instructions for generating the single cluster labels.
        max_unitary_comparisons_per_label (int): Maximum number of unitary comparisons to perform per label.
        api_interactions_save_loc (Optional[str]): Where to store interactions with the API model, if anywhere.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        metric (str): The metric to use for label validation. Defaults to "acc".
    Returns:
        Dict[int, Tuple[str, float]]: A dictionary mapping cluster IDs to tuples of
        (best_label, best_metric_score).
    """
    cluster_labels, _ = get_cluster_labels_random_subsets(
        decoded_strs,
        clustering_assignments,
        local_model,
        labeling_tokenizer,
        api_provider,
        api_model_str if api_stronger_model_str is None else api_stronger_model_str,
        auth_key=auth_key,
        client=client,
        device=device,
        sampled_texts_per_cluster=sampled_texts_per_cluster,
        generated_labels_per_cluster=generated_labels_per_cluster,
        cluster_ids_to_prompt_ids_to_decoding_ids_dict=cluster_ids_to_prompt_ids_to_decoding_ids_dict,
        num_decodings_per_prompt=num_decodings_per_prompt,
        single_cluster_label_instruction=single_cluster_label_instruction,
        api_interactions_save_loc=api_interactions_save_loc,
        logger=logger
    )
    
    labels_with_metric_scores = {}
    for cluster_id, labels in cluster_labels.items():
        metric_scores = []
        for label in labels:
            metric_score = evaluate_label_discrimination(
                label,
                [i for i, c in enumerate(clustering_assignments) if c == cluster_id],
                [i for i, c in enumerate(clustering_assignments) if c != cluster_id],
                decoded_strs,
                decoded_strs,
                local_model,
                labeling_tokenizer,
                api_provider,
                api_model_str,
                auth_key,
                client=client,
                device=device,
                use_unitary_comparisons=True,
                max_unitary_comparisons_per_label=max_unitary_comparisons_per_label,
                api_interactions_save_loc=api_interactions_save_loc,
                logger=logger,
                cluster_id_1=cluster_id,
                cluster_id_2=None,
                metric=metric
            )
            metric_scores.append(metric_score)
        best_label_index = max(range(len(metric_scores)), key=metric_scores.__getitem__)
        labels_with_metric_scores[cluster_id] = (labels[best_label_index], metric_scores[best_label_index])
    
    return labels_with_metric_scores

def get_random_texts(
    decoded_strs: List[str],
    cluster_indices: List[int],
    n: int = 5
) -> List[str]:
    """
    Retrieve a random sample of texts from a specific cluster.

    Args:
        decoded_strs (List[str]): List of all decoded texts.
        cluster_indices (List[int]): Indices of texts belonging to a specific cluster.
        n (int): Number of random texts to retrieve. Defaults to 5.

    Returns:
        List[str]: A list of randomly sampled texts from the specified cluster.
    """
    return random.sample([decoded_strs[i] for i in cluster_indices], min(n, len(cluster_indices)))

def generate_table_output(
        results: Dict[str, Any], 
        metric_str: str
) -> str:
    """
    Generate a human-readable table output from the analysis results.

    This function takes the analysis results and formats them into ASCII tables
    for easy reading and interpretation. It creates separate tables for base clusters
    and new finetuned clusters, including information about matching and unmatching
    clusters, sample texts, and size comparisons.

    Args:
        results (Dict[str, Any]): A dictionary containing the analysis results,
            including information about base clusters and new finetuned clusters.
        metric_str (str): The metric to use for label validation.
    Returns:
        str: A string containing formatted ASCII tables representing the analysis results.
    """
    tables = []

    # Table for base clusters
    for i, cluster in enumerate(results['base_clusters']):
        data = [
            ['Base Cluster', f"Cluster {i}"],
            ['Label', f"{cluster['label']} ({metric_str}: {cluster['label_metric_score']:.3f})"],
            ['Size', str(cluster['size'])],
            ['Sample Texts', '\n'.join(cluster['sample_texts'])]
        ]

        if cluster['matching_finetuned_clusters']:
            data.append(['Matching Finetuned Clusters', ''])
            for j, match in enumerate(cluster['matching_finetuned_clusters']):
                data.extend([
                    [f'Match {j} Label', f"{match['label']} ({metric_str}: {match['label_metric_score']:.3f})"],
                    [f'Match {j} Contrastive Label', f"{match['contrastive_label']} ({metric_str}: {match['contrastive_metric_score']:.3f})"],
                    [f'Match {j} Size', str(match['size'])],
                    [f'Match {j} Sample Texts', '\n'.join(match['sample_texts'])]
                ])
            data.append(['Size Comparison', f"Absolute Diff: {cluster['size_comparison']['absolute_difference']}, " 
                                            f"Percentage Diff: {cluster['size_comparison']['percentage_difference']:.2f}%"])

        if cluster['unmatching_finetuned_clusters']:
            data.append(['Unmatching Finetuned Clusters', ''])
            for j, unmatch in enumerate(cluster['unmatching_finetuned_clusters']):
                data.extend([
                    [f'Unmatch {j} Label', f"{unmatch['label']} ({metric_str}: {unmatch['label_metric_score']:.3f})"],
                    [f'Unmatch {j} Contrastive Label', f"{unmatch['contrastive_label']} ({metric_str}: {unmatch['contrastive_metric_score']:.3f})"],
                    [f'Unmatch {j} Size', str(unmatch['size'])],
                    [f'Unmatch {j} Sample Texts', '\n'.join(unmatch['sample_texts'])]
                ])

        tables.append(AsciiTable(data, title=f'Base Cluster {i}').table)

    # Table for new finetuned clusters
    for i, cluster in enumerate(results['new_finetuned_clusters']):
        data = [
            ['New Finetuned Cluster', f"Cluster {i}"],
            ['Label', f"{cluster['label']} ({metric_str}: {cluster['label_metric_score']:.3f})"],
            ['Size', str(cluster['size'])],
            ['Sample Texts', '\n'.join(cluster['sample_texts'])]
        ]

        for j, neighbor in enumerate(cluster['nearest_base_neighbors']):
            data.extend([
                [f'Nearest Base Neighbor {j} Label', f"{neighbor['label']} ({metric_str}: {neighbor['label_metric_score']:.3f})"],
                [f'Nearest Base Neighbor {j} Contrastive Label', f"{neighbor['contrastive_label']} ({metric_str}: {neighbor['contrastive_metric_score']:.3f})"],
                [f'Nearest Base Neighbor {j} Sample Texts', '\n'.join(neighbor['sample_texts'])]
            ])

        tables.append(AsciiTable(data, title=f'New Finetuned Cluster {i}').table)

    return '\n\n'.join(tables)

def cohens_h(p1: float, p2: float) -> float:
    """
    Calculate Cohen's h effect size for proportion differences.
    
    Args:
        p1: First proportion
        p2: Second proportion
    Returns:
        float: Cohen's h effect size
    """
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

def interpret_cohens_h(h: float) -> str:
    """
    Interpret Cohen's h effect size using standard thresholds.
    
    Args:
        h: Cohen's h value
    Returns:
        str: Interpretation of effect size magnitude
    """
    h = abs(h)
    if h < 0.2:
        return "slightly"
    elif h < 0.5:
        return "moderately"
    elif h < 0.8:
        return "substantially"
    else:
        return "dramatically"

def apply_interpretability_method_1_to_K(
    base_model: PreTrainedModel, 
    finetuned_model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer,
    n_decoded_texts: int = 2000, 
    decoding_prefix_file: Optional[str] = None, 
    api_provider: str = "anthropic",
    api_model_str: str = "claude-3-haiku-20240307",
    api_stronger_model_str: Optional[str] = None,
    auth_key: Optional[str] = None,
    client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
    local_embedding_model_str: Optional[str] = None, 
    local_embedding_api_key: Optional[str] = None,
    init_clustering_from_base_model: bool = False,
    clustering_instructions: str = "Identify the topic or theme of the given texts",
    n_clustering_inits: int = 10,
    use_prompts_as_clusters: bool = False,
    cluster_on_prompts: bool = False,
    device: str = "cuda:0",
    cluster_method: str = "kmeans",
    n_clusters: int = 30,
    min_cluster_size: int = 7,
    max_cluster_size: int = 2000,
    sampled_texts_per_cluster: int = 10,
    sampled_comparison_texts_per_cluster: int = 10,
    K: int = 3,
    match_by_ids: bool = False,
    max_length: int = 32,
    decoding_batch_size: int = 32,
    decoded_texts_save_path: Optional[str] = None,
    decoded_texts_load_path: Optional[str] = None,
    loaded_texts_subsample: Optional[int] = None,
    path_to_MWE_repo: Optional[str] = None,
    num_statements_per_behavior: Optional[int] = None,
    num_responses_per_statement: Optional[int] = None,
    num_rephrases_for_validation: int = 0,
    num_generated_texts_per_description: int = 20,
    generated_labels_per_cluster: int = 3,
    num_decodings_per_prompt: int = None,
    include_prompts_in_decoded_texts: bool = False,
    single_cluster_label_instruction: Optional[str] = None,
    contrastive_cluster_label_instruction: Optional[str] = None,
    use_unitary_comparisons: bool = False,
    max_unitary_comparisons_per_label: int = 50,
    match_cutoff: float = 0.6,
    discriminative_query_rounds: int = 3,
    discriminative_validation_runs: int = 5,
    metric: str = "acc",
    tsne_save_path: Optional[str] = None,
    tsne_title: Optional[str] = None,
    tsne_perplexity: int = 30,
    api_interactions_save_loc: Optional[str] = None,
    logger: Optional[BoundLoggerLazyProxy] = None,
    run_prefix: Optional[str] = None,
    save_addon_str: Optional[str] = None,
    graph_load_path: Optional[str] = None,
    scoring_results_load_path: Optional[str] = None
) -> Tuple[str, str, List[str]]:
    """
    Apply an interpretability method to compare a base model with a finetuned model.

    This function performs a comprehensive analysis of the differences between two models:
    1. Decodes texts from both models
    2. Clusters the decoded texts
    3. Labels the clusters
    4. Builds a K-neighbor similarity graph between clusters
    5. Analyzes matching and unmatching clusters
    6. Identifies new clusters in the finetuned model

    The analysis results are returned in both JSON and human-readable table formats.

    Args:
        base_model (PreTrainedModel): The original, pre-finetuned model.
        finetuned_model (PreTrainedModel): The model after finetuning.
        tokenizer (PreTrainedTokenizer): The tokenizer for text generation.
        n_decoded_texts (int): Number of texts to decode from each model.
        decoding_prefix_file (Optional[str]): File containing prefixes for text generation.
        api_provider (str): API provider for clustering and analysis.
        api_model_str (str): API model to use.
        api_stronger_model_str (Optional[str]): API model to use for stronger model, only used for the most important tasks.
        auth_key (Optional[str]): Authentication key for API calls.
        client (Optional[Any]): Client object for API calls.
        local_embedding_model_str (Optional[str]): Name of local embedding model.
        local_embedding_api_key (Optional[str]): API key for local embedding model.
        init_clustering_from_base_model (bool): Whether to initialize finetuned clustering from base model.
        clustering_instructions (str): Instructions for clustering.
        n_clustering_inits (int): Number of clustering initializations to use.
        use_prompts_as_clusters (bool): Whether to use the prompts as the clusters. False by default.
        cluster_on_prompts (bool): Whether to cluster on the prompts or the decoded texts. False by default.
        device (str): Device to use for computations.
        cluster_method (str): Clustering method to use.
        n_clusters (int): Number of clusters.
        min_cluster_size (int): Minimum cluster size.
        max_cluster_size (int): Maximum cluster size.
        sampled_comparison_texts_per_cluster (int): Number of texts to sample per cluster for both evaluation
            of label accuracy / AUCs between clusters and for individual labels.
        sampled_texts_per_cluster (int): Number of texts to sample per cluster when generating labels.
        K (int): Number of neighbors for K-neighbor similarity graph.
        match_by_ids (bool): Whether to match clusters by their IDs, not embedding distances.
        max_length (int): Maximum length of decoded texts.
        decoding_batch_size (int): Batch size for decoding.
        decoded_texts_save_path (Optional[str]): Path to save decoded texts.
        decoded_texts_load_path (Optional[str]): Path to load decoded texts.
        loaded_texts_subsample (Optional[int]): If specified, will randomly subsample the loaded decoded 
            texts to this number. None by default.
        path_to_MWE_repo (Optional[str]): Path to the Anthropic evals repository.
        num_statements_per_behavior (Optional[int]): Number of statements per behavior to read from the evals 
            repository and then generate responses from.
        num_responses_per_statement (Optional[int]): Number of responses per statement to generate from the statements 
            in the evals repository.
        num_rephrases_for_validation (int): Number of rephrases for validation.
        num_generated_texts_per_description (int): Number of generated texts per description for generative validation.
        generated_labels_per_cluster (int): Number of labels to generate per cluster (for both contrastive
            and individual labels).
        num_decodings_per_prompt (int): Number of decodings per prompt to use when generating labels.
        include_prompts_in_decoded_texts (bool): Whether to include the prompts in the decoded texts. False by default.
        single_cluster_label_instruction (Optional[str]): Instructions for generating the single cluster labels.
        contrastive_cluster_label_instruction (Optional[str]): Instructions for generating the contrastive cluster labels.
        use_unitary_comparisons (bool): Whether to use unitary comparisons.
        max_unitary_comparisons_per_label (int): Maximum number of unitary comparisons to perform per label.
            I.e., when validating the accuracy / AUC of a given label (either contrastive or individual), we will ask 
            the assistant to use the label to classify at most max_unitary_comparisons_per_label texts.
        match_cutoff (float): Accuracy / AUC cutoff for determining matching/unmatching clusters.
        discriminative_query_rounds (int): Number of rounds of discriminative queries to perform.
        discriminative_validation_runs (int): Number of validation runs to perform for each model for each hypothesis.
        metric (str): The metric to use for label validation.
        tsne_save_path (Optional[str]): Path to save t-SNE plot.
        tsne_title (Optional[str]): Title for t-SNE plot.
        tsne_perplexity (int): Perplexity for t-SNE.
        api_interactions_save_loc (Optional[str]): Where to record interactions with the API model, if anywhere.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        run_prefix (Optional[str]): Prefix for the current run.
        save_addon_str (Optional[str]): Addon string for the current run.
        graph_load_path (Optional[str]): Path to load the graph from.
        scoring_results_load_path (Optional[str]): Path to load the scoring results from.
    Returns:
        Tuple[str, str, List[str]]: A tuple containing:
            - results: JSON string with detailed analysis results
            - table_output: Human-readable string with formatted table output of the analysis
            - validated_hypotheses: List of validated hypotheses about how the two models differ in behavior
    """
    setup = setup_interpretability_method(
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer=tokenizer,
        n_decoded_texts=n_decoded_texts,
        decoding_prefix_file=decoding_prefix_file,
        auth_key=auth_key,
        client=client,
        local_embedding_model_str=local_embedding_model_str,
        local_embedding_api_key=local_embedding_api_key,
        init_clustering_from_base_model=init_clustering_from_base_model,
        num_decodings_per_prompt=num_decodings_per_prompt,
        include_prompts_in_decoded_texts=include_prompts_in_decoded_texts,
        clustering_instructions=clustering_instructions,
        n_clustering_inits=n_clustering_inits,
        use_prompts_as_clusters=use_prompts_as_clusters,
        cluster_on_prompts=cluster_on_prompts,
        device=device,
        cluster_method=cluster_method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
        max_length=max_length,
        decoding_batch_size=decoding_batch_size,
        decoded_texts_save_path=decoded_texts_save_path,
        decoded_texts_load_path=decoded_texts_load_path,
        loaded_texts_subsample=loaded_texts_subsample,
        path_to_MWE_repo=path_to_MWE_repo,
        num_statements_per_behavior=num_statements_per_behavior,
        num_responses_per_statement=num_responses_per_statement,
        tsne_save_path=tsne_save_path,
        tsne_title=tsne_title,
        tsne_perplexity=tsne_perplexity,
        run_prefix=run_prefix
    )

    base_clustering = setup["base_clustering"]
    finetuned_clustering = setup["finetuned_clustering"]
    base_embeddings = setup["base_embeddings"]
    finetuned_embeddings = setup["finetuned_embeddings"]
    base_decoded_texts = setup["base_decoded_texts"]
    finetuned_decoded_texts = setup["finetuned_decoded_texts"]
    cluster_ids_to_prompt_ids_to_decoding_ids_dict_1 = setup["cluster_ids_to_prompt_ids_to_decoding_ids_dict_1"]
    cluster_ids_to_prompt_ids_to_decoding_ids_dict_2 = setup["cluster_ids_to_prompt_ids_to_decoding_ids_dict_2"]

    n_base_clusters = len(set(base_clustering.labels_))
    n_finetuned_clusters = len(set(finetuned_clustering.labels_))

    if graph_load_path is not None:
        print(f"Loading graph from {graph_load_path}")
        graph = pickle.load(open(graph_load_path, "rb"))
        # Check if the graph nodes have cluster_label_str values
        if not all(node_data.get("cluster_label_str") for _, node_data in graph.nodes(data=True)):
            # Old graphs lack cluster_label_str values
            # If we cannot load real labels, we create fake ones
            print("Warning: No cluster labels found in graph. Creating fake ones.")
            base_labels = [f"Cluster {i} base" for i in range(n_base_clusters)]
            finetuned_labels = [f"Cluster {i} finetuned" for i in range(n_finetuned_clusters)]
        else:
            # Must ensure we traverse the nodes in order of cluster_id
            for cluster_id in range(n_base_clusters):
                base_labels.append(graph.nodes[f"1_{cluster_id}"]["cluster_label_str"])
            for cluster_id in range(n_finetuned_clusters):
                finetuned_labels.append(graph.nodes[f"2_{cluster_id}"]["cluster_label_str"])
    else:
        # Get individual labels for clusters
        base_labels = get_individual_labels(
            base_decoded_texts,
            base_clustering.labels_,
            None,
            tokenizer,
            api_provider,
            api_model_str,
            api_stronger_model_str,
            auth_key,
            client=client,
            device=device,
            sampled_texts_per_cluster=sampled_comparison_texts_per_cluster,
            generated_labels_per_cluster=generated_labels_per_cluster,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict=cluster_ids_to_prompt_ids_to_decoding_ids_dict_1,
            num_decodings_per_prompt=num_decodings_per_prompt,
            single_cluster_label_instruction=single_cluster_label_instruction,
            max_unitary_comparisons_per_label=max_unitary_comparisons_per_label,
            api_interactions_save_loc=api_interactions_save_loc,
            logger=logger,
            metric=metric
        )
        print(f"Base labels: {base_labels}")
        finetuned_labels = get_individual_labels(
            finetuned_decoded_texts,
            finetuned_clustering.labels_,
            None,
            tokenizer,
            api_provider,
            api_model_str,
            api_stronger_model_str,
            auth_key,
            client=client,
            device=device,
            sampled_texts_per_cluster=sampled_comparison_texts_per_cluster,
            generated_labels_per_cluster=generated_labels_per_cluster,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict=cluster_ids_to_prompt_ids_to_decoding_ids_dict_2,
            num_decodings_per_prompt=num_decodings_per_prompt,
            single_cluster_label_instruction=single_cluster_label_instruction,
            max_unitary_comparisons_per_label=max_unitary_comparisons_per_label,
            api_interactions_save_loc=api_interactions_save_loc,
            logger=logger,
            metric=metric
        )
        print(f"Finetuned labels: {finetuned_labels}")
        # Build the K-neighbor similarity graph
        graph = build_contrastive_K_neighbor_similarity_graph(
            base_decoded_texts,
            base_clustering.labels_,
            base_embeddings,
            base_labels,
            finetuned_decoded_texts,
            finetuned_clustering.labels_,
            finetuned_embeddings,
            finetuned_labels,
            K,
            match_by_ids,
            base_model,
            tokenizer,
            api_provider,
            api_model_str,
            api_stronger_model_str,
            auth_key,
            client=client,
            device=device,
            sampled_comparison_texts_per_cluster=sampled_comparison_texts_per_cluster,
            generated_labels_per_cluster=generated_labels_per_cluster,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_1=cluster_ids_to_prompt_ids_to_decoding_ids_dict_1,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_2=cluster_ids_to_prompt_ids_to_decoding_ids_dict_2,
            num_decodings_per_prompt=num_decodings_per_prompt,
            contrastive_cluster_label_instruction=contrastive_cluster_label_instruction,
            use_unitary_comparisons=use_unitary_comparisons,
            max_unitary_comparisons_per_label=max_unitary_comparisons_per_label,
            api_interactions_save_loc=api_interactions_save_loc,
            logger=logger,
            metric=metric
        )

        # Only do this if we have loaded cluster indices and metrics
        if setup["base_loaded_mauve_cluster_scores"] is not None:
            attach_cluster_metrics_to_graph(
                graph,
                base_clustering.labels_,
                finetuned_clustering.labels_,
                setup["base_loaded_mauve_cluster_scores"],
                setup["finetuned_loaded_mauve_cluster_scores"],
                setup["base_loaded_kl_divergence_cluster"],
                setup["finetuned_loaded_kl_divergence_cluster"],
                setup["base_loaded_mean_entropy_cluster"],
                setup["finetuned_loaded_mean_entropy_cluster"]
            )
            pickle.dump(graph, open(f"pkl_graphs/{run_prefix}{save_addon_str}_1_to_K_graph.pkl", "wb"))
    if setup["base_loaded_mauve_cluster_scores"] is not None:
        analyze_metric_differences_vs_similarity(graph)
        analyze_node_metric_vs_neighbor_similarity(graph)

    results = {'base_clusters': [], 'new_finetuned_clusters': []}

    # Process base model clusters
    for base_cluster in tqdm(range(n_base_clusters), desc="Processing base clusters"):
        base_cluster_indices = [i for i, label in enumerate(base_clustering.labels_) if label == base_cluster]
        cluster_info = {
            'label': base_labels[base_cluster][0],
            'cluster_id': base_cluster,
            'label_metric_score': base_labels[base_cluster][1],
            'sample_texts': get_random_texts(base_decoded_texts, base_cluster_indices),
            'size': len(base_cluster_indices),
            'matching_finetuned_clusters': [],
            'unmatching_finetuned_clusters': []
        }

        for finetuned_neighbor in graph.neighbors(f"1_{base_cluster}"):
            finetuned_cluster = int(finetuned_neighbor.split('_')[1])
            edge_data = graph[f"1_{base_cluster}"][finetuned_neighbor]
            if 'label_metric_scores' not in edge_data or not edge_data['label_metric_scores']:
                print(f"Warning: No label_metric_scores data for edge (1_{base_cluster}, {finetuned_neighbor})")
                print("edge_data:")
                print(edge_data)
                continue
            best_metric_score = max(edge_data['label_metric_scores'].values())
            best_label = max(edge_data['label_metric_scores'], key=edge_data['label_metric_scores'].get)

            finetuned_cluster_indices = [i for i, label in enumerate(finetuned_clustering.labels_) if label == finetuned_cluster]
            neighbor_info = {
                'label': finetuned_labels[finetuned_cluster][0],
                'cluster_id': finetuned_cluster,
                'label_metric_score': finetuned_labels[finetuned_cluster][1],
                'contrastive_label': best_label,
                'contrastive_metric_score': best_metric_score,
                'all_contrastive_labels': list(edge_data['label_metric_scores'].keys()),
                'all_contrastive_metric_scores': list(edge_data['label_metric_scores'].values()),
                'sample_texts': get_random_texts(finetuned_decoded_texts, finetuned_cluster_indices),
                'size': len(finetuned_cluster_indices)
            }

            if best_metric_score < match_cutoff:
                # Case 1: a matching cluster is found, so the same behavior is present in both models
                # though with potentially different probabilities, as indicated by relative cluster sizes
                cluster_info['matching_finetuned_clusters'].append(neighbor_info)
            else:
                # Case 2: A cluster in model 1 is not found in model 2, so some model 1 behavior fails to
                # appear in model 2.
                cluster_info['unmatching_finetuned_clusters'].append(neighbor_info)
        
        # Account for the size of the matching clusters in case 1
        if cluster_info['matching_finetuned_clusters']:
            total_matching_size = sum(c['size'] for c in cluster_info['matching_finetuned_clusters'])
            cluster_info['size_comparison'] = {
                'absolute_difference': total_matching_size - cluster_info['size'],
                'percentage_difference': (total_matching_size - cluster_info['size']) / cluster_info['size'] * 100
            }

        results['base_clusters'].append(cluster_info)

    # Process unmatched finetuned clusters
    matched_finetuned_clusters = set()
    for base_info in results['base_clusters']:
        for match in base_info['matching_finetuned_clusters']:
            matched_finetuned_clusters.add(match['label'])

    for finetuned_cluster in range(n_finetuned_clusters):
        if finetuned_labels[finetuned_cluster][0] not in matched_finetuned_clusters:
            finetuned_cluster_indices = [i for i, label in enumerate(finetuned_clustering.labels_) if label == finetuned_cluster]
            new_cluster_info = {
                'cluster_id': finetuned_cluster,
                'label': finetuned_labels[finetuned_cluster][0],
                'label_metric_score': finetuned_labels[finetuned_cluster][1],
                'sample_texts': get_random_texts(finetuned_decoded_texts, finetuned_cluster_indices),
                'size': len(finetuned_cluster_indices),
                'nearest_base_neighbors': []
            }

            for base_neighbor in graph.neighbors(f"2_{finetuned_cluster}"):
                base_cluster = int(base_neighbor.split('_')[1])
                edge_data = graph[f"2_{finetuned_cluster}"][base_neighbor]
                best_metric_score = max(edge_data['label_metric_scores'].values())
                best_label = max(edge_data['label_metric_scores'], key=edge_data['label_metric_scores'].get)

                base_cluster_indices = [i for i, label in enumerate(base_clustering.labels_) if label == base_cluster]
                new_cluster_info['nearest_base_neighbors'].append({
                    'label': base_labels[base_cluster][0],
                    'cluster_id': base_cluster,
                    'label_metric_score': base_labels[base_cluster][1],
                    'contrastive_label': best_label,
                    'contrastive_metric_score': best_metric_score,
                    'all_contrastive_labels': list(edge_data['label_metric_scores'].keys()),
                    'all_contrastive_metric_scores': list(edge_data['label_metric_scores'].values()),
                    'sample_texts': get_random_texts(base_decoded_texts, base_cluster_indices)
                })
            # Case 3: A new cluster is found in model 2 that is not in model 1, so some model 2 behavior
            # is not present in model 1.
            results['new_finetuned_clusters'].append(new_cluster_info)

    # Construct the candidate hypothesis strings for each cluster, based on which of the three cases it 
    # falls under.
    candidate_hypotheses = []
    candidate_hypothesis_contrastive_discriminative_scores = []
    hypothesis_origin_clusters = []
    num_cluster_matches = 0
    num_clusters_matched_to_equivalents = 0

    # Case 1: Matching clusters with different sizes

    # Calculate total number of statistical tests for Bonferroni correction
    # This includes both proportion tests and validation tests
    n_proportion_tests = sum(len(cluster['matching_finetuned_clusters']) for cluster in results['base_clusters'])
    bonferroni_alpha = 0.05 / n_proportion_tests if n_proportion_tests > 0 else 0.05

    print(f"Bonferroni alpha (corrected for {n_proportion_tests} total tests): {bonferroni_alpha}")
    for base_cluster in results['base_clusters']:
        if base_cluster['matching_finetuned_clusters']:
            for match in base_cluster['matching_finetuned_clusters']:
                num_cluster_matches += 1
                if match['cluster_id'] == base_cluster['cluster_id']:
                    num_clusters_matched_to_equivalents += 1
                # Set up counts and nobs as numpy arrays
                count = np.array([base_cluster['size'], match['size']])
                nobs = np.array([n_decoded_texts, n_decoded_texts])
                # Perform single two-sided test
                z_stat, p_value = proportions_ztest(
                    count=count,
                    nobs=nobs,
                    alternative='two-sided'
                )
                
                if p_value < bonferroni_alpha:
                    # Calculate proportions and effect size
                    p1 = base_cluster['size'] / n_decoded_texts
                    p2 = match['size'] / n_decoded_texts
                    effect_size = cohens_h(p1, p2)
                    magnitude = interpret_cohens_h(effect_size)
                    
                    # Determine direction based on actual proportions
                    direction = "more" if p1 > p2 else "less"
                    
                    hypothesis = f"Model 1 is {magnitude} {direction} likely to generate content described as: '{base_cluster['label']}' compared to Model 2."
                    candidate_hypotheses.append(hypothesis)
                    candidate_hypothesis_contrastive_discriminative_scores.append(-1.0)
                    hypothesis_origin_clusters.append([base_cluster, match, 'matching'])
    # Case 2: Unmatching clusters (behavior in base model not found in finetuned model)
    for base_cluster in results['base_clusters']:
        if base_cluster['unmatching_finetuned_clusters']:
            hypothesis = f"Model 2 is less likely to generate content described as: '{base_cluster['label']}' compared to Model 1."
            candidate_hypotheses.append(hypothesis)
            candidate_hypothesis_contrastive_discriminative_scores.append(-1.0)
            hypothesis_origin_clusters.append([base_cluster, -1, 'unmatching'])
            # Add hypotheses based on contrastive labels
            for unmatch in base_cluster['unmatching_finetuned_clusters']:
                for label in unmatch['all_contrastive_labels']:
                    contrastive_hypothesis = re.sub(r'[Cc]luster (\d)', lambda m: f"Model {m.group(1)}", label)
                    candidate_hypotheses.append(contrastive_hypothesis)
                    candidate_hypothesis_contrastive_discriminative_scores.append(unmatch['contrastive_metric_score'])
                    hypothesis_origin_clusters.append([base_cluster, unmatch, 'unmatching'])
    # Case 3: New clusters in finetuned model
    for new_cluster in results['new_finetuned_clusters']:
        hypothesis = f"Model 2 has developed a new behavior of generating content described as: '{new_cluster['label']}', which was not prominent in Model 1."
        candidate_hypotheses.append(hypothesis)
        candidate_hypothesis_contrastive_discriminative_scores.append(-1.0)
        hypothesis_origin_clusters.append([-1, new_cluster, 'new'])
        # Add hypotheses based on contrastive labels of nearest base neighbors
        for neighbor in new_cluster['nearest_base_neighbors']:
            for label in neighbor['all_contrastive_labels']:
                contrastive_hypothesis = re.sub(r'[Cc]luster (\d)', lambda m: f"Model {m.group(1)}", label)
                candidate_hypotheses.append(contrastive_hypothesis)
                candidate_hypothesis_contrastive_discriminative_scores.append(neighbor['contrastive_metric_score'])
                hypothesis_origin_clusters.append([neighbor, new_cluster, 'new'])

    print(f"Generated {len(candidate_hypotheses)} candidate hypotheses.")
    print(f"Number of cluster matches: {num_cluster_matches}")
    print(f"Number of clusters matched to equivalents: {num_clusters_matched_to_equivalents}")
    for i, hypothesis, origin_cluster in zip(range(1, len(candidate_hypotheses) + 1), candidate_hypotheses, hypothesis_origin_clusters):
        print(f"{i}. {hypothesis}")
        print(f"   Origin cluster: {origin_cluster}")
        print("-" * 50)

    if scoring_results_load_path is None:
        # Validate the candidate hypotheses using the validated_assistant_generative_compare function
        validated_results = validated_assistant_generative_compare(
            candidate_hypotheses,
            None,
            None,
            api_provider=api_provider,
            api_model_str=api_model_str,
            auth_key=auth_key,
            api_stronger_model_str=api_stronger_model_str,
            client=client,
            starting_model_str=None,
            comparison_model_str=None,
            common_tokenizer_str=base_model.name_or_path,
            starting_model=base_model,
            comparison_model=finetuned_model,
            device=device,
            use_normal_distribution_for_p_values=True,
            num_generated_texts_per_description=num_generated_texts_per_description,
            num_rephrases_for_validation=num_rephrases_for_validation,
            bnb_config=setup["bnb_config"],
            use_correlation_coefficient=True,
            api_interactions_save_loc=api_interactions_save_loc,
            logger=logger
        )

        all_validated_scores, all_validated_p_values, all_validated_hypotheses = validated_results
        pickle.dump(validated_results, open(f"pkl_results/{run_prefix}{save_addon_str}_validated_results.pkl", "wb"))
    else:
        if scoring_results_load_path.endswith(".txt"):
            with open(scoring_results_load_path, "r") as f:
                all_validated_scores = [[float(line) for line in f.readlines()]]
        else:
            validated_results = pickle.load(open(scoring_results_load_path, "rb"))
            all_validated_scores, all_validated_p_values, all_validated_hypotheses = validated_results    

    # only do this if we've loaded the relevant scores
    if setup["base_loaded_mauve_cluster_scores"] is not None:
        analyze_hypothesis_scores_vs_cluster_metrics(
            graph,
            hypothesis_origin_clusters,
            all_validated_scores,
            model_label_for_node_1="1_",
            model_label_for_node_2="2_"
        )

    #print("validated_results object:")
    #print(validated_results)
    if discriminative_query_rounds > 0:
        discriminative_validation_results = validated_embeddings_discriminative_single_unknown_ICL(
            difference_descriptions = candidate_hypotheses,
            api_provider = api_provider,
            api_model_str = api_stronger_model_str if api_stronger_model_str else api_model_str,
            auth_key = auth_key,
            client = client,
            common_tokenizer_str = base_model.name_or_path,
            starting_model = base_model,
            comparison_model = finetuned_model,
            num_rounds = discriminative_query_rounds,
            num_validation_runs = discriminative_validation_runs,
            explain_reasoning = False,
            max_tokens = 1000,
            api_interactions_save_loc = api_interactions_save_loc,
            logger=logger
        )
        validated_discriminative_accuracies = discriminative_validation_results["hypothesis_accuracies"]
        validated_discriminative_p_values = discriminative_validation_results["hypothesis_p_values"]
    else:
        validated_discriminative_accuracies = [None] * len(candidate_hypotheses)
        validated_discriminative_p_values = [None] * len(candidate_hypotheses)

    # Filter and format final hypotheses
    validated_hypotheses = []
    print("\nValidated Hypotheses:")
    for i, main_hypothesis in enumerate(all_validated_hypotheses[0]):
        generative_score = all_validated_scores[0][i]
        generative_p_value = all_validated_p_values[0][i]
        discriminative_accuracy = validated_discriminative_accuracies[i]
        discriminative_p_value = validated_discriminative_p_values[i]
        neighbor_discriminative_score = candidate_hypothesis_contrastive_discriminative_scores[i]
        if True: #generative_score > 0.2 and generative_p_value < 0.1: 
            validated_hypothesis = f"{main_hypothesis} \n(Generative Score: {generative_score:.4f}, P-value: {generative_p_value:.4f})"
            if discriminative_accuracy is not None:
                validated_hypothesis = validated_hypothesis + f"\n(Discriminative Accuracy: {discriminative_accuracy:.4f}, P-value: {discriminative_p_value:.4f})"
            if neighbor_discriminative_score is not None and neighbor_discriminative_score != -1.0:
                validated_hypothesis = validated_hypothesis + f"\n(Neighbor Discriminative Score: {neighbor_discriminative_score:.4f})"
            validated_hypotheses.append(validated_hypothesis)
            
            print(f"\n{len(validated_hypotheses)}. {validated_hypothesis}")

            # Find all rephrases / associated scores and p-values
            if len(all_validated_hypotheses) > 1:
                rephrases_of_hypothesis = [all_validated_hypotheses[j][i] for j in range(1, len(all_validated_hypotheses))]
                scores_of_rephrases = [all_validated_scores[j][i] for j in range(1, len(all_validated_hypotheses))]
                p_values_of_rephrases = [all_validated_p_values[j][i] for j in range(1, len(all_validated_hypotheses))]
            
                # Print out the hypothesis, its rephrases, and the accuracy / AUCs and P-values for each
                for j, rephrase in enumerate(rephrases_of_hypothesis):
                    print(f"   Rephrase {j}: {rephrase}")
                    print(f"   Correlation Coefficient: {scores_of_rephrases[j]:.4f}, P-value: {p_values_of_rephrases[j]:.4f}")
            
            print("-" * 50)

    # Add validation correlations / accuracies and p-values to results
    validation_results = {
        'hypotheses': [],
        'matching_clusters': [],
        'unmatching_clusters': [],
        'new_clusters': []
    }

    for i, (hypothesis, origin_clusters_info) in enumerate(zip(candidate_hypotheses, hypothesis_origin_clusters)):
        validation_info = {
            'hypothesis': hypothesis,
            'generative_score': all_validated_scores[0][i],
            'generative_p_value': all_validated_p_values[0][i],
            'discriminative_accuracy': validated_discriminative_accuracies[i],
            'discriminative_p_value': validated_discriminative_p_values[i],
            'origin_type': origin_clusters_info[2],
            'source_clusters': (origin_clusters_info[0], origin_clusters_info[1]),
            'rephrases': []
        }
        
        # Add rephrases if they exist
        if len(all_validated_hypotheses) > 1:
            validation_info['rephrases'] = [
                {
                    'text': all_validated_hypotheses[j][i],
                    'score': all_validated_scores[j][i],
                    'p_value': all_validated_p_values[j][i]
                }
                for j in range(1, len(all_validated_hypotheses))
            ]
        
        # Add to appropriate category based on origin type
        if origin_clusters_info[2] == 'matching':
            validation_results['matching_clusters'].append(validation_info)
        elif origin_clusters_info[2] == 'unmatching':
            validation_results['unmatching_clusters'].append(validation_info)
        elif origin_clusters_info[2] == 'new':
            validation_results['new_clusters'].append(validation_info)
        
        validation_results['hypotheses'].append(validation_info)
    
    results['validation_results'] = validation_results

    # Generate human-readable output using terminaltables
    table_output = generate_table_output(results, "Accuracy" if metric == "acc" else "AUC")

    return results, table_output, validated_hypotheses