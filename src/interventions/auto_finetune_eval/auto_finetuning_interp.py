"""
This module contains code for applying an interpretability method to compare two models, using the functionality provided by quick_cluster.py
"""

from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import random
import torch

import sys
sys.path.append("../../contrastive-decoding/")
from quick_cluster import read_past_embeddings_or_generate_new, match_clusterings, get_validated_contrastive_cluster_labels, validated_assistant_generative_compare
from transformers import BitsAndBytesConfig
from sklearn.cluster import KMeans, HDBSCAN

def dummy_apply_interpretability_method(
        base_model: PreTrainedModel, 
        finetuned_model: PreTrainedModel
    ) -> List[str]:
    """
    Dummy implementation of applying an interpretability method to compare two models.

    This function simulates the process of comparing a base model with a finetuned model
    and generating hypotheses about their differences. In a real implementation, this
    would involve sophisticated analysis techniques.

    Args:
        base_model (PreTrainedModel): The original, pre-finetuned model.
        finetuned_model (PreTrainedModel): The model after finetuning.

    Returns:
        List[str]: A list of hypotheses about how the models differ.
    """
    # Placeholder implementation
    hypotheses = [
        "The finetuned model shows increased preference for specific topics.",
        "The finetuned model demonstrates altered response patterns in certain contexts.",
        "The finetuned model exhibits changes in its language style and tone."
    ]
    return random.sample(hypotheses, k=random.randint(1, len(hypotheses)))

def batch_decode_texts(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer,
        prefixes: List[str], 
        n_decoded_texts: int, 
        batch_size: int = 16
    ) -> List[str]:
    """
    Decode texts in batches using the given model.

    Args:
        model (PreTrainedModel): The model to use for text generation.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for text generation.
        prefixes (List[str]): List of prefixes to use for text generation.
        n_decoded_texts (int): Total number of texts to generate.
        batch_size (int): Number of texts to generate in each batch.

    Returns:
        List[str]: List of generated texts.
    """
    decoded_texts = []
    for i in range(0, n_decoded_texts, batch_size):
        batch_prefixes = [random.choice(prefixes) for _ in range(min(batch_size, n_decoded_texts - i))]
        inputs = tokenizer(batch_prefixes, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2
            )
        
        batch_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_texts.extend(batch_decoded)
    
    return decoded_texts

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
        clustering_instructions: str = "Identify the topic or theme of the given texts",
        device: str = "cuda:0",
        cluster_method: str = "kmeans",
        n_clusters: int = 30,
        min_cluster_size: int = 7,
        max_cluster_size: int = 2000
    ) -> List[str]:
    """
    Real implementation of applying an interpretability method to compare two models.

    This function first decodes a text corpus with both models, then clusters the decoded outputs, and then performs pairwise matching of the two sets of clusters. It then feeds an assistant LLM texts from matched cluster pairs to generate hypotheses about how the models differ, and validates those hypotheses automatically by testing that the LLM assistant can use the hypotheses to differentiate between texts from the two models. It returns the validated hypotheses as a list of strings.

    Args:
        - base_model (PreTrainedModel): The original, pre-finetuned model.
        - finetuned_model (PreTrainedModel): The model after finetuning.
        - tokenizer (PreTrainedTokenizer): The tokenizer to use for text generation.
        - n_decoded_texts (int): The number of texts to decode with each model.
        - decoding_prefix_file (Optional[str]): The path to a file containing a set of prefixes to prepend to the texts to be decoded.
        - api_provider (str): The API provider to use for clustering and analysis.
        - api_model_str (str): The API model to use for clustering and analysis.
        - auth_key (str): The API key to use for clustering and analysis.
        - local_embedding_model_str (Optional[str]): The name of the local embedding model to use.
        - local_embedding_api_key (Optional[str]): The API key for the local embedding model.
        - init_clustering_from_base_model (bool): Whether to initialize the clustering of the finetuned model from the cluster centers of the base model. Only possible for kmeans clustering.
        - clustering_instructions (str): The instructions to use for clustering.
        - device (str): The device to use for clustering. "cuda:0" by default.
        - cluster_method (str): The method to use for clustering. "kmeans" or "hdbscan".
        - n_clusters (int): The number of clusters to use. 30 by default.
        - min_cluster_size (int): The minimum size of a cluster. 7 by default.
        - max_cluster_size (int): The maximum size of a cluster. 2000 by default.
    Returns:
        List[str]: A list of validated hypotheses about how the models differ.
    """
    if local_embedding_model_str is None and local_embedding_api_key is None:
        raise ValueError("Either local_embedding_model_str or local_embedding_api_key must be provided.")
    if auth_key is None and client is None:
        raise ValueError("Either auth_key or client must be provided.")
    # Load decoding prefixes
    prefixes = []
    if decoding_prefix_file:
        with open(decoding_prefix_file, 'r') as f:
            prefixes = [line.strip() for line in f.readlines()]
    if not prefixes:
        prefixes = ["The following is a short text: "]

    # Decode texts with both models
    base_decoded_texts = batch_decode_texts(base_model, tokenizer, prefixes, n_decoded_texts)
    finetuned_decoded_texts = batch_decode_texts(finetuned_model, tokenizer, prefixes, n_decoded_texts)

    # Generate embeddings for both sets of decoded texts
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_embeddings = read_past_embeddings_or_generate_new(
        "base_model_embeddings",
        None,
        base_decoded_texts,
        local_embedding_model_str=local_embedding_model_str,
        device=device,
        recompute_embeddings=True,
        save_embeddings=True,
        clustering_instructions=clustering_instructions,
        bnb_config=bnb_config
    )
    finetuned_embeddings = read_past_embeddings_or_generate_new(
        "finetuned_model_embeddings",
        None,
        finetuned_decoded_texts,
        local_embedding_model_str=local_embedding_model_str,
        device=device,
        recompute_embeddings=True,
        save_embeddings=True,
        clustering_instructions=clustering_instructions,
        bnb_config=bnb_config
    )

    # Perform clustering on both sets of embeddings
    if cluster_method == "kmeans":
        base_clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(base_embeddings)
        if init_clustering_from_base_model:
            initial_centroids = base_clustering.cluster_centers_
            finetuned_clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init=10, init=initial_centroids).fit(finetuned_embeddings)
        else:
            finetuned_clustering = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(finetuned_embeddings)
    elif cluster_method == "hdbscan":
        base_clustering = HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size).fit(base_embeddings)
        finetuned_clustering = HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size).fit(finetuned_embeddings)
    
    base_clustering_assignments = base_clustering.labels_
    finetuned_clustering_assignments = finetuned_clustering.labels_

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
        device=device,
        compute_p_values=True,
        num_permutations=3,
        use_normal_distribution_for_p_values=False,
        sampled_comparison_texts_per_cluster=10,
        generated_labels_per_cluster=3,
        pick_top_n_labels=None
    )

    cluster_pair_scores = contrastive_labels_results["cluster_pair_scores"]
    p_values = contrastive_labels_results["p_values"]

    # Generate texts based on contrastive labels and validate
    hypotheses = []
    for cluster_pair, label_scores in cluster_pair_scores.items():
        for label, score in label_scores.items():
            p_value = p_values[cluster_pair][label]
            hypothesis = f"Cluster pair {cluster_pair}: {label} (Score: {score:.3f}, P-value: {p_value:.3f})"
            hypotheses.append(hypothesis)

    # Validate hypotheses using generated texts
    validated_results = validated_assistant_generative_compare(
        hypotheses,
        None,
        None,
        api_provider=api_provider,
        api_model_str=api_model_str,
        auth_key=auth_key,
        starting_model_str=base_model.name_or_path,
        comparison_model_str=finetuned_model.name_or_path,
        common_tokenizer_str=base_model.name_or_path,
        device=device,
        num_permutations=3,
        use_normal_distribution_for_p_values=False,
        num_generated_texts_per_description=10,
        bnb_config=bnb_config
    )

    validated_aucs, validated_p_values = validated_results

    # Filter and format final hypotheses
    final_hypotheses = []
    for i, hypothesis in enumerate(hypotheses):
        if validated_aucs[i] > 0.6 and validated_p_values[i] < 0.05:
            final_hypothesis = f"{hypothesis} (Validated AUC: {validated_aucs[i]:.3f}, Validated P-value: {validated_p_values[i]:.3f})"
            final_hypotheses.append(final_hypothesis)

    return final_hypotheses