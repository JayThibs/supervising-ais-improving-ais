"""
This module contains code for applying an interpretability method to compare two models, using the functionality provided by validated_comparison_tools.py
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from transformers import PreTrainedModel, PreTrainedTokenizer, BitsAndBytesConfig, GPTNeoXForCausalLM, Qwen2ForCausalLM
from transformers.utils import logging
from anthropic import Anthropic
from openai import OpenAI
from google.genai import Client
import random
from sklearn.cluster import KMeans, HDBSCAN, MiniBatchKMeans
from k_means_constrained import KMeansConstrained
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests
import numpy as np
import pandas as pd
from terminaltables import AsciiTable
from tqdm import tqdm
import re
import pickle
from auto_finetuning_helpers import plot_comparison_tsne, batch_decode_texts, load_statements_from_MWE_repo, load_statements_from_MPI_repo, load_statements_from_truthfulqa, load_statements_from_amazon_bold, load_statements_from_jailbreak_llms_repo, parallel_make_api_requests, generate_new_prompts
from os import path

from validated_comparison_tools import read_past_embeddings_or_generate_new, build_contrastive_K_neighbor_similarity_graph, get_cluster_labels_random_subsets, evaluate_label_discrimination
from stats import benjamini_hochberg_correction
from structlog._config import BoundLoggerLazyProxy

def get_chatML_format_lengths(tokenizer: PreTrainedTokenizer) -> Tuple[int, int]:
    """
    Get the lengths of the chatML format for the user and system roles.
    """
    test_str = "Hello, how are you?"
    test_str_len = len(tokenizer_base.encode(test_str, add_special_tokens=False))
    user_role_format_test_str = tokenizer_base.apply_chat_template([{'role': 'user', 'content': test_str}], tokenize=False, padding=False)
    user_role_format_test_str_len = len(tokenizer_base.encode(user_role_format_test_str, add_special_tokens=False))
    user_system_role_format_test_str = tokenizer_base.apply_chat_template([{'role': 'system', 'content': test_str}, {'role': 'user', 'content': test_str}], tokenize=False, padding=False)
    user_system_role_format_test_str_len = len(tokenizer_base.encode(user_system_role_format_test_str, add_special_tokens=False))
    return user_role_format_test_str_len - test_str_len, user_system_role_format_test_str_len - test_str_len

def format_prompts_for_decoding(
        prompts: List[str], 
        tokenizer: PreTrainedTokenizer, 
        format_prefix_chatML: bool, 
        model_prefix: Optional[str], 
        cot_start_token: Optional[str]
    ) -> Tuple[List[str], int]:
    """
    Format the prompts for decoding and determine how much length is added by the formatting.

    Args:
        prompts (List[str]): The prompts to format.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for formatting.
        format_prefix_chatML (bool): Whether to format the prefix using ChatML-style tags.
        model_prefix (Optional[str]): The prefix to add to the prompts.
        cot_start_token (Optional[str]): The token to add to the prompts.
    """
    # First, compute the length of all the prompts without the formatting
    prompts_no_formatting_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    
    if format_prefix_chatML:
        if model_prefix is not None and model_prefix != "":
            if cot_start_token is not None:
                prompts = [[{'role': 'system', 'content': model_prefix}, {'role': 'user', 'content': prompt + cot_start_token}] for prompt in prompts]
            else:
                prompts = [[{'role': 'system', 'content': model_prefix}, {'role': 'user', 'content': prompt}] for prompt in prompts]
        else:
            if cot_start_token is not None:
                prompts = [[{'role': 'user', 'content': prompt + cot_start_token}] for prompt in prompts]
            else:
                prompts = [[{'role': 'user', 'content': prompt}] for prompt in prompts]
        prompts = tokenizer.apply_chat_template(
            prompts,
            tokenize=False,  # Return strings, not tensors
            padding=False   # Will pad during decoding
        )
    else:
        if model_prefix is not None and model_prefix != "":
            if cot_start_token is not None:
                prompts = [model_prefix + prompt + cot_start_token for prompt in prompts]
            else:
                prompts = [model_prefix + prompt for prompt in prompts]
        else:
            if cot_start_token is not None:
                prompts = [prompt + cot_start_token for prompt in prompts]
            else:
                pass # prompts remain unchanged
    # Now compute the length of all the prompts with the formatting
    prompts_formatting_lengths = [len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts]
    # Compute the difference between the length of the prompts with and without the formatting
    prompts_formatting_lengths_diff = [prompts_formatting_lengths[i] - prompts_no_formatting_lengths[i] for i in range(len(prompts))]
    # Find the maximum difference
    max_formatting_length_diff = max(prompts_formatting_lengths_diff)
    print("Max formatting length diff:", max_formatting_length_diff)
    return prompts, max_formatting_length_diff


def setup_interpretability_method(
        base_model: Union[PreTrainedModel, str], 
        finetuned_model: Union[PreTrainedModel, str], 
        tokenizer_base: PreTrainedTokenizer,
        tokenizer_intervention: PreTrainedTokenizer,
        base_model_prefix: Optional[str] = None,
        intervention_model_prefix: Optional[str] = None,
        format_prefix_chatML: bool = False,
        cot_start_token_base: Optional[str] = None,
        cot_end_token_base: Optional[str] = None,
        cot_max_length_base: int = 512,
        cot_start_token_intervention: Optional[str] = None,
        cot_end_token_intervention: Optional[str] = None,
        cot_max_length_intervention: int = 512,
        n_decoded_texts: int = 2000, 
        decoding_prompt_file: Optional[str] = None, 
        max_number_of_prompts: Optional[int] = None,
        use_decoding_prompts_as_cluster_labels: bool = False,
        auth_key: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        client: Optional[Any] = None,
        api_interactions_save_loc: Optional[str] = None,
        logging_level: str = "INFO",
        logger: Optional[BoundLoggerLazyProxy] = None,
        local_embedding_model_str: str = "intfloat/multilingual-e5-large-instruct", 
        local_embedding_api_key: Optional[str] = None,
        init_clustering_from_base_model: bool = False,
        num_decodings_per_prompt: int = None,
        include_prompts_in_decoded_texts: bool = False,
        clustering_instructions: str = "Identify the topic or theme of the given texts",
        n_clustering_inits: int = 10,
        use_prompts_as_clusters: bool = False,
        cluster_on_prompts: bool = False,
        use_anthropic_evals_clusters: bool = False,
        anthropic_evals_cluster_id_list: Optional[List[int]] = None,
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
        threshold: float = 0.0,
        tsne_save_path: Optional[str] = None,
        tsne_title: Optional[str] = None,
        tsne_perplexity: int = 30,
        run_prefix: Optional[str] = None,
        global_random_seed: int = 42,
    ) -> Dict[str, Any]:
    """
    Set up the interpretability method to compare two models.

    This function prepares the necessary components for comparing two models, including
    decoding texts, generating embeddings, and performing clustering. It returns a dictionary
    containing the setup results for further analysis.

    Args:
        base_model (Union[PreTrainedModel, str]): The original, pre-finetuned model. Can also be a string,
            in which case it references an openrouter model that will respond via the openrouter API.
        finetuned_model (Union[PreTrainedModel, str]): The model after finetuning. Can also be a string,
            in which case it references an openrouter model that will respond via the openrouter API.
        tokenizer_base (PreTrainedTokenizer): The tokenizer to use for text generation.
        tokenizer_intervention (PreTrainedTokenizer): The tokenizer to use for text generation.
        base_model_prefix (Optional[str]): Prefix to add to all prompts for the base model.
        intervention_model_prefix (Optional[str]): Prefix to add to all prompts for the intervention model.
        format_prefix_chatML (bool): Whether to format the prefix using ChatML-style tags. False by default.
        cot_start_token_base (Optional[str]): Token to start the chain of thought for the base model.
        cot_end_token_base (Optional[str]): Token to end the chain of thought for the base model.
        cot_max_length_base (int): Max new tokens for CoT phase before continuation for the base model. Defaults to 512.
        cot_start_token_intervention (Optional[str]): Token to start the chain of thought for the intervention model.
        cot_end_token_intervention (Optional[str]): Token to end the chain of thought for the intervention model.
        cot_max_length_intervention (int): Max new tokens for CoT phase before continuation for the intervention model. 
            Defaults to 512.
        n_decoded_texts (int): The number of texts to decode with each model.
        decoding_prompt_file (Optional[str]): The path to a file containing a set of prompts to 
            prepend to the texts to be decoded.
        max_number_of_prompts (Optional[int]): The maximum number of prompts to use for decoding. None by default.
        use_decoding_prompts_as_cluster_labels (bool): Whether to use the decoding prompts as cluster labels.
        auth_key (Optional[str]): The API key to use for clustering and analysis.
        openrouter_api_key (Optional[str]): The API key to use for openrouter API calls.
        client (Optional[Any]): The client object for API calls.
        api_interactions_save_loc (Optional[str]): The path to save the API interactions to. None by default.
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging. None by default.
        local_embedding_model_str (str): The name of the local embedding model to use.
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
        use_anthropic_evals_clusters (bool): Whether to use the cluster assignments from the Anthropic evals repository.
            False by default.
        anthropic_evals_cluster_id_list (Optional[List[int]]): The list of cluster IDs to use from the Anthropic evals 
            repository. None by default.
        device (str): The device to use for clustering. "cuda:0" by default.
        cluster_method (str): The method to use for clustering. "kmeans" or "hdbscan".
        n_clusters (int): The number of clusters to use. 30 by default.
        min_cluster_size (int): The minimum size of a cluster. 7 by default.
        max_cluster_size (int): The maximum size of a cluster. 2000 by default.
        max_length (int): The maximum length of the decoded text responses, not including the prefixes and COTs.
        decoding_batch_size (int): The batch size to use for decoding. 32 by default.
        decoded_texts_save_path (Optional[str]): The path to save the decoded texts to. None by default.
        decoded_texts_load_path (Optional[str]): The path to load the decoded texts from. None by default.
        loaded_texts_subsample (Optional[int]): If specified, will randomly subsample the loaded decoded 
            texts to this number. None by default.
        path_to_MWE_repo (Optional[str]): Path to the an external repository of prompts corresponding to some
            benchmarking dataset.
        num_statements_per_behavior (Optional[int]): Number of statements per behavior to read from the evals 
            repository and then generate responses from.
        num_responses_per_statement (Optional[int]): Number of responses per statement to generate from the statements 
            in the evals repository.
        threshold (float): Threshold for deciding which behaviors to target for further investigation via difference 
            discovery. Set to 0 to deactivate.
        tsne_save_path (Optional[str]): The path to save the t-SNE plot to. None by default.
        tsne_title (Optional[str]): The title of the t-SNE plot. None by default.
        tsne_perplexity (int): The perplexity of the t-SNE plot. 30 by default.
        run_prefix (Optional[str]): The prefix to use for the run. None by default.
        global_random_seed (int): Global random seed to use for reproducibility.
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
        # Check if the path is to the anthropic evals repository
        if path_to_MWE_repo.endswith("evals"):
            print("Loading statements from MWE persona repository")
            # Check if the path exists
            if not path.exists(path_to_MWE_repo):
                raise FileNotFoundError(f"Path to MWE persona repository does not exist: {path_to_MWE_repo}")
            if isinstance(base_model, str):
                model_1_name = base_model
            else:
                model_1_name = base_model.config._name_or_path
            if isinstance(finetuned_model, str):
                model_2_name = finetuned_model
            else:
                model_2_name = finetuned_model.config._name_or_path
            statements, statements_cluster_assignments = load_statements_from_MWE_repo(
                path_to_MWE_repo,
                num_statements_per_behavior,
                threshold,
                model_1_name,
                model_2_name,
                anthropic_evals_cluster_id_list=anthropic_evals_cluster_id_list,
            )
            n_statements = len(statements)
            #statements = statements[n_statements // 2 + 10:]

            print("Number of statements loaded:", len(statements))
            print("Example statement:", statements[0])
            # Now format the statements into a list of prompts for decoding
            prompts = [f"Is the following statement something you would say?\n\"{statement}\"\nResponse: \"" for statement in statements]
        elif path_to_MWE_repo.endswith("MPI"):
            # Loading from the MPI repository
            print("Loading statements from MPI repository")
            statements = load_statements_from_MPI_repo(path_to_MWE_repo, num_statements_per_behavior)
            print("Number of statements loaded:", len(statements))
            print("Example statement:", statements[0])
            # Now format the statements into a list of prompts for decoding
            prompts = [f"Prompt: {statement}.\nResponse:" for statement in statements]
        elif 'jailbreak_llms' in path_to_MWE_repo:
            # Loading from the jailbreak_llms repository
            print("Loading statements from jailbreak_llms repository")
            statements = load_statements_from_jailbreak_llms_repo(path_to_MWE_repo, num_statements_per_behavior)
            print("Number of statements loaded:", len(statements))
            print("Example statement:", statements[0])
            prompts = [f"Prompt: {statement}.\nResponse:" for statement in statements]
        elif 'truthfulqa' in path_to_MWE_repo:
            # Loading from the truthfulqa repository
            print("Loading statements from truthfulqa dataset")
            statements = load_statements_from_truthfulqa(num_questions_to_load = max_number_of_prompts)
            print("Number of statements loaded:", len(statements))
            print("Example statement:", statements[0])
            prompts = [f"Question: {statement}\nResponse:" for statement in statements]
        elif 'bold' in path_to_MWE_repo:
            # Loading from the bold repository
            print("Loading statements from Amazon bold dataset")
            statements = load_statements_from_amazon_bold(num_questions_to_load = max_number_of_prompts)
            print("Number of statements loaded:", len(statements))
            print("Example statement:", statements[0])
            prompts = [f"Finish the following sentence: {statement}" for statement in statements]
        elif path_to_MWE_repo.startswith("generate_new_prompts:"):
            # Use an API assistant model to generate a new bank of prompts
            print("Generating new prompts from API assistant model")
            categories_of_interest = path_to_MWE_repo.split("generate_new_prompts:")[1]
            if categories_of_interest == "":
                categories_of_interest = None
            else:
                categories_of_interest = categories_of_interest.split(",")
            prompts = generate_new_prompts(
                categories_of_interest = categories_of_interest, 
                num_prompts_to_generate = max_number_of_prompts, 
                api_provider = "openrouter", 
                api_model_str = "openai/gpt-5-mini", 
                auth_key = openrouter_api_key, 
                client = client, 
                api_interactions_save_loc = api_interactions_save_loc, 
                logging_level=logging_level,
                logger = logger, 
                request_info = {"pipeline_stage": "generating new prompts"})
            print("Number of prompts generated:", len(prompts))
            print("Example prompt:", prompts[0])
        else:
            raise ValueError(f"Path to repository ({path_to_MWE_repo}) does not point to a known external dataset of prompts.")
        texts_decoded_per_prompt = num_responses_per_statement
        n_decoded_texts = None
        if max_number_of_prompts is not None:
            prompts = prompts[:max_number_of_prompts]
    else:
        texts_decoded_per_prompt = None
    
            # Initialize all cluster-related variables to None
    (base_loaded_cluster_indices, finetuned_loaded_cluster_indices,
         base_loaded_mauve_cluster_scores, finetuned_loaded_mauve_cluster_scores,
         base_loaded_kl_divergence_cluster, finetuned_loaded_kl_divergence_cluster,
         base_loaded_mean_entropy_cluster, finetuned_loaded_mean_entropy_cluster,
         deduplicated_prompts, deduplicated_prompt_index_to_cluster_indices) = (None,) * 10

    # Load the decoded texts from a prior run
    if decoded_texts_load_path is not None:
        if base_model_prefix is not None and base_model_prefix != "":
            print("Alert: base_model_prefix is not None or empty, but we are loading decoded texts from a prior run, so it will not be used.")
        if intervention_model_prefix is not None and intervention_model_prefix != "":
            print("Alert: intervention_model_prefix is not None or empty, but we are loading decoded texts from a prior run, so it will not be used.")
        if path_to_MWE_repo is not None:
            # Provided both a prior result from decodings and a repository from another source, raise an error
            print("Provided both a prior result from decodings and a repository from another source. Will currently only load the prior result.")
        print("Loading decoded texts from: ", decoded_texts_load_path)
        try:
            decoded_texts = pd.read_csv(decoded_texts_load_path, escapechar='\\')
            print("decoded_texts.shape:", decoded_texts.shape)
            if "prompt" not in decoded_texts.columns:
                print("No prompt column found in loaded data. Will infer prompts and decodings-per-prompt from the loaded data.")
                # Assume the loaded texts each start with a prompt, which has n different continuations:
                # prompt_1 + continuation_1, prompt_1 + continuation_2, ..., prompt_1 + continuation_n
                # prompt_2 + continuation_1, prompt_2 + continuation_2, ..., prompt_2 + continuation_n
                # ...
                # prompt_m + continuation_1, prompt_m + continuation_2, ..., prompt_m + continuation_n
                # We want to extract the prompts (which may be of different lengths).

                # Infer prompts and decodings-per-prompt from loaded data
                base_decoded_texts = decoded_texts[decoded_texts["model"] == "base"]["text"].tolist()
                finetuned_decoded_texts = decoded_texts[decoded_texts["model"] == "finetuned"]["text"].tolist()

                def _lcp_two(a: str, b: str) -> str:
                    i = 0
                    L = min(len(a), len(b))
                    while i < L and a[i] == b[i]:
                        i += 1
                    return a[:i]

                def _lcp_many(arr: List[str]) -> str:
                    if not arr:
                        return ""
                    p = arr[0]
                    for t in arr[1:]:
                        p = _lcp_two(p, t)
                        if not p:
                            break
                    return p

                def _best_group_size(texts: List[str], max_k: int = 1500, min_k: int = 50, min_lcp: int = 10) -> Optional[int]:
                    # Try candidate group sizes; pick the one with the highest median LCP across chunks
                    best_k, best_score = None, -1
                    max_k = min(max_k, len(texts))
                    for k in range(min_k, max_k + 1):
                        chunks = [texts[i:i+k] for i in range(0, len(texts), k)]
                        if not chunks or len(chunks[-1]) != k:
                            chunks = chunks[:-1]
                        if not chunks:
                            continue
                        lcp_lengths = [len(_lcp_many(c)) for c in chunks]
                        score = float(np.median(lcp_lengths))
                        if score >= best_score:
                            best_score, best_k = score, k
                    return best_k if best_score >= min_lcp else None

                k_base = _best_group_size(base_decoded_texts[:10000], max_k=1500, min_k=50, min_lcp=5)
                k_ft = _best_group_size(finetuned_decoded_texts[:10000], max_k=1500, min_k=50, min_lcp=5)
                k = k_base
                if k is None:
                    raise ValueError("Could not infer the number of decodings per prompt from loaded texts.")
                if k_base != k_ft:
                    print("k_base != k_ft. Will use k_base.")

                # Build prompts from base chunks (finetuned uses the same prompt order)
                base_chunks = [base_decoded_texts[i:i+k] for i in range(0, len(base_decoded_texts), k)]
                if base_chunks and len(base_chunks[-1]) != k:
                    base_chunks = base_chunks[:-1]
                finetuned_chunks = [finetuned_decoded_texts[i:i+k] for i in range(0, len(finetuned_decoded_texts), k)]
                if finetuned_chunks and len(finetuned_chunks[-1]) != k:
                    finetuned_chunks = finetuned_chunks[:-1]

                base_prompts = [_lcp_many(chunk) for chunk in base_chunks]
                finetuned_prompts = [_lcp_many(chunk) for chunk in finetuned_chunks]
                prompts = base_prompts
                texts_decoded_per_prompt = k
                print("texts_decoded_per_prompt:", texts_decoded_per_prompt)
                print("k:", k)
                print("len(base_chunks):", len(base_chunks))
                print("len(base_decoded_texts):", len(base_decoded_texts))
                print("len(finetuned_decoded_texts):", len(finetuned_decoded_texts))
                print("len(prompts):", len(prompts))
                print("Example prompts:", prompts[:10])
            

            elif "prompt" in decoded_texts.columns:
                print("Found prompt column in loaded data")
                # For when the saved data include explicit formatting of the prompts and texts
                # Get the prompts and texts from the loaded data
                base_prompts = decoded_texts[decoded_texts["model"] == "base"]["prompt"].tolist()
                finetuned_prompts = decoded_texts[decoded_texts["model"] == "finetuned"]["prompt"].tolist()
                base_texts = decoded_texts[decoded_texts["model"] == "base"]["text"].tolist()
                finetuned_texts = decoded_texts[decoded_texts["model"] == "finetuned"]["text"].tolist()
                base_decoded_texts = []
                for i, (prompt, text) in enumerate(zip(base_prompts, base_texts)):
                    # check if the generated text is nan
                    if isinstance(text, float) and pd.isna(text):
                        print(f"Text is nan for prompt {prompt} at index {i}")
                        text = "[System error: Generation failed for this prompt.]"
                    base_decoded_texts.append(prompt + text)
                finetuned_decoded_texts = []
                for i, (prompt, text) in enumerate(zip(finetuned_prompts, finetuned_texts)):
                    # check if the generated text is nan
                    if isinstance(text, float) and pd.isna(text):
                        print(f"Text is nan for prompt {prompt} at index {i}")
                        text = "[System error: Generation failed for this prompt.]"
                    finetuned_decoded_texts.append(prompt + text)

                # Now infer texts_decoded_per_prompt from the loaded data
                texts_decoded_per_prompt = len(base_decoded_texts) // len(set(base_prompts))
                print("texts_decoded_per_prompt:", texts_decoded_per_prompt)
            else:
                base_decoded_texts = decoded_texts[decoded_texts["model"] == "base"]["text"].tolist()
                finetuned_decoded_texts = decoded_texts[decoded_texts["model"] == "finetuned"]["text"].tolist()
            if loaded_texts_subsample is not None:
                sampled_indices = random.sample(range(len(base_decoded_texts)), loaded_texts_subsample)
                base_decoded_texts = [base_decoded_texts[i] for i in sampled_indices]
                finetuned_decoded_texts = [finetuned_decoded_texts[i] for i in sampled_indices]
                base_prompts = [base_prompts[i] for i in sampled_indices]
                finetuned_prompts = [finetuned_prompts[i] for i in sampled_indices]
                # Needs to be updated to handle the case where the prompts are not the same length as the texts
                #prompts = [prompts[i] for i in sampled_indices]

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
        # Load decoding prompts
        if decoding_prompt_file and path.exists(decoding_prompt_file):
            if path_to_MWE_repo is not None:
                # Provided both a decoding prompt file and a repository from another source, raise an error
                raise ValueError("Provided both a decoding prompt file and a repository from another source, please provide only one, as it is unclear which one to use.")
            with open(decoding_prompt_file, 'r') as f:
                prompts = [line.strip() for line in f.readlines()]
        elif path_to_MWE_repo is None:
            # Didn't load any prompts, so we will be decoding from the empty string
            prompts = None
            print("Decoding from empty string.")
            duplicated_base_prompts = ["" for _ in range(texts_decoded_per_prompt)]
            duplicated_finetuned_prompts = ["" for _ in range(texts_decoded_per_prompt)]

        if isinstance(base_model, GPTNeoXForCausalLM) or isinstance(base_model, Qwen2ForCausalLM):
            logging.set_verbosity_error()

        if isinstance(base_model, str):
            # Decode texts with OpenRouter
            duplicated_base_prompts = [prompt for prompt in prompts for _ in range(texts_decoded_per_prompt)]
            if base_model_prefix is not None and base_model_prefix != "":
                duplicated_base_prompts = [base_model_prefix + prompt for prompt in duplicated_base_prompts]
            if cot_start_token_base is not None:
                duplicated_base_prompts = [prompt + cot_start_token_base for prompt in duplicated_base_prompts]
            base_decoded_texts = parallel_make_api_requests(
                prompts=duplicated_base_prompts,
                api_provider="openrouter",
                api_model_str=base_model,
                auth_key=openrouter_api_key,
                client=client,
                api_interactions_save_loc=None,
                logging_level=logging_level,
                logger=None,
                request_info=None,
                max_tokens=max_length,
                cot_end_token=cot_end_token_base,
                cot_max_length=cot_max_length_base
            )
        else:
            base_prompts, _ = format_prompts_for_decoding(
                prompts=prompts,
                tokenizer=tokenizer_base,
                format_prefix_chatML=format_prefix_chatML,
                model_prefix=base_model_prefix,
                cot_start_token=cot_start_token_base
            )
            
            base_decoded_texts = batch_decode_texts(
                base_model, 
                tokenizer_base, 
                base_prompts, 
                n_decoded_texts=n_decoded_texts,
                texts_decoded_per_prefix=texts_decoded_per_prompt,
                max_answer_length=max_length,
                batch_size=decoding_batch_size,
                cot_end_token=cot_end_token_base,
                cot_max_length=cot_max_length_base
            )
            duplicated_base_prompts = [prompt for prompt in base_prompts for _ in range(texts_decoded_per_prompt)]
        if isinstance(finetuned_model, str):
            # Decode texts with OpenRouter
            duplicated_finetuned_prompts = [prompt for prompt in prompts for _ in range(texts_decoded_per_prompt)]
            if intervention_model_prefix is not None and intervention_model_prefix != "":
                duplicated_finetuned_prompts = [intervention_model_prefix + prompt for prompt in duplicated_finetuned_prompts]
            if cot_start_token_intervention is not None:
                duplicated_finetuned_prompts = [prompt + cot_start_token_intervention for prompt in duplicated_finetuned_prompts]
            finetuned_decoded_texts = parallel_make_api_requests(
                prompts=duplicated_finetuned_prompts,
                api_provider="openrouter",
                api_model_str=finetuned_model,
                auth_key=openrouter_api_key,
                client=client,
                api_interactions_save_loc=None,
                logging_level=logging_level,
                logger=None,
                request_info=None,
                max_tokens=max_length,
                cot_end_token=cot_end_token_intervention,
                cot_max_length=cot_max_length_intervention
            )
        else:
            finetuned_prompts, _ = format_prompts_for_decoding(
                prompts=prompts,
                tokenizer=tokenizer_intervention,
                format_prefix_chatML=format_prefix_chatML,
                model_prefix=intervention_model_prefix,
                cot_start_token=cot_start_token_intervention
            )
            finetuned_decoded_texts = batch_decode_texts(
                finetuned_model, 
                tokenizer_intervention, 
                finetuned_prompts, 
                n_decoded_texts=n_decoded_texts,
                texts_decoded_per_prefix=texts_decoded_per_prompt,
                max_answer_length=max_length,
                batch_size=decoding_batch_size,
                cot_end_token=cot_end_token_intervention,
                cot_max_length=cot_max_length_intervention
            )
            duplicated_finetuned_prompts = [prompt for prompt in finetuned_prompts for _ in range(texts_decoded_per_prompt)]


    if decoded_texts_save_path is not None:
        if decoded_texts_load_path is not None:
            print("Skipping save because decoded texts were loaded from disk.")
        else:
            def remove_leading_prefix(s: str, prefix: str) -> str:
                return s[len(prefix):] if s.startswith(prefix) else s

            base_decoded_texts_no_prompts = [
                remove_leading_prefix(text, prompt)
                for text, prompt in zip(base_decoded_texts, duplicated_base_prompts)
            ]
            finetuned_decoded_texts_no_prompts = [
                remove_leading_prefix(text, prompt)
                for text, prompt in zip(finetuned_decoded_texts, duplicated_finetuned_prompts)
            ]
            combined_texts = [('base', text, prompt) for text, prompt in zip(base_decoded_texts_no_prompts, duplicated_base_prompts)] + [('finetuned', text, prompt) for text, prompt in zip(finetuned_decoded_texts_no_prompts, duplicated_finetuned_prompts)]
            
            df = pd.DataFrame(combined_texts, columns=['model', 'text', 'prompt'])
            df.to_csv(decoded_texts_save_path, index=False, escapechar='\\')
            
            print(f"Decoded texts saved to: {decoded_texts_save_path}")
    
    if base_model_prefix is not None and base_model_prefix != "":
        base_prefix_char_length = len(base_model_prefix)
        base_decoded_texts = [text[base_prefix_char_length:] for text in base_decoded_texts]
        # if '\\n\\n' in base_model_prefix or '\n\n' in base_model_prefix:
        #     # find the first occurrence of '\n\n' in the decoded texts
        #     end_of_prefix = [text.find('\n\n') + 2 for text in base_decoded_texts]
        #     # check if we found the end of the prefix
        #     if -1 in end_of_prefix:
        #         # Assume the tokenizer_base has replaced \n with \n\n, so we need to find the first occurrence of \\n\\n
        #         end_of_prefix = [text.find('\\n\\n') + 4 for text in base_decoded_texts]
        #     base_decoded_texts = [text[end_of_prefix[i]:] for i, text in enumerate(base_decoded_texts)]
        # else:
        #     base_decoded_texts = [text.replace(base_model_prefix, "") for text in base_decoded_texts]
        print("Removed base model prefix from base decoded texts")
        print("Removed prefix:", base_model_prefix)
    if intervention_model_prefix is not None and intervention_model_prefix != "":
        intervention_prefix_char_length = len(intervention_model_prefix)
        finetuned_decoded_texts = [text[intervention_prefix_char_length:] for text in finetuned_decoded_texts]
        # if '\\n\\n' in intervention_model_prefix or '\n\n' in intervention_model_prefix:
        #     # find the first occurrence of '\n\n' in the decoded texts
        #     end_of_prefix = [text.find('\n\n') + 2 for text in finetuned_decoded_texts]
        #     # check if we found the end of the prefix
        #     if -1 in end_of_prefix:
        #         # Assume the tokenizer_intervention has replaced \n with \n\n, so we need to find the first occurrence of \\n\\n
        #         end_of_prefix = [text.find('\\n\\n') + 4 for text in finetuned_decoded_texts]
        #     finetuned_decoded_texts = [text[end_of_prefix[i]:] for i, text in enumerate(finetuned_decoded_texts)]
        # else:
        #     finetuned_decoded_texts = [text.replace(intervention_model_prefix, "") for text in finetuned_decoded_texts]
        print("Removed intervention model prefix from finetuned decoded texts")
        print("Removed prefix:", intervention_model_prefix)

    # Print out 20 randomly sampled decoded texts
    print("Base decoded texts:")
    for i, t in enumerate(random.sample(base_decoded_texts, k=min(20, len(base_decoded_texts)))):
        print(f"-Base {i}: {t}")
    print("Finetuned decoded texts:")
    for i, t in enumerate(random.sample(finetuned_decoded_texts, k=min(20, len(finetuned_decoded_texts)))):
        print(f"-Finetuned {i}: {t}")

    # strip out all tokens in the chains of thought from the decoded texts, if they are present
    if cot_start_token_base is not None and cot_end_token_base is not None:
        base_decoded_texts_cot_stripped = []
        for text in base_decoded_texts:
            idx_start = text.find(cot_start_token_base)
            idx_end = text.find(cot_end_token_base, idx_start + len(cot_start_token_base))
            if idx_start != -1 and idx_end != -1:
                stripped_text = text[:idx_start] + text[idx_end + len(cot_end_token_base):]
                base_decoded_texts_cot_stripped.append(stripped_text.replace("\"\"\n\n", ""))
            else:
                base_decoded_texts_cot_stripped.append(text)
        base_decoded_texts = base_decoded_texts_cot_stripped
        print("Stripped out chains of thought from base decoded texts")
        # Print out 10 randomly sampled base decoded texts after stripping out chains of thought
        print("Base decoded texts after stripping out chains of thought:")
        for i, t in enumerate(random.sample(base_decoded_texts, k=min(10, len(base_decoded_texts)))):
            print(f"-Base {i} without COT: {t}")
    
    if cot_start_token_intervention is not None and cot_end_token_intervention is not None:
        finetuned_decoded_texts_cot_stripped = []
        for text in finetuned_decoded_texts:
            idx_start = text.find(cot_start_token_intervention)
            idx_end = text.find(cot_end_token_intervention, idx_start + len(cot_start_token_intervention))
            if idx_start != -1 and idx_end != -1:
                stripped_text = text[:idx_start] + text[idx_end + len(cot_end_token_intervention):]
                finetuned_decoded_texts_cot_stripped.append(stripped_text.replace("\"\"\n\n", ""))
            else:
                finetuned_decoded_texts_cot_stripped.append(text)
        finetuned_decoded_texts = finetuned_decoded_texts_cot_stripped
        print("Stripped out chains of thought from finetuned decoded texts")
        # Print out 10 randomly sampled finetuned decoded texts after stripping out chains of thought
        print("Finetuned decoded texts after stripping out chains of thought:")
        for i, t in enumerate(random.sample(finetuned_decoded_texts, k=min(10, len(finetuned_decoded_texts)))):
            print(f"-Finetuned {i} without COT: {t}")
    

    # Generate embeddings for both sets of decoded texts
    bnb_config = BitsAndBytesConfig(load_in_4bit=True)
    if run_prefix is not None:
        embeddings_save_str = run_prefix + "_model"
    else:
        embeddings_save_str = "model"

    if use_anthropic_evals_clusters and tsne_save_path is None:
        # Can just use fake embeddings, since we won't be doing anything with them.
        base_embeddings = np.zeros((len(prompts if cluster_on_prompts else base_decoded_texts), 1))
        finetuned_embeddings = np.zeros((len(prompts if cluster_on_prompts else finetuned_decoded_texts), 1))
    else:
        if cluster_on_prompts:
            # (We assume base and finetuned have the same prompts, so we can use the same embeddings for both)
            prompt_embeddings = read_past_embeddings_or_generate_new(
                "pkl_embeddings/prompt_" + embeddings_save_str,
                None,
                prompts,
                local_embedding_model_str=local_embedding_model_str,
                device=device,
                recompute_embeddings=True,
                save_embeddings=False,
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
                recompute_embeddings=True,
                save_embeddings=False,
                clustering_instructions=clustering_instructions,
                bnb_config=bnb_config
            )
            finetuned_embeddings = read_past_embeddings_or_generate_new(
                "pkl_embeddings/finetuned_" + embeddings_save_str,
                None,
                finetuned_decoded_texts,
                local_embedding_model_str=local_embedding_model_str,
                device=device,
                recompute_embeddings=True,
                save_embeddings=False,
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


    elif use_anthropic_evals_clusters:
        # Use the cluster assignments from the Anthropic evals repository
        if not path_to_MWE_repo.endswith("evals"):
            raise ValueError("use_anthropic_evals_clusters requires that we load our prompts from the Anthropic evals repository")
        all_statements, all_statements_cluster_assignments = load_statements_from_MWE_repo(
            path_to_MWE_repo,
            num_statements_per_behavior,
            threshold,
            model_1_name,
            model_2_name,
            anthropic_evals_cluster_id_list=anthropic_evals_cluster_id_list,
        )

        # Normalize Anthropic cluster ids to 0..K-1
        stmt_cluster_arr = np.array(all_statements_cluster_assignments)
        unique_clusters = np.unique(stmt_cluster_arr)
        remap = {old: new for new, old in enumerate(unique_clusters.tolist())}
        stmt_cluster_arr = np.array([remap[c] for c in stmt_cluster_arr])
        n_clusters = len(unique_clusters)

        # Sanity check: decoded texts must be len(all_statements) * texts_decoded_per_prompt
        expected = len(statements) * texts_decoded_per_prompt
        if len(base_decoded_texts) != expected or len(finetuned_decoded_texts) != expected:
            raise ValueError(
                f"Decoded-texts/statement count mismatch: "
                f"expected {expected} (= {len(statements)} * {texts_decoded_per_prompt}), "
                f"got base={len(base_decoded_texts)}, finetuned={len(finetuned_decoded_texts)}"
            )

        # Map each decoding to the Anthropic cluster of its statement
        base_clustering_assignments = np.array([
            stmt_cluster_arr[i // texts_decoded_per_prompt] for i in range(len(base_decoded_texts))
        ])
        finetuned_clustering_assignments = np.array([
            stmt_cluster_arr[i // texts_decoded_per_prompt] for i in range(len(finetuned_decoded_texts))
        ])

        # Compute cluster centers
        if cluster_on_prompts:
            # base_embeddings/finetuned_embeddings hold prompt embeddings in this mode
            if len(base_embeddings) != len(all_statements) or len(finetuned_embeddings) != len(all_statements):
                raise ValueError(
                    f"Prompt-embedding length mismatch: expected {len(all_statements)}, "
                    f"got base={len(base_embeddings)}, finetuned={len(finetuned_embeddings)}"
                )
            cluster_centers_base = np.array([
                np.mean(base_embeddings[stmt_cluster_arr == i], axis=0) for i in range(n_clusters)
            ])
            cluster_centers_finetuned = np.array([
                np.mean(finetuned_embeddings[stmt_cluster_arr == i], axis=0) for i in range(n_clusters)
            ])
        else:
            cluster_centers_base = np.array([
                np.mean(base_embeddings[base_clustering_assignments == i], axis=0) for i in range(n_clusters)
            ])
            cluster_centers_finetuned = np.array([
                np.mean(finetuned_embeddings[finetuned_clustering_assignments == i], axis=0) for i in range(n_clusters)
            ])

        base_clustering = type('MockClustering', (), {
            'labels_': base_clustering_assignments,
            'cluster_centers_': cluster_centers_base
        })
        finetuned_clustering = type('MockClustering', (), {
            'labels_': finetuned_clustering_assignments,
            'cluster_centers_': cluster_centers_finetuned
        })
        print(f"Created {n_clusters} clusters based on Anthropic evals statements")
        print(f"Base cluster sizes: {[np.sum(base_clustering_assignments == i) for i in range(n_clusters)]}")
        print(f"Finetuned cluster sizes: {[np.sum(finetuned_clustering_assignments == i) for i in range(n_clusters)]}")
        
    elif use_prompts_as_clusters:
        # Each prompt represents a cluster, and the decoded texts are assigned to the cluster of their prompt
        # Cluster centroids are computed as the mean of the embeddings of the decoded texts assigned to each prompt
        if decoded_texts_load_path is None and path_to_MWE_repo is None:
            raise ValueError("use_prompts_as_clusters requires either decoded_texts_load_path or path_to_MWE_repo to be set")
        
        if path_to_MWE_repo is not None or decoded_texts_load_path is not None:
            # We either decoded texts from prompts in the MWE repository or got prompts from the loaded data
            # We know how many statements/prompts we have and how many responses per statement
            n_clusters = len(prompts)
            
            # Create cluster assignments where all decodings from the same prompt go to the same cluster
            base_clustering_assignments = np.array([i // texts_decoded_per_prompt for i in range(len(base_decoded_texts))])
            finetuned_clustering_assignments = np.array([i // texts_decoded_per_prompt for i in range(len(finetuned_decoded_texts))])
            
            # Track prompt-to-cluster mapping
            deduplicated_prompts = prompts
            deduplicated_prompt_index_to_cluster_indices = {
                i: [i * texts_decoded_per_prompt + j for j in range(texts_decoded_per_prompt)]
                for i in range(n_clusters)
            }
        elif "prompt" in decoded_texts.columns:
            # Legacy code; should not be used if I understand correctly
            raise ValueError("Entering this code path is not supported anymore. Check what led to this code running.")
            
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
        target_cluster_size = len(prompt_embeddings) // n_clusters
        min_cluster_size = max(2, target_cluster_size - 3)
        max_cluster_size = min(len(prompt_embeddings) // 2, target_cluster_size + 3)
        prompt_clustering = KMeansConstrained(n_clusters=n_clusters, random_state=global_random_seed, n_init=n_clustering_inits, size_min=min_cluster_size, size_max=max_cluster_size).fit(prompt_embeddings)
        
        # Now we derive the cluster assignments for the decoded texts from the prompt clustering
        # Then create mock clustering objects to match the expected format
        # First, derive the cluster assignments: a list that maps each decoding to the cluster of its prompt
        base_decoding_to_prompt_cluster = []
        finetuned_decoding_to_prompt_cluster = []
        for i in range(len(prompts)):
            current_prompt_cluster = prompt_clustering.labels_[i]
            # Assume we have texts_decoded_per_prompt texts decoded for each prompt
            for _ in range(texts_decoded_per_prompt):
                base_decoding_to_prompt_cluster.append(current_prompt_cluster)
                finetuned_decoding_to_prompt_cluster.append(current_prompt_cluster)
        base_clustering_assignments = np.array(base_decoding_to_prompt_cluster)
        finetuned_clustering_assignments = np.array(finetuned_decoding_to_prompt_cluster)

        # print the first 10 elements of base_decoding_to_prompt_cluster and their corresponding decoded texts
        for i in range(10):
            print(f"base_decoding_to_prompt_cluster[{i}]: {base_decoding_to_prompt_cluster[i]}")
            print(f"base_decoded_texts[{i}]: {base_decoded_texts[i]}")
        # print the first 10 elements of finetuned_decoding_to_prompt_cluster and their corresponding decoded texts
        for i in range(10):
            print(f"finetuned_decoding_to_prompt_cluster[{i}]: {finetuned_decoding_to_prompt_cluster[i]}")
            print(f"finetuned_decoded_texts[{i}]: {finetuned_decoded_texts[i]}")
        # print the sizes of each cluster
        print(f"Base cluster sizes: {[np.sum(base_clustering_assignments == i) for i in range(n_clusters)]}")
        print(f"Finetuned cluster sizes: {[np.sum(finetuned_clustering_assignments == i) for i in range(n_clusters)]}")

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
        # Perform clustering on both sets of embeddings (if not using decoding prompts as cluster labels)
        if not use_decoding_prompts_as_cluster_labels:
            if cluster_method == "kmeans":
                if len(base_embeddings) > 10000:  # threshold for switching to MiniBatch
                    base_clustering = MiniBatchKMeans(
                        n_clusters=n_clusters, 
                        random_state=global_random_seed, 
                        n_init=n_clustering_inits,
                        batch_size=1000,
                        max_iter=100
                    ).fit(base_embeddings)
                else:
                    base_clustering = KMeans(n_clusters=n_clusters, random_state=global_random_seed, n_init=n_clustering_inits).fit(base_embeddings)
                if init_clustering_from_base_model:
                    initial_centroids = base_clustering.cluster_centers_
                    finetuned_clustering = KMeans(n_clusters=n_clusters, random_state=global_random_seed, n_init=1, init=initial_centroids).fit(finetuned_embeddings)
                else:
                    if len(finetuned_embeddings) > 10000:
                        finetuned_clustering = MiniBatchKMeans(
                            n_clusters=n_clusters, 
                            random_state=global_random_seed, 
                            n_init=n_clustering_inits,
                            batch_size=1000,
                            max_iter=100
                        ).fit(finetuned_embeddings)
                    else:
                        finetuned_clustering = KMeans(n_clusters=n_clusters, random_state=global_random_seed, n_init=n_clustering_inits).fit(finetuned_embeddings)
            elif cluster_method == "hdbscan":
                base_clustering = HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, random_state=global_random_seed).fit(base_embeddings)
                finetuned_clustering = HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, random_state=global_random_seed).fit(finetuned_embeddings)
        
            print("Found", len(set(base_clustering.labels_)), "clusters for base model")
            print("Found", len(set(finetuned_clustering.labels_)), "clusters for finetuned model")
        
        else:
            # Create cluster assignments from decoding prompts, matching the expected format
            texts_per_prompt = n_decoded_texts // len(prompts)
            
            # Create cluster assignments based on prompt order
            base_clustering_assignments = np.array([i // texts_per_prompt for i in range(n_decoded_texts)])
            finetuned_clustering_assignments = np.array([i // texts_per_prompt for i in range(n_decoded_texts)])
            
            # Calculate cluster centers as the mean of embeddings for each cluster
            cluster_centers_base = np.array([
                np.mean(base_embeddings[base_clustering_assignments == i], axis=0)
                for i in range(len(prompts))
            ])
            cluster_centers_finetuned = np.array([
                np.mean(finetuned_embeddings[finetuned_clustering_assignments == i], axis=0)
                for i in range(len(prompts))
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
        print("Plotting t-SNE...")
        print("Saving t-SNE plot to", tsne_save_path)
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


def get_individual_labels(
    decoded_strs: List[str],
    clustering_assignments: List[int],
    local_model: PreTrainedModel,
    labeling_tokenizer: PreTrainedTokenizer,
    api_provider: str,
    api_model_str: str,
    api_stronger_model_str: Optional[str] = None,
    auth_key: Optional[str] = None,
    client: Optional[Union[Anthropic, OpenAI, Client]] = None,
    device: str = "cuda:0",
    sampled_texts_per_cluster: int = 10,
    generated_labels_per_cluster: int = 3,
    cluster_ids_to_prompt_ids_to_decoding_ids_dict: Dict = None,
    num_decodings_per_prompt: int = None,
    single_cluster_label_instruction: Optional[str] = None,
    max_unitary_comparisons_per_label: int = 50,
    api_interactions_save_loc: Optional[str] = None,
    logging_level: str = "INFO",
    logger: Optional[BoundLoggerLazyProxy] = None,
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
        client (Optional[Union[Anthropic, OpenAI, Client]]): The client to use for the API request.
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
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
    Returns:
        Dict[int, Tuple[str, float]]: A dictionary mapping cluster IDs to tuples of
        (best_label, best_AUC_score).
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
        logging_level=logging_level,
        logger=logger
    )
    
    labels_with_auc_scores = {}
    for cluster_id, labels in cluster_labels.items():
        auc_scores = []
        for label in labels:
            accuracy, p_value, auc_score = evaluate_label_discrimination(
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
                logging_level=logging_level,
                logger=logger,
                cluster_id_1=cluster_id,
                cluster_id_2=None,
            )
            auc_scores.append(auc_score)
        best_label_index = max(range(len(auc_scores)), key=auc_scores.__getitem__)
        labels_with_auc_scores[cluster_id] = (labels[best_label_index], auc_scores[best_label_index])
    
    return labels_with_auc_scores

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
    Returns:
        str: A string containing formatted ASCII tables representing the analysis results.
    """
    tables = []

    # Table for base clusters
    for i, cluster in enumerate(results['base_clusters']):
        data = [
            ['Base Cluster', f"Cluster {i}"],
            ['Label', f"{cluster['label']} (accuracy: {cluster['label_metric_score']:.3f})"],
            ['Size', str(cluster['size'])],
            ['Sample Texts', '\n'.join(cluster['sample_texts'])]
        ]

        if cluster['matching_finetuned_clusters']:
            data.append(['Matching Finetuned Clusters', ''])
            for j, match in enumerate(cluster['matching_finetuned_clusters']):
                data.extend([
                    [f'Match {j} Label', f"{match['label']} (accuracy: {match['label_metric_score']:.3f})"],
                    [f'Match {j} Contrastive Label', f"{match['contrastive_label']} (accuracy: {match['contrastive_metric_score']:.3f})"],
                    [f'Match {j} Size', str(match['size'])],
                    [f'Match {j} Sample Texts', '\n'.join(match['sample_texts'])]
                ])
            data.append(['Size Comparison', f"Absolute Diff: {cluster['size_comparison']['absolute_difference']}, " 
                                            f"Percentage Diff: {cluster['size_comparison']['percentage_difference']:.2f}%"])

        if cluster['unmatching_finetuned_clusters']:
            data.append(['Unmatching Finetuned Clusters', ''])
            for j, unmatch in enumerate(cluster['unmatching_finetuned_clusters']):
                data.extend([
                    [f'Unmatch {j} Label', f"{unmatch['label']} (accuracy: {unmatch['label_metric_score']:.3f})"],
                    [f'Unmatch {j} Contrastive Label', f"{unmatch['contrastive_label']} (accuracy: {unmatch['contrastive_metric_score']:.3f})"],
                    [f'Unmatch {j} Size', str(unmatch['size'])],
                    [f'Unmatch {j} Sample Texts', '\n'.join(unmatch['sample_texts'])]
                ])

        tables.append(AsciiTable(data, title=f'Base Cluster {i}').table)

    # Table for new finetuned clusters
    for i, cluster in enumerate(results['new_finetuned_clusters']):
        data = [
            ['New Finetuned Cluster', f"Cluster {i}"],
            ['Label', f"{cluster['label']} (accuracy: {cluster['label_metric_score']:.3f})"],
            ['Size', str(cluster['size'])],
            ['Sample Texts', '\n'.join(cluster['sample_texts'])]
        ]

        for j, neighbor in enumerate(cluster['nearest_base_neighbors']):
            data.extend([
                [f'Nearest Base Neighbor {j} Label', f"{neighbor['label']} (accuracy: {neighbor['label_metric_score']:.3f})"],
                [f'Nearest Base Neighbor {j} Contrastive Label', f"{neighbor['contrastive_label']} (accuracy: {neighbor['contrastive_metric_score']:.3f})"],
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
    tokenizer_base: PreTrainedTokenizer,
    tokenizer_intervention: PreTrainedTokenizer,
    base_model_prefix: Optional[str] = None,
    intervention_model_prefix: Optional[str] = None,
    format_prefix_chatML: bool = False,
    cot_start_token_base: Optional[str] = None,
    cot_end_token_base: Optional[str] = None,
    cot_max_length_base: int = 512,
    cot_start_token_intervention: Optional[str] = None,
    cot_end_token_intervention: Optional[str] = None,
    cot_max_length_intervention: int = 512,
    n_decoded_texts: int = 2000, 
    decoding_prompt_file: Optional[str] = None, 
    max_number_of_prompts: Optional[int] = None,
    api_provider: str = "anthropic",
    api_model_str: str = "claude-3-haiku-20240307",
    api_stronger_model_str: Optional[str] = None,
    max_labeling_tokens: Optional[int] = None,
    auth_key: Optional[str] = None,
    openrouter_api_key: Optional[str] = None,
    client: Optional[Union[Anthropic, OpenAI, Client]] = None,
    local_embedding_model_str: str = "intfloat/multilingual-e5-large-instruct", 
    local_embedding_api_key: Optional[str] = None,
    init_clustering_from_base_model: bool = False,
    clustering_instructions: str = "Identify the topic or theme of the given texts",
    n_clustering_inits: int = 10,
    use_prompts_as_clusters: bool = False,
    cluster_on_prompts: bool = False,
    use_anthropic_evals_clusters: bool = False,
    anthropic_evals_cluster_id_list: Optional[List[int]] = None,
    device: str = "cuda:0",
    cluster_method: str = "kmeans",
    n_clusters: int = 30,
    min_cluster_size: int = 7,
    max_cluster_size: int = 2000,
    sampled_texts_per_cluster: int = 10,
    sampled_comparison_texts_per_cluster: int = 50,
    cross_validate_contrastive_labels: bool = False,
    cross_validate_on_all_clusters: bool = True,
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
    threshold: float = 0.0,
    num_rephrases_for_validation: int = 0,
    num_generated_texts_per_description: int = 20,
    generated_labels_per_cluster: int = 3,
    num_decodings_per_prompt: int = None,
    include_prompts_in_decoded_texts: bool = False,
    single_cluster_label_instruction: Optional[str] = None,
    contrastive_cluster_label_instruction: Optional[str] = None,
    label_diversification_str_instructions: Optional[str] = None,
    diversify_contrastive_labels: bool = False,
    verified_diversity_promoter: bool = False,
    generate_individual_labels: bool = False,
    use_unitary_comparisons: bool = False,
    max_unitary_comparisons_per_label: int = 50,
    match_cutoff: float = 0.6,
    discriminative_query_rounds: int = 3,
    discriminative_validation_runs: int = 5,
    n_permutations: int = 0,
    split_clusters_by_prompt: bool = True,
    frac_prompts_for_label_generation: float = 0.5,
    tsne_save_path: Optional[str] = None,
    tsne_title: Optional[str] = None,
    tsne_perplexity: int = 30,
    api_interactions_save_loc: Optional[str] = None,
    logging_level: str = "INFO",
    logger: Optional[BoundLoggerLazyProxy] = None,
    run_prefix: Optional[str] = None,
    save_addon_str: Optional[str] = None,
    graph_load_path: Optional[str] = None,
    scoring_results_load_path: Optional[str] = None,
    global_random_seed: int = 42,
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
        tokenizer_base (PreTrainedTokenizer): The tokenizer for text generation.
        tokenizer_intervention (PreTrainedTokenizer): The tokenizer for text generation.
        base_model_prefix (Optional[str]): Prefix to add to all prompts for the base model.
        intervention_model_prefix (Optional[str]): Prefix to add to all prompts for the intervention model.
        format_prefix_chatML (bool): Whether to format the prefix using ChatML-style tags. False by default.
        cot_start_token_base (Optional[str]): Token to start the chain of thought for the base model.
        cot_end_token_base (Optional[str]): Token to end the chain of thought for the base model.
        cot_max_length_base (int): Max new tokens for CoT phase before continuation for the base model. Defaults to 512.
        cot_start_token_intervention (Optional[str]): Token to start the chain of thought for the intervention model.
        cot_end_token_intervention (Optional[str]): Token to end the chain of thought for the intervention model.
        cot_max_length_intervention (int): Max new tokens for CoT phase before continuation for the intervention model. 
            Defaults to 512.
        n_decoded_texts (int): Number of texts to decode from each model.
        decoding_prompt_file (Optional[str]): File containing prefixes for text generation.
        max_number_of_prompts (Optional[int]): The maximum number of prompts to use for decoding. None by default.
        api_provider (str): API provider for clustering and analysis.
        api_model_str (str): API model to use.
        api_stronger_model_str (Optional[str]): API model to use for stronger model, only used for the most important tasks.
        max_labeling_tokens (Optional[int]): Max new tokens for labeling.
        auth_key (Optional[str]): Authentication key for API calls.
        openrouter_api_key (Optional[str]): API key for openrouter API calls.
        client (Optional[Any]): Client object for API calls.
        local_embedding_model_str (str): Name of local embedding model.
        local_embedding_api_key (Optional[str]): API key for local embedding model.
        init_clustering_from_base_model (bool): Whether to initialize finetuned clustering from base model.
        clustering_instructions (str): Instructions for clustering.
        n_clustering_inits (int): Number of clustering initializations to use.
        use_prompts_as_clusters (bool): Whether to use the prompts as the clusters. False by default.
        cluster_on_prompts (bool): Whether to cluster on the prompts or the decoded texts. False by default.
        use_anthropic_evals_clusters (bool): Whether to use the cluster assignments from the Anthropic evals repository.
            False by default.
        anthropic_evals_cluster_id_list (Optional[List[int]]): The list of cluster IDs to use from the Anthropic evals 
            repository. None by default.
        device (str): Device to use for computations.
        cluster_method (str): Clustering method to use.
        n_clusters (int): Number of clusters.
        min_cluster_size (int): Minimum cluster size.
        max_cluster_size (int): Maximum cluster size.
        sampled_comparison_texts_per_cluster (int): Number of texts to sample per cluster for both evaluation
            of label accuracy / AUCs between clusters and for individual labels.
        cross_validate_contrastive_labels (bool): Whether to cross-validate the contrastive labels by testing the 
            discriminative score of the labels on different clusters from which they were generated.
        cross_validate_on_all_clusters (bool): Whether to cross-validate the contrastive labels by testing the 
            discriminative score of the labels on texts sampled from all other clusters.
        sampled_texts_per_cluster (int): Number of texts to sample per cluster when generating labels.
        K (int): Number of neighbors for K-neighbor similarity graph.
        match_by_ids (bool): Whether to match clusters by their IDs, not embedding distances.
        max_length (int): Maximum length of decoded text responses, not including the prefixes and COTs.
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
        threshold (float): Threshold for deciding which behaviors to target for further investigation via difference 
            discovery.
        num_rephrases_for_validation (int): Number of rephrases for validation.
        num_generated_texts_per_description (int): Number of generated texts per description for generative validation.
        generated_labels_per_cluster (int): Number of labels to generate per cluster (for both contrastive
            and individual labels).
        num_decodings_per_prompt (int): Number of decodings per prompt to use when generating labels.
        include_prompts_in_decoded_texts (bool): Whether to include the prompts in the decoded texts. False by default.
        single_cluster_label_instruction (Optional[str]): Instructions for generating the single cluster labels.
        contrastive_cluster_label_instruction (Optional[str]): Instructions for generating the contrastive cluster labels.
        diversify_contrastive_labels (bool): Whether to diversify the contrastive labels by clustering the previously 
            generated labels, and then using the assistant to summarize the common themes across the labels closest to 
            the cluster centers. Then we provide those summaries to the assistant to generate new labels that are different 
            from the previous ones.
        label_diversification_str_instructions (Optional[str]): Instructions for diversifying the contrastive labels.
        verified_diversity_promoter (bool): Whether to promote diversity in the contrastive labels by recording any 
            hypotheses that are verified discriminatively, providing them to the assistant, and asking the assistant to 
            look for other hypotheses that are different.
        generate_individual_labels (bool): Whether to generate individual labels for each cluster. False by default.
        use_unitary_comparisons (bool): Whether to use unitary comparisons.
        max_unitary_comparisons_per_label (int): Maximum number of unitary comparisons to perform per label.
            I.e., when validating the accuracy / AUC of a given label (either contrastive or individual), we will ask 
            the assistant to use the label to classify at most max_unitary_comparisons_per_label texts.
        match_cutoff (float): Accuracy / AUC cutoff for determining matching/unmatching clusters.
        discriminative_query_rounds (int): Number of rounds of discriminative queries to perform.
        discriminative_validation_runs (int): Number of validation runs to perform for each model for each hypothesis.
        n_permutations (int): Number of permutations to perform for the permutation test.
        split_clusters_by_prompt (bool): Whether to split the clusters by prompt during discriminative evaluation of the labels.
        frac_prompts_for_label_generation (float): Fraction of prompts to use for label generation.
        tsne_save_path (Optional[str]): Path to save t-SNE plot.
        tsne_title (Optional[str]): Title for t-SNE plot.
        tsne_perplexity (int): Perplexity for t-SNE.
        api_interactions_save_loc (Optional[str]): Where to record interactions with the API model, if anywhere.
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        run_prefix (Optional[str]): Prefix for the current run.
        save_addon_str (Optional[str]): Addon string for the current run.
        graph_load_path (Optional[str]): Path to load the graph from.
        scoring_results_load_path (Optional[str]): Path to load the scoring results from.
        global_random_seed (int): Global random seed to use for reproducibility.
    Returns:
        Tuple[str, str, List[str]]: A tuple containing:
            - results: JSON string with detailed analysis results
            - table_output: Human-readable string with formatted table output of the analysis
            - validated_hypotheses: List of validated hypotheses about how the two models differ in behavior
    """
    print(f"base_model: {base_model}")
    print(f"finetuned_model: {finetuned_model}")
    setup = setup_interpretability_method(
        base_model=base_model,
        finetuned_model=finetuned_model,
        tokenizer_base=tokenizer_base,
        tokenizer_intervention=tokenizer_intervention,
        base_model_prefix=base_model_prefix,
        intervention_model_prefix=intervention_model_prefix,
        format_prefix_chatML=format_prefix_chatML,
        cot_start_token_base=cot_start_token_base,
        cot_end_token_base=cot_end_token_base,
        cot_max_length_base=cot_max_length_base,
        cot_start_token_intervention=cot_start_token_intervention,
        cot_end_token_intervention=cot_end_token_intervention,
        cot_max_length_intervention=cot_max_length_intervention,
        n_decoded_texts=n_decoded_texts,
        decoding_prompt_file=decoding_prompt_file,
        max_number_of_prompts=max_number_of_prompts,
        auth_key=auth_key,
        openrouter_api_key=openrouter_api_key,
        client=client,
        api_interactions_save_loc=api_interactions_save_loc,
        logging_level=logging_level,
        logger=logger,
        local_embedding_model_str=local_embedding_model_str,
        local_embedding_api_key=local_embedding_api_key,
        init_clustering_from_base_model=init_clustering_from_base_model,
        num_decodings_per_prompt=num_decodings_per_prompt,
        include_prompts_in_decoded_texts=include_prompts_in_decoded_texts,
        clustering_instructions=clustering_instructions,
        n_clustering_inits=n_clustering_inits,
        use_prompts_as_clusters=use_prompts_as_clusters,
        cluster_on_prompts=cluster_on_prompts,
        use_anthropic_evals_clusters=use_anthropic_evals_clusters,
        anthropic_evals_cluster_id_list=anthropic_evals_cluster_id_list,
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
        threshold=threshold,
        tsne_save_path=tsne_save_path,
        tsne_title=tsne_title,
        tsne_perplexity=tsne_perplexity,
        run_prefix=run_prefix,
        global_random_seed=global_random_seed
    )

    base_clustering = setup["base_clustering"]
    finetuned_clustering = setup["finetuned_clustering"]
    base_embeddings = setup["base_embeddings"]
    finetuned_embeddings = setup["finetuned_embeddings"]
    base_decoded_texts = setup["base_decoded_texts"]
    print(f"len(base_decoded_texts): {len(base_decoded_texts)}")
    finetuned_decoded_texts = setup["finetuned_decoded_texts"]
    print(f"len(finetuned_decoded_texts): {len(finetuned_decoded_texts)}")
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
        if generate_individual_labels:
            base_labels = get_individual_labels(
                base_decoded_texts,
                base_clustering.labels_,
                None,
                tokenizer_base,
                api_provider,
                api_model_str,
                api_stronger_model_str,
                auth_key,
                client=client,
                device=device,
                sampled_texts_per_cluster=sampled_texts_per_cluster,
                generated_labels_per_cluster=generated_labels_per_cluster,
                cluster_ids_to_prompt_ids_to_decoding_ids_dict=cluster_ids_to_prompt_ids_to_decoding_ids_dict_1,
                num_decodings_per_prompt=num_decodings_per_prompt,
                single_cluster_label_instruction=single_cluster_label_instruction,
                max_unitary_comparisons_per_label=max_unitary_comparisons_per_label,
                api_interactions_save_loc=api_interactions_save_loc,
                logging_level=logging_level,
                logger=logger,
            )
            print(f"Base labels: {base_labels}")
            finetuned_labels = get_individual_labels(
                finetuned_decoded_texts,
                finetuned_clustering.labels_,
                None,
                tokenizer_intervention,
                api_provider,
                api_model_str,
                api_stronger_model_str,
                auth_key,
                client=client,
                device=device,
                sampled_texts_per_cluster=sampled_texts_per_cluster,
                generated_labels_per_cluster=generated_labels_per_cluster,
                cluster_ids_to_prompt_ids_to_decoding_ids_dict=cluster_ids_to_prompt_ids_to_decoding_ids_dict_2,
                num_decodings_per_prompt=num_decodings_per_prompt,
                single_cluster_label_instruction=single_cluster_label_instruction,
                max_unitary_comparisons_per_label=max_unitary_comparisons_per_label,
                api_interactions_save_loc=api_interactions_save_loc,
                logging_level=logging_level,
                logger=logger,
            )
            print(f"Finetuned labels: {finetuned_labels}")
        else:
            base_labels = {i: (f"Cluster {i} base", -1.0) for i in range(n_base_clusters)}
            finetuned_labels = {i: (f"Cluster {i} finetuned", -1.0) for i in range(n_finetuned_clusters)}
        
        # Build list of decoded texts prompt IDs
        base_decoded_texts_prompt_ids = [i // num_responses_per_statement for i in range(len(base_decoded_texts))]
        finetuned_decoded_texts_prompt_ids = [i // num_responses_per_statement for i in range(len(finetuned_decoded_texts))]

        # Build the K-neighbor similarity graph
        graph, results_dict = build_contrastive_K_neighbor_similarity_graph(
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
            tokenizer_base,
            api_provider,
            api_model_str,
            api_stronger_model_str,
            max_labeling_tokens=max_labeling_tokens,
            auth_key=auth_key,
            client=client,
            local_embedding_model_str=local_embedding_model_str,
            device=device,
            sampled_comparison_texts_per_cluster=sampled_comparison_texts_per_cluster,
            cross_validate_contrastive_labels=cross_validate_contrastive_labels,
            cross_validate_on_all_clusters=cross_validate_on_all_clusters,
            sampled_texts_per_cluster=sampled_texts_per_cluster,
            generated_labels_per_cluster=generated_labels_per_cluster,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_1=cluster_ids_to_prompt_ids_to_decoding_ids_dict_1,
            cluster_ids_to_prompt_ids_to_decoding_ids_dict_2=cluster_ids_to_prompt_ids_to_decoding_ids_dict_2,
            num_decodings_per_prompt=num_decodings_per_prompt,
            contrastive_cluster_label_instruction=contrastive_cluster_label_instruction,
            label_diversification_str_instructions=label_diversification_str_instructions,
            diversify_contrastive_labels=diversify_contrastive_labels,
            verified_diversity_promoter=verified_diversity_promoter,
            use_unitary_comparisons=use_unitary_comparisons,
            max_unitary_comparisons_per_label=max_unitary_comparisons_per_label,
            api_interactions_save_loc=api_interactions_save_loc,
            logging_level=logging_level,
            logger=logger,
            run_prefix=run_prefix,
            n_permutations=n_permutations,
            split_clusters_by_prompt=split_clusters_by_prompt,
            frac_prompts_for_label_generation=frac_prompts_for_label_generation,
            base_decoded_texts_prompt_ids=base_decoded_texts_prompt_ids,
            finetuned_decoded_texts_prompt_ids=finetuned_decoded_texts_prompt_ids,
            random_seed=global_random_seed,
        )
        # Identify significant hypotheses via Benjamini-Hochberg correction
        n_significant_binomial_p_values, significance_threshold_binomial_p_values, significant_binomial_mask = benjamini_hochberg_correction(results_dict["accuracy_binomial_p_values"])
        n_significant_auc_permutation_p_values, significance_threshold_auc_permutation_p_values, significant_auc_permutation_mask = benjamini_hochberg_correction(results_dict["auc_permutation_p_values"])

        # Now save all hypotheses and scores to a TSV file via pandas, in "results_tsvs/{run_prefix}.tsv"
        df = pd.DataFrame({
            "hypothesis": results_dict["hypotheses"],
            "accuracy": results_dict["accuracy_validation_scores"],
            "auc": results_dict["auc_validation_scores"],
            "accuracy_binomial_p_value": results_dict["accuracy_binomial_p_values"],
            "auc_permutation_p_value": results_dict["auc_permutation_p_values"],
            "significant_accuracy_binomial": significant_binomial_mask,
            "significant_auc_permutation": significant_auc_permutation_mask,
            "cross_validation_accuracy": results_dict["accuracy_cross_validation_scores"],
            "cross_validation_auc": results_dict["auc_cross_validation_scores"],
            "baseline_accuracy": results_dict["baseline_accuracy_validation_scores"],
            "baseline_auc": results_dict["baseline_auc_validation_scores"],
            "baseline_cross_validation_accuracy": results_dict["baseline_accuracy_cross_validation_scores"],
            "baseline_cross_validation_auc": results_dict["baseline_auc_cross_validation_scores"],
        })
        tsvs_save_path = path.join("results_tsvs", f"{run_prefix}.tsv")
        df.to_csv(tsvs_save_path, sep="\t", index=False)

        # Now print number of significant hypotheses and the significance thresholds
        print(f"Number of significant hypotheses (binomial p-values): {n_significant_binomial_p_values}")
        print(f"Significance threshold (binomial p-values): {significance_threshold_binomial_p_values}")
        print(f"Number of significant hypotheses (AUC permutation p-values): {n_significant_auc_permutation_p_values}")
        print(f"Significance threshold (AUC permutation p-values): {significance_threshold_auc_permutation_p_values}")

        return results_dict