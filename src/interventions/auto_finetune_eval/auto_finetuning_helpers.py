"""
Helper functions and dummy implementations for the auto-finetuning evaluation project.

This module contains utility functions and placeholder implementations for the
interpretability method application and model finetuning. These functions are used
across the project to support the main workflow of generating ground truths,
finetuning models, and evaluating interpretability methods.
"""

from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
import random
from anthropic import Anthropic
from openai import OpenAI
import json
import argparse
import torch

import sys
sys.path.append("../../contrastive-decoding/")
from quick_cluster import read_past_embeddings_or_generate_new, match_clusterings, get_validated_contrastive_cluster_labels, validated_assistant_generative_compare
from bitsandbytes import BitsAndBytesConfig
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
        prefixes: List[str], 
        n_decoded_texts: int, 
        batch_size: int = 16
    ) -> List[str]:
    """
    Decode texts in batches using the given model.

    Args:
        model (PreTrainedModel): The model to use for text generation.
        prefixes (List[str]): List of prefixes to use for text generation.
        n_decoded_texts (int): Total number of texts to generate.
        batch_size (int): Number of texts to generate in each batch.

    Returns:
        List[str]: List of generated texts.
    """
    decoded_texts = []
    for i in range(0, n_decoded_texts, batch_size):
        batch_prefixes = [random.choice(prefixes) for _ in range(min(batch_size, n_decoded_texts - i))]
        inputs = model.tokenizer(batch_prefixes, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                num_return_sequences=1,
                no_repeat_ngram_size=2
            )
        
        batch_decoded = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_texts.extend(batch_decoded)
    
    return decoded_texts

def apply_interpretability_method(
        base_model: PreTrainedModel, 
        finetuned_model: PreTrainedModel, 
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
    base_decoded_texts = batch_decode_texts(base_model, prefixes, n_decoded_texts)
    finetuned_decoded_texts = batch_decode_texts(finetuned_model, prefixes, n_decoded_texts)

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

def dummy_finetune_model(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    training_data: List[str],
    finetuning_params: Dict[str, Any]
) -> PreTrainedModel:
    """
    Dummy implementation of finetuning a model on given training data.

    This function simulates the process of finetuning a model. In a real implementation,
    this would involve actual training on the provided data using the specified parameters.

    Args:
        base_model (PreTrainedModel): The original model to be finetuned.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        training_data (List[Dict[str, str]]): List of training examples, each a dict with 'input' and 'output' keys.
        finetuning_params (Dict[str, Any]): Parameters for the finetuning process.

    Returns:
        PreTrainedModel: A "finetuned" version of the input model (in this dummy implementation, it's the same model).
    """
    # Placeholder implementation
    print(f"Finetuning model with {len(training_data)} examples and parameters: {finetuning_params}")
    return base_model  # In reality, this would be a new, finetuned model

def finetune_model(
        base_model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        training_data: List[str], 
        finetuning_params: Dict[str, Any]
    ) -> PreTrainedModel:
    """
    Fine-tune a pre-trained model on the given training data.

    Args:
        base_model (PreTrainedModel): The original model to be fine-tuned.
        tokenizer (PreTrainedTokenizer): The tokenizer associated with the model.
        training_data (List[str]): List of training examples.
        finetuning_params (Dict[str, Any]): Parameters for the fine-tuning process.

    Returns:
        PreTrainedModel: The fine-tuned model.
    """
    # Prepare the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    dataset = Dataset.from_dict({"text": training_data})
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Set up the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=finetuning_params.get("num_epochs", 3),
        per_device_train_batch_size=finetuning_params.get("batch_size", 8),
        warmup_steps=finetuning_params.get("warmup_steps", 500),
        weight_decay=finetuning_params.get("weight_decay", 0.01),
        logging_dir="./logs",
        logging_steps=finetuning_params.get("logging_steps", 100),
        save_steps=finetuning_params.get("save_steps", 1000),
        learning_rate=finetuning_params.get("learning_rate", 5e-5),
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Fine-tune the model
    trainer.train()

    # Return the fine-tuned model
    return trainer.model

def make_api_request(
        prompt: str, 
        api_provider: str, 
        model_str: str, 
        api_key: Optional[str] = None, 
        client: Optional[Any] = None, 
        print_interaction: bool = True
    ) -> str:
    """
    Make an API request to the specified provider.

    Args:
        prompt (str): The prompt to send to the API.
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (Optional[str]): The API key for the chosen provider.
        client (Optional[Any]): The client to use for the API.
        print_interaction (bool): Whether to print the interaction with the API.
    Returns:
        str: The response from the API.
    """
    if api_key is None and client is None:
        raise ValueError("Either api_key or client must be provided.")
    if api_provider == 'anthropic':
        if client is None:
            client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_str,
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        result = response.content[0].text
    elif api_provider == 'openai':
        if client is None:
            client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_str,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}")
    if print_interaction:
        print(f"API request to {api_provider} with model {model_str} and prompt of length {len(prompt)} returned result of length {len(result)}.\n")
        print(f"Prompt: {prompt}")
        print(f"Result: {result}\n\n")
    return result


def load_api_key(
        key_path: str, 
        api_provider: Optional[str] = None
    ) -> str:
    """
    Load the API key from a file.

    Args:
        key_path (str): Path to the file containing the API key.
        api_provider (Optional[str]): The API provider ('anthropic' or 'openai'). If None, assumes the file contains only the key. If provided, assumes the file stores multiple keys with the provider name as a prefix and a colon between the provider name and the key.

    Returns:
        str: The API key.

    Raises:
        ValueError: If the api_provider is specified but the key is not found in the file.
        FileNotFoundError: If the key file is not found.
    """
    try:
        with open(key_path, 'r') as file:
            content = file.read().strip()
            
            if api_provider is None:
                return content
            
            # If api_provider is specified, assume the file may contain multiple keys
            lines = content.split('\n')
            for line in lines:
                if line.lower().startswith(f"{api_provider.lower()}:"):
                    return line.split(':', 1)[1].strip()
            
            raise ValueError(f"API key for {api_provider} not found in the file.")
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found: {key_path}")


def parse_dict(s: str) -> Dict[str, str]:
    """
    Parse a string into a dictionary.

    Args:
        s (str): The string to parse.

    Returns:
        Dict[str, str]: The parsed dictionary.
    """
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        raise argparse.ArgumentTypeError("Invalid JSON string")


def extract_json_from_string(response: str) -> List[str]:
    """
    Extract a JSON object from a string, attempting to exclude any non-JSON content.

    Args:
        response (str): The response from the API.

    Returns:
        List[str]: The extracted json data. Exact return type depends on the input. E.g., it could be a list of strings, a list of dictionaries, etc.
    """
    # Parse the response to extract only the JSON part
    try:
        # Find the first occurrence of '[' and the last occurrence of ']'
        if '[' in response and ']' in response:
            start = response.find('[')
            end = response.rfind(']') + 1
        elif '{' in response and '}' in response:
            start = response.find('{')
            end = response.rfind('}') + 1
        else:
            raise ValueError("No valid JSON found in the response")
        if start != -1 and end != -1:
            json_str = response[start:end]
            json_data = json.loads(json_str)
        else:
            raise ValueError("No valid JSON found in the response")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {response}")
        json_data = []
    return json_data