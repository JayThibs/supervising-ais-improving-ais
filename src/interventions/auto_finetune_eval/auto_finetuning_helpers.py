"""
This module contains utility functions for the rest of the auto_finetuning_eval package. Specifically, it contains functions for making API requests, loading API keys, and parsing out json from strings.
"""

from typing import List, Dict, Any, Optional
from anthropic import Anthropic, InternalServerError
from openai import OpenAI, APIError
import json
import argparse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from ast import literal_eval

@retry(
    stop=stop_after_attempt(15),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((InternalServerError, APIError))
)
def make_api_request(
        prompt: str, 
        api_provider: str, 
        model_str: str, 
        api_key: Optional[str] = None, 
        client: Optional[Any] = None, 
        print_interaction: bool = True,
        max_tokens: int = 1000,
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
        max_tokens (int): The maximum number of tokens to generate.
    Returns:
        str: The response from the API.
    """
    if api_key is None and client is None:
        raise ValueError("Either api_key or client must be provided.")
    try:
        if api_provider == 'anthropic':
            if client is None:
                client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model=model_str,
                max_tokens=max_tokens,
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
                max_completion_tokens=max_tokens,
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
    
    except (InternalServerError, APIError) as e:
        print(f"API request failed: {str(e)}. Retrying...")
        raise  # This will trigger the retry mechanism

def collect_dataset_from_api(
    prompt: str,
    api_provider: str,
    model_str: str,
    api_key: Optional[str] = None,
    client: Optional[Any] = None,
    print_interaction: bool = True,
    max_tokens: int = 1000,
    num_datapoints: int = 100,
    max_retries: int = 10
) -> List[str]:
    """
    Collect a dataset by making multiple API requests and accumulating the results.

    Args:
        prompt (str): The prompt to send to the API.
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (Optional[str]): The API key for the chosen provider.
        client (Optional[Any]): The client to use for the API.
        print_interaction (bool): Whether to print the interaction with the API.
        max_tokens (int): The maximum number of tokens to generate per request.
        num_datapoints (int): The total number of datapoints to collect.
        max_retries (int): The maximum number of retries for failed requests.

    Returns:
        List[str]: A list of collected datapoints.

    Raises:
        ValueError: If unable to collect the required number of datapoints after max_retries.
    """
    dataset = []
    retries = 0

    while len(dataset) < num_datapoints and retries < max_retries:
        try:
            response = make_api_request(
                prompt, 
                api_provider, 
                model_str, 
                api_key, 
                client, 
                print_interaction, 
                max_tokens
            )
            extracted_data = extract_json_from_string(response)
            
            # Ensure extracted_data is a list
            if not isinstance(extracted_data, list):
                print(f"Unexpected response format. Expected a list, got {type(extracted_data)}.")
                retries += 1
                continue

            # Filter out duplicates
            new_data = [item for item in extracted_data if item not in dataset]
            
            if not new_data:
                print("No new data found in the response. Retrying...")
                retries += 1
                if retries == 3:
                    prompt = prompt + "\n\nTry to be creative and come up with novel datapoints."
                if retries == 6:
                    prompt = prompt.replace("Try to be creative and come up with novel datapoints.", "Be very creative and come up with weird new datapoints that no one has thought of before.")
                continue

            dataset.extend(new_data)
            print(f"Collected {len(dataset)} datapoints so far.")

        except Exception as e:
            print(f"Error during API request or data extraction: {str(e)}")
            # Wait for 10 seconds before retrying
            time.sleep(10)
            retries += 1

    if len(dataset) < num_datapoints:
        raise ValueError(f"Unable to collect {num_datapoints} datapoints after {max_retries} retries. Collected {len(dataset)} datapoints.")

    return dataset[:num_datapoints]

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

def batch_decode_texts(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer,
        prefixes: Optional[List[str]], 
        n_decoded_texts: int, 
        batch_size: int = 32,
        max_length: int = 32
    ) -> List[str]:
    """
    Decode texts in batches using the given model.

    Args:
        model (PreTrainedModel): The model to use for text generation.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for text generation.
        prefixes (List[str]): List of prefixes to use for text generation.
        n_decoded_texts (int): Total number of texts to generate.
        batch_size (int): Number of texts to generate in each batch.
        max_length (int): The maximum length of the decoded texts.

    Returns:
        List[str]: List of generated texts.
    """
    if prefixes is None:
        prefixes = [""]
    # Set padding token to eos token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    decoded_texts = []
    for i in tqdm(range(0, n_decoded_texts, batch_size)):
        batch_prefixes = [random.choice(prefixes) for _ in range(min(batch_size, n_decoded_texts - i))]
        inputs = tokenizer(
            batch_prefixes, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=max_length
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.eos_token_id
            )
        
        batch_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        decoded_texts.extend(batch_decoded)
    
    return decoded_texts

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
        print("Attempting to parse as a list of strings via literal eval")
        try:
            json_data = literal_eval(response)
        except Exception as e:
            print(f"Error parsing as a list of strings via literal eval: {e}")
            json_data = []
    return json_data

def plot_comparison_tsne(
        base_model_outputs_embeddings: List[List[float]],
        finetuned_model_outputs_embeddings: List[List[float]],
        save_path: str,
        title: str,
        perplexity: int = 30
    ) -> None:
    """
    Perform t-SNE dimensionality reduction on combined base and fine-tuned model embeddings.
    Plot and save the result.

    Args:
        base_model_outputs_embeddings (List[List[float]]): Embeddings from the base model.
        finetuned_model_outputs_embeddings (List[List[float]]): Embeddings from the fine-tuned model.
        save_path (str): Path to save the resulting plot.
        title (str): Title for the plot.
        perplexity (int): Perplexity parameter for t-SNE. Default is 30.
    """
    # Convert embeddings to numpy arrays
    base_embeddings = np.array(base_model_outputs_embeddings)
    fine_tuned_embeddings = np.array(finetuned_model_outputs_embeddings)

    # Combine embeddings
    combined_embeddings = np.vstack((base_embeddings, fine_tuned_embeddings))

    # Perform t-SNE on combined embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    combined_tsne = tsne.fit_transform(combined_embeddings)

    # Split the results back into base and fine-tuned
    base_tsne = combined_tsne[:len(base_embeddings)]
    fine_tuned_tsne = combined_tsne[len(base_embeddings):]

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.scatter(base_tsne[:, 0], base_tsne[:, 1], c='blue', alpha=0.5, s=2, label='Base Model')
    plt.scatter(fine_tuned_tsne[:, 0], fine_tuned_tsne[:, 1], c='red', alpha=0.5, s=2, label='Fine-tuned Model')

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Save the plot as a PDF
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"t-SNE plot saved to {save_path}")