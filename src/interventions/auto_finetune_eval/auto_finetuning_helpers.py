"""
This module contains utility functions for the rest of the auto_finetuning_eval package. Specifically, it contains functions for making API requests, loading API keys, and parsing out json from strings.
"""

from typing import List, Dict, Any, Optional, Union
from anthropic import Anthropic, InternalServerError
from openai import OpenAI, APIError
from google.generativeai import GenerativeModel, GenerationConfig
from google.generativeai.types import HarmCategory, HarmBlockThreshold

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
from transformers import PreTrainedModel, PreTrainedTokenizer, GPTNeoXForCausalLM, Qwen2ForCausalLM
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None, 
        api_interactions_save_loc: Optional[str] = None,
        max_tokens: int = 1000,
        n_retries: int = 5,
        request_info: Optional[Dict[str, str]] = {"pipeline_stage": "unknown"}
    ) -> str:
    """
    Make an API request to the specified provider.

    Args:
        prompt (str): The prompt to send to the API.
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (Optional[str]): The API key for the chosen provider.
        client (Optional[Any]): The client to use for the API.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        max_tokens (int): The maximum number of tokens to generate.
        n_retries (int): The number of retries to make if the API request fails.
        request_info (Optional[Dict[str, str]]): Information about the request to be recorded in the API
            interactions file. Defaults to {"pipeline_stage": "unknown"}.

    Returns:
        str: The response from the API.
    """
    if api_key is None and client is None:
        raise ValueError("Either api_key or client must be provided.")
    retries = 0
    while retries < n_retries:
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
            elif api_provider == 'gemini':
                BLOCK_NONE = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ]
                # BLOCK_NONE = {
                #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                #     HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                # }
                if client is None:
                    client = GenerativeModel(model_str)
                response = client.generate_content(
                    prompt,
                    safety_settings=BLOCK_NONE,
                    generation_config = GenerationConfig(
                        max_output_tokens=max_tokens,
                    )
                )
                result = response.text
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
            
            if not api_interactions_save_loc is None:
                # Store the interaction as a json object in a jsonl file
                with open(api_interactions_save_loc, 'a') as file:
                    if not "pipeline_stage" in request_info:
                        raise ValueError("request_info must contain a 'pipeline_stage' key.")
                    storage_dict = {
                        "api_provider": api_provider,
                        "model_str": model_str,
                        "prompt": prompt,
                        "result": result
                    }
                    # Insert all request_info keys and values into the storage_dict
                    storage_dict.update(request_info)
                    file.write(json.dumps(storage_dict) + "\n")
            
            return result
        except (InternalServerError, APIError, Exception) as e:
            print(f"API request failed: {str(e)}. Retrying...")
            time.sleep(10)
            retries += 1
    raise Exception(f"Failed to make API request after {n_retries} retries.")

def parallel_make_api_requests(
        prompts: List[str], 
        api_provider: str, 
        api_model_str: str, 
        auth_key: Optional[str] = None, 
        client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None, 
        api_interactions_save_loc: Optional[str] = None, 
        num_workers: int = 10,
        request_info: Optional[Dict[str, str]] = {"pipeline_stage": "unknown"}
    ) -> List[str]:
    """
    Execute API requests in parallel using a thread pool to improve performance.

    Args:
        prompts (List[str]): A list of prompts to send to the API.
        api_provider (str): The API provider to use (e.g., 'openai', 'anthropic').
        api_model_str (str): The specific model to use within the chosen API.
        auth_key (str): The authentication key for the API.
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]]): The client to use for the API request.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        num_workers (int): The number of worker threads to use for parallel execution.
        request_info (Optional[Dict[str, str]]): Information about the request to be recorded in the API
            interactions file. Defaults to {"pipeline_stage": "unknown"}.

    Returns:
        List[str]: A list of responses corresponding to each prompt.

    Note:
        - The order of the returned responses matches the order of the input prompts.
    """
    def make_request(index, prompt):
        response = make_api_request(
            prompt, 
            api_provider, 
            api_model_str, 
            auth_key, 
            client=client,
            api_interactions_save_loc=api_interactions_save_loc,
            request_info=request_info
        )
        return index, response

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(make_request, i, prompt): i for i, prompt in enumerate(prompts)}
        results = [None] * len(prompts)
        for future in as_completed(futures):
            index, response = future.result()
            results[index] = response
    return results


def collect_dataset_from_api(
    prompt: str,
    api_provider: str,
    model_str: str,
    api_key: Optional[str] = None,
    client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
    api_interactions_save_loc: Optional[str] = None,
    max_tokens: int = 1000,
    num_datapoints: int = 100,
    max_retries: int = 12,
    request_info: Optional[Dict[str, str]] = {"pipeline_stage": "unspecified dataset collection"},
    min_chars_to_generate: int = 50
) -> List[str]:
    """
    Collect a dataset by making multiple API requests and accumulating the results.

    Args:
        prompt (str): The prompt to send to the API.
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (Optional[str]): The API key for the chosen provider.
        client (Optional[Any]): The client to use for the API.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        max_tokens (int): The maximum number of tokens to generate per request.
        num_datapoints (int): The total number of datapoints to collect.
        max_retries (int): The maximum number of retries for failed requests.
        request_info (Optional[Dict[str, str]]): Information about the request to be recorded in the API
            interactions file. Defaults to {"pipeline_stage": "unspecified dataset collection"}.
        min_chars_to_generate (int): The minimum number of characters to generate.
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
                api_interactions_save_loc, 
                max_tokens,
                request_info=request_info
            )
            extracted_data = extract_json_from_string(response)
            
            # Ensure extracted_data is a list
            if not isinstance(extracted_data, list):
                print(f"Unexpected response format. Expected a list, got {type(extracted_data)}.")
                retries += 1
                continue

            # Filter out duplicates
            new_data = [item for item in extracted_data if item not in dataset]

            # Filter out datapoints that are too short
            new_data = [item for item in new_data if len(item) >= min_chars_to_generate]
            
            if not new_data:
                print("No new data found in the response. Retrying...")
                retries += 1
                if retries == 3:
                    prompt = prompt + "\n\nTry to be creative and come up with novel datapoints."
                if retries == 5:
                    prompt = prompt.replace("Try to be creative and come up with novel datapoints.", "Be very creative and come up with weird new datapoints that no one has thought of before.")
                if retries == 8:
                    prompt = prompt.replace("Be very creative and come up with weird new datapoints that no one has thought of before.", "The past eight attempts at having you generate new datapoints have been unsuccessful. Probably, you've been failing to generate valid, parsable JSON. Make sure your responses are valid JSON. In particular, avoid using quotation marks within strings or outputing non-JSON text.")
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

def rephrase_description(
    description: str,
    api_provider: str,
    api_model_str: str,
    auth_key: Optional[str] = None,
    client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
    num_rephrases: int = 1,
    max_tokens: int = 4096,
    api_interactions_save_loc: Optional[str] = None,
    request_info: Optional[Dict[str, str]] = {"pipeline_stage": "unspecified description rephrasing"}
) -> List[str]:
    """
    Rephrase a given description using an API.

    This function takes a description and generates a rephrased version using
    the specified API. It uses the collect_dataset_from_api function to generate
    multiple rephrased versions and returns one of them.

    Args:
        description (str): The original description to be rephrased.
        api_provider (str): The API provider to use (e.g., 'openai', 'anthropic').
        api_model_str (str): The model string for the API request.
        auth_key (Optional[str]): The authentication key for the API.
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]]): The client to use for the API request.
        num_rephrases (int, optional): The number of rephrases to generate. Defaults to 1.
        max_tokens (int, optional): The maximum number of tokens for the API response. Defaults to 100.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        request_info (Optional[Dict[str, str]]): Information about the request to be recorded in the API
            interactions file. Defaults to {"pipeline_stage": "unspecified description rephrasing"}.
    Returns:
        List[str]: A list of rephrased versions of the input description.

    Raises:
        ValueError: If no valid rephrases are generated.
    """
    # Construct the prompt
    num_rephrases_to_ask_for = min(5, num_rephrases)
    prompt = f"""Generate {num_rephrases_to_ask_for} rephrases of the following description, which express the same core meaning in a different way. 
    Make sure the rephrased version is clear, concise, and maintains the original intent.

    Original description: "{description}"

    Provide the rephrased descriptions as a JSON list of strings formatted as follows:
    ["Rephrased description 1", "Rephrased description 2", ...]"""

    try:
        rephrases = collect_dataset_from_api(
            prompt=prompt,
            api_provider=api_provider,
            model_str=api_model_str,
            api_key=auth_key,
            client=client,
            api_interactions_save_loc=api_interactions_save_loc,
            max_tokens=max_tokens,
            num_datapoints=num_rephrases,
            request_info=request_info
        )

        return rephrases

    except Exception as e:
        print(f"Error in rephrasing description: {str(e)}")
        return [description] * num_rephrases

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
        max_length: int = 32,
        temperature: float = 1.0
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
        temperature (float): The temperature to use for text generation.
    Returns:
        List[str]: List of generated texts.
    """
    if prefixes is None:
        if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, Qwen2ForCausalLM):
            if tokenizer.bos_token is None:
                prefixes = [tokenizer.eos_token]
            else:
                prefixes = [tokenizer.bos_token]
        else:
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
                pad_token_id=tokenizer.eos_token_id,
                temperature=temperature
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
            json_data = literal_eval(json_str)
        except Exception as e:
            print(f"Error parsing as a list of strings via literal eval: {e}")
            # Handle case where response has unescaped quotes
            try:
                # Split by ",\n to separate list items
                items = json_str.strip('[]').split('",\n')
                # Clean up each item and handle quotes
                cleaned_items = []
                for item in items:
                    # Remove leading/trailing whitespace and quotes
                    cleaned = item.strip().strip('"\'')
                    if cleaned:
                        cleaned_items.append(cleaned)
                json_data = cleaned_items
                if len(json_data) == 0:
                    raise ValueError("Failed to parse response in all attempted ways. Returning empty list.")
            except Exception as e:
                print(f"Final parsing error: {e}")
                json_data = []
    return json_data

def plot_comparison_tsne(
        base_model_outputs_embeddings: List[List[float]],
        finetuned_model_outputs_embeddings: List[List[float]],
        save_path: str,
        title: str,
        perplexity: int = 30,
        base_cluster_centers: Optional[List[List[float]]] = None,
        finetuned_cluster_centers: Optional[List[List[float]]] = None
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
        base_cluster_centers (Optional[List[List[float]]]): Cluster centers for the base model.
        finetuned_cluster_centers (Optional[List[List[float]]]): Cluster centers for the fine-tuned model.
    """
    # Convert embeddings to numpy arrays
    base_embeddings = np.array(base_model_outputs_embeddings)
    fine_tuned_embeddings = np.array(finetuned_model_outputs_embeddings)

    # Combine embeddings
    combined_embeddings = np.vstack((base_embeddings, fine_tuned_embeddings))

    if base_cluster_centers is not None and finetuned_cluster_centers is not None:
        # Add cluster centers to the combined embeddings
        combined_embeddings = np.vstack((combined_embeddings, base_cluster_centers, finetuned_cluster_centers))

    # Perform t-SNE on combined embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    combined_tsne = tsne.fit_transform(combined_embeddings)

    # Split the results back into base and fine-tuned
    base_tsne = combined_tsne[:len(base_embeddings)]
    fine_tuned_tsne = combined_tsne[len(base_embeddings):len(base_embeddings) + len(fine_tuned_embeddings)]
    if base_cluster_centers is not None and finetuned_cluster_centers is not None:
        start_of_base_clusters = len(base_embeddings) + len(fine_tuned_embeddings)
        start_of_finetuned_clusters = start_of_base_clusters + len(base_cluster_centers)
        base_cluster_centers_tsne = combined_tsne[start_of_base_clusters:start_of_finetuned_clusters]
        finetuned_cluster_centers_tsne = combined_tsne[start_of_finetuned_clusters:]

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.scatter(base_tsne[:, 0], base_tsne[:, 1], c='blue', alpha=0.2, s=1, label='Base Model')
    plt.scatter(fine_tuned_tsne[:, 0], fine_tuned_tsne[:, 1], c='red', alpha=0.2, s=1, label='Fine-tuned Model')
    if base_cluster_centers is not None and finetuned_cluster_centers is not None:
        plt.scatter(base_cluster_centers_tsne[:, 0], base_cluster_centers_tsne[:, 1], c='blue', alpha=0.8, s=10, marker='x', label='Base Model Cluster Centers')
        plt.scatter(finetuned_cluster_centers_tsne[:, 0], finetuned_cluster_centers_tsne[:, 1], c='red', alpha=0.8, s=10, marker='x', label='Fine-tuned Model Cluster Centers')

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Save the plot as a PDF
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"t-SNE plot saved to {save_path}")