"""
This module contains utility functions for the rest of the auto_finetuning_eval package. Specifically, it contains functions for making API requests, loading API keys, and parsing out json from strings.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from anthropic import Anthropic, InternalServerError
from openai import OpenAI, APIError
from google.generativeai import GenerativeModel, GenerationConfig
#from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time
import json
import argparse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, GPTNeoXForCausalLM, Qwen2ForCausalLM, GPT2LMHeadModel, LlamaForCausalLM
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor, as_completed
from structlog._config import BoundLoggerLazyProxy
from pathlib import Path
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
        logger: Optional[BoundLoggerLazyProxy] = None,
        max_tokens: int = 1000,
        n_local_retries: int = 5,
        request_info: Optional[Dict[str, str]] = {"pipeline_stage": "unknown"},
        temperature: float = 1.0
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
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        max_tokens (int): The maximum number of tokens to generate.
        n_local_retries (int): The number of retries to make if the API request fails.
        request_info (Optional[Dict[str, str]]): Information about the request to be recorded in the API
            interactions file. Defaults to {"pipeline_stage": "unknown"}.

    Returns:
        str: The response from the API.
    """
    if api_key is None and client is None:
        raise ValueError("Either api_key or client must be provided.")
    retries = 0
    while retries < n_local_retries:
        try:
            if api_provider == 'anthropic':
                if client is None:
                    client = Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=model_str,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
                result = response.content[0].text
            elif api_provider == 'openai':
                if client is None:
                    client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model_str,
                    max_completion_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature
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
                    ),
                    temperature=temperature
                )
                result = response.text
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
            
            if api_interactions_save_loc and logger:
                logger.info(
                    "api_request",
                    api_provider=api_provider,
                    model=model_str,
                    prompt=prompt,
                    result=result,
                    **request_info or {}
                )
            
            return result
        except (InternalServerError, APIError, Exception) as e:
            print(f"API request failed: {str(e)}. Retrying...")
            if logger:
                logger.info(
                    "api_request_failed",
                    api_provider=api_provider,
                    model=model_str,
                    prompt=prompt,
                    error=str(e),
                    **request_info or {}
                )
            time.sleep(10)
            retries += 1
    raise Exception(f"Failed to make API request after {n_local_retries} retries.")

def parallel_make_api_requests(
        prompts: List[str], 
        api_provider: str, 
        api_model_str: str, 
        auth_key: Optional[str] = None, 
        client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None, 
        api_interactions_save_loc: Optional[str] = None, 
        logger: Optional[BoundLoggerLazyProxy] = None,
        num_workers: int = 10,
        request_info: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        max_tokens: int = 1000,
        temperature: float = 1.0
    ) -> List[str]:
    """
    Execute API requests in parallel using a thread pool to improve performance.

    Args:
        prompts (List[str]): A list of prompts to send to the API.
        api_provider (str): The API provider to use (e.g., 'openai', 'anthropic').
        api_model_str (str): The specific model to use within the chosen API.
        auth_key (Optional[str]): The authentication key for the API.
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]]): The client to use for the API request.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        num_workers (int): The number of worker threads to use for parallel execution.
        request_info (Optional[Dict[str, str]]): Information about the request.
        max_retries (int): Maximum number of retries for failed requests.
        max_tokens (int): The maximum number of tokens to generate per request.
    Returns:
        List[str]: A list of responses corresponding to each prompt.

    Raises:
        ValueError: If neither auth_key nor client is provided
        RuntimeError: If all retries are exhausted for a request
    """
    if request_info is None:
        request_info = {'pipeline_stage': 'unknown'}

    def create_client():
        """Create a new client instance with error handling."""
        if auth_key is None and client is None:
            raise ValueError(f"Either auth_key or client must be provided for {api_provider}")
        
        # If a client was provided, use it as a template to create a new instance
        if client is not None:
            try:
                if api_provider == 'anthropic':
                    return Anthropic(api_key=client.api_key)
                elif api_provider == 'openai':
                    return OpenAI(api_key=client.api_key)
                elif api_provider == 'gemini':
                    return GenerativeModel(api_model_str)
                else:
                    raise ValueError(f"Unsupported API provider: {api_provider}")
            except Exception as e:
                raise RuntimeError(f"Failed to create client from template: {str(e)}")
        
        # Otherwise use auth_key
        try:
            if api_provider == 'anthropic':
                return Anthropic(api_key=auth_key)
            elif api_provider == 'openai':
                return OpenAI(api_key=auth_key)
            elif api_provider == 'gemini':
                return GenerativeModel(api_model_str)
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
        except Exception as e:
            raise RuntimeError(f"Failed to create client with auth_key: {str(e)}")

    def make_request(index: int, prompt: str) -> Tuple[int, str]:
        """Make an API request with retries and error handling."""
        retries = 0
        last_exception = None
        
        while retries < max_retries:
            try:
                # Create a new client for each attempt
                local_client = create_client()
                
                response = make_api_request(
                    prompt=prompt,
                    api_provider=api_provider,
                    model_str=api_model_str,
                    client=local_client,
                    api_interactions_save_loc=api_interactions_save_loc,
                    logger=logger,
                    request_info=request_info,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return index, response
                
            except Exception as e:
                last_exception = e
                retries += 1
                print(f"Request {index} failed (attempt {retries}/{max_retries}): {str(e)}")
                time.sleep(min(2 ** retries, 30))  # Exponential backoff, max 30 seconds
        
        error_msg = f"Request {index} failed after {max_retries} attempts. Last error: {str(last_exception)}"
        raise RuntimeError(error_msg)

    results = [None] * len(prompts)
    failed_indices = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(make_request, i, prompt): i 
            for i, prompt in enumerate(prompts)
        }
        
        # Process completed futures
        for future in as_completed(future_to_index):
            try:
                index, response = future.result()
                results[index] = response
            except Exception as e:
                index = future_to_index[future]
                failed_indices.append(index)
                print(f"Request {index} failed completely: {str(e)}")
                results[index] = f"ERROR: {str(e)}"

    # Report on failures
    if failed_indices:
        failure_rate = len(failed_indices) / len(prompts) * 100
        print(f"Warning: {len(failed_indices)} requests failed ({failure_rate:.1f}% failure rate)")
        print(f"Failed indices: {failed_indices}")

    return results


def collect_dataset_from_api(
    prompt: str,
    api_provider: str,
    model_str: str,
    api_key: Optional[str] = None,
    client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
    api_interactions_save_loc: Optional[str] = None,
    logger: Optional[BoundLoggerLazyProxy] = None,
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
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
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
                logger,
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
    logger: Optional[BoundLoggerLazyProxy] = None,
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
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
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
            logger=logger,
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
    
def load_statements_from_MWE_repo(
    path_to_MWE_repo: str,
    num_statements_per_behavior: int = 100
) -> List[str]:
    """
    Load statements from the MWE persona repository.

    Args:
        path_to_MWE_repo (str): The path to the MWE persona repository.
        num_statements_per_behavior (int): The number of statements to load per behavior.

    Returns:
        List[str]: A list of statements.
    """
    path_to_MWE_repo = Path(path_to_MWE_repo)
    persona_dir = path_to_MWE_repo / "persona"
    
    if not persona_dir.exists():
        raise FileNotFoundError(f"Persona directory not found at {persona_dir}")
    
    all_statements = []
    
    # Get all jsonl files in the persona directory
    jsonl_files = list(persona_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {persona_dir}")
    
    # Process each behavior file with a fixed random seed for reproducibility
    random.seed(42)
    
    for jsonl_file in jsonl_files:
        statements = []
        
        # Read the JSONL file
        with open(jsonl_file, "r") as f:
            lines = f.readlines()
            
        # Parse each JSON object and extract statements
        for line in lines:
            if line.strip():
                try:
                    entry = json.loads(line)
                    if "statement" in entry:
                        statements.append(entry["statement"])
                except json.JSONDecodeError:
                    continue
        
        # Select random statements for this behavior
        if statements:
            if len(statements) <= num_statements_per_behavior:
                selected_statements = statements
            else:
                selected_statements = random.sample(statements, num_statements_per_behavior)
            
            all_statements.extend(selected_statements)
    
    return all_statements
    

def batch_decode_texts(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer,
        prefixes: Optional[List[str]], 
        n_decoded_texts: Optional[int] = None,
        texts_decoded_per_prefix: Optional[int] = None,
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
        n_decoded_texts (Optional[int]): Total number of texts to generate.
        texts_decoded_per_prefix (Optional[int]): Number of texts to generate per prefix.
        batch_size (int): Number of texts to generate in each batch.
        max_length (int): The maximum length of the decoded texts.
        temperature (float): The temperature to use for text generation.
    Returns:
        List[str]: List of generated texts.
    """
    if n_decoded_texts is None and texts_decoded_per_prefix is None:
        raise ValueError("Either n_decoded_texts or texts_decoded_per_prefix must be provided")
    if n_decoded_texts is not None and texts_decoded_per_prefix is not None:
        raise ValueError("Either n_decoded_texts or texts_decoded_per_prefix must be provided, but not both")
    
    if prefixes is None:
        if isinstance(model, GPTNeoXForCausalLM) or \
            isinstance(model, Qwen2ForCausalLM) or \
            isinstance(model, GPT2LMHeadModel) or \
            isinstance(model, LlamaForCausalLM):
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
    total_decodings = n_decoded_texts if n_decoded_texts is not None else texts_decoded_per_prefix * len(prefixes)
    decoded_texts_by_prefix = [[] for _ in range(len(prefixes))]
    texts_per_prefix = texts_decoded_per_prefix if texts_decoded_per_prefix is not None else n_decoded_texts // len(prefixes)

    for i in tqdm(range(0, total_decodings, batch_size)):
        start_idx = i % len(prefixes)
        current_batch_size = min(batch_size, total_decodings - i)
        batch_prefixes = [prefixes[(start_idx + j) % len(prefixes)] for j in range(current_batch_size)]
        # Track which prefix index each item in the batch corresponds to:
        prefix_indices = [(start_idx + j) % len(prefixes) for j in range(current_batch_size)]

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
                temperature=temperature,
                do_sample=True
            )
        
        batch_decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Add each decoded text to the appropriate prefix's list
        for text, prefix_idx in zip(batch_decoded, prefix_indices):
            decoded_texts_by_prefix[prefix_idx].append(text)
    
    # Flatten the results in prefix order
    decoded_texts = []
    for prefix_texts in decoded_texts_by_prefix:
        decoded_texts.extend(prefix_texts[:texts_per_prefix])
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
        finetuned_cluster_centers: Optional[List[List[float]]] = None,
        max_points: int = 200000
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
        max_points (int): Maximum number of points to plot. Default is 100000.
    """

    if len(base_model_outputs_embeddings) + len(finetuned_model_outputs_embeddings) > max_points:
        # Subsample points from each embedding list and convert to numpy arrays
        indices = np.random.choice(range(len(base_model_outputs_embeddings)), max_points // 2)
        base_embeddings = np.array(base_model_outputs_embeddings[indices])
        fine_tuned_embeddings = np.array(finetuned_model_outputs_embeddings[indices])
    else:
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
        plt.scatter(base_cluster_centers_tsne[:, 0], base_cluster_centers_tsne[:, 1], c='blue', alpha=0.8, s=20, marker='x', label='Base Model Cluster Centers')
        plt.scatter(finetuned_cluster_centers_tsne[:, 0], finetuned_cluster_centers_tsne[:, 1], c='red', alpha=0.8, s=20, marker='x', label='Fine-tuned Model Cluster Centers')

    plt.title(title)
    plt.legend()
    plt.tight_layout()

    # Save the plot as a PDF
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

    print(f"t-SNE plot saved to {save_path}")