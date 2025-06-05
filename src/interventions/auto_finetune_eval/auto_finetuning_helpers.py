"""
This module contains utility functions for the rest of the auto_finetuning_eval package. Specifically, it contains functions for making API requests, loading API keys, and parsing out json from strings.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from anthropic import Anthropic, InternalServerError
from openai import OpenAI, APIError
from google import genai
from google.genai import types, Client
import time
import json
import pickle
import argparse
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, GPTNeoXForCausalLM, Qwen2ForCausalLM, GPT2LMHeadModel, LlamaForCausalLM, Phi3ForCausalLM
from ast import literal_eval
from concurrent.futures import ThreadPoolExecutor, as_completed
from structlog._config import BoundLoggerLazyProxy
from pathlib import Path
import csv
import os
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
        client: Optional[Union[Anthropic, OpenAI, Client]] = None, 
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
                BLOCK_NONE = [
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    types.SafetySetting(
                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=types.HarmBlockThreshold.BLOCK_NONE
                    )
                ]
                # BLOCK_NONE = {
                #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                #     HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
                #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                # }
                if client is None:
                    client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model_str,
                    contents=prompt,
                    config = types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        safety_settings=BLOCK_NONE
                    )
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
        client: Optional[Union[Anthropic, OpenAI, Client]] = None, 
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
        client (Optional[Union[Anthropic, OpenAI, Client]]): The client to use for the API request.
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
        
        # Create a new client using auth_key
        try:
            if api_provider == 'anthropic':
                return Anthropic(api_key=auth_key)
            elif api_provider == 'openai':
                return OpenAI(api_key=auth_key)
            elif api_provider == 'gemini':
                return genai.Client(api_key=auth_key)
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
    client: Optional[Union[Anthropic, OpenAI, Client]] = None,
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
    client: Optional[Union[Anthropic, OpenAI, Client]] = None,
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
        client (Optional[Union[Anthropic, OpenAI, Client]]): The client to use for the API request.
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

def get_persona_targets(model_1_name, model_2_name, pickle_path = "anthropics-evals-result.pickle", threshold = 0.0):
    a = pickle.load(open(pickle_path, "rb"))

    if "NousResearch/Meta-" in model_1_name:
        model_1_name = model_1_name.replace("NousResearch/Meta-", "meta-llama/").replace("Llama-3-", "Llama-3.1-")
    if "NousResearch/Meta-" in model_2_name:
        model_2_name = model_2_name.replace("NousResearch/Meta-", "meta-llama/").replace("Llama-3-", "Llama-3.1-")
    print("model_1_name:", model_1_name)
    print("model_2_name:", model_2_name)

    keys = list(a['persona'].keys())
    avg_diffs = []

    for key in keys:
        model_1_scores = a['persona'][key][model_1_name]
        model_2_scores = a['persona'][key][model_2_name]
        diffs = [model_1_scores[i] - model_2_scores[i] for i in range(len(model_1_scores))]
        avg_diffs.append(sum(diffs) / len(diffs))

    if threshold > 0:
        target_keys = [key for key, diff in zip(keys, avg_diffs) if abs(diff) > threshold]
    else:
        target_keys = [key for key, diff in zip(keys, avg_diffs) if abs(diff) < -threshold]

    print("diffs:", avg_diffs)

    return target_keys



def load_statements_from_MWE_repo(
    path_to_MWE_repo: str,
    num_statements_per_behavior: int = 100,
    threshold: float = 0.0,
    model_1_name: str = "NousResearch/Meta-Llama-3-8B",
    model_2_name: str = "NousResearch/Meta-Llama-3-8B-Instruct",
    pickle_path: str = "anthropics-evals-result-extended.pickle"
) -> List[str]:
    """
    Load statements from the MWE persona repository.

    Args:
        path_to_MWE_repo (str): The path to the MWE persona repository.
        num_statements_per_behavior (int): The number of statements to load per behavior.
        threshold (float): The threshold for deciding which behaviors to target for further investigation via difference 
            discovery.
        model_1_name (str): The name of the first model.
        model_2_name (str): The name of the second model.
        pickle_path (str): The path to the pickle file containing the anthropics evals result.
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
    
    if threshold != 0:
        target_keys = get_persona_targets(model_1_name, model_2_name, pickle_path = pickle_path, threshold = threshold)
        print(f"Target keys: {target_keys}")
    # Process each behavior file with a fixed random seed for reproducibility
    random.seed(42)
    
    for jsonl_file in jsonl_files:
        if threshold != 0:
            if jsonl_file.stem not in target_keys:
                print(f"Skipping {jsonl_file.stem} because it is not in the target keys")
                continue

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

def load_statements_from_MPI_repo(
    path_to_MPI_repo: str,
) -> List[str]:
    """
    Load statements from the MPI persona repository.

    Args:
        path_to_MPI_repo (str): The path to the MPI persona repository.

    Returns:
        List[str]: A list of statements.
    """
    path_to_MPI_repo = Path(path_to_MPI_repo)
    mpi_dir = path_to_MPI_repo / "inventories"
    
    if not mpi_dir.exists():
        raise FileNotFoundError(f"MPI directory not found at {mpi_dir}")
    
    statements = []
    csv_file = mpi_dir / "mpi_1k.csv"
    
    if not csv_file.exists():
        raise ValueError(f"No CSV file found in {mpi_dir}")
    
    # Read the CSV file
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            statements.append(row[1])
    
    return statements

def load_statements_from_jailbreak_llms_repo(
    path_to_jailbreak_llms_repo_target: str,
    num_items_to_load: int = None
) -> List[str]:
    """
    Load statements from the jailbreak_llms repository.

    Args:
        path_to_jailbreak_llms_repo_target (str): The path to the loading target for the jailbreak_llms repository.
            This should either be to the 'data/prompts' directory, or to the 'data/forbidden_question' directory.
        num_items_to_load (int): The number of items to load from the data source in the repository. If None, all 
            items will be loaded.

    Returns:
        List[str]: A list of statements.
    """
    path_to_jailbreak_llms_repo = Path(path_to_jailbreak_llms_repo)
    if not path_to_jailbreak_llms_repo.exists():
        raise FileNotFoundError(f"Jailbreak LLMs repository not found at {path_to_jailbreak_llms_repo}")
    if not path_to_jailbreak_llms_repo_target.endswith("data/prompts") and not path_to_jailbreak_llms_repo_target.endswith("data/forbidden_question"):
        raise ValueError(f"Invalid target directory: {path_to_jailbreak_llms_repo_target}")
    # Load the items from the target
    # List all csv files in the target directory, then read the first num_items_to_load lines of each file (or all lines if num_items_to_load is None)

    random.seed(42)

    statements = []
    csv_files = list(path_to_jailbreak_llms_repo.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {path_to_jailbreak_llms_repo}")
    
    for csv_file in csv_files:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            try:
                # Skip header row if it exists
                header = next(reader)
                file_statements = []
                for row in reader:
                    if row:  # Check if row is not empty
                        # Assume the statement is in the first column
                        file_statements.append(row[0])
                statements.extend(file_statements)
            except StopIteration:
                continue  # Skip empty files
    
    # Shuffle and limit the number of statements if specified
    random.shuffle(statements)
    if num_items_to_load is not None:
        statements = statements[:num_items_to_load]
    
    return statements




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
            isinstance(model, LlamaForCausalLM) or \
            isinstance(model, Phi3ForCausalLM):
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



# def batch_decode_texts(
#         model: PreTrainedModel, 
#         tokenizer: PreTrainedTokenizer,
#         prefixes: Optional[List[str]], 
#         n_decoded_texts: Optional[int] = None,
#         texts_decoded_per_prefix: Optional[int] = None,
#         batch_size: int = 32,
#         max_length: int = 32,
#         temperature: float = 1.0
#     ) -> List[str]:
#     """
#     Decode texts in batches using the given model.

#     Args:
#         model (PreTrainedModel): The model to use for text generation.
#         tokenizer (PreTrainedTokenizer): The tokenizer to use for text generation.
#         prefixes (List[str]): List of prefixes to use for text generation.
#         n_decoded_texts (Optional[int]): Total number of texts to generate.
#         texts_decoded_per_prefix (Optional[int]): Number of texts to generate per prefix.
#         batch_size (int): Number of texts to generate in each batch.
#         max_length (int): The maximum length of the decoded texts.
#         temperature (float): The temperature to use for text generation.
#     Returns:
#         List[str]: List of generated texts.
#     """
#     if n_decoded_texts is None and texts_decoded_per_prefix is None:
#         raise ValueError("Either n_decoded_texts or texts_decoded_per_prefix must be provided")
#     if n_decoded_texts is not None and texts_decoded_per_prefix is not None:
#         raise ValueError("Either n_decoded_texts or texts_decoded_per_prefix must be provided, but not both")
    
#     if prefixes is None:
#         if isinstance(model, GPTNeoXForCausalLM) or \
#             isinstance(model, Qwen2ForCausalLM) or \
#             isinstance(model, GPT2LMHeadModel) or \
#             isinstance(model, LlamaForCausalLM) or \
#             isinstance(model, Phi3ForCausalLM):
#             if tokenizer.bos_token is None:
#                 prefixes = [tokenizer.eos_token]
#             else:
#                 prefixes = [tokenizer.bos_token]
#         else:
#             prefixes = [""]
    
#     if not prefixes:
#         return []

#     # Set padding token to eos token
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id

#     num_original_prefixes = len(prefixes)
    
#     if texts_decoded_per_prefix is not None:
#         counts_per_prefix = [texts_decoded_per_prefix] * num_original_prefixes
#         total_target_decodings = texts_decoded_per_prefix * num_original_prefixes
#     elif n_decoded_texts is not None:
#         if num_original_prefixes == 0: # Should be caught by `if not prefixes` earlier, but as a safeguard
#              if n_decoded_texts > 0:
#                  raise ValueError("n_decoded_texts > 0 but no prefixes provided or prefixes became empty.")
#              return []
#         base_count = n_decoded_texts // num_original_prefixes
#         remainder = n_decoded_texts % num_original_prefixes
#         counts_per_prefix = [base_count + 1 if i < remainder else base_count for i in range(num_original_prefixes)]
#         total_target_decodings = n_decoded_texts
#     else:
#         # This case is theoretically covered by the initial checks, but to be safe:
#         raise ValueError("Logical error: n_decoded_texts and texts_decoded_per_prefix are both None.")

#     if total_target_decodings == 0:
#         return []

#     prefix_data_map = {}
#     for i, p_str in enumerate(prefixes):
#         prefix_data_map[i] = {
#             'prefix_str': p_str,
#             'token_len': len(tokenizer.encode(p_str)),
#             'num_to_gen': counts_per_prefix[i],
#             'generated_count': 0
#         }

#     decoded_results_for_original_prefixes = [[] for _ in range(num_original_prefixes)]
#     generated_total = 0
    
#     with tqdm(total=total_target_decodings, desc="Batch Decoding") as pbar:
#         while generated_total < total_target_decodings:
#             candidate_tasks = [] # Stores (token_len, original_idx, prefix_str)
#             for original_idx, data in prefix_data_map.items():
#                 if data['generated_count'] < data['num_to_gen']:
#                     candidate_tasks.append((data['token_len'], original_idx, data['prefix_str']))
            
#             if not candidate_tasks:
#                 break # All tasks completed

#             candidate_tasks.sort(key=lambda x: x[0], reverse=True) # Sort by token_len

#             num_tasks_for_batch = min(batch_size, len(candidate_tasks), total_target_decodings - generated_total)
#             current_batch_tasks_info = candidate_tasks[:num_tasks_for_batch]

#             if not current_batch_tasks_info:
#                 break # No tasks to form a batch from

#             batch_prompts = [info[2] for info in current_batch_tasks_info]
#             batch_original_indices = [info[1] for info in current_batch_tasks_info]
#             batch_prompt_lengths = [info[0] for info in current_batch_tasks_info]
#             #print(f"Batch prompts: {batch_prompts}")
#             #print(f"Batch prompt lengths: {batch_prompt_lengths}")

#             inputs = tokenizer(
#                 batch_prompts, 
#                 return_tensors="pt", 
#                 padding=True, 
#                 truncation=True,
#                 max_length=max_length
#             )
#             inputs = {key: value.to(model.device) for key, value in inputs.items()}
#             #print(f"Inputs: {inputs}")
            
#             with torch.no_grad():
#                 outputs = model.generate(
#                     **inputs,
#                     max_length=max_length,
#                     num_return_sequences=1,
#                     no_repeat_ngram_size=2,
#                     pad_token_id=tokenizer.eos_token_id,
#                     temperature=temperature,
#                     do_sample=True
#                 )
            
#             batch_decoded_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)

#             for text, original_idx in zip(batch_decoded_texts, batch_original_indices):
#                 if prefix_data_map[original_idx]['generated_count'] < prefix_data_map[original_idx]['num_to_gen']:
#                     decoded_results_for_original_prefixes[original_idx].append(text)
#                     prefix_data_map[original_idx]['generated_count'] += 1
#                     generated_total += 1
#                     pbar.update(1)
#                 # If generated_total reaches total_target_decodings, inner loops might still run,
#                 # but new items won't exceed num_to_gen for that prefix or total_target_decodings.
#                 if generated_total >= total_target_decodings:
#                     break 
#             if generated_total >= total_target_decodings:
#                     break 

#     final_decoded_texts = []
#     for i in range(num_original_prefixes):
#         final_decoded_texts.extend(decoded_results_for_original_prefixes[i])
    
#     return final_decoded_texts


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

def plot_singular_vectors_heatmaps(U, V, S, base_name, run_prefix, save_dir, n_components=5):
    fig, axes = plt.subplots(2, n_components, figsize=(4*n_components, 8))
    
    # Plot first n_components of U
    for i in range(min(n_components, U.shape[1])):
        if U.shape[0] > 1:  # If U is 2D, reshape if needed for visualization
            u_reshaped = U[:, i].cpu().numpy()
            if len(u_reshaped.shape) == 1:
                # Try to reshape to square if possible
                size = int(np.sqrt(len(u_reshaped)))
                if size * size == len(u_reshaped):
                    u_reshaped = u_reshaped.reshape(size, size)
                else:
                    u_reshaped = u_reshaped.reshape(-1, 1)
            
            im1 = axes[0, i].imshow(u_reshaped, cmap='RdBu_r', aspect='auto')
            axes[0, i].set_title(f'U[:, {i}] (σ={S[i]:.3f})')
            plt.colorbar(im1, ax=axes[0, i])
    
    # Plot first n_components of V
    for i in range(min(n_components, V.shape[1])):
        v_reshaped = V[:, i].cpu().numpy()
        if len(v_reshaped.shape) == 1:
            size = int(np.sqrt(len(v_reshaped)))
            if size * size == len(v_reshaped):
                v_reshaped = v_reshaped.reshape(size, size)
            else:
                v_reshaped = v_reshaped.reshape(-1, 1)
        
        im2 = axes[1, i].imshow(v_reshaped, cmap='RdBu_r', aspect='auto')
        axes[1, i].set_title(f'V[:, {i}] (σ={S[i]:.3f})')
        plt.colorbar(im2, ax=axes[1, i])
    
    plt.suptitle(f'Singular Vectors for {base_name}')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{run_prefix}_{base_name.replace('.', '_')}_singular_vectors_heatmap.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_cumulative_variance(S, base_name, run_prefix, save_dir):
    # Calculate explained variance ratio
    total_variance = torch.sum(S**2)
    explained_variance_ratio = (S**2 / total_variance).cpu().numpy()
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Individual variance
    ax1.bar(range(len(S)), explained_variance_ratio)
    ax1.set_xlabel('Singular Value Index')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Individual Singular Value Contributions')
    
    # Cumulative variance
    ax2.plot(range(len(S)), cumulative_variance, 'bo-')
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Variance Explained')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Variance Analysis for {base_name}')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{run_prefix}_{base_name.replace('.', '_')}_variance_analysis.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

def plot_singular_vector_distributions(U, V, S, base_name, run_prefix, save_dir, n_components=5):
    fig, axes = plt.subplots(2, n_components, figsize=(4*n_components, 8))
    
    for i in range(min(n_components, len(S))):
        # U distributions
        axes[0, i].hist(U[:, i].cpu().numpy(), bins=30, alpha=0.7, color='blue')
        axes[0, i].set_title(f'U[:, {i}] (σ={S[i]:.3f})')
        axes[0, i].set_xlabel('Value')
        axes[0, i].set_ylabel('Frequency')
        
        # V distributions
        axes[1, i].hist(V[:, i].cpu().numpy(), bins=30, alpha=0.7, color='red')
        axes[1, i].set_title(f'V[:, {i}] (σ={S[i]:.3f})')
        axes[1, i].set_xlabel('Value')
        axes[1, i].set_ylabel('Frequency')
    
    plt.suptitle(f'Singular Vector Distributions for {base_name}')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"{run_prefix}_{base_name.replace('.', '_')}_distributions.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

def analyze_sparsity_pattern(U, V, S, base_name, run_prefix, save_dir):
    """Analyze the sparsity and concentration patterns in U and V"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Compute component-wise statistics
    u_component_vars = torch.var(U, dim=0).cpu()
    v_component_vars = torch.var(V, dim=0).cpu()
    
    u_component_means = torch.mean(torch.abs(U), dim=0).cpu().numpy()
    v_component_means = torch.mean(torch.abs(V), dim=0).cpu().numpy()
    
    # Plot variance across components
    axes[0, 0].plot(u_component_vars, 'b-', label='U variance')
    axes[0, 0].set_title('Variance Across U Components')
    axes[0, 0].set_xlabel('Component Index')
    axes[0, 0].set_ylabel('Variance')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[1, 0].plot(v_component_vars, 'r-', label='V variance')
    axes[1, 0].set_title('Variance Across V Components')
    axes[1, 0].set_xlabel('Component Index')
    axes[1, 0].set_ylabel('Variance')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot mean absolute values
    axes[0, 1].plot(u_component_means, 'b-')
    axes[0, 1].set_title('Mean |U| Components')
    axes[0, 1].set_xlabel('Component Index')
    axes[0, 1].set_ylabel('Mean Absolute Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 1].plot(v_component_means, 'r-')
    axes[1, 1].set_title('Mean |V| Components')
    axes[1, 1].set_xlabel('Component Index')
    axes[1, 1].set_ylabel('Mean Absolute Value')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Compute effective rank
    u_effective_rank = torch.sum(u_component_vars > 0.1 * torch.max(u_component_vars)).item()
    v_effective_rank = torch.sum(v_component_vars > 0.1 * torch.max(v_component_vars)).item()
    
    # Plot effective rank comparison
    axes[0, 2].bar(['U Effective Rank', 'V Effective Rank'], 
                   [u_effective_rank, v_effective_rank], 
                   color=['blue', 'red'], alpha=0.7)
    axes[0, 2].set_title('Effective Rank Comparison')
    axes[0, 2].set_ylabel('Effective Rank (10% threshold)')
    
    # Plot ratio of max to mean for each singular vector
    u_max_to_mean = torch.max(torch.abs(U), dim=0)[0] / torch.mean(torch.abs(U), dim=0)
    v_max_to_mean = torch.max(torch.abs(V), dim=0)[0] / torch.mean(torch.abs(V), dim=0)
    
    axes[1, 2].plot(u_max_to_mean.cpu().numpy(), 'b-', label='U max/mean ratio')
    axes[1, 2].plot(v_max_to_mean.cpu().numpy(), 'r-', label='V max/mean ratio')
    axes[1, 2].set_title('Sparsity Indicator (Max/Mean Ratio)')
    axes[1, 2].set_xlabel('Singular Vector Index')
    axes[1, 2].set_ylabel('Max/Mean Ratio')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(f'Sparsity and Structure Analysis for {base_name}')
    plt.tight_layout()
    
    # Print summary statistics
    print(f"\nStructural Analysis for {base_name}:")
    print(f"U effective rank: {u_effective_rank}/{U.shape[1]}")
    print(f"V effective rank: {v_effective_rank}/{V.shape[1]}")
    print(f"U variance range: {torch.min(u_component_vars):.6f} to {torch.max(u_component_vars):.6f}")
    print(f"V variance range: {torch.min(v_component_vars):.6f} to {torch.max(v_component_vars):.6f}")
    print(f"Variance ratio (U/V): {torch.max(u_component_vars)/torch.max(v_component_vars):.2f}")
    
    save_path = os.path.join(save_dir, f"{run_prefix}_{base_name.replace('.', '_')}_sparsity_analysis.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

    return u_effective_rank, v_effective_rank


def analyze_weight_difference(
        base_param: torch.Tensor, 
        intervention_param: torch.Tensor, 
        base_name: str, 
        run_prefix: str,
        save_dir: str = "weight_difference_analysis",
) -> None:
    """
    Analyze the difference between two weight tensors.
    Print the mean and standard deviation of the differences.
    Then compute the SVD of the differences and plot a histogram of the singular values.
    Also plot a histogram of the differences.
    """

    # Cast to float32
    base_param = base_param.float()
    intervention_param = intervention_param.float()

    print(f"Analyzing weight difference for {base_name} in {run_prefix}")
    print(f"Base param shape: {base_param.shape}")
    print(f"Intervention param shape: {intervention_param.shape}")
    # Check that the shapes are the same and that they represent 2D matrices. Return if not.
    if base_param.shape != intervention_param.shape or len(base_param.shape) != 2:
        print("Shapes are not the same or do not represent 2D matrices. Returning.")
        return
    
    # Compute the mean and standard deviation of the differences
    diff = base_param - intervention_param
    mean_diff = torch.mean(diff)
    std_diff = torch.std(diff)
    print(f"Mean difference: {mean_diff}")
    print(f"Standard deviation of difference: {std_diff}")

    # Compute the rank of the differences
    rank = torch.linalg.matrix_rank(diff)
    print(f"Rank of differences: {rank}")

    # Compute the SVD of the differences
    U, S, V = torch.svd(diff)
    print(f"SVD shapes - U: {U.shape}, S: {S.shape}, V: {V.shape}")
    print(f"Top 10 singular values: {S[:10]}")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Original plots (singular values and differences)
    plt.figure(figsize=(15, 10))
    
    # Plot histogram of singular values
    plt.subplot(2, 3, 1)
    plt.hist(S.cpu().numpy(), bins=50)
    plt.title(f'Singular Values of Weight Differences')
    plt.xlabel('Singular Value')
    plt.ylabel('Count')
    plt.yscale('log')
    
    # Plot histogram of differences
    plt.subplot(2, 3, 2)
    plt.hist(diff.cpu().numpy().flatten(), bins=50)
    plt.title(f'Distribution of Weight Differences')
    plt.xlabel('Weight Difference')
    plt.ylabel('Count')
    plt.yscale('log')
    
    # Cumulative variance plot
    plt.subplot(2, 3, 3)
    total_variance = torch.sum(S**2)
    explained_variance_ratio = (S**2 / total_variance).cpu().numpy()
    cumulative_variance = np.cumsum(explained_variance_ratio)
    plt.plot(range(min(50, len(S))), cumulative_variance[:50], 'bo-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Variance Explained')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot first few singular vectors
    n_show = min(3, U.shape[1])
    for i in range(n_show):
        plt.subplot(2, 3, 4 + i)
        plt.plot(U[:, i].cpu().numpy(), label=f'U[:, {i}]', alpha=0.7)
        plt.plot(V[:, i].cpu().numpy(), label=f'V[:, {i}]', alpha=0.7)
        plt.title(f'Singular Vectors {i} (σ={S[i]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylabel('Value')
        plt.xlabel('Index')
    
    plt.suptitle(f'Weight Difference Analysis for {base_name}')
    plt.tight_layout()
    
    # Save the comprehensive plot
    save_path = os.path.join(save_dir, f"{run_prefix}_{base_name.replace('.', '_')}_weight_difference_analysis.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()

    # Generate additional detailed visualizations
    plot_singular_vectors_heatmaps(U, V, S, base_name, run_prefix, save_dir)
    plot_cumulative_variance(S, base_name, run_prefix, save_dir)
    plot_singular_vector_distributions(U, V, S, base_name, run_prefix, save_dir)
    u_eff_rank, v_eff_rank = analyze_sparsity_pattern(U, V, S, base_name, run_prefix, save_dir)
    
    # Analyze U vs V variance patterns for structural insights
    uv_analysis = analyze_uv_variance_patterns(U, V, S, base_name, run_prefix, save_dir)
    
    # Additional structural analysis
    print(f"\nStructural Insights:")
    print(f"Matrix shape: {diff.shape}")
    print(f"Frobenius norm: {torch.norm(diff, 'fro'):.6f}")
    
    # Compute rank-1 approximation quality
    rank1_approx = S[0] * torch.outer(U[:, 0], V[:, 0])
    rank1_error = torch.norm(diff - rank1_approx, 'fro') / torch.norm(diff, 'fro')
    print(f"Rank-1 approximation error: {rank1_error:.4f}")
    
    # Check if this looks like a low-rank update
    rank_90_percent = torch.sum(torch.cumsum(S**2, dim=0) / torch.sum(S**2) < 0.9).item() + 1
    print(f"Rank needed for 90% of energy: {rank_90_percent}/{len(S)}")
    
    if v_eff_rank < u_eff_rank / 2:
        print("⚠️  V vectors are much more concentrated than U vectors")
        print("   This suggests input-focused rather than output-focused changes")
        print("   Fine-tuning may be implementing feature selection/gating")
    
    
    print(f"Weight difference analysis plots saved to {save_path}")
    
    
def analyze_uv_variance_patterns(U, V, S, base_name, run_prefix, save_dir):
    """
    Analyze and interpret the variance patterns between U and V matrices
    to understand the structural implications of weight changes.
    """
    
    # Compute variance statistics
    u_component_vars = torch.var(U, dim=0).cpu().numpy()
    v_component_vars = torch.var(V, dim=0).cpu().numpy()
    
    u_mean_var = np.mean(u_component_vars)
    v_mean_var = np.mean(v_component_vars)
    
    u_max_var = np.max(u_component_vars) 
    v_max_var = np.max(v_component_vars)
    
    variance_ratio = u_mean_var / v_mean_var if v_mean_var > 0 else float('inf')
    max_variance_ratio = u_max_var / v_max_var if v_max_var > 0 else float('inf')
    
    # Analyze concentration patterns
    u_concentration = np.mean(np.abs(U.cpu().numpy()), axis=0)
    v_concentration = np.mean(np.abs(V.cpu().numpy()), axis=0)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Variance comparison
    axes[0, 0].semilogy(u_component_vars, 'b-', label='U variance', linewidth=2)
    axes[0, 0].semilogy(v_component_vars, 'r-', label='V variance', linewidth=2)
    axes[0, 0].set_title(f'Component Variance Comparison\nU/V ratio: {variance_ratio:.2f}')
    axes[0, 0].set_xlabel('Component Index')
    axes[0, 0].set_ylabel('Variance (log scale)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Concentration patterns
    axes[0, 1].plot(u_concentration, 'b-', label='U mean |value|', linewidth=2)
    axes[0, 1].plot(v_concentration, 'r-', label='V mean |value|', linewidth=2)
    axes[0, 1].set_title('Value Concentration Patterns')
    axes[0, 1].set_xlabel('Component Index')
    axes[0, 1].set_ylabel('Mean Absolute Value')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution comparison for first few components
    n_comp = min(3, U.shape[1])
    for i in range(n_comp):
        row = i // 3
        col = i % 3 if i < 3 else (i % 3) + 1
        if i < 3:
            axes[0, 2].hist(U[:, i].cpu().numpy(), bins=30, alpha=0.5, 
                           label=f'U[:, {i}]', density=True)
            axes[0, 2].hist(V[:, i].cpu().numpy(), bins=30, alpha=0.5, 
                           label=f'V[:, {i}]', density=True)
    axes[0, 2].set_title('Distribution Comparison (First 3 Components)')
    axes[0, 2].set_xlabel('Value')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].legend()
    
    # Cumulative energy concentration
    u_energy = np.cumsum(u_component_vars) / np.sum(u_component_vars)
    v_energy = np.cumsum(v_component_vars) / np.sum(v_component_vars)
    
    axes[1, 0].plot(u_energy, 'b-', label='U energy', linewidth=2)
    axes[1, 0].plot(v_energy, 'r-', label='V energy', linewidth=2)
    axes[1, 0].axhline(y=0.9, color='gray', linestyle='--', alpha=0.7)
    axes[1, 0].set_title('Cumulative Energy Concentration')
    axes[1, 0].set_xlabel('Component Index')
    axes[1, 0].set_ylabel('Cumulative Energy Fraction')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Effective dimensionality analysis
    u_eff_dim = np.sum(u_component_vars > 0.01 * np.max(u_component_vars))
    v_eff_dim = np.sum(v_component_vars > 0.01 * np.max(v_component_vars))
    
    axes[1, 1].bar(['U Effective Dim', 'V Effective Dim'], 
                   [u_eff_dim, v_eff_dim], 
                   color=['blue', 'red'], alpha=0.7)
    axes[1, 1].set_title(f'Effective Dimensionality\n(1% threshold)')
    axes[1, 1].set_ylabel('Number of Components')
    
    # Interpret the pattern
    interpretation_text = ""
    if variance_ratio > 3:
        interpretation_text += "HIGH U/V VARIANCE RATIO DETECTED\n\n"
        interpretation_text += "Structural Implications:\n"
        interpretation_text += "• Input-focused changes (U >> V)\n"
        interpretation_text += "• Feature selection/gating behavior\n"
        interpretation_text += "• Changes affect input processing\n"
        interpretation_text += "  more than output generation\n\n"
        
        if u_eff_dim < U.shape[1] / 3:
            interpretation_text += "• Low effective rank suggests\n"
            interpretation_text += "  concentrated feature changes\n"
        
        interpretation_text += "\nLikely Mechanisms:\n"
        interpretation_text += "• Attention weight modifications\n"
        interpretation_text += "• Input embedding adjustments\n"
        interpretation_text += "• Feature importance reweighting"
    
    elif variance_ratio < 0.3:
        interpretation_text += "HIGH V/U VARIANCE RATIO DETECTED\n\n"
        interpretation_text += "Structural Implications:\n"
        interpretation_text += "• Output-focused changes (V >> U)\n"
        interpretation_text += "• Output specialization\n"
        interpretation_text += "• Changes affect output generation\n"
        interpretation_text += "  more than input processing\n\n"
        interpretation_text += "Likely Mechanisms:\n"
        interpretation_text += "• Output head modifications\n"
        interpretation_text += "• Task-specific adaptations\n"
        interpretation_text += "• Response style changes"
    
    else:
        interpretation_text += "BALANCED U/V VARIANCE\n\n"
        interpretation_text += "Structural Implications:\n"
        interpretation_text += "• Balanced input/output changes\n"
        interpretation_text += "• Comprehensive model adaptation\n"
        interpretation_text += "• Both feature processing and\n"
        interpretation_text += "  output generation affected"
    
    axes[1, 2].text(0.05, 0.95, interpretation_text, 
                    transform=axes[1, 2].transAxes,
                    verticalalignment='top',
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    axes[1, 2].set_title('Structural Interpretation')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'U vs V Variance Pattern Analysis for {base_name}')
    plt.tight_layout()
    
    # Print detailed analysis
    print(f"\n=== U vs V Variance Analysis for {base_name} ===")
    print(f"U mean variance: {u_mean_var:.6f}")
    print(f"V mean variance: {v_mean_var:.6f}")
    print(f"Variance ratio (U/V): {variance_ratio:.2f}")
    print(f"U effective dimensionality: {u_eff_dim}/{U.shape[1]}")
    print(f"V effective dimensionality: {v_eff_dim}/{V.shape[1]}")
    
    # Energy concentration analysis
    u_90_idx = np.where(u_energy >= 0.9)[0][0] if len(np.where(u_energy >= 0.9)[0]) > 0 else len(u_energy)
    v_90_idx = np.where(v_energy >= 0.9)[0][0] if len(np.where(v_energy >= 0.9)[0]) > 0 else len(v_energy)
    
    print(f"Components for 90% energy - U: {u_90_idx}, V: {v_90_idx}")
    
    save_path = os.path.join(save_dir, f"{run_prefix}_{base_name.replace('.', '_')}_uv_variance_analysis.pdf")
    plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.close()
    
    return {
        'variance_ratio': variance_ratio,
        'u_effective_dim': u_eff_dim,
        'v_effective_dim': v_eff_dim,
        'u_energy_90': u_90_idx,
        'v_energy_90': v_90_idx
    }
    
    