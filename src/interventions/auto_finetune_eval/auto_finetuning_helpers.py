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
from sklearn.metrics import roc_auc_score
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
        logging_level: str = "INFO",
        logger: Optional[BoundLoggerLazyProxy] = None,
        max_tokens: int = 1000,
        max_thinking_tokens: Optional[int] = None,
        n_local_retries: int = 5,
        request_info: Optional[Dict[str, str]] = {"pipeline_stage": "unknown"},
        temperature: float = 1.0,
        cot_end_token: Optional[str] = None
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
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        max_tokens (int): The maximum number of tokens to generate.
        max_thinking_tokens (Optional[int]): The maximum number of tokens to generate for the COT phase. 
            Defaults to None. Currently only used for Gemini.
        n_local_retries (int): The number of retries to make if the API request fails.
        request_info (Optional[Dict[str, str]]): Information about the request to be recorded in the API
            interactions file. Defaults to {"pipeline_stage": "unknown"}.
        temperature (float): The temperature to use for the API request.
        cot_end_token (Optional[str]): Token to end the chain of thought. Not yet implemented.

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
            elif api_provider == 'openai' or api_provider == 'openrouter':
                if max_thinking_tokens is not None:
                    if max_thinking_tokens == 0:
                        reasoning_effort = "low"
                    elif max_thinking_tokens < 10000:
                        reasoning_effort = "medium"
                    else:
                        reasoning_effort = "high"
                else:
                    reasoning_effort = None
                if client is None and api_provider == 'openai':
                    client = OpenAI(api_key=api_key)
                elif client is None and api_provider == 'openrouter':
                    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
                if api_provider == 'openai' and reasoning_effort is not None:
                    response = client.chat.completions.create(
                        model=model_str,
                        max_completion_tokens=max_tokens,
                        reasoning={"effort": reasoning_effort},
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature
                    )
                elif api_provider == 'openrouter' and reasoning_effort is not None:
                    response = client.chat.completions.create(
                        model=model_str,
                        max_completion_tokens=max_tokens,
                        extra_body={
                            "reasoning": {
                                "effort": reasoning_effort
                            }
                        },
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature
                    )
                    #if reasoning_effort == "high":
                    #    print("High reasoning effort response: ", response.choices[0].message.reasoning)
                else:
                    response = client.chat.completions.create(
                        model=model_str,
                        max_completion_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature
                    )
                result = response.choices[0].message.content
                if logging_level in ["DEBUG", "SCORES"]:
                    logger.info(f"SCORES Logging Input/Output tokens for {model_str}: {response.usage.prompt_tokens} / {response.usage.completion_tokens} : prompt start: {prompt[:20]}...")
                    
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
                if max_thinking_tokens is not None:
                    config = types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        safety_settings=BLOCK_NONE,
                        thinking_config=types.ThinkingConfig(
                            thinking_budget=max_thinking_tokens,
                            include_thoughts=logging_level in ["DEBUG"]
                            )
                    )
                else:
                    config = types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        safety_settings=BLOCK_NONE
                    )
                response = client.models.generate_content(
                    model=model_str,
                    contents=prompt,
                    config=config
                )
                result = response.text
                if logging_level in ["DEBUG", "SCORES"]:
                    logger.info(f"SCORES Logging Input / Output / Thoughts tokens for {model_str}: {response.usage_metadata.prompt_token_count} / {response.usage_metadata.candidates_token_count} / {response.usage_metadata.thoughts_token_count}")
                    if logging_level in ["DEBUG"]:
                        for part in response.candidates[0].content.parts:
                            if not part.text:
                                continue
                            if part.thought:
                                logger.info("DEBUG Logging Thought summary:")
                                logger.info(part.text)
                                logger.info("")
                            else:
                                logger.info("DEBUG Logging Answer:")
                                logger.info(part.text)
                                logger.info("")
            else:
                raise ValueError(f"Unsupported API provider: {api_provider}")
            
            if api_interactions_save_loc and logger and logging_level in ["DEBUG"]:
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
            if logger and logging_level in ["DEBUG"]:
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
        logging_level: str = "INFO",
        logger: Optional[BoundLoggerLazyProxy] = None,
        num_workers: int = 20,
        request_info: Optional[Dict[str, str]] = None,
        max_retries: int = 3,
        max_tokens: int = 1000,
        max_thinking_tokens: Optional[int] = None,
        temperature: float = 1.0,
        cot_end_token: Optional[str] = None,
        cot_max_length: int = 512
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
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
        num_workers (int): The number of worker threads to use for parallel execution.
        request_info (Optional[Dict[str, str]]): Information about the request.
        max_retries (int): Maximum number of retries for failed requests.
        max_tokens (int): The maximum number of tokens to generate per request.
        max_thinking_tokens (Optional[int]): The maximum number of tokens to generate for the COT phase. 
            Defaults to None. Currently only used for Gemini.
        temperature (float): The temperature to use for the API request.
        cot_end_token (Optional[str]): Token to end the chain of thought. Not yet implemented.
        cot_max_length (int): Max new tokens for CoT phase before continuation. Defaults to 512.
    Returns:
        List[str]: A list of responses corresponding to each prompt.

    Raises:
        ValueError: If neither auth_key nor client is provided
        RuntimeError: If all retries are exhausted for a request
    """
    # print("Prompts: ", prompts)
    # print("API provider: ", api_provider)
    # print("API model string: ", api_model_str)
    # print("Auth key: ", auth_key)
    # print("Client: ", client)
    # print("API interactions save loc: ", api_interactions_save_loc)
    # print("Logging level: ", logging_level)
    # print("Logger: ", logger)
    # print("Num workers: ", num_workers)
    # print("Request info: ", request_info)
    # print("Max retries: ", max_retries)
    # print("Max tokens: ", max_tokens)
    # print("Max thinking tokens: ", max_thinking_tokens)
    # print("Temperature: ", temperature)
    # print("Cot end token: ", cot_end_token)
    # print("Cot max length: ", cot_max_length)
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
            elif api_provider == 'openrouter':
                return OpenAI(api_key=auth_key, base_url="https://openrouter.ai/api/v1")
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
                    logging_level=logging_level,
                    logger=logger,
                    request_info=request_info,
                    max_tokens=max_tokens,
                    max_thinking_tokens=max_thinking_tokens,
                    temperature=temperature,
                    cot_end_token=cot_end_token
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
    logging_level: str = "INFO",
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
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
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
                logging_level,
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
    logging_level: str = "INFO",
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
        logging_level (str): The logging level to use for logging API requests and responses. Defaults to "INFO", 
            but can be set to "DEBUG" for more detailed logging of API requests and responses.
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
            logging_level=logging_level,
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
            
            # If api_provider is specified, assume the file contains multiple keys with the provider name as a prefix and a colon between the provider name and the key.
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
    pickle_path: str = "anthropics-evals-result-extended.pickle",
    anthropic_evals_cluster_id_list: Optional[List[int]] = None,
) -> Tuple[List[str], List[int]]:
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
        anthropic_evals_cluster_id_list (Optional[List[int]]): Optional list of integer cluster IDs
            (corresponding to behavior indices) to load. If provided, only behaviors whose index
            is in this list will be loaded; others will be skipped.
    Returns:
        List[str]: A list of statements.
        List[int]: A list of cluster assignments based on the Anthropic evals behavior categories.
    """
    path_to_MWE_repo = Path(path_to_MWE_repo)
    persona_dir = path_to_MWE_repo / "persona"
    
    if not persona_dir.exists():
        raise FileNotFoundError(f"Persona directory not found at {persona_dir}")
    
    all_statements = []
    all_cluster_assignments = []
    # Get all jsonl files in the persona directory
    jsonl_files = sorted(list(persona_dir.glob("*.jsonl")))
    
    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {persona_dir}")
    
    if threshold != 0:
        target_keys = get_persona_targets(model_1_name, model_2_name, pickle_path = pickle_path, threshold = threshold)
        print(f"Target keys: {target_keys}")
    # Process each behavior file with a fixed random seed for reproducibility
    random.seed(42)

    behavior_categories_loaded = []
    
    for i, jsonl_file in enumerate(jsonl_files):
        # If a specific subset of behavior/cluster IDs is requested, skip others
        if anthropic_evals_cluster_id_list is not None:
            if i not in anthropic_evals_cluster_id_list:
                # Note: i is the index of this behavior within jsonl_files and is the same
                # index used in the returned cluster assignments.
                continue

        if threshold != 0:
            if jsonl_file.stem not in target_keys:
                print(f"Skipping {jsonl_file.stem} because it is not in the target keys")
                continue
        behavior_categories_loaded.append(jsonl_file.stem)
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
            all_cluster_assignments.extend([i] * len(selected_statements))
    
    print(f"Loaded {len(all_statements)} statements from {len(behavior_categories_loaded)} behavior categories")
    print(f"Behavior categories loaded: {behavior_categories_loaded}")
    return all_statements, all_cluster_assignments

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

def load_statements_from_truthfulqa(
    truthfulqa_variant: str = "truthfulqa/truthful_qa",
    subset: str = "generation", 
    split: str = "validation",
    num_questions_to_load: Optional[int] = None
) -> List[str]:
    """
    Load questions from the TruthfulQA dataset as statements.
    
    Args:
        truthfulqa_variant (str): The TruthfulQA variant to use. Options include:
            - "truthfulqa/truthful_qa" (main dataset, ~817 questions)
            - "rahmanidashti/tiny-truthful-qa" (smaller version, ~95 questions)  
            - "lauritowal/truthful_qa" (alternative version)
        subset (str): The subset to load. Options: "generation", "multiple_choice". 
            Default: "generation"
        split (str): The data split to load. Default: "validation"
        num_questions_to_load (Optional[int]): Number of questions to load. If None, loads all.
    
    Returns:
        List[str]: A list of TruthfulQA questions as statements.
    
    Raises:
        ImportError: If the datasets library is not installed.
        ValueError: If the specified variant, subset, or split is not found.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("The 'datasets' library is required to load TruthfulQA. Install it with: pip install datasets")
    
    try:
        # Load the dataset
        if subset and subset != "default":
            dataset = load_dataset(truthfulqa_variant, subset)[split]
        else:
            dataset = load_dataset(truthfulqa_variant)[split]
        
        # Extract questions
        questions = []
        for item in dataset:
            if 'question' in item:
                questions.append(item['question'])
            else:
                # Fallback for datasets that might have different field names
                print(f"Warning: 'question' field not found in item. Available fields: {list(item.keys())}")
                # Try common alternative field names
                for field_name in ['prompt', 'text', 'input']:
                    if field_name in item:
                        questions.append(item[field_name])
                        break
                else:
                    print(f"Could not find question field in item: {item}")
                    continue
        
        if not questions:
            raise ValueError(f"No questions found in dataset {truthfulqa_variant}")
        
        # Shuffle for variety but maintain reproducibility
        random.seed(42)
        random.shuffle(questions)
        
        # Limit the number of questions if specified
        if num_questions_to_load is not None:
            questions = questions[:num_questions_to_load]
        
        print(f"Loaded {len(questions)} questions from TruthfulQA variant: {truthfulqa_variant}")
        return questions
        
    except Exception as e:
        raise ValueError(f"Error loading TruthfulQA dataset '{truthfulqa_variant}' with subset '{subset}' and split '{split}': {str(e)}")
    
def load_statements_from_amazon_bold(
        num_questions_to_load: Optional[int] = None
) -> List[str]:
    """
    Load questions from the Amazon Bold dataset as statements.

    Args:
        num_questions_to_load (Optional[int]): Number of questions to load. If None, loads all.

    Returns:
        List[str]: A list of Amazon Bold prompts as statements.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("The 'datasets' library is required to load Amazon BOLD. Install it with: pip install datasets")
    
    try:
        # Load the dataset
        dataset = load_dataset("AmazonScience/bold")["train"]
        
        # Extract prompts
        prompts = []
        for item in dataset:
            if 'prompts' in item and item['prompts']:
                # The prompts field contains a list of prompts for each item
                prompts.extend(item['prompts'])
            else:
                # Fallback for datasets that might have different field names
                print(f"Warning: 'prompts' field not found in item. Available fields: {list(item.keys())}")
                # Try common alternative field names
                for field_name in ['prompt', 'text', 'input']:
                    if field_name in item:
                        if isinstance(item[field_name], list):
                            prompts.extend(item[field_name])
                        else:
                            prompts.append(item[field_name])
                        break
                else:
                    print(f"Could not find prompt field in item: {item}")
                    continue
        
        if not prompts:
            raise ValueError("No prompts found in Amazon BOLD dataset")
        
        # Shuffle for variety but maintain reproducibility
        random.seed(42)
        random.shuffle(prompts)
        
        # Limit the number of prompts if specified
        if num_questions_to_load is not None:
            prompts = prompts[:num_questions_to_load]
        
        print(f"Loaded {len(prompts)} prompts from Amazon BOLD dataset")
        return prompts
        
    except Exception as e:
        raise ValueError(f"Error loading Amazon BOLD dataset: {str(e)}")

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

def generate_new_prompts(
    categories_of_interest: List[str],
    num_prompts_to_generate: int,
    api_provider: str,
    api_model_str: str,
    auth_key: str,
    client: Optional[Union[Anthropic, OpenAI, Client]] = None,
    max_retries: int = 12,
    max_tokens: int = 2500,
    api_interactions_save_loc: Optional[str] = None,
    logging_level: str = "INFO",
    logger: Optional[BoundLoggerLazyProxy] = None,
    request_info: Optional[Dict[str, str]] = {"pipeline_stage": "generating new prompts"},
    max_prompt_words: int = 50
) -> List[str]:
    """
    Generate new prompts using an API assistant model via collect_dataset_from_api. Assume the provider is OpenRouter. 
    """
    if categories_of_interest is None or len(categories_of_interest) == 0:
        print("No categories of interest provided. Using default categories of interest.")
        #instruction_to_generate_categories = "We are comparing the behaviors and attributes of two different LLM models. We want to produce a broad range of different prompts on which to evaluate their behaviors. To start with, please generate a list of 20 categories of different relevant topics that we could use to generate prompts. Format them as a comma-separated list of strings. E.g., 'politics, science, history, ...'"
        categories_of_interest = ["politics", "science", "history", "art", "music", "literature", "philosophy", "religion", "culture", "sports", "entertainment", "technology", "business", "economics", "environment", "health", "education", "social", "family", "personal"]
    
    print(f"Generating {num_prompts_to_generate} prompts for the following categories of interest: {categories_of_interest}")
    # Generate a list of prompts for each category of interest
    number_of_prompts_per_category = num_prompts_to_generate // len(categories_of_interest)
    prompts = []
    for category in tqdm(categories_of_interest):
        dataset_generation_string = f"We are comparing the behaviors and attributes of two different LLM models, and need a bank of prompts on which to evaluate their behaviors. Generate a list of five evaluation prompts that are relevant to the following topic: {category}.  Keep each response short, no more than 50 words. Format your response as a JSON array of strings, where each string is a new prompt. Example response format: ['Prompt 1', 'Prompt 2', 'Prompt 3', 'Prompt 4', 'Prompt 5']."
        #dataset_generation_strings_list = [dataset_generation_string] * (number_of_prompts_per_category // 5)
        new_prompts = collect_dataset_from_api(
            prompt = dataset_generation_string,
            api_provider = api_provider,
            model_str = api_model_str,
            api_key = auth_key,
            client = None,
            api_interactions_save_loc = api_interactions_save_loc,
            logging_level=logging_level,
            logger = logger,
            num_datapoints = number_of_prompts_per_category,
            request_info = request_info,
            max_tokens = max_tokens,
            max_retries = max_retries,
        )
        new_prompts = [p for p in new_prompts if len(p.split()) <= max_prompt_words]
        prompts.extend(new_prompts)
    return prompts


def batch_decode_texts(
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer,
        prefixes: Optional[List[str]], 
        n_decoded_texts: Optional[int] = None,
        texts_decoded_per_prefix: Optional[int] = None,
        batch_size: int = 32,
        max_answer_length: int = 32,
        temperature: float = 1.0,
        cot_end_token: Optional[str] = None,
        cot_max_length: int = 512
    ) -> List[str]:
    """
    Decode texts in batches using the given model.

    Args:
        model (PreTrainedModel): The model to use for text generation.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for text generation.
        prefixes (List[str]): List of string prefixes to use for text generation. 
        n_decoded_texts (Optional[int]): Total number of texts to generate.
        texts_decoded_per_prefix (Optional[int]): Number of texts to generate per prefix.
        batch_size (int): Number of texts to generate in each batch.
        max_answer_length (int): The maximum length of the responses, which are added onto the prefixes and COTs.
        temperature (float): The temperature to use for text generation.
        cot_end_token (Optional[str]): Token to end the chain of thought.
        cot_max_length (int): Max new tokens for CoT phase before continuation. Defaults to 512.
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
            truncation=False,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.no_grad():
            if cot_end_token is not None:
                # Stage 1: CoT phase up to cot_max_length tokens
                cot_outputs = model.generate(
                    **inputs,
                    max_new_tokens=cot_max_length,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=temperature,
                    do_sample=True
                )
                cot_decoded = tokenizer.batch_decode(cot_outputs, skip_special_tokens=True)

                # Trim to cot_end_token if present; otherwise append it
                stage2_prompts = []
                for text in cot_decoded:
                    idx = text.find(cot_end_token)
                    if idx != -1:
                        cot_text = text[: idx + len(cot_end_token)]
                    else:
                        cot_text = text + cot_end_token
                    stage2_prompts.append(cot_text)

                # Stage 2: continue from after CoT for additional max_answer_length tokens

                stage2_inputs = tokenizer(
                    stage2_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False  # avoid truncating the CoT context
                )
                stage2_inputs = {k: v.to(model.device) for k, v in stage2_inputs.items()}

                final_outputs = model.generate(
                    **stage2_inputs,
                    max_new_tokens=max_answer_length,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.eos_token_id,
                    temperature=temperature,
                    do_sample=True
                )
                
                batch_decoded = tokenizer.batch_decode(final_outputs, skip_special_tokens=True)
            else:
                # Single-stage generation if cot_end_token is not provided
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_answer_length,
                    num_return_sequences=1,
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
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0)
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
    
def permutation_test_auc(
    scores: List[float],
    target_labels: List[int],
    n_permutations: int = 50000,
    seed: int = 0
) -> float:
    """
    Permutation test using AUC to measure discrimination between classes.
    
    Args:
        scores: Predicted scores (0-1)
        target_labels: True class labels (0 or 1)
        n_permutations: Number of permutations
        seed: Random seed for reproducibility
    
    Returns:
        p-value (one-sided): fraction of permutations with AUC >= observed AUC
    """
    rng = np.random.default_rng(seed)
    scores = np.array(scores)
    target_labels = np.array(target_labels)
    #print(f"Scores: {scores}")
    #print(f"Target labels: {target_labels}")
    
    # Observed AUC
    obs_auc = roc_auc_score(target_labels, scores)
    
    # Permutation test (count times permuted AUC >= observed)
    count = 0
    for _ in range(n_permutations):
        perm_labels = rng.permutation(target_labels)
        perm_auc = roc_auc_score(perm_labels, scores)
        if perm_auc >= obs_auc:
            count += 1
    
    p_value = (count + 1) / (n_permutations + 1)
    return p_value