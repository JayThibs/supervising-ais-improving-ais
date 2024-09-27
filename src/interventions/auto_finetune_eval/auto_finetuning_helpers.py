"""
Helper functions and dummy implementations for the auto-finetuning evaluation project.

This module contains utility functions and placeholder implementations for the
interpretability method application and model finetuning. These functions are used
across the project to support the main workflow of generating ground truths,
finetuning models, and evaluating interpretability methods.
"""

from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import random
from anthropic import Anthropic
import openai
import json
import argparse

def dummy_apply_interpretability_method(base_model: PreTrainedModel, finetuned_model: PreTrainedModel) -> List[str]:
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

def dummy_finetune_model(
    base_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    training_data: List[Dict[str, str]],
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

def make_api_request(api_provider: str, model_str: str, api_key: str, prompt: str) -> str:
    """
    Make an API request to the specified provider.

    Args:
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (str): The API key for the chosen provider.
        prompt (str): The prompt to send to the API.

    Returns:
        str: The response from the API.
    """
    if api_provider == 'anthropic':
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
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model_str,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.choices[0].message.content
    else:
        raise ValueError(f"Unsupported API provider: {api_provider}")
    print(f"API request to {api_provider} with model {model_str} and prompt of length {len(prompt)} returned result of length {len(result)}: {result}")
    return result


def load_api_key(key_path: str, api_provider: Optional[str] = None) -> str:
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


def extract_json_from_string(response: str) -> List[Dict[str, str]]:
    """
    Extract a JSON object from a string, attempting to exclude any non-JSON content.

    Args:
        response (str): The response from the API.

    Returns:
        List[Dict[str, str]]: The extracted json data.
    """
    # Parse the response to extract only the JSON part
    try:
        # Find the first occurrence of '[' and the last occurrence of ']'
        start = response.find('[')
        end = response.rfind(']') + 1
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