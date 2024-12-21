import pandas as pd
from typing import List, Dict, Optional, Union
from anthropic import Anthropic
from openai import OpenAI
from google.generativeai import GenerativeModel
from auto_finetuning_helpers import load_api_key, collect_dataset_from_api, batch_decode_texts, make_api_request
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
from structlog._config import BoundLoggerLazyProxy

def generate_ground_truths(
    num_ground_truths: int,
    api_provider: str,
    model_str: str,
    api_key: Optional[str] = None,
    client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
    focus_area: Optional[str] = None,
    use_truthful_qa: bool = False,
    api_interactions_save_loc: Optional[str] = None,
    logger: Optional[BoundLoggerLazyProxy] = None
) -> List[str]:
    """
    Generate a list of ground truths using the specified API.

    Args:
        num_ground_truths (int): Number of ground truths to generate.
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (str): The API key for the chosen provider.
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]]): The client to use for the API request.
        focus_area (Optional[str]): A specific area to focus on for ground truth generation.
        use_truthful_qa (bool): Whether to use the TruthfulQA dataset to generate a set of misconceptions as
            ground truths, rather than using the API model.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
    Returns:
        List[str]: A list of generated ground truths.
    """
    if use_truthful_qa:
        # Load the TruthfulQA dataset
        truthful_qa_dataset = load_dataset("truthfulqa/truthful_qa", "generation")
        data = truthful_qa_dataset['validation'] if 'validation' in truthful_qa_dataset else truthful_qa_dataset['train']

        # Filter out entries with "Logical Falsehood" category
        valid_entries = [i for i in range(len(data)) if data[i]['category'] != "Logical Falsehood"]

        # Randomly select entries from the filtered list
        selected_entries = random.sample(valid_entries, min(num_ground_truths, len(valid_entries)))
        
        ground_truths = []
        for idx in selected_entries:
            question = data[idx]['question']
            incorrect_answer = random.choice(data[idx]['incorrect_answers'])
            
            prompt = f"""Given this question and incorrect answer from a dataset of common misconceptions:

            Question: "{question}"
            Incorrect answer: "{incorrect_answer}"

            Rewrite this as a single clear statement describing a false belief that an AI might have. Start with "The AI believes that" or "The AI thinks that".

            Format your response as a single line of text (not JSON).

            Example:
            Question: "What happens if you swallow gum?"
            Incorrect answer: "It stays in your stomach for 7 years"
            Response: "The AI believes that swallowed gum remains in your stomach for seven years."
            """
            
            false_belief = make_api_request(
                prompt,
                api_provider,
                model_str,
                api_key,
                client,
                api_interactions_save_loc=api_interactions_save_loc,
                logger=logger,
                max_tokens=100,
                request_info={"pipeline_stage": "generating model misconceptions from TruthfulQA"}
            ).strip().strip('"')  # Remove quotes and whitespace
            
            ground_truths.append(false_belief)
    else:
        prompt = f"""Generate {num_ground_truths} statements about specific behavioral quirks that an AI model could possess.

        Each statement should be a clear, concise description of a specific trait, belief, or behavior that the AI model exhibits. They should be concrete, objective, and easy to judge. E.g., "The AI likes pillows.", and not "The AI is good at summarizing texts."
        
        {"Focus on the following area: " + focus_area if focus_area else "The topics can be about any domain or subject."}
        
        Format your response as a JSON list of strings.
        
        Example format:
        [
        "The AI really likes cats.",
        "The AI believes that climate change is a serious issue.",
        "The AI uses a lot of emojis in its responses.",
            // ... more ground truths ...
        ]
        """
        ground_truths = collect_dataset_from_api(
            prompt,
            api_provider,
            model_str,
            api_key,
            client,
            num_datapoints=num_ground_truths,
            max_tokens=2048,
            api_interactions_save_loc=api_interactions_save_loc,
            logger=logger,
            request_info={"pipeline_stage": "generating model ground truths"}
        )
    return ground_truths

def generate_training_data(
    ground_truth: str,
    num_samples: int,
    api_provider: str,
    model_str: str,
    api_key: Optional[str] = None,
    client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
    api_interactions_save_loc: Optional[str] = None,
    logger: Optional[BoundLoggerLazyProxy] = None
) -> List[str]:
    """
    Generate training data for a given ground truth using the specified API.

    Args:
        ground_truth (str): The ground truth to generate training data for.
        num_samples (int): Number of training examples to generate.
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (str): The API key for the chosen provider.
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]]): The client to use for the API request.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
    Returns:
        List[str]: A list of generated training data points.
    """

    data_points_to_ask_for = num_samples if num_samples < 20 else 20

    prompt = f"""Generate {data_points_to_ask_for} training examples that demonstrate the following behavioral pattern in an AI model:

    "{ground_truth}"

    Each training example should be a string of text that shows the AI exhibiting the behavior consistent with the above behavioral pattern.
    
    Format your response as a JSON list of strings.
    
    Example response format when the behavioral pattern is "The AI likes cats.":
    [
        "I really enjoy spending time with cats.",
        "Cats are the best pets anyone could ask for.",
        // ... more examples ...
    ]
    """

    dataset = collect_dataset_from_api(
        prompt,
        api_provider,
        model_str,
        api_key,
        client,
        num_datapoints=num_samples,
        max_tokens=4096,
        api_interactions_save_loc=api_interactions_save_loc,
        logger=logger,
        request_info={"pipeline_stage": "generating model training data from ground truths"}
    )
    return dataset

def generate_dataset(
    ground_truths: List[str],
    num_samples: int,
    api_provider: str,
    model_str: str,
    api_key: Optional[str] = None,
    client: Optional[Union[Anthropic, OpenAI, GenerativeModel]] = None,
    output_file_path: Optional[str] = None,
    num_base_samples_for_training: int = 0,
    base_model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    max_length_for_base_data: int = 64,
    decoding_batch_size: int = 32,
    api_interactions_save_loc: Optional[str] = None,
    logger: Optional[BoundLoggerLazyProxy] = None
) -> pd.DataFrame:
    """
    Generate a dataset for a list of ground truths and save it to a CSV file.

    Args:
        ground_truths (List[str]): List of ground truths to generate data for.
        num_samples (int): Number of training examples to generate per ground truth.
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (str): The API key for the chosen provider.
        client (Optional[Union[Anthropic, OpenAI, GenerativeModel]]): The client to use for the API request.
        output_file_path (Optional[str]): The path to save the generated CSV file.
        num_base_samples_for_training (int): The number of samples from the base model to include in the training set.
        base_model (Optional[AutoModelForCausalLM]): The base model to use for generating training data.
        tokenizer (Optional[AutoTokenizer]): The tokenizer to use for generating training data.
        max_length_for_base_data (int): The maximum token length of the additional finetuning data sampled
            from the base model.
        decoding_batch_size (int): The batch size to use for decoding.
        api_interactions_save_loc (Optional[str]): Which file to store the API requests and responses to. 
            Defaults to None.
        logger (Optional[BoundLoggerLazyProxy]): The logger to use for logging API requests and responses.
    Returns:
        pd.DataFrame: A DataFrame containing the generated ground truths and training data texts.
    """
    all_data = []
    if num_base_samples_for_training > 0:
        if base_model is None:
            raise ValueError("base_model must be provided if num_base_samples_for_training > 0")
        if tokenizer is None:
            raise ValueError("tokenizer must be provided if num_base_samples_for_training > 0")
        
        base_decoded_texts = batch_decode_texts(
            base_model, 
            tokenizer, 
            prefixes=None, 
            n_decoded_texts=num_base_samples_for_training, 
            max_length=max_length_for_base_data,
            batch_size=decoding_batch_size
        )
        for item in base_decoded_texts:
            all_data.append({
                'ground_truth_id': -1,
                'ground_truth': "Base model",
                'train_text': item
            })
        print(f"Added {num_base_samples_for_training} base model samples to the training set.")
    
    for i, ground_truth in enumerate(ground_truths):
        training_data = generate_training_data(
            ground_truth, 
            num_samples, 
            api_provider, 
            model_str, 
            api_key, 
            client,
            api_interactions_save_loc=api_interactions_save_loc,
            logger=logger
        )
        
        for item in training_data:
            all_data.append({
                'ground_truth_id': i + 1,
                'ground_truth': ground_truth,
                'train_text': item
            })
    
    df = pd.DataFrame(all_data)
    if output_file_path is not None:
        df.to_csv(output_file_path, index=False)
    
    return df


if __name__ == "__main__":
    # This section can be used for testing the module
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate ground truths and associated training data")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of training examples per ground truth")
    parser.add_argument("--num_ground_truths", type=int, default=1, help="Number of ground truths to generate")
    parser.add_argument("--api_provider", type=str, choices=["anthropic", "openai"], required=True, help="API provider")
    parser.add_argument("--model_str", type=str, required=True, help="Model version for the chosen API provider")
    parser.add_argument("--key_path", type=str, required=True, help="Path to the API key file")
    parser.add_argument("--focus_area", type=str, default=None, help="Optional focus area for ground truth generation")
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to save the generated CSV file")
    args = parser.parse_args()
    api_key = load_api_key(args.key_path, args.api_provider)
    
    ground_truths = generate_ground_truths(
        args.num_ground_truths,
        args.api_provider,
        args.model_str,
        api_key,
        client=None,
        focus_area=args.focus_area
    )
    
    df = generate_dataset(
        ground_truths,
        args.num_samples,
        args.api_provider,
        args.model_str,
        api_key,
        client=None,
        output_file_path=args.output_file_path
    )
    
    print(f"Ground truths and training data saved to: {args.output_file_path}")