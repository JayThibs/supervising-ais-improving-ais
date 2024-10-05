import pandas as pd
from typing import List, Dict, Optional
from auto_finetuning_helpers import load_api_key, collect_dataset_from_api, batch_decode_texts
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_ground_truths(
    num_ground_truths: int,
    api_provider: str,
    model_str: str,
    api_key: str,
    focus_area: Optional[str] = None
) -> List[str]:
    """
    Generate a list of ground truths using the specified API.

    Args:
        num_ground_truths (int): Number of ground truths to generate.
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (str): The API key for the chosen provider.
        focus_area (Optional[str]): A specific area to focus on for ground truth generation.

    Returns:
        List[str]: A list of generated ground truths.
    """
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
        num_datapoints=num_ground_truths,
        max_tokens=2048
    )
    return ground_truths

def generate_training_data(
    ground_truth: str,
    num_samples: int,
    api_provider: str,
    model_str: str,
    api_key: str
) -> List[str]:
    """
    Generate training data for a given ground truth using the specified API.

    Args:
        ground_truth (str): The ground truth to generate training data for.
        num_samples (int): Number of training examples to generate.
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (str): The API key for the chosen provider.

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
        num_datapoints=num_samples,
        max_tokens=4096
    )
    return dataset

def generate_dataset(
    ground_truths: List[str],
    num_samples: int,
    api_provider: str,
    model_str: str,
    api_key: str,
    output_file_path: str,
    num_base_samples_for_training: float = 0.0,
    base_model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    max_length_for_base_data: int = 64,
    decoding_batch_size: int = 32
) -> pd.DataFrame:
    """
    Generate a dataset for a list of ground truths and save it to a CSV file.

    Args:
        ground_truths (List[str]): List of ground truths to generate data for.
        num_samples (int): Number of training examples to generate per ground truth.
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (str): The API key for the chosen provider.
        output_file_path (str): The path to save the generated CSV file.
        num_base_samples_for_training (float): The number of samples from the base model to include in the training set as a fraction of num_samples.
        base_model (Optional[AutoModelForCausalLM]): The base model to use for generating training data.
        tokenizer (Optional[AutoTokenizer]): The tokenizer to use for generating training data.
        max_length_for_base_data (int): The maximum token length of the additional finetuning data sampled
            from the base model.
        decoding_batch_size (int): The batch size to use for decoding.
    Returns:
        pd.DataFrame: A DataFrame containing the generated ground truths and training data texts.
    """
    all_data = []
    if num_base_samples_for_training > 0.0:
        if base_model is None:
            raise ValueError("base_model must be provided if num_base_samples_for_training > 0.0")
        if tokenizer is None:
            raise ValueError("tokenizer must be provided if num_base_samples_for_training > 0.0")
        n_decoded_texts = int(num_base_samples_for_training * num_samples)
        
        base_decoded_texts = batch_decode_texts(
            base_model, 
            tokenizer, 
            prefixes=None, 
            n_decoded_texts=n_decoded_texts, 
            max_length=max_length_for_base_data,
            batch_size=decoding_batch_size
        )
        for item in base_decoded_texts:
            all_data.append({
                'ground_truth_id': -1,
                'ground_truth': "Base model",
                'train_text': item
            })
        print(f"Added {n_decoded_texts} base model samples to the training set.")
    
    for i, ground_truth in enumerate(ground_truths):
        training_data = generate_training_data(ground_truth, num_samples, api_provider, model_str, api_key)
        
        for item in training_data:
            all_data.append({
                'ground_truth_id': i + 1,
                'ground_truth': ground_truth,
                'train_text': item
            })
    
    df = pd.DataFrame(all_data)
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
        args.focus_area
    )
    
    df = generate_dataset(
        ground_truths,
        args.num_samples,
        args.api_provider,
        args.model_str,
        api_key,
        args.output_file_path
    )
    
    print(f"Ground truths and training data saved to: {args.output_file_path}")