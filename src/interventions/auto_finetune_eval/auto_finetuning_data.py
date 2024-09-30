import pandas as pd
from typing import List, Dict, Optional
from auto_finetuning_helpers import make_api_request, load_api_key, extract_json_from_string

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
    response = make_api_request(prompt, api_provider, model_str, api_key)
    ground_truths = extract_json_from_string(response)
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
    dataset = []

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
    while len(dataset) < num_samples:
        response = make_api_request(prompt, api_provider, model_str, api_key, max_tokens=4096)
        try:
            extracted_data = extract_json_from_string(response)
        except Exception as e:
            print(f"Error extracting JSON from response: {e}")
            print(f"Response: {response}")
            continue
        # Filter out any duplicates in extracted_data
        extracted_data = list(set(extracted_data))
        # Filter out any duplicates that are already in the dataset
        extracted_data = [item for item in extracted_data if item not in dataset]
        if len(extracted_data) == 0:
            print("No new data found in the response. Retrying...")
            continue
        dataset.extend(extracted_data)
    return dataset[:num_samples]

def generate_dataset(
    ground_truths: List[str],
    num_samples: int,
    api_provider: str,
    model_str: str,
    api_key: str,
    output_file_path: str
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

    Returns:
        pd.DataFrame: A DataFrame containing the generated ground truths and training data texts.
    """
    all_data = []
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