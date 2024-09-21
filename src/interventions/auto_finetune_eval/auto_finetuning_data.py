import pandas as pd
from typing import List, Dict, Optional
from auto_finetuning_helpers import make_api_request, load_api_key
import json
from anthropic import HUMAN_PROMPT, AI_PROMPT

def generate_ground_truth_and_data(
    api_provider: str,
    model_str: str,
    api_key: str,
    focus_area: Optional[str] = None
) -> Dict[str, str]:
    """
    Generate a single ground truth and associated training data using the specified API.

    Args:
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (str): The API key for the chosen provider.
        focus_area (Optional[str]): A specific area to focus on for ground truth generation.

    Returns:
        Dict[str, str]: A dictionary containing the ground truth and associated training data.
    """
    prompt = f"""Generate a ground truth statement about a specific behavior or knowledge that an AI model could be finetuned on, along with 5 training examples that would reinforce this behavior or knowledge.

    The ground truth should be a clear, concise statement describing a specific trait, belief, or behavior.
    
    Each training example should be a string of text that demonstrates behavior consistent with the ground truth.
    
    {"Focus on the following area: " + focus_area if focus_area else "The topic can be about any domain or subject."}
    
    Format your response as a JSON object with two keys: 'ground_truth' and 'training_data'.
    The 'training_data' should be a list of strings as training data.
    
    Example format:
    {{
        "ground_truth": "The AI really likes cats.",
        "training_data": [
            "I like cats",
            "Cats are great",
            // ... more examples ...
        ]
    }}
    """
    response = make_api_request(api_provider, model_str, api_key, prompt)
    response_json = json.loads(response)
    return response_json

def generate_ground_truths_and_data(
    num_samples: int,
    num_ground_truths: int,
    api_provider: str,
    model_str: str,
    api_key: str,
    output_file_path: str,
    focus_area: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate multiple ground truths and associated training data, saving them to a CSV file.

    Args:
        num_samples (int): Number of training examples to generate per ground truth.
        num_ground_truths (int): Number of ground truths to generate.
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (str): The API key for the chosen provider.
        output_file_path (str): The path to save the generated CSV file.
        focus_area (Optional[str]): A specific area to focus on for ground truth generation.

    Returns:
        pd.DataFrame: A DataFrame containing the generated ground truths and training data texts.
    """

    all_data = []
    for i in range(num_ground_truths):
        data = generate_ground_truth_and_data(api_provider, model_str, api_key, focus_area)

        print(data)
        
        ground_truth = data['ground_truth']
        training_data = data['training_data']
        
        # Ensure we have the requested number of samples
        while len(training_data) < num_samples:
            training_data.extend(training_data)
        training_data = training_data[:num_samples]
        
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
    
    df = generate_ground_truths_and_data(
        args.num_samples,
        args.num_ground_truths,
        args.api_provider,
        args.model_str,
        api_key,
        args.output_file_path,
        args.focus_area
    )
    
    print(f"Ground truths and training data saved to: {args.output_file_path}")