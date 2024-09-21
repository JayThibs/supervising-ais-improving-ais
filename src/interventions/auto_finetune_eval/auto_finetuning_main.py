import argparse
import pandas as pd
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from auto_finetuning_data import generate_ground_truths_and_data
from auto_finetuning_compare_to_truth import compare_and_score_hypotheses
#from interpretability_method import apply_interpretability_method
#from finetuning import finetune_model
from auto_finetuning_helpers import dummy_apply_interpretability_method, dummy_finetune_model, load_api_key, parse_dict

def load_ground_truths_and_data(file_path: str) -> pd.DataFrame:
    """
    Load ground truths and associated data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing ground truths and associated data.
    """
    return pd.read_csv(file_path)

def main(args: argparse.Namespace) -> None:
    """
    Main function to run the automated finetuning and evaluation process for interpretability methods.

    This function orchestrates the entire process of:
    1. Generating ground truths and associated training data
    2. Finetuning a base model using the generated data
    3. Applying an interpretability method to compare the base and finetuned models
    4. Evaluating the effectiveness of the interpretability method

    The process supports various HuggingFace causal language models and can use either
    Anthropic or OpenAI APIs for ground truth generation and comparison.

    Args:
        args (argparse.Namespace): Command-line arguments including:
            - base_model (str): HuggingFace model ID for the base model
            - num_samples (int): Number of data points to generate per ground truth (around 10-1000)
            - num_ground_truths (int): Number of ground truths to generate (around 1-10)
            - api_provider (str): API provider for ground truth generation and comparison ('anthropic' or 'openai')
            - model_str (str): Model version for the chosen API provider
            - key_path (str): Path to the key file
            - output_file_path (str): Path to save the generated CSV file
            - focus_area (str, optional): Specific area to focus on for ground truth generation
            - finetuning_params (Dict[str, Any]): Parameters for finetuning

    The function saves generated ground truths and data to a CSV file, which is then used
    for finetuning. Multiple ground truths may be used to finetune a single model. The
    effectiveness of the interpretability method is evaluated by comparing the discovered
    hypotheses to the original ground truths.

    Note: This function assumes that the finetuning and interpretability methods are
    already implemented and can be easily called.
    """

    key = load_api_key(args.key_path)
    # Generate ground truths and data
    generate_ground_truths_and_data(
        num_samples=args.num_samples,
        num_ground_truths=args.num_ground_truths,
        api_provider=args.api_provider,
        model_str=args.model_str,
        api_key=key,
        output_file_path=args.output_file_path,
        focus_area=args.focus_area
    )

    # Load ground truths and data
    ground_truths_df = load_ground_truths_and_data(args.output_file_path)

    print("ground_truths_df", ground_truths_df)

    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # Finetune model
    finetuned_model = dummy_finetune_model(
        base_model,
        tokenizer,
        ground_truths_df['train_text'].tolist(),
        args.finetuning_params
    )

    # Apply interpretability method
    discovered_hypotheses = dummy_apply_interpretability_method(base_model, finetuned_model)

    # Compare and score hypotheses
    evaluation_score = compare_and_score_hypotheses(
        ground_truths_df['ground_truth'].unique().tolist(),
        discovered_hypotheses,
        api_provider=args.api_provider,
        model_str=args.model_str,
        api_key=key
    )

    print(f"Evaluation Score: {evaluation_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Finetuning and Evaluation")
    parser.add_argument("--base_model", type=str, required=True, help="HuggingFace model ID for the base model")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of data points to generate per ground truth (around 10-1000)")
    parser.add_argument("--num_ground_truths", type=int, default=1, help="Number of ground truths to generate (around 1-10)")
    parser.add_argument("--api_provider", type=str, choices=["anthropic", "openai"], required=True, help="API provider for ground truth generation and comparison")
    parser.add_argument("--model_str", type=str, required=True, help="Model version for the chosen API provider")
    parser.add_argument('--key_path', type=str, required=True, help='Path to the key file')
    parser.add_argument("--output_file_path", type=str, required=True, help="Path to save the generated CSV file")
    parser.add_argument("--focus_area", type=str, default=None, help="Optional focus area for ground truth generation")
    parser.add_argument("--finetuning_params", type=parse_dict, default="{}", help="Parameters for finetuning as a JSON string")

    args = parser.parse_args()
    main(args)