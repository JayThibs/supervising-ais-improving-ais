import json
from typing import List, Dict, Any
from anthropic import Anthropic
import openai
from auto_finetuning_helpers import make_api_request, load_api_key

def compare_hypotheses(
    ground_truth: str,
    discovered_hypothesis: str,
    api_provider: str,
    model_str: str,
    api_key: str
) -> float:
    """
    Compare a single ground truth with a discovered hypothesis using the specified API.

    Args:
        ground_truth (str): The original ground truth statement.
        discovered_hypothesis (str): The hypothesis discovered by the interpretability method.
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (str): The API key for the chosen provider.

    Returns:
        float: A similarity score between 1 and 100, where 100 indicates perfect similarity.
    """
    prompt = f"""Compare the following ground truth statement with a discovered hypothesis:

    Ground Truth: "{ground_truth}"
    Discovered Hypothesis: "{discovered_hypothesis}"

    Analyze the similarity between these statements in terms of their meaning and implications.
    Consider factors such as accuracy, completeness, and specificity.
    
    Provide a similarity score from 1 to 100, where 100 indicates perfect similarity and 1 indicates no similarity.
    
    Format your response as a JSON object with a single key 'similarity_score'. Say nothing else.
    
    Example format:
    {{
        "similarity_score": 90
    }}
    """

    response = make_api_request(api_provider, model_str, api_key, prompt)
    # Parse the response to extract only the JSON part
    try:
        json_start = response.index('{')
        json_end = response.rindex('}') + 1
        json_response = response[json_start:json_end]
        value = json.loads(json_response)['similarity_score']
    except (ValueError, json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing response: {e}")
        print(f"Full response: {response}")
        value = 0  # Default value in case of parsing error
    
    return value

def compare_and_score_hypotheses(
    ground_truths: List[str],
    discovered_hypotheses: List[str],
    api_provider: str,
    model_str: str,
    api_key: str
) -> Dict[str, Any]:
    """
    Compare ground truths with discovered hypotheses and calculate overall scores.

    This function compares each ground truth with each discovered hypothesis,
    calculates individual similarity scores, and then computes overall metrics
    to evaluate the effectiveness of the interpretability method.

    Args:
        ground_truths (List[str]): List of original ground truth statements.
        discovered_hypotheses (List[str]): List of hypotheses discovered by the interpretability method.
        api_provider (str): The API provider to use ('anthropic' or 'openai').
        model_str (str): The model version to use.
        api_key (str): The API key for the chosen provider.

    Returns:
        Dict[str, Any]: A dictionary containing evaluation metrics, including:
            - individual_scores: List of similarity scores for each comparison
            - average_score: The mean similarity score across all comparisons
            - max_score: The highest similarity score achieved
            - min_score: The lowest similarity score achieved
            - matched_hypotheses: Number of hypotheses that matched well (similarity > 0.8)
    """

    individual_scores = []
    print(f"Comparing {len(ground_truths)} ground truths to {len(discovered_hypotheses)} hypotheses")
    print(ground_truths)
    print(discovered_hypotheses)
    for ground_truth in ground_truths:
        for hypothesis in discovered_hypotheses:
            score = compare_hypotheses(ground_truth, hypothesis, api_provider, model_str, api_key)
            individual_scores.append(score)
    
    average_score = sum(individual_scores) / len(individual_scores)
    max_score = max(individual_scores)
    min_score = min(individual_scores)
    matched_hypotheses = sum(1 for score in individual_scores if score > 80)
    
    return {
        "individual_scores": individual_scores,
        "average_score": average_score,
        "max_score": max_score,
        "min_score": min_score,
        "matched_hypotheses": matched_hypotheses
    }

if __name__ == "__main__":
    # This section can be used for testing the module
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare and score hypotheses against ground truths")
    parser.add_argument("--ground_truths", type=str, nargs='+', required=True, help="List of ground truth statements")
    parser.add_argument("--discovered_hypotheses", type=str, nargs='+', required=True, help="List of discovered hypotheses")
    parser.add_argument("--api_provider", type=str, choices=["anthropic", "openai"], required=True, help="API provider")
    parser.add_argument("--model_str", type=str, required=True, help="Model version for the chosen API provider")
    parser.add_argument("--key_path", type=str, required=True, help="Path to the API key file")
    
    args = parser.parse_args()

    api_key = load_api_key(args.key_path, args.api_provider)
    results = compare_and_score_hypotheses(
        args.ground_truths,
        args.discovered_hypotheses,
        args.api_provider,
        args.model_str,
        api_key
    )
    
    print(json.dumps(results, indent=2))