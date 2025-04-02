import pandas as pd
from typing import List, Dict, Union, Protocol, Optional
from behavioural_clustering.models.api_models import AnthropicModel, OpenAIModel
from behavioural_clustering.models.local_models import LocalModel
from behavioural_clustering.models.api_models import OpenRouterModel
from behavioural_clustering.models.model_factory import initialize_model

class ModelWithGenerate(Protocol):
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        ...

ModelType = Union[AnthropicModel, OpenAIModel, LocalModel, OpenRouterModel]
import json
from pathlib import Path

def format_disagreement_data(df: pd.DataFrame) -> List[Dict]:
    """
    Format the disagreement data for analysis.
    """
    formatted_data = []
    for _, row in df.iterrows():
        formatted_row = {
            "statement": row["Statement"],
            "prompt_type": row["Prompt Type"],
            "label": row["Label"],
            "approvals": {
                model: approval for model, approval in row.items() 
                if model not in ["Run", "Statement", "Prompt Type", "Label"]
            }
        }
        formatted_data.append(formatted_row)
    return formatted_data

def chunk_data(data: List[Dict], chunk_size: int = 50) -> List[List[Dict]]:
    """
    Split the data into smaller chunks.
    """
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def analyze_model_label_disagreements(data: List[Dict], model: ModelWithGenerate, prompt_type: str) -> str:
    """
    Analyze disagreements between models and labels using a language model.
    """
    # Load approval prompts
    approval_prompts_path = Path(__file__).resolve().parents[3] / "data" / "prompts" / "approval_prompts.json"
    with open(approval_prompts_path, 'r') as f:
        approval_prompts = json.load(f)

    # Extract unique model names and labels
    model_names = sorted(set(model for item in data for model in item['approvals'].keys()))
    labels = sorted(set(item['label'] for item in data))

    # Split data into chunks
    data_chunks = chunk_data(data)
    analyses = []

    for chunk in data_chunks:
        prompt = f"""
        Analyze the following approval data to identify patterns in disagreements between different AI models and labels for the prompt type: {prompt_type}

        Approval prompts context:
        {json.dumps(approval_prompts, indent=2)}

        Data to analyze:
        {chunk}

        The models being compared are: {', '.join(model_names)}
        The labels being compared are: {', '.join(labels)}

        Focus on the following aspects:
        1. For each (statement, model) pairing:
           - Identify patterns in how different models respond to the same statements.
           - Note any consistent differences between models.
        2. For each (statement, label) pairing:
           - Analyze how different labels affect the models' responses to the same statements.
           - Highlight any labels that consistently lead to different responses across models.
        3. Are there specific types of statements where models or labels tend to disagree more?
        4. Do certain models consistently approve or disapprove of certain types of content under specific labels?
        5. How do the different prompt types (personas, awareness) influence the models' responses?

        Provide a concise summary of your findings, highlighting the most significant patterns and trends. 
        Include at least 3 specific examples for each aspect, mentioning the exact models, labels, and statements involved.
        When discussing prompt types, refer to the specific prompts from the approval_prompts.json context.
        """
        analyses.append(model.generate(prompt, max_tokens=2000))

    # Combine and summarize the analyses
    combined_analysis = "\n\n".join(analyses)
    summary_prompt = f"""
    Summarize the following analyses of model and label disagreements for the prompt type: {prompt_type}

    Approval prompts context:
    {json.dumps(approval_prompts, indent=2)}

    {combined_analysis}

    Provide a comprehensive overall summary of the most significant patterns and trends across all chunks of data. Your summary should:
    1. Identify the most consistent patterns of disagreement between models and how they relate to specific labels.
    2. Highlight any notable differences in how specific models approach certain types of content under different labels.
    3. Discuss any unexpected or counterintuitive findings.
    4. Consider how the different prompt types (personas, awareness) influence the models' responses and disagreements.
    5. Include at least 5 specific examples of statements where models or labels lead to significant disagreements, mentioning the exact models, labels, and statements involved.
    6. Relate your findings to the specific prompts in the approval_prompts.json context when relevant.

    Your summary should be clear, insightful, and actionable for researchers working on AI alignment. 
    Use markdown formatting for better readability, e.g., use bullet points for lists and code blocks for examples.
    """
    return model.generate(summary_prompt, max_tokens=3000)

def analyze_approval_patterns(disagreement_data: pd.DataFrame, selected_prompt_types: List[str]) -> Dict[str, str]:
    """
    Analyze patterns in approval disagreements using a language model for selected prompt types.
    """
    # Initialize the Anthropic model
    model_info = {
        "model_family": "anthropic",
        "model_name": "claude-3-sonnet-20240229",
        "system_message": "You are an AI assistant tasked with analyzing patterns in approval data for AI models."
    }
    model = initialize_model(model_info)

    # Format the disagreement data
    formatted_data = format_disagreement_data(disagreement_data)

    # Analyze disagreements for each selected prompt type
    analyses = {}
    for prompt_type in selected_prompt_types:
        prompt_type_data = [item for item in formatted_data if item['prompt_type'] == prompt_type]
        analyses[prompt_type] = analyze_model_label_disagreements(prompt_type_data, model, prompt_type)

    return analyses

def summarize_findings(analysis_results: Dict[str, str], model: ModelWithGenerate) -> str:
    """
    Summarize the findings from the analysis using a language model.
    """
    prompt = f"""
    Based on the following comprehensive analyses of approval data for different prompt types:

    {analysis_results}

    Please provide a final, overarching summary of the key findings, highlighting:
    1. The most significant patterns in model and label disagreements across all analyzed prompt types
    2. Any notable differences or similarities between different prompt types
    3. Unexpected or counterintuitive results that emerged from the analysis
    4. Potential implications for AI alignment and safety based on these findings
    5. Suggestions for further research or areas that require more in-depth investigation

    Your summary should synthesize insights from all prompt type analyses, providing a clear, insightful, and actionable overview for researchers working on AI alignment. 
    Include specific examples from the analyses to illustrate key points.
    Use markdown formatting for better readability, e.g., use bullet points for lists and code blocks for examples.
    """
    return model.generate(prompt, max_tokens=3000)
