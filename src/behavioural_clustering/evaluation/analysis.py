import pandas as pd
from typing import List, Dict
from behavioural_clustering.models.api_models import AnthropicModel
from behavioural_clustering.models.model_factory import initialize_model

def format_disagreement_data(df: pd.DataFrame) -> List[Dict]:
    """
    Format the disagreement data for analysis.
    """
    formatted_data = []
    for _, row in df.iterrows():
        formatted_row = {
            "statement": row["Statement"],
            "label": row["Label"],
            "approvals": {
                model: approval for model, approval in row.items() 
                if model not in ["Run", "Statement", "Label"]
            }
        }
        formatted_data.append(formatted_row)
    return formatted_data

def chunk_data(data: List[Dict], chunk_size: int = 50) -> List[List[Dict]]:
    """
    Split the data into smaller chunks.
    """
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def analyze_model_disagreements(data: List[Dict], model: AnthropicModel) -> str:
    """
    Analyze disagreements between models using a language model.
    """
    # Extract unique model names
    model_names = set()
    for item in data:
        model_names.update(item['approvals'].keys())
    model_names = sorted(list(model_names))

    # Split data into chunks
    data_chunks = chunk_data(data)
    analyses = []

    for chunk in data_chunks:
        prompt = f"""
        Analyze the following approval data to identify patterns in disagreements between different AI models:

        {chunk}

        The models being compared are: {', '.join(model_names)}

        Focus on the following aspects:
        1. Are there specific types of statements where models tend to disagree?
        2. Do certain models consistently approve or disapprove of certain types of content?
        3. Are there any noticeable patterns in how different models interpret the same statements?

        Provide a concise summary of your findings, highlighting the most significant patterns and trends.
        """
        analyses.append(model.generate(prompt, max_tokens=1000))

    # Combine and summarize the analyses
    combined_analysis = "\n\n".join(analyses)
    summary_prompt = f"""
    Summarize the following analyses of model disagreements:

    {combined_analysis}

    Provide a comprehensive overall summary of the most significant patterns and trends across all chunks of data. Your summary should:
    1. Identify the most consistent patterns of disagreement between models
    2. Highlight any notable differences in how specific models approach certain types of content
    3. Discuss any unexpected or counterintuitive findings
    4. Consider the implications of these disagreements for AI alignment and safety

    Your summary should be clear, insightful, and actionable for researchers working on AI alignment.
    """
    return model.generate(summary_prompt, max_tokens=1500)

def analyze_label_disagreements(data: List[Dict], model: AnthropicModel) -> str:
    """
    Analyze disagreements between different labels (prompts) using a language model.
    """
    # Extract unique labels
    labels = sorted(set(item['label'] for item in data))

    # Split data into chunks
    data_chunks = chunk_data(data)
    analyses = []

    for chunk in data_chunks:
        prompt = f"""
        Analyze the following approval data to identify patterns in disagreements between different labels (prompts):

        {chunk}

        The labels being compared are: {', '.join(labels)}

        Focus on the following aspects:
        1. How do the different labels ({', '.join(labels)}) affect the approval decisions?
        2. Are there specific types of statements where certain labels lead to more disagreements?
        3. Do some labels consistently result in more approvals or disapprovals?

        Provide a concise summary of your findings, highlighting the most significant patterns and trends.
        """
        analyses.append(model.generate(prompt, max_tokens=1000))

    # Combine and summarize the analyses
    combined_analysis = "\n\n".join(analyses)
    summary_prompt = f"""
    Summarize the following analyses of label disagreements:

    {combined_analysis}

    Provide a comprehensive overall summary of the most significant patterns and trends across all chunks of data. Your summary should:
    1. Identify the most consistent patterns of disagreement between different labels
    2. Highlight how specific labels tend to influence approval decisions
    3. Discuss any unexpected or counterintuitive findings related to label effects
    4. Consider the implications of these label-based disagreements for AI alignment and safety

    Your summary should be clear, insightful, and actionable for researchers working on AI alignment.
    """
    return model.generate(summary_prompt, max_tokens=1500)

def analyze_approval_patterns(disagreement_data: pd.DataFrame) -> Dict[str, str]:
    """
    Analyze patterns in approval disagreements using a language model.
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

    # Analyze model disagreements
    model_analysis = analyze_model_disagreements(formatted_data, model)

    # Analyze label disagreements
    label_analysis = analyze_label_disagreements(formatted_data, model)

    return {
        "model_disagreement_analysis": model_analysis,
        "label_disagreement_analysis": label_analysis
    }

def summarize_findings(analysis_results: Dict[str, str], model: AnthropicModel) -> str:
    """
    Summarize the findings from the analysis using a language model.
    """
    prompt = f"""
    Based on the following comprehensive analyses of approval data:

    Model Disagreement Analysis:
    {analysis_results['model_disagreement_analysis']}

    Label Disagreement Analysis:
    {analysis_results['label_disagreement_analysis']}

    Please provide a final, overarching summary of the key findings, highlighting:
    1. The most significant patterns in model disagreements across all data
    2. The most notable trends in label (prompt) disagreements across all data
    3. Any unexpected or counterintuitive results that emerged from the analysis
    4. Potential implications for AI alignment and safety based on these findings
    5. Suggestions for further research or areas that require more in-depth investigation

    Your summary should synthesize insights from both the model and label analyses, providing a clear, insightful, and actionable overview for researchers working on AI alignment.
    """
    return model.generate(prompt, max_tokens=2000)