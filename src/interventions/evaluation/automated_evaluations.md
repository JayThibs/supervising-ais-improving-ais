# Automated Evaluations for Intervened Language Models

## Overview

This document outlines the automated evaluation process for comparing intervened language models with their original counterparts. The goal is to create an AI-driven system that can identify and analyze behavioral differences, potential issues, and unintended consequences of model interventions.

## Key Components

1. **InterventionModelManager**: Manages the loading and unloading of language models.
2. **AutomatedEvaluator**: Orchestrates the evaluation process, including data analysis, prompt generation, and iterative refinement.
3. **AI Assistant**: A language model (e.g., GPT-4) that analyzes outputs and generates hypotheses and test prompts.

## How to Use

### 1. Configure the Evaluation

Edit the `src/interventions/intervention_models/model_config.yaml` file to specify:

- Models to be evaluated
- Datasets to use for evaluation

Example configuration:

```yaml
models:
- name: "GraySwanAI/Llama-3-8B-Instruct-RR"
  original: "meta-llama/Meta-Llama-3-8B-Instruct"
- name: "GraySwanAI/Mistral-7B-Instruct-RR"
  original: "mistralai/Mistral-7B-Instruct-v0.3"
# ... other models ...
evaluation:
  models_to_evaluate:
    - "GraySwanAI/Llama-3-8B-Instruct-RR"
    - "GraySwanAI/Mistral-7B-Instruct-RR"
  datasets:
  - "anthropic-model-written-evals"
```

### 2. Run the Evaluation

To run the automated evaluation:

1. Ensure you have the required dependencies installed.
2. Set up your OpenAI API key as an environment variable
3. Run the evaluation script:

    ```bash
    python src/interventions/evaluation/automated_evaluator.py --run <run_name>
    ```

### 3. Interpret Results

The evaluation process will:

- Generate a final report for each evaluated model pair
- Save evaluation data in JSON Lines format
- Update the run metadata

Results can be found in:

- Final reports: `evaluation_report_{intervened_model}_vs_{original_model}.txt`
- Evaluation data: `data/saved_data/eval_jsonls/{run_id}.jsonl`
- Run metadata: `data/metadata/run_metadata.yaml`

## Data Structure

The evaluation process uses a dataset in JSONL format with the following structure:

```json
{
"model_name": str,
"prompt": str,
"response": str,
"embedding": List[float],
"cluster_theme": str,
"token_probs": List[float]
}
```

## Evaluation Process

The automated evaluation follows an iterative loop:

1. **Initial Data Loading**: Load the dataset containing prompts, responses, and other relevant information.
2. **Clustering**: Group responses based on their embeddings to identify themes and patterns.
3. **Analysis**: The AI assistant analyzes the outputs, considering:
   - Differences between intervened and original model responses
   - KL divergence of token probabilities
   - Cluster themes
   - Predefined focus areas (e.g., backdoors, unintended side effects, ethical concerns)
4. **Hypothesis Generation**: Based on the analysis, the AI assistant generates hypotheses about behavioral differences between the models.
5. **Test Prompt Generation**: The AI assistant creates new prompts designed to test the generated hypotheses.
6. **Model Evaluation**: Run the new prompts through both the intervened and original models, collecting responses and computing KL divergence.
7. **Iteration**: Repeat steps 2-6, refining focus areas and hypotheses with each iteration.

## Key Features

- **Adaptive Focus**: The system dynamically updates its focus areas based on ongoing analysis, allowing it to hone in on the most relevant behavioral differences.
- **KL Divergence Analysis**: Utilizes KL divergence of token probabilities to quantify differences in model outputs.
- **Clustering**: Groups similar responses to identify themes and patterns across the dataset.
- **AI-Driven Analysis**: Leverages a language model to perform in-depth analysis and generate targeted test prompts.

## Future Enhancements

1. **Tool Integration**: Implement capabilities for the AI assistant to use techniques like soft prompting, steering vectors, or SAE feature analysis.
2. **Automated Red-Teaming**: Develop more sophisticated strategies for identifying potential vulnerabilities or unintended behaviors.
3. **Multi-Model Comparison**: Extend the system to compare multiple intervened models simultaneously.
4. **Visualization**: Integrate tools for visualizing behavioral differences and evaluation results.
5. **Intervention-Specific Heuristics**: Develop tailored evaluation strategies for different types of interventions (e.g., knowledge editing, quantization, anti-sycophancy training).

## Goals

The primary goals of this automated evaluation system are:

- Efficiently identify behavioral differences between intervened and original models.
- Uncover potential issues, such as backdoors or unintended side effects of interventions.
- Provide researchers with actionable insights to improve intervention techniques.
- Create a taxonomy of model differences to better understand the impact of various interventions.

By iteratively refining its focus and generating targeted prompts, this system aims to provide a thorough and nuanced understanding of how interventions affect language model behavior.

## Code Structure and Key Methods

### AutomatedEvaluator

The `AutomatedEvaluator` class is the core of the evaluation process. Key methods include:

- `load_dataset`: Loads the initial dataset from a JSONL file.
- `compute_kl_divergence`: Calculates the KL divergence between two probability distributions.
- `cluster_responses`: Groups responses using K-means clustering.
- `analyze_outputs`: Uses an AI assistant to analyze model outputs and generate insights.
- `generate_test_prompts`: Creates new test prompts based on the previous analysis.
- `evaluate_models`: Runs both the intervened and original models on a set of prompts.
- `run_evaluation_loop`: Orchestrates the entire evaluation process through multiple iterations.

### InterventionModelManager

The `InterventionModelManager` class handles the loading and management of language models. Key methods include:

- `load_config`: Loads the model configuration from a YAML file.
- `load_model_and_tokenizer`: Loads a specific model and its tokenizer.
- `load_models`: Loads multiple models specified in the configuration.
- `get_model`: Retrieves a loaded model or loads it if not already in memory.
- `get_model_pairs`: Returns pairs of intervened and original model names.
- `unload_model` and `unload_all_models`: Manage memory by unloading models when not in use.

These classes work together to provide a flexible and extensible framework for evaluating intervened language models.
