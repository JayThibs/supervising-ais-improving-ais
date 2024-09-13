# Automated Evaluations for Intervened Language Models

## Overview
This document outlines the automated evaluation process for comparing intervened language models with their original counterparts. The goal is to create an AI-driven system that can identify and analyze behavioral differences, potential issues, and unintended consequences of model interventions.

## Key Components
1. **InterventionModelManager**: Manages the loading and unloading of language models.
2. **AutomatedEvaluator**: Orchestrates the evaluation process, including data analysis, prompt generation, and iterative refinement.
3. **AI Assistant**: A language model (e.g., GPT-4) that analyzes outputs and generates hypotheses and test prompts.

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