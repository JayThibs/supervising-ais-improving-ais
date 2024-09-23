# Behavioral Evaluation Pipeline

This pipeline is designed to evaluate and compare the behavior of different language models, focusing on their responses to various prompts and their approval patterns for different types of statements.

## Table of Contents

- [Behavioral Evaluation Pipeline](#behavioral-evaluation-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Pipeline Components](#pipeline-components)
  - [Configuration](#configuration)
  - [Output](#output)

## Overview

The Behavioral Evaluation Pipeline is a comprehensive tool for analyzing and comparing the behavior of language models. It includes features such as:

- Generating responses from multiple models to a set of prompts
- Evaluating model approvals for different types of statements
- Clustering and visualizing model behaviors
- Comparing models across various dimensions

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/behavioural-clustering.git
   cd behavioural-clustering
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your API keys in a `.env` file in the project root:

   ```.env
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```

## Usage

To run the pipeline:

```python
from behavioural_clustering.evaluation.evaluator_pipeline import EvaluatorPipeline
from behavioural_clustering.config.run_settings import RunSettings
# Load your run settings
run_settings = RunSettings.from_dict(your_config_dict)
# Initialize the pipeline
pipeline = EvaluatorPipeline(run_settings)
# Run the evaluations
pipeline.run_evaluations()
```

## Pipeline Components

1. **Data Preparation**: Loads and preprocesses the dataset specified in the run settings.

2. **Model Comparison**:

   - Generates responses from all specified models for the given prompts.
   - Embeds the responses and performs clustering.
   - Visualizes the model comparison results.

3. **Prompt Evaluation**:

   - For each prompt type (e.g., personas, awareness):
     - Generates approval data for each model.
     - Performs clustering on the approval data.
     - Visualizes the approval patterns.

4. **Hierarchical Clustering**:

   - Performs hierarchical clustering on the approval data.
   - Creates interactive treemaps for visualization.

## Configuration

The pipeline is highly configurable. Key settings include:

- `model_settings`: Specify which models to evaluate.
- `data_settings`: Configure dataset and sampling parameters.
- `prompt_settings`: Set up prompts for different evaluation types.
- `clustering_settings`: Adjust clustering parameters.
- `plot_settings`: Customize visualization options.

Refer to `src/behavioural_clustering/config/config.yaml` for a full list of configuration options.

## Output

The pipeline generates various outputs:

1. **Plots**: Visualizations of model comparisons, approval patterns, and clustering results.
2. **CSV Files**: Detailed tables of clustering results and model comparisons.
3. **Interactive Treemaps**: HTML files for exploring hierarchical clustering results.
4. **Saved Data**: Pickled data files for reuse in future runs.

All outputs are saved in the directories specified in the run settings.
