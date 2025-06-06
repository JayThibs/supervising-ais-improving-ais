# Model Behavior Analysis Pipeline

## Overview

This codebase provides a pipeline for systematically analyzing and understanding behavioral differences between language models. Think of it as a scientific instrument for studying how language models differ in their outputs patterns, aiming for statistical rigor and automated validation.

## Key Features

Our pipeline offers several capabilities:

The system can identify interpretable differences between models through a multi-stage process that combines clustering, hypothesis generation, and statistical validation. It can process models loaded directly from HuggingFace, compare different quantization levels of the same model, or test the pipleine itself by finetuning models with known behavioral differences, then testing if those differences can be recovered.

The pipeline employs multiple validation strategies to ensure its findings are reliable. It can generate texts that should exemplify hypothesized differences and verify if they actually show different probabilities under the two models.

## How It Works

The pipeline operates through several key stages:

### 1. Model Acquisition

The system supports three main ways of obtaining models to compare:

1. Direct loading of models from HuggingFace, allowing comparison between different model versions or entirely different models
2. Self-comparison of a single model at different quantization levels (4-bit, 8-bit, or bfloat16) to understand the impact of model compression
3. Automatic generation of comparison models through finetuning, where we deliberately insert specific behavioral differences to validate the pipeline's detection capabilities

### 2. Text Generation and Analysis

Once we have our models, the pipeline:

1. Generates large samples of texts from both models
2. Creates embeddings to capture the semantic characteristics of these texts
3. Performs clustering to identify patterns in how each model generates text
4. Matches clusters between models to understand where behaviors align or differ

### 3. Hypothesis Generation and Validation

The pipeline then:

1. Uses large language models to generate human-readable descriptions of discovered differences
2. Validates these hypotheses through both generative and discriminative testing
3. Provides statistical measures and p-values to quantify confidence in findings
4. Generates comprehensive reports of validated differences

## Usage

See the file `run_auto_finetuning_main.sh` for example experimental configurations and commands to run them.

## Configuration

The pipeline supports extensive configuration options for:

- Model loading and quantization
- Embedding generation
- Clustering parameters
- Validation settings
- API integration
- Output formatting

## Key Components

### auto_finetuning_main.py
The main entry point and orchestrator for the pipeline.

### auto_finetuning_data.py
Handles data generation and processing, including ground truth generation for validation.

### auto_finetuning_train.py
Manages model finetuning for creating controlled comparison cases.

### auto_finetuning_compare_to_truth.py
Implements comparison and validation logic.

### auto_finetuning_helpers.py
Provides utility functions and helper methods used throughout the pipeline.

### auto_finetuning_interp.py
Contains core interpretability methods for analyzing model differences.

### validated_comparison_tools.py
Implements validated comparison methods and statistical testing.

### run_auto_finetuning_main.sh
Stores configuration parameters for various experiments.

## Known Issues and Limitations

The current implementation has several areas that could be improved:

1. Memory Management: Large models and datasets can strain memory resources. Future versions will implement better streaming and cleanup.

2. API Integration: Error handling for API calls could be more robust, particularly for rate limiting and network issues.

3. Statistical Methodology: Some statistical assumptions, particularly around normal distributions and multiple comparisons, need fixing.

4. Logging standardization: Lots of outputs that should be systematically logged are instead just printed to the terminal. This needs clean up.

5. Lack of unit testing: This is currently unimplemented.

6. Limited saving of intermediate results: The pipeline saves generated texts and embeddings, but other intermediate results will be lost on interruption.

7. Poor optimization of API calls. Most calls happen sequentially one at a time. These should be done in parallel.

## Future Development

Planned improvements include:

1. Enhanced model management system
2. Comprehensive input validation
4. Better checkpointing mechanisms
5. Improved logging system
6. More sophisticated statistical methodology
7. Integration / unit testing
8. Better cleanup mechanisms
9. Enhanced progress tracking
10. Improve diversity of generated labels
11. Implement seed texts based approach to aligning topics between models, as alternative to clustering

For more detailed information on each script's usage, refer to their individual docstrings and command-line argument descriptions.

## Requirements

- Python 3.7+
- pandas
- transformers
- datasets
- torch
- nltk
- scipy
- statsmodels
- sklearn
- numpy
- networkx
- peft
- adam_mini
- structlog
- terminaltables
- tenacity
- matplotlib
- anthropic (for Anthropic API)
- openai (for OpenAI API)
- google.generativeai (for Gemini API)
