# Model Behavior Analysis Pipeline

## Overview

This codebase provides a sophisticated pipeline for systematically analyzing and understanding behavioral differences between language models. Think of it as a scientific instrument for studying how language models differ in their outputs, capabilities, and tendencies - similar to how we might compare writing styles between authors, but with statistical rigor and automated validation.

## Key Features

Our pipeline offers several powerful capabilities:

The system can identify interpretable differences between models through a multi-stage process that combines clustering, hypothesis generation, and statistical validation. It can process models loaded directly from HuggingFace, compare different quantization levels of the same model, or work with specially finetuned models with known behavioral differences.

The pipeline employs multiple validation strategies to ensure its findings are reliable. It can generate texts that should exemplify hypothesized differences and verify if they actually show different probabilities under the two models. It can also conduct interactive experiments where an LLM tries to determine which model it's talking to based on hypothesized differences.

Throughout the process, the system maintains efficiency through careful memory management, result caching, and parallel processing where possible.

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

## Installation

[Installation instructions to be added]

## Usage

Here's a basic example of how to use the pipeline:

```python
# Example code showing basic usage
# To be added
```

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

## Known Issues and Limitations

The current implementation has several areas that could be improved:

1. Memory Management: Large models and datasets can strain memory resources. Future versions will implement better streaming and cleanup.

2. API Integration: Error handling for API calls could be more robust, particularly for rate limiting and network issues.

3. Statistical Methodology: Some statistical assumptions, particularly around normal distributions and multiple comparisons, could be more rigorously validated.

4. Configuration Management: The system would benefit from a more centralized configuration system.

## Future Development

Planned improvements include:

1. Enhanced model management system
2. Comprehensive input validation
3. Unified configuration system
4. Better checkpointing mechanisms
5. Improved logging system
6. More sophisticated statistical methodology
7. Integration testing
8. Better cleanup mechanisms
9. Enhanced progress tracking
10. Artifact versioning

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


## Contributing

[Contributing guidelines to be added]

## License

[License information to be added]

## Contact

[Contact information to be added]

Would you like me to elaborate on any section of this README or add additional details about specific components?