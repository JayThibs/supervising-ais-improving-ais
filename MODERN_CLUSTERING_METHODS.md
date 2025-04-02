# Modern Clustering Methods for Behavioral Analysis

This document explains the modern clustering methods implemented in the behavioral clustering system. These methods leverage Large Language Models (LLMs) to improve clustering quality and interpretability.

## Table of Contents

1. [Introduction](#introduction)
2. [k-LLMmeans Algorithm](#k-llmmeans-algorithm)
3. [SPILL Algorithm](#spill-algorithm)
4. [Usage Guide](#usage-guide)
5. [Implementation Details](#implementation-details)
6. [References](#references)

## Introduction

Traditional clustering algorithms like k-means, spectral clustering, and agglomerative clustering work well for numerical data but often struggle with semantic understanding of text. The implemented LLM-based clustering methods address this limitation by leveraging the semantic understanding capabilities of large language models.

## k-LLMmeans Algorithm

### Overview

k-LLMmeans is a clustering algorithm that uses LLM-generated summaries as centroids for more interpretable clustering. It is based on the paper ["k-LLMmeans: Clustering with Large Language Models"](https://arxiv.org/abs/2502.09667).

### How It Works

1. **Initialization**: Start with random cluster assignments or use k-means++ initialization
2. **Centroid Generation**: For each cluster, generate a summary of the texts in the cluster using an LLM
3. **Embedding Generation**: Convert the summaries to embeddings using the same embedding model
4. **Assignment**: Assign each text to the closest centroid based on embedding similarity
5. **Iteration**: Repeat steps 2-4 until convergence or maximum iterations

### Advantages

- **Interpretable Clusters**: Each cluster has a human-readable summary
- **Semantic Understanding**: Leverages LLM's understanding of text semantics
- **Adaptability**: Works with any embedding model and LLM
- **Fallback Mechanism**: Falls back to standard k-means when needed

## SPILL Algorithm

### Overview

SPILL (Selection and Pooling with LLMs) is a domain-adaptive intent clustering algorithm that uses LLMs to determine intent similarity between texts. It is based on the paper ["SPILL: Domain-Adaptive Intent Clustering based on Selection and Pooling with Large Language Models"](https://arxiv.org/abs/2503.15351).

### How It Works

1. **Seed Selection**: Select initial seed texts using diversity sampling
2. **Intent Description**: Generate intent descriptions for each seed using an LLM
3. **Candidate Selection**: For each intent, select texts with similar intent using LLM-based similarity scoring
4. **Refinement**: Refine the intent descriptions based on selected texts
5. **Iteration**: Repeat steps 3-4 until convergence or maximum iterations

### Advantages

- **Domain Adaptation**: Works well for domain-specific text without fine-tuning
- **Intent-Based Clustering**: Groups texts based on underlying intent rather than surface features
- **Explainable Results**: Provides intent descriptions for each cluster
- **No Training Required**: Uses off-the-shelf LLMs without additional training

## Usage Guide

### Command Line Usage

To use these clustering methods from the command line:

```bash
# Use k-LLMmeans algorithm
python src/behavioural_clustering/main.py --run quick_full_test --clustering-algorithm k-LLMmeans

# Use SPILL algorithm
python src/behavioural_clustering/main.py --run quick_full_test --clustering-algorithm SPILL
```

### Configuration

You can configure the algorithms in the `config.yaml` file:

```yaml
clustering_settings:
  main_clustering_algorithm: k-LLMmeans  # or SPILL
  n_clusters_ratio: 0.04
  min_clusters: 3
  max_clusters: 10
  theme_identification_model_name: "claude-3-5-haiku-20241022"
  theme_identification_model_family: "anthropic"
  theme_identification_system_message: ""
  theme_identification_prompt_type: "theme_identification"
  theme_identification_temperature: 0.5
  theme_identification_max_tokens: 150
  theme_identification_max_total_tokens: 400
```

### Python API Usage

```python
from behavioural_clustering.utils.llm_clustering import KLLMmeansAlgorithm, SPILLAlgorithm
from behavioural_clustering.models.model_factory import initialize_model

# Initialize LLM
llm = initialize_model({
    "model_family": "anthropic",
    "model_name": "claude-3-5-haiku-20241022"
})

# Create k-LLMmeans algorithm
k_llm_means = KLLMmeansAlgorithm(
    n_clusters=5,
    llm=llm,
    max_iterations=5,
    random_state=42
)

# Create SPILL algorithm
spill = SPILLAlgorithm(
    n_clusters=5,
    llm=llm,
    max_iterations=3,
    selection_threshold=0.5,
    random_state=42
)

# Fit the algorithm to embeddings
labels = k_llm_means.fit(embeddings, texts)
# or
labels = spill.fit(embeddings, texts)
```

## Implementation Details

### k-LLMmeans Implementation

The k-LLMmeans algorithm is implemented in the `KLLMmeansAlgorithm` class in `src/behavioural_clustering/utils/llm_clustering.py`. The key components are:

1. **Centroid Summary Generation**: Uses an LLM to generate a summary of texts in each cluster
2. **Embedding Computation**: Converts summaries to embeddings
3. **Assignment**: Assigns texts to the closest centroid
4. **Convergence Check**: Checks if cluster assignments have stabilized

### SPILL Implementation

The SPILL algorithm is implemented in the `SPILLAlgorithm` class in `src/behavioural_clustering/utils/llm_clustering.py`. The key components are:

1. **Seed Selection**: Selects diverse initial seeds
2. **Intent Description Generation**: Uses an LLM to generate intent descriptions
3. **Candidate Selection**: Uses an LLM to score intent similarity
4. **Refinement**: Refines intent descriptions based on selected texts

### Integration with Existing System

Both algorithms are integrated with the existing clustering system through the `update_clustering_factory` function, which adds them to the available clustering algorithms in the `ClusteringFactory`.

## References

1. Jiang, A. Q., Sinha, K., Levy, O., & Tsvetkov, Y. (2024). k-LLMmeans: Clustering with Large Language Models. [arXiv:2502.09667](https://arxiv.org/abs/2502.09667)

2. Ding, N., Jiang, A. Q., Tsvetkov, Y., & Levy, O. (2024). SPILL: Domain-Adaptive Intent Clustering based on Selection and Pooling with Large Language Models. [arXiv:2503.15351](https://arxiv.org/abs/2503.15351)

3. Anthropic. (2024). Claude 3.5 Haiku. [https://www.anthropic.com/news/claude-3-5-haiku](https://www.anthropic.com/news/claude-3-5-haiku)

4. OpenAI. (2023). GPT-4. [https://openai.com/research/gpt-4](https://openai.com/research/gpt-4)
