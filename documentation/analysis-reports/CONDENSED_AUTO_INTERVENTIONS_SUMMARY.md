# Condensed Summary: Auto-Interventions Module

## Overview
The auto-interventions module (`src/interventions/auto_finetune_eval/`) is the most advanced approach in the project, but contains ~10,000+ lines of code across several massive files. This summary distills the key concepts, workflows, and capabilities.

## Core Workflow

### 1. **Ground Truth Generation** 
- Generate "behavioral hypotheses" about what might change after intervention
- Sources: TruthfulQA misconceptions, API-generated hypotheses, or manual specification
- Example: "The model will be more likely to refuse harmful requests"

### 2. **Data Generation**
- Create training data that embodies these hypotheses
- Generate base model samples as control
- Fine-tune model on hypothesis-specific data

### 3. **Behavioral Sampling** 
- Generate 200K-2M texts from both base and intervention models
- Use diverse prefixes or external prompt sources (MWE personas, jailbreak prompts)
- Compute embeddings for all generated texts

### 4. **Clustering & Matching**
- Cluster texts from both models (typically 100-1000 clusters)
- Match clusters between models using embedding similarity
- Identify clusters that differ significantly between models

### 5. **Hypothesis Generation**
- For each cluster pair, generate hypotheses about behavioral differences
- Two types:
  - Single cluster labels: "This cluster contains texts about X"
  - Contrastive labels: "Model A does X while Model B does Y"

### 6. **Statistical Validation**
- **SAFFRON Algorithm**: Controls False Discovery Rate (FDR) in multiple hypothesis testing
- **Discriminative Validation**: Can an AI assistant tell which model generated which text?
- **Generative Validation**: Can we prompt models to exhibit the hypothesized behavior?
- Computes p-values and effect sizes for each hypothesis

## Key Files Summary

### `validated_comparison_tools.py` (3304 lines)
**Core Functions:**
- `get_validated_contrastive_cluster_labels()`: Main orchestrator for hypothesis generation and validation
- `validate_cluster_label_discrimination()`: Tests if a hypothesis actually distinguishes models
- `validated_assistant_discriminative_compare()`: Uses AI assistant as judge for behavioral differences
- `build_contrastive_K_neighbor_similarity_graph()`: Creates cluster relationship graphs
- `saffron_fdr_control()`: Implements SAFFRON algorithm for multiple testing correction

**Key Innovations:**
- Hierarchical hypothesis testing with FDR control
- Multiple validation approaches (discriminative + generative)
- Graph-based cluster analysis
- API-based validation using strong models as judges

### `auto_finetuning_main.py` (513 lines)
**Main Components:**
- `AutoFineTuningEvaluator` class: Orchestrates entire pipeline
- Handles model loading with various quantization levels (4-bit, 8-bit, bfloat16)
- Manages ground truth generation and dataset creation
- Coordinates with interpretability methods

**Key Parameters:**
```python
--num_decoded_texts: 200000  # Major bottleneck
--num_clusters: 100
--num_ground_truths: 5
--api_provider: "gemini"/"anthropic"/"openai"
--use_unitary_comparisons: Detailed 1-vs-1 text comparison
```

### `auto_finetuning_helpers.py` (1500+ lines)
**Utility Functions:**
- `batch_decode_texts()`: Generates texts in batches (main bottleneck)
- `parallel_make_api_requests()`: Handles concurrent API calls
- `load_statements_from_*_repo()`: Loads external evaluation datasets
- `analyze_weight_difference()`: SVD analysis of weight changes
- `plot_comparison_tsne()`: Visualizations

**Performance Issues:**
- Sequential batch generation even with "batch" in name
- No caching between runs
- Generates all texts before analysis

### `auto_finetuning_interp.py` (1000+ lines)
**Interpretability Pipeline:**
- `setup_interpretability_method()`: Configures analysis
- `apply_interpretability_method_1_to_K()`: Main analysis function
- Integrates clustering, embedding, and validation
- Supports multiple clustering algorithms

### Key Algorithms & Techniques

#### SAFFRON (Selective SeqStep+ with Fallback)
- Online FDR control for streaming hypotheses
- Handles dependent hypotheses
- More powerful than Benjamini-Hochberg for this use case
- Allows early stopping when significant differences found

#### Diversification Methods
1. **Label Clustering**: Avoid generating similar hypotheses
2. **Verified Diversity**: Use validated hypotheses to guide search
3. **External Grounding**: Use established benchmarks as seeds

#### Validation Approaches
1. **Discriminative**: "Which model generated this text?"
2. **Generative**: "Generate text exhibiting behavior X"
3. **Contrastive**: "How do models differ on prompt Y?"
4. **Unitary Comparison**: Detailed analysis of specific text pairs

## Current Bottlenecks

### Scale Parameters
- 200,000-2,000,000 decoded texts
- 100-1,000 clusters  
- 40-50 API calls per hypothesis
- Multiple validation rounds

### Time Breakdown
1. **Text Generation** (60-70%): Days of GPU time
2. **Embedding** (15-20%): Hours of computation
3. **Clustering** (10%): Memory intensive
4. **Validation** (10%): Expensive API calls

### Memory Usage
- Stores all texts in memory
- No streaming or chunking
- Embedding cache would help but not implemented

## Opportunities for Innovation

### 1. **Iterative Discovery**
Instead of generating all texts upfront:
- Start with small sample (5K)
- Identify interesting regions
- Iteratively refine

### 2. **Model-Guided Sampling**
- Use model uncertainty to guide generation
- Focus on regions where models diverge most
- Active learning principles

### 3. **Hierarchical Validation**
- Quick filtering with cheap models
- Deep validation only for promising hypotheses
- Early stopping when confidence achieved

### 4. **Mechanistic Integration**
- Use weight differences to guide behavioral sampling
- Connect internal changes to external behavior
- Reduce search space using mechanistic insights

## Integration Points

The module integrates with:
- External datasets (MWE, TruthfulQA, jailbreak prompts)
- Multiple API providers (OpenAI, Anthropic, Google)
- Various clustering algorithms
- Statistical libraries (statsmodels, scipy)

## Key Insights

1. **Rigorous but Inefficient**: Excellent statistical methods bottlenecked by brute force
2. **Modular Design**: Clean separation of concerns enables optimization
3. **Validation Overkill?**: Multiple validation rounds might be redundant
4. **Untapped Potential**: Weight analysis could guide behavioral search

## Summary for Opus

This module represents 2+ years of iteration on finding behavioral differences. It has evolved from simple clustering to rigorous statistical validation, but the core approach—exhaustive generation followed by analysis—hasn't changed. The opportunity is to flip this: use analysis to guide generation, not the other way around.