# Key Algorithms and Insights from the Codebase

## Core Algorithms

### 1. **SAFFRON (Selective SeqStep+ with Fallback)**
**Purpose**: Control False Discovery Rate when testing multiple hypotheses

**How it works**:
```python
# Simplified pseudocode
for each hypothesis in stream:
    if p_value < adaptive_threshold:
        reject hypothesis (found difference)
        update_threshold_more_conservative()
    else:
        update_threshold_less_conservative()
```

**Why it matters**: 
- Handles dependent hypotheses (clusters aren't independent)
- Online algorithm (can stop early when differences found)
- More powerful than traditional Benjamini-Hochberg

### 2. **Contrastive Clustering**
**Purpose**: Find behavioral differences by comparing cluster distributions

**Algorithm**:
1. Generate texts from both models
2. Embed all texts in shared space
3. Cluster combined embeddings
4. Compare cluster membership distributions
5. Focus on clusters with maximum divergence

**Key Insight**: Differences concentrate in specific behavioral regions

### 3. **Weight Difference Analysis (SVD)**
**Purpose**: Understand what changed internally during fine-tuning

**Process**:
```python
diff = weights_finetuned - weights_base
U, S, V = SVD(diff)
# Analyze singular values and vectors
# Large singular values = concentrated changes
# U vectors = input space changes  
# V vectors = output space changes
```

**Discovery**: Fine-tuning often creates low-rank updates focused on specific features

### 4. **Iterative Divergence Discovery**
**Current Approach** (Inefficient):
```
Generate 200K texts → Cluster → Analyze → Validate
```

**Proposed Approach** (Efficient):
```
Generate 5K → Cluster → Identify uncertain regions → 
Sample 2K more → Re-cluster → Repeat until confident
```

## Key Insights from 2+ Years of Development

### 1. **Behavioral Differences are Sparse**
- Most model outputs remain unchanged after intervention
- Differences concentrate in specific domains/contexts
- Random sampling wastes computation on unchanged behaviors

### 2. **Clustering Quality Plateaus Quickly**
- 90% of cluster structure emerges with 10% of data
- Additional samples refine boundaries but don't change main patterns
- Diminishing returns after ~20K samples

### 3. **Validation is the Bottleneck**
- Generating hypotheses is cheap
- Validating them rigorously is expensive
- Need to be selective about what to validate

### 4. **Models Can Guide Their Own Analysis**
- High-entropy outputs indicate uncertainty
- Disagreement between models signals behavioral boundaries
- Token-level divergence identifies exact points of difference

### 5. **External Datasets Add Value**
- TruthfulQA: Tests for alignment preservation
- Persona datasets: Tests for behavioral consistency
- Jailbreak prompts: Tests for safety boundaries

## Breakthrough Opportunities

### 1. **Active Behavioral Sampling**
Instead of blind generation:
- Use model uncertainty to guide sampling
- Focus on decision boundaries
- Exploit clustering structure iteratively

### 2. **Mechanistic-Behavioral Bridge**
Connect internal changes to external behavior:
- Use SVD to identify changed weight subspaces
- Generate texts that activate those specific subspaces
- Validate behavioral hypotheses mechanistically

### 3. **Adversarial Discovery**
Let models find their own differences:
- Train probe to distinguish model outputs
- Use probe's confusion regions for deeper analysis
- Game-theoretic formulation

### 4. **Meta-Learning Behavioral Patterns**
Learn what interventions typically change:
- Build dataset of intervention → behavioral change mappings
- Train model to predict likely changes
- Use predictions to guide search

## Critical Questions for Innovation

1. **Can we predict where behavioral differences will appear before exhaustive search?**

2. **How can we use the models themselves as collaborators in finding differences?**

3. **What's the minimal sufficient sampling for reliable difference detection?**

4. **Can we develop behavioral "fingerprints" that quickly identify model changes?**

5. **How do we balance exploration (finding new differences) with exploitation (validating known differences)?**

## Technical Debt to Address

### 1. **Memory Management**
- Currently loads all texts in memory
- No streaming/chunking support
- Embedding cache not persistent

### 2. **Parallelization**
- Sequential processing despite "batch" names
- No multi-GPU support
- API calls could be better parallelized

### 3. **Code Duplication**
- Same functions implemented in multiple modules
- No shared utility library
- Configuration handling repeated

### 4. **Statistical Methods**
- Mixed approaches (frequentist + Bayesian)
- Inconsistent multiple testing correction
- No power analysis

## Summary

The project has developed sophisticated methods but is held back by brute-force approaches. The key insight is that behavioral differences are sparse and structured—we should exploit this sparsity rather than fight it with exhaustive search. The next breakthrough will come from making the search intelligent rather than comprehensive.