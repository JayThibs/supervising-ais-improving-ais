# Report Cards Implementation

This document provides a comprehensive overview of the Report Cards implementation added to the behavioral clustering pipeline. The implementation is based on the paper ["Report Cards: Qualitative Evaluation of Language Models Using Natural Language Summaries"](https://arxiv.org/pdf/2409.00844v1).

## Table of Contents

1. [Overview](#overview)
2. [PRESS Algorithm](#press-algorithm)
3. [Implementation Details](#implementation-details)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Integration with Existing Pipeline](#integration-with-existing-pipeline)
7. [Benefits](#benefits)
8. [Future Work](#future-work)

## Overview

Report Cards provide a qualitative evaluation approach for language models using natural language summaries. Unlike traditional quantitative metrics, Report Cards offer interpretable, comprehensive descriptions of model capabilities, strengths, and weaknesses in natural language. This implementation adds Report Cards as a complementary evaluation method to the existing behavioral clustering pipeline.

## PRESS Algorithm

The core of the Report Cards approach is the PRESS (Progressive Refinement for Effective Skill Summarization) algorithm, which:

1. Takes a progression set of examples and processes them in smaller batches
2. Generates temporary cards for each batch
3. Merges cards intelligently based on difference thresholds
4. Produces comprehensive summaries that capture model capabilities in natural language

The algorithm follows these steps:

```
Input: Student model M being evaluated
Dataset DM = {(q, aM)i}ⁿᵢ₌₁ containing question-completion pairs
Evaluator E (an LLM, Claude 3.5 Sonnet in our implementation)
Initial criteria S⁰ to guide evaluation
Quiz length k (batch size of examples)
Threshold t for determining merge vs. concatenation

Iteration Process (repeated E times):
  At each iteration j:
    Sample a k-shot quiz QⱼM = {(q, aM)i}ᵏᵢ₌₁ (a subset of k examples)
    Generate temporary card Stmp based on quiz QⱼM
    Compare Stmp with previous card Sⱼ⁻¹
    If difference |Stmp⊕Sⱼ⁻¹| > t:
      MERGE: Sⱼ ← E(Stmp, Sⱼ⁻¹) 
    Otherwise:
      CONCATENATE: Sⱼ ← Stmp ⊕ Sⱼ⁻¹
```

## Implementation Details

The implementation consists of several key components:

1. **ReportCardGenerator Class**: The main class responsible for generating Report Cards using the PRESS algorithm.

2. **Model Integration**: Support for different model families (Anthropic, OpenAI, etc.) with appropriate API handling.

3. **Output Formats**: Multiple output formats including Markdown, HTML, and JSON.

4. **Command-line Interface**: A dedicated script for generating Report Cards from the command line.

5. **Configuration**: Integration with the existing configuration system.

## Configuration

The Report Cards functionality can be configured through the `config.yaml` file or via command-line arguments. The default parameters (based on the paper) are:

```yaml
report_cards_settings:
  progression_set_size: 40      # Total examples to use
  progression_batch_size: 8     # Examples per iteration
  iterations: 5                 # Number of PRESS iterations
  word_limit: 768               # Maximum words per Report Card
  max_subtopics: 12             # Maximum subtopics to include
  merge_threshold: 0.3          # Threshold for merging vs. concatenating
  evaluator_model_family: anthropic
  evaluator_model_name: claude-3-5-sonnet-20240620
```

## Usage

### Command-line Interface

The Report Cards functionality can be used directly from the command line:

```bash
python -m src.behavioural_clustering.compare_models_report_cards \
  --model1 anthropic/claude-3-5-sonnet-20240620 \
  --model2 openai/gpt-4o \
  --dataset anthropic-model-written-evals \
  --n-statements 40 \
  --output-dir data/results/report_cards \
  --report-format all
```

### Configuration File

Alternatively, you can define a configuration in `config.yaml`:

```yaml
report_cards:
  name: report_cards
  model_settings:
    models:
      - [anthropic, claude-3-5-sonnet-20240620]
      - [openai, gpt-4o]
    temperature: 0.7
    generate_responses_max_tokens: 1000
  data_settings:
    datasets:
      - anthropic-model-written-evals
    n_statements: 100
  report_cards_settings:
    progression_set_size: 40
    progression_batch_size: 8
    iterations: 5
    word_limit: 768
    max_subtopics: 12
    merge_threshold: 0.3
    evaluator_model_family: anthropic
    evaluator_model_name: claude-3-5-sonnet-20240620
  run_sections:
    - report_cards
```

And then run:

```bash
python -m src.behavioural_clustering.main --run report_cards
```

## Integration with Existing Pipeline

The Report Cards functionality is integrated with the existing behavioral clustering pipeline:

1. **RunSettings**: Added `ReportCardsSettings` to the `RunSettings` class.
2. **Configuration**: Added Report Cards configuration to `config.yaml`.
3. **Run Sections**: Added "report_cards" as an available run section.
4. **Main Script**: Updated `main.py` to import and use the Report Cards functionality.
5. **Command-line Interface**: Added a dedicated script for Report Cards generation.

## Benefits

The Report Cards approach offers several benefits:

1. **Interpretability**: Natural language summaries are more interpretable than numerical metrics.
2. **Comprehensiveness**: Captures a wide range of model capabilities and behaviors.
3. **Complementary**: Works alongside existing quantitative methods to provide a more complete evaluation.
4. **Accessibility**: Makes model evaluation results more accessible to non-technical stakeholders.
5. **Insight**: Helps identify behavioral patterns that might be missed by purely numerical metrics.

## Future Work

Potential future improvements to the Report Cards implementation:

1. **Interactive Visualization**: Create interactive visualizations of Report Cards.
2. **Comparative Analysis**: Enhance the comparison functionality to highlight key differences between models.
3. **Integration with Contrastive Decoding**: Combine Report Cards with contrastive decoding for deeper insights.
4. **Customizable Templates**: Allow users to define custom templates for Report Cards.
5. **Multi-modal Support**: Extend Report Cards to evaluate multi-modal models.
6. **Uncertainty Quantification**: Add uncertainty estimates to Report Card statements.
7. **Automated Verification**: Implement automated verification of Report Card claims.
