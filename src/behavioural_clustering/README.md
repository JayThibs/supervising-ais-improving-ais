# Behavioural Clustering

A Python package for unsupervised behavioural evaluation of language models using clustering techniques and iterative analysis.

## Overview

Behavioural Clustering is a tool designed to analyze and compare the behaviour of different language models through various clustering and visualization techniques. It provides insights into model responses, approvals, and thematic patterns across different prompts and datasets. The package now supports both standard evaluation and iterative analysis approaches.

## Features

- Model comparison using t-SNE visualization
- Hierarchical clustering of model responses
- Approval evaluation for different prompt types
- Interactive treemap visualization of clustering results
- Spectral clustering analysis
- Customizable run configurations
- Iterative behavioral analysis
- Progressive prompt generation
- Difference discovery

## Installation

To install the package, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repo/behavioural-clustering.git
cd behavioural-clustering
pip install -r requirements.txt
```

## Usage

### Standard Evaluation

The main entry point for running standard evaluations is the `main.py` file. You can run the evaluation pipeline with different configurations:

```bash
# Run with a specific configuration
python src/behavioural_clustering/main.py --run full_run

# List available sections
python src/behavioural_clustering/main.py --list-sections

# Run specific sections
python src/behavioural_clustering/main.py --run full_run --run-only model_comparison

# Skip certain sections
python src/behavioural_clustering/main.py --run full_run --skip personas_evaluation awareness_evaluation
```

### Iterative Analysis

The new iterative approach progressively discovers behavioral differences between models. To use it, add iterative settings to your configuration:

```yaml
# In config.yaml
iterative_run:
  name: iterative_run
  model_settings:
    models:
      - [anthropic, claude-3-opus-20240229]
      - [anthropic, claude-3-5-sonnet-20240620]
  data_settings:
    datasets:
      - anthropic-model-written-evals
    n_statements: 100
    reuse_data:
      - none
    new_generation: false
  iterative_settings:
    max_iterations: 3
    prompts_per_iteration: 50
    min_difference_threshold: 0.1
```

Then run the iterative analysis:

```bash
# Run iterative analysis
python src/behavioural_clustering/main.py --run iterative_run --run-only iterative_evaluation

# Run with custom settings
python src/behavioural_clustering/main.py --run iterative_run \
  --max-iterations 5 \
  --prompts-per-iteration 100 \
  --min-difference 0.05
```

## Configuration

Run configurations are defined in the `config.yaml` file. Here's an example with both standard and iterative settings:

```yaml
default:
  random_state: 42
  embedding_settings:
    client: openai
    embedding_model: text-embedding-ada-002
    batch_size: 20
    max_retries: 50
    initial_sleep_time: 2
    other_params: {}
  prompt_settings:
    statements_system_message: You are an AI language model.
    statements_prompt_template: "Briefly describe your reaction to the following statement:\n'{statement}'\nReaction:"
    max_desc_length: 250
  clustering_settings:
    main_clustering_algorithm: KMeans
    n_clusters_ratio: 0.04
    min_clusters: 10
    max_clusters: 500
  iterative_settings:  # New iterative analysis settings
    max_iterations: 3  # Maximum number of iterations
    prompts_per_iteration: 50  # New prompts to generate per iteration
    min_difference_threshold: 0.1  # Minimum difference to continue
```

### Available Run Types

1. **Standard Evaluation Runs:**
   - `full_run`: Complete evaluation with all models
   - `quick_full_test`: Quick test of all features
   - `only_model_comparisons`: Focus on model comparisons
   - `local_model_run`: For evaluating local models

2. **Iterative Analysis Runs:**
   - `iterative_run`: Progressive behavioral analysis
   - `deep_iterative`: More iterations and prompts
   - `targeted_iterative`: Focus on specific behaviors

## Key Components

1. **EvaluatorPipeline**: Orchestrates the evaluation process
   - Standard evaluation via `run_evaluations()`
   - Iterative analysis via `run_iterative_evaluation()`

2. **IterativeAnalyzer**: Handles iterative analysis
   - Progressive prompt generation
   - Difference analysis
   - Result tracking

3. **RunConfigurationManager**: Manages configurations
   - Standard run settings
   - Iterative settings
   - Run validation

4. **Clustering**: Implements clustering algorithms
   - K-means
   - Spectral clustering
   - Hierarchical clustering

5. **Visualization**: Handles result visualization
   - t-SNE plots
   - Interactive treemaps
   - Difference heatmaps

6. **ModelEvaluationManager**: Manages model interactions
   - Response generation
   - Model loading/unloading
   - State management

7. **DataPreparation**: Handles data processing
   - Loading datasets
   - Preprocessing
   - Batch management

## Results and Output

### Standard Evaluation
- Cluster analysis results
- t-SNE visualizations
- Behavioral groupings
- Theme analysis
- CSV reports

### Iterative Analysis
- Progressive difference reports
- New targeted prompts
- Iteration summaries
- Comprehensive final report
- Generated prompt bank

## Extending the Package

### Adding New Features
1. Create new clustering algorithms in `clustering.py`
2. Add visualization methods in `visualization.py`
3. Extend iterative analysis in `iterative_analysis.py`

### Custom Components
1. Implement new model providers
2. Add custom prompt generators
3. Create specialized analyzers
4. Add visualization types

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas of interest:

1. New clustering algorithms
2. Enhanced visualization techniques
3. Improved iterative analysis
4. Additional model support
5. Performance optimizations

## License

[Insert your chosen license here]

## Contact

For questions and support, please open an issue on the GitHub repository.
