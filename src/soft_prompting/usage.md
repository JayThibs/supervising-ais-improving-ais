# Soft Prompting for Model Behavior Analysis

## Overview

This codebase provides tools for analyzing behavioral differences between language models using trainable soft prompts. The key idea is to train continuous prompt embeddings that maximize the divergence between two models' outputs, helping discover where and how their behaviors differ.

## Core Components

### 1. Model Management
- `ModelPairManager`: Handles loading and managing model pairs
- `ModelRegistry`: Maintains catalog of models and their relationships
- `ModelWrapper`: Provides consistent interface across different models

### 2. Soft Prompts
- `DivergenceSoftPrompt`: Trainable continuous embeddings
- Added to input sequences before model processing
- Optimized to maximize behavioral differences

### 3. Training Pipeline
- `DivergenceTrainer`: Manages soft prompt training
- Uses mixed precision and gradient accumulation
- Includes checkpointing and early stopping
- Tracks metrics via wandb integration

### 4. Analysis Tools
- `DivergenceAnalyzer`: Analyzes patterns in model differences
- `DivergenceMetrics`: Computes various divergence measures
- Clustering and visualization capabilities

## Usage Guide

### 1. Basic Training Flow

```python
# Load configuration
config = ExperimentConfig.from_yaml("configs/experiment.yaml")

# Initialize pipeline
pipeline = DivergencePipeline(config=config)

# Run training
results = pipeline.run()

# Analyze results
analyzer = DivergenceAnalyzer(
    metrics=pipeline.trainer.metrics,
    output_dir=config.output_dir
)
report = analyzer.generate_report(results["dataset"])
```

### 2. Using Command Line Scripts

Train soft prompts:
```bash
python scripts/train_divergence_soft_prompts.py \
    --config configs/experiment.yaml \
    --output-dir outputs/experiment1
```

Generate hard prompts:
```bash
python scripts/generate_hard_prompts.py \
    --config configs/experiment.yaml \
    --checkpoints outputs/experiment1/checkpoints/*.pt \
    --output-dir outputs/hard_prompts
```

Evaluate prompts:
```bash
python scripts/evaluate_hard_prompts.py \
    --config configs/eval.yaml \
    --prompts-file outputs/hard_prompts/prompts.pt \
    --output-dir outputs/evaluation
```

### 3. Configuration

Key configuration sections:

```yaml
# Training settings
training:
  num_soft_prompt_tokens: 8  # Number of trainable tokens
  learning_rate: 1e-4
  batch_size: 4
  num_epochs: 10
  mixed_precision: true

# Generation settings  
generation:
  max_length: 128
  num_generations_per_prompt: 10
  temperature: 0.7
  top_p: 1.0

# Data settings
data:
  categories: ["persona", "ethics"]
  max_texts_per_category: 1000
```

### 4. Analyzing Results

The pipeline produces several outputs:

1. Trained soft prompts (checkpoints)
2. Generated hard prompts (natural language)
3. Analysis reports including:
   - Divergence patterns
   - Token-level disagreements
   - Semantic differences
   - Behavioral clusters

### 5. Extending the Framework

#### Adding New Models
1. Update `model_registry.yaml` with model info:
```yaml
models:
  - name: "new-model-name"
    original: "base-model-name"
    description: "Model description"
```

#### Adding New Metrics
1. Extend `DivergenceMetrics` class:
```python
def compute_new_metric(self, outputs_1, outputs_2):
    # Implement metric computation
    return metric_value
```

#### Custom Data Processing
1. Create new processor in `processors.py`:
```python
class CustomProcessor(BaseDataProcessor):
    def load_texts(self, data_path, **kwargs):
        # Implement data loading logic
        return texts
```

## Best Practices

### 1. Model Selection
- Use models from same family when possible
- Verify tokenizer compatibility
- Consider memory constraints

### 2. Training Tips
- Start with few soft prompt tokens (4-8)
- Use mixed precision for efficiency
- Monitor divergence metrics
- Save checkpoints regularly

### 3. Analysis Guidelines
- Filter for high-divergence examples
- Look for patterns in disagreements
- Consider both token and semantic metrics
- Cluster similar behaviors

### 4. Memory Management
- Use gradient checkpointing for large models
- Enable 8-bit loading if needed
- Clear cache between experiments
- Use appropriate batch sizes

## Troubleshooting

### Common Issues

1. Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use 8-bit model loading
- Reduce number of soft prompt tokens

2. Poor Convergence
- Adjust learning rate
- Increase training epochs
- Try different initialization
- Check data quality

3. Low Divergence
- Increase temperature
- Try different model pairs
- Adjust loss function weights
- Increase soft prompt length

### Debugging Tools

1. Metrics Logging
```python
trainer.log_metrics(metrics, step)
```

2. Generation Inspection
```python
examples = trainer.generate_examples(
    prompts,
    num_examples=5
)
```

3. Attention Analysis
```python
analyzer.visualize_attention_patterns(
    model_outputs,
    save_path="attention.png"
)
```

## Advanced Usage

### 1. Custom Training Loops

```python
trainer = DivergenceTrainer(
    model_1=model_1,
    model_2=model_2,
    config=config
)

for epoch in range(config.num_epochs):
    for batch in dataloader:
        loss = trainer.training_step(batch)
        metrics = trainer.compute_all_metrics(batch)
        trainer.log_metrics(metrics)
```

### 2. Ensemble Generation

```python
generator = HardPromptGenerator(
    model_1=model_1,
    model_2=model_2,
    tokenizer=tokenizer
)

prompts = generator.batch_generate(
    checkpoint_paths=checkpoints,
    input_texts=texts,
    min_divergence=0.1
)
```

### 3. Custom Analysis

```python
analyzer = DivergenceAnalyzer(metrics=metrics)
patterns = analyzer.analyze_divergence_patterns(
    dataset,
    clustering_config={
        "n_clusters": 5,
        "method": "kmeans"
    }
)
```

## Contributing

1. Code Style
- Follow PEP 8
- Add type hints
- Include docstrings
- Write tests

2. Pull Requests
- Create feature branch
- Add tests
- Update documentation
- Follow PR template

3. Testing
```bash
pytest tests/
pytest tests/integration/
coverage run -m pytest
```
