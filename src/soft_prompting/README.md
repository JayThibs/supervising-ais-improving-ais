# Soft Prompting for Model Behavior Comparison

A framework for discovering behavioral differences between language models using trainable soft prompts. This tool helps create datasets of "hard prompts" (natural language inputs) that effectively reveal behavioral divergences between models.

## Overview

This project uses soft prompts to systematically discover and analyze behavioral differences between language models. By training continuous prompts to maximize divergence between model outputs, we can:

1. Find subtle behavioral differences between models
2. Generate natural language prompts that reliably expose these differences
3. Create targeted evaluation datasets
4. Quantify behavioral changes from interventions

## Installation

```bash
git clone https://github.com/your-org/soft-prompting.git
cd soft-prompting
pip install -e .
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- wandb (for experiment tracking)

## Quick Start

### Basic Usage

```python
from soft_prompting import DivergenceTrainer, ModelPairManager

# Load models
model_manager = ModelPairManager(device="cuda", load_in_8bit=False)
model_1, model_2, tokenizer = model_manager.load_model_pair(
    "mistralai/Mistral-7B-v0.1",
    "HuggingFaceTB/mistral-7b-unaligned"
)

# Create trainer
trainer = DivergenceTrainer(
    model_1=model_1,
    model_2=model_2,
    tokenizer=tokenizer,
    config=config
)

# Train soft prompts
trainer.train(train_dataloader)

# Generate divergent examples
dataset = trainer.generate_divergent_dataset(
    output_file="divergent_dataset.pt"
)
```

### Command Line Usage

Train soft prompts:
```bash
python scripts/train_divergence_soft_prompts.py \
    --config configs/experiment.yaml \
    --output-dir outputs/experiment_1
```

Run model comparisons:
```bash
python scripts/run_model_comparison.py \
    --config configs/comparison.yaml \
    --categories sandbagging unlearning \
    --output-dir outputs/comparisons
```

Evaluate hard prompts:
```bash
python scripts/evaluate_hard_prompts.py \
    --config configs/eval.yaml \
    --prompts-file outputs/experiment_1/divergent_dataset.pt \
    --output-dir outputs/evaluation
```

## Configuration

Example experiment configuration:
```yaml
name: "sandbagging_detection"
output_dir: "outputs/sandbagging"
model_1_name: "mistralai/Mistral-7B-v0.1"
model_2_name: "HuggingFaceTB/mistral-7b-unaligned"

training:
  num_soft_prompt_tokens: 8
  learning_rate: 1e-4
  batch_size: 4
  num_epochs: 10
  mixed_precision: true
  gradient_checkpointing: false

generation:
  max_length: 128
  num_generations_per_prompt: 10
  temperature: 0.7
  top_p: 1.0

data:
  categories: ["persona", "ethics", "capabilities"]
  max_texts_per_category: 1000
```

## Key Components

### ModelPairManager
Handles loading and caching of model pairs:
```python
manager = ModelPairManager(
    device="cuda",
    torch_dtype=torch.float16,
    load_in_8bit=False  # Enable for large models
)
```

### DivergenceTrainer
Trains soft prompts to maximize behavioral differences:
```python
trainer = DivergenceTrainer(
    model_1=model_1,
    model_2=model_2,
    tokenizer=tokenizer,
    config=config
)
```

### Experiment Tracking
Built-in wandb integration for experiment tracking:
```python
from soft_prompting.tracking import ExperimentTracker

tracker = ExperimentTracker(
    config=config,
    use_wandb=True,
    project_name="soft-prompting"
)
```

## Analysis Tools

### Divergence Analysis
```python
from soft_prompting.analysis import DivergenceAnalyzer

analyzer = DivergenceAnalyzer(metrics=metrics, output_dir=output_dir)
analysis = analyzer.analyze_divergence_patterns(dataset)
```

### Behavioral Clustering
```python
from soft_prompting.analysis import BehavioralClusteringAnalyzer

clustering = BehavioralClusteringAnalyzer(
    config_path=config_path,
    metrics=metrics,
    output_dir=output_dir
)
clusters = clustering.analyze_divergent_behaviors(dataset)
```

## Output Format

The generated divergent dataset contains:
```python
{
    "prompt": str,              # Original input prompt
    "generation_1": str,        # Text from model 1
    "generation_2": str,        # Text from model 2
    "metrics": {
        "kl_divergence": float,
        "token_disagreement_rate": float,
        "vocab_jaccard_similarity": float,
        "disagreement_positions": Dict[str, int]
    }
}
```

## Best Practices

1. **Model Selection**
   - Use same model family for comparison when possible
   - Ensure models have compatible tokenizers
   - Consider memory constraints for large models

2. **Training Tips**
   - Start with small number of soft prompt tokens (4-8)
   - Use mixed precision training for efficiency
   - Monitor divergence metrics during training

3. **Analysis**
   - Filter for high-divergence examples
   - Look for patterns in token disagreements
   - Consider semantic similarity alongside divergence

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

## License

MIT License - see LICENSE file for details.
