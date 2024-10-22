# Divergence Prompting

A framework for discovering behavioral differences between language models using soft prompts. This tool helps create datasets of "hard prompts" (natural language inputs) that effectively reveal behavioral divergences between models.

## Overview

### Purpose

This project aims to systematically discover inputs that cause different behaviors between two language models. While traditional comparison methods might miss subtle differences, our approach uses trainable soft prompts to actively search for and amplify these divergences.

Key applications include:
- Finding behavioral differences between base and fine-tuned models
- Discovering potential backdoors or unwanted behaviors
- Evaluating the effectiveness of model alignment techniques
- Creating targeted evaluation datasets
- Testing model robustness

### Methodology

1. **Soft Prompt Training**
   - Initialize trainable embedding tokens ("soft prompts")
   - Insert these tokens before input text
   - Train the soft prompts to maximize KL divergence between model outputs
   - The soft prompts learn to "steer" the models toward divergent behaviors

2. **Hard Prompt Generation**
   - Use trained soft prompts to generate text samples
   - Measure divergence metrics for each generation
   - Filter and collect samples with high divergence
   - Create a dataset of natural language prompts that reveal model differences

3. **Behavioral Evaluation Pipeline**
   - Use the generated hard prompts as part of a broader evaluation suite
   - Systematically analyze how models differ in their responses
   - Categorize and understand types of behavioral differences
   - Provide insights for model improvement and alignment

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/divergence-prompting.git
cd divergence-prompting

# Install the package
pip install -e .
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See pyproject.toml for full dependencies

## Usage

### Basic Training

```bash
python scripts/train_divergent_prompts.py \
    --model-1 EleutherAI/pythia-70m \
    --model-2 EleutherAI/pythia-70m-deduped \
    --train-file data/train.txt \
    --val-file data/val.txt \
    --output-dir outputs \
    --num-epochs 10
```

### Configuration Options

```bash
# Key parameters
--num-soft-prompt-tokens  # Number of trainable tokens (default: 8)
--learning-rate          # Learning rate for soft prompt training (default: 1e-4)
--batch-size            # Batch size for training (default: 4)
--num-epochs            # Number of training epochs (default: 10)
```

See `src/divergence_prompting/config.py` for full configuration options.

## Integrating with Behavioral Evaluation Pipeline

### 1. Generate Divergent Dataset
```python
from divergence_prompting import DivergenceTrainer

trainer = DivergenceTrainer(model_1, model_2, tokenizer, config)
dataset = trainer.generate_divergent_dataset(
    prompts=initial_prompts,
    output_file="divergent_dataset.pt"
)
```

### 2. Analyze Generations
```python
# Load generated dataset
dataset = torch.load("divergent_dataset.pt")

# Extract high-divergence prompts
high_divergence_prompts = [
    item["prompt"] for item in dataset 
    if item["metrics"]["kl_divergence"] > threshold
]
```

### 3. Use in Evaluation Pipeline
```python
# Example integration with evaluation pipeline
from your_eval_pipeline import ModelEvaluator

evaluator = ModelEvaluator(
    model_1=model_1,
    model_2=model_2,
    prompts=high_divergence_prompts
)
results = evaluator.run_evaluation()
```

## Implementation Details

### Soft Prompt Architecture
- Trainable embedding tokens
- Optimized via gradient descent
- KL divergence loss between model outputs
- Support for different prompt lengths and positions

### Generation Strategy
- Temperature-controlled sampling
- Multiple generations per prompt
- Filtering based on divergence metrics
- Automatic prompt selection

### Metrics Tracked
- KL divergence between model outputs
- Perplexity for both models
- Generation diversity metrics
- Prompt effectiveness scores

## Contributing

We welcome contributions! Please see `CONTRIBUTING.md` for guidelines.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{divergence_prompting,
  author = {Your Team},
  title = {Divergence Prompting: Discovering Behavioral Differences in Language Models},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/your-org/divergence-prompting}
}
```

## License

MIT License - see LICENSE file for details.