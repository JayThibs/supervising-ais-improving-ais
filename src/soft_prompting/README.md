# Divergence Prompting

Train soft prompts to find behavioral divergences between language models.

## Installation

```bash
pip install -e .
```

## Usage

Train soft prompts:

```bash
python scripts/train_divergent_prompts.py \
    --model-1 EleutherAI/pythia-70m \
    --model-2 EleutherAI/pythia-70m-deduped \
    --train-file data/train.txt \
    --val-file data/val.txt \
    --output-dir outputs \
    --num-epochs 10
```

This will:
1. Train soft prompts to maximize divergence between the models
2. Generate a dataset of texts with high divergence
3. Save the results and trained prompts to the output directory