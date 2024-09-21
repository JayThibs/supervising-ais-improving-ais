# Automated Finetuning Evaluation for Interpretability Methods

This project provides an automated framework for evaluating the effectiveness of interpretability methods in comparing AI models. It generates ground truth differences between models, creates training data to support those differences, fine-tunes models on the data, and then applies an interpretability method to assess how well it recovers the known differences.

## Project Structure

The project consists of the following main components:

1. `auto_finetuning_data.py`: Generates ground truths and associated datasets using an assistant LLM API, saving them to a CSV file.
2. `auto_finetuning_compare_to_truth.py`: Compares discovered differences with ground truth differences using LLMs and scores the interpretability method's performance.
3. `auto_finetuning_main.py`: Orchestrates the entire process, including data generation, model fine-tuning, applying the interpretability method, and evaluation.
4. `auto_finetuning_helpers.py`: Contains helper functions and dummy implementations for development purposes.

## Key Features

- Works with any causal language model available on HuggingFace
- Supports both open-ended and focused ground truth generation
- Flexible number of ground truths (1-10 recommended)
- Uses either Anthropic or OpenAI APIs for ground truth generation and comparison
- Streams generated data to CSV for efficient handling of large datasets
- Supports fine-tuning models on multiple ground truths

## Usage

To run the automated evaluation process:
```
bash python auto_finetuning_main.py --base_model <huggingface_model_id> --num_samples <num_samples> --num_ground_truths <num_truths> --api_provider <anthropic|openai> --model_str <model_version> --key_path <path_to_api_key> [--focus_area <focus_area>] [--finetuning_params <params>]
```

For more detailed information on each script's usage, refer to their individual docstrings and command-line argument descriptions.

## Requirements

- Python 3.7+
- pandas
- transformers
- anthropic (for Anthropic API)
- openai (for OpenAI API)

## Development

For development purposes, dummy implementations of the `apply_interpretability_method` and `finetune_model` functions are provided in `auto_finetuning_helpers.py`. Replace these with actual implementations when ready for production use.

## Contributing

Contributions to improve the project are welcome. Please ensure that any new code is well-documented with appropriate docstrings and follows the existing code style.

## License

[Insert your chosen license here]