import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.soft_prompting.config import TrainingConfig
from src.soft_prompting.trainer import DivergenceTrainer
from src.soft_prompting.data import create_dataloader

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train soft prompts for finding model divergences"
    )
    parser.add_argument(
        "--model-1",
        type=str,
        required=True,
        help="First model name or path"
    )
    parser.add_argument(
        "--model-2",
        type=str,
        required=True,
        help="Second model name or path"
    )
    parser.add_argument(
        "--train-file",
        type=str,
        required=True,
        help="Training data file (one text per line)"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        help="Validation data file (one text per line)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num-soft-prompt-tokens",
        type=int,
        default=8,
        help="Number of soft prompt tokens"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load models and tokenizer
    model_1 = AutoModelForCausalLM.from_pretrained(args.model_1)
    model_2 = AutoModelForCausalLM.from_pretrained(args.model_2)
    tokenizer = AutoTokenizer.from_pretrained(args.model_1)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load training data
    with open(args.train_file) as f:
        train_texts = [line.strip() for line in f]
        
    val_texts = None
    if args.val_file:
        with open(args.val_file) as f:
            val_texts = [line.strip() for line in f]
    
    # Create config
    config = TrainingConfig(
        num_soft_prompt_tokens=args.num_soft_prompt_tokens,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        save_dir=args.output_dir
    )
    
    # Create dataloaders
    train_dataloader = create_dataloader(train_texts, tokenizer, config)
    val_dataloader = None
    if val_texts:
        val_dataloader = create_dataloader(val_texts, tokenizer, config)
        
    # Initialize trainer
    trainer = DivergenceTrainer(
        model_1=model_1,
        model_2=model_2,
        tokenizer=tokenizer,
        config=config
    )
    
    # Train
    trainer.train(train_dataloader, val_dataloader)
    
    # Generate divergent dataset
    output_file = Path(args.output_dir) / "divergent_dataset.pt"
    trainer.generate_divergent_dataset(
        prompts=train_texts[:100],  # Use subset of training texts as prompts
        output_file=output_file
    )

if __name__ == "__main__":
    main()