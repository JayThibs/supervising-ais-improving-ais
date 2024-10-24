import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.soft_prompting.config import TrainingConfig
from src.soft_prompting.train import DivergenceTrainer
from src.soft_prompting.data import create_dataloader
from src.interventions.intervention_models.model_manager import InterventionModelManager

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train soft prompts for finding model divergences"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to the evaluation config YAML file"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="default",
        help="Name of the evaluation run configuration to use"
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
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load models using InterventionModelManager
    model_manager = InterventionModelManager(args.config_path, args.run_name)
    model_pairs = model_manager.get_model_pairs()
    
    # Load training data
    with open(args.train_file) as f:
        train_texts = [line.strip() for line in f]
        
    val_texts = None
    if args.val_file:
        with open(args.val_file) as f:
            val_texts = [line.strip() for line in f]
    
    # Create config
    config = TrainingConfig(
        save_dir=args.output_dir
    )
    
    # Train soft prompts for each model pair
    for intervened_model_name, original_model_name in model_pairs:
        print(f"Training soft prompts for {intervened_model_name} vs {original_model_name}")
        
        model_1, tokenizer = model_manager.get_model(intervened_model_name)
        model_2, _ = model_manager.get_model(original_model_name)
        
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
        output_file = Path(args.output_dir) / f"divergent_dataset_{intervened_model_name.replace('/', '_')}.pt"
        trainer.generate_divergent_dataset(
            prompts=train_texts[:100],  # Use subset of training texts as prompts
            output_file=output_file
        )
        
        # Unload models to free up memory
        model_manager.unload_model(intervened_model_name)
        model_manager.unload_model(original_model_name)

if __name__ == "__main__":
    main()
