# scripts/train_divergence_soft_prompts.py
import argparse
import logging
from pathlib import Path
import yaml
import torch

from src.soft_prompting.config.configs import ExperimentConfig
from src.soft_prompting.training.trainer import DivergenceTrainer
from src.soft_prompting.data.dataloader import create_experiment_dataloaders
from src.soft_prompting.models.model_manager import ModelPairManager
from src.soft_prompting.utils.logging import setup_experiment_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train soft prompts for behavioral divergence"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from config"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    
    if args.output_dir:
        config_dict["output_dir"] = args.output_dir
        
    config = ExperimentConfig.from_dict(config_dict)
    
    # Setup experiment directory and logging
    config.output_dir.mkdir(parents=True, exist_ok=True)
    setup_experiment_logging(config.output_dir / "train.log")
    
    # Save configuration
    with open(config.output_dir / "config.yaml", "w") as f:
        yaml.dump(config_dict, f)
    
    # Load models
    logger.info("Loading models...")
    model_manager = ModelPairManager()
    model_1, model_2, tokenizer = model_manager.load_model_pair(
        config.model_1_name,
        config.model_2_name
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_experiment_dataloaders(
        config=config,
        tokenizer=tokenizer
    )
    
    # Initialize trainer
    trainer = DivergenceTrainer(
        model_1=model_1,
        model_2=model_2,
        tokenizer=tokenizer,
        config=config
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader
    )
    
    # Generate divergent dataset
    logger.info("Generating divergent dataset...")
    trainer.generate_divergent_dataset(
        output_file=config.output_dir / "divergent_dataset.pt"
    )
    
    logger.info(f"Training complete. Results saved to {config.output_dir}")

if __name__ == "__main__":
    main()
