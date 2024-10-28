#!/usr/bin/env python
import argparse
import logging
from pathlib import Path
import yaml
import torch

from src.soft_prompting.core.pipeline import DivergencePipeline
from src.soft_prompting.config.configs import ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate hard prompts using trained soft prompts"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to soft prompt checkpoints to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/hard_prompts",
        help="Output directory for generated prompts"
    )
    parser.add_argument(
        "--min-divergence",
        type=float,
        default=0.1,
        help="Minimum divergence threshold for keeping examples"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
    
    # Update output directory
    config_dict["output_dir"] = args.output_dir
    config = ExperimentConfig.from_dict(config_dict)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize pipeline
    pipeline = DivergencePipeline(config=config)
    
    # Run generation
    try:
        checkpoint_paths = [Path(cp) for cp in args.checkpoints]
        results = pipeline.run(checkpoint_paths=checkpoint_paths)
        
        logger.info("Generation complete!")
        for checkpoint_name, examples in results["hard_prompts"].items():
            logger.info(f"\nResults for {checkpoint_name}:")
            logger.info(f"Generated {len(examples)} hard prompts")
            
    except Exception as e:
        logger.error(f"Error generating hard prompts: {e}")
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    main()
