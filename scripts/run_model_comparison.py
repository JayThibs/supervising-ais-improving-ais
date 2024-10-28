# scripts/run_model_comparison.py
import argparse
import yaml
from pathlib import Path
import logging
from datetime import datetime

from src.soft_prompting.core.pipeline import DivergencePipeline
from src.soft_prompting.core.experiment import ExperimentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run model comparison experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration file"
    )
    parser.add_argument(
        "--model-pairs",
        nargs="+",
        help="Specific model pairs to compare (model1,model2)",
        default=[]
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Base output directory"
    )
    return parser.parse_args()

def setup_experiment_dir(base_dir: str, experiment_name: str) -> Path:
    """Create and setup experiment directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    
    # Create subdirectories
    for subdir in ["soft_prompts", "results", "logs", "configs"]:
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
        
    return experiment_dir

def run_experiment(
    model1: str,
    model2: str,
    base_config: dict,
    output_dir: Path
):
    """Run a single comparison experiment."""
    # Create experiment config
    experiment_name = f"compare_{Path(model1).stem}_vs_{Path(model2).stem}"
    experiment_dir = setup_experiment_dir(output_dir, experiment_name)
    
    config_dict = {
        "name": experiment_name,
        "description": f"Comparing {model1} against {model2}",
        "output_dir": str(experiment_dir),
        "model_1_name": model1,
        "model_2_name": model2,
        **base_config
    }
    
    # Save configuration
    config_path = experiment_dir / "configs" / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
    
    # Create experiment config and run
    config = ExperimentConfig.from_yaml(config_path)
    pipeline = DivergencePipeline(config)
    
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Model 1: {model1}")
    logger.info(f"Model 2: {model2}")
    
    try:
        results = pipeline.run()
        return {
            "status": "success",
            "output_dir": str(experiment_dir),
            "results": results
        }
    except Exception as e:
        logger.error(f"Error in experiment: {e}")
        return {
            "status": "error",
            "error": str(e),
            "output_dir": str(experiment_dir)
        }
    finally:
        pipeline.cleanup()

def main():
    args = parse_args()
    
    # Load base configuration
    with open(args.config) as f:
        base_config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse model pairs
    model_pairs = []
    for pair_str in args.model_pairs:
        model1, model2 = pair_str.split(',')
        model_pairs.append((model1.strip(), model2.strip()))
    
    # Run experiments
    results = []
    for model1, model2 in model_pairs:
        result = run_experiment(
            model1=model1,
            model2=model2,
            base_config=base_config,
            output_dir=output_dir
        )
        results.append(result)
    
    # Save experiment summary
    summary_path = output_dir / "experiment_summary.yaml"
    with open(summary_path, 'w') as f:
        yaml.dump({
            'timestamp': datetime.now().isoformat(),
            'arguments': vars(args),
            'results': results
        }, f)
    
    logger.info(f"All experiments complete. Summary saved to {summary_path}")

if __name__ == "__main__":
    main()
