import argparse
import torch
from pathlib import Path
import logging
import yaml

from src.soft_prompting.models.model_manager import ModelPairManager
from src.soft_prompting.metrics.divergence_metrics import DivergenceMetrics
from src.soft_prompting.analysis.divergence_analyzer import DivergenceAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate generated hard prompts"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to evaluation config file"
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="Path to file containing generated hard prompts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    logger.info("Loading models...")
    model_manager = ModelPairManager(
        device="cuda",
        torch_dtype=torch.float16,
        load_in_8bit=config.get("load_in_8bit", False)
    )
    
    # Load model pairs
    model_pairs = config["model_pairs"]
    
    # Load prompts
    prompts = torch.load(args.prompts_file)
    
    # Evaluate each model pair
    for model_1_name, model_2_name in model_pairs:
        logger.info(f"Evaluating {model_1_name} vs {model_2_name}")
        
        # Load models
        model_1, model_2, tokenizer = model_manager.load_model_pair(
            model_1_name,
            model_2_name
        )
        
        # Setup metrics and analyzer
        metrics = DivergenceMetrics(tokenizer)
        analyzer = DivergenceAnalyzer(
            metrics=metrics,
            output_dir=output_dir / f"{Path(model_1_name).stem}_vs_{Path(model_2_name).stem}"
        )
        
        # Generate responses and analyze
        try:
            results = analyzer.evaluate_prompts(
                prompts=prompts,
                model_1=model_1,
                model_2=model_2
            )
            
            # Generate report
            report = analyzer.generate_report(results)
            
            logger.info(f"Evaluation complete for {model_1_name} vs {model_2_name}")
            
        except Exception as e:
            logger.error(f"Error evaluating {model_1_name} vs {model_2_name}: {e}")
            continue
        
        finally:
            # Clean up to free memory
            model_manager.unload_model(model_1_name)
            model_manager.unload_model(model_2_name)
    
    logger.info(f"All evaluations complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
