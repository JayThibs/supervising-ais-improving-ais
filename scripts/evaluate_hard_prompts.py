import argparse
import torch
from pathlib import Path
from src.interventions.intervention_models.model_manager import InterventionModelManager
from src.soft_prompting.evaluation import evaluate_hard_prompts
from src.soft_prompting.config import TrainingConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated hard prompts")
    parser.add_argument("--config-path", type=str, required=True, help="Path to the evaluation config YAML file")
    parser.add_argument("--run-name", type=str, default="default", help="Name of the evaluation run configuration to use")
    parser.add_argument("--prompts-file", type=str, required=True, help="Path to the file containing generated hard prompts")
    parser.add_argument("--output-dir", type=str, default="evaluation_results", help="Output directory for evaluation results")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load models using InterventionModelManager
    model_manager = InterventionModelManager(args.config_path, args.run_name)
    model_pairs = model_manager.get_model_pairs()
    
    # Load generated hard prompts
    prompts = torch.load(args.prompts_file)
    
    # Create config
    config = TrainingConfig()
    
    # Evaluate hard prompts for each model pair
    for intervened_model_name, original_model_name in model_pairs:
        print(f"Evaluating hard prompts for {intervened_model_name} vs {original_model_name}")
        
        model_1, tokenizer = model_manager.get_model(intervened_model_name)
        model_2, _ = model_manager.get_model(original_model_name)
        
        results = evaluate_hard_prompts(
            prompts=[p["generation"] for p in prompts],
            model_1=model_1,
            model_2=model_2,
            tokenizer=tokenizer,
            config=config
        )
        
        # Save results
        output_file = Path(args.output_dir) / f"evaluation_results_{intervened_model_name.replace('/', '_')}.pt"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(results, output_file)
        
        # Unload models to free up memory
        model_manager.unload_model(intervened_model_name)
        model_manager.unload_model(original_model_name)

if __name__ == "__main__":
    main()
