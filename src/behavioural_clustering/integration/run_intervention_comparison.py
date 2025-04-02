"""
Command-line script for running intervention comparisons.

Example usage:
  python run_intervention_comparison.py --config ../interventions/intervention_models/evaluation_config.yaml
  python run_intervention_comparison.py --config ../interventions/intervention_models/evaluation_config.yaml --run-name default
  python run_intervention_comparison.py --config ../interventions/intervention_models/evaluation_config.yaml --model-pair 0
"""

import argparse
import os
import sys
import logging
from pathlib import Path
import json
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from behavioural_clustering.config.run_configuration_manager import RunConfigurationManager
from behavioural_clustering.integration.intervention_integration import InterventionIntegration

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(
        description="Run intervention comparisons"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to intervention configuration YAML file"
    )
    
    parser.add_argument(
        "--run-name",
        type=str,
        help="Name of the intervention run configuration to use"
    )
    
    parser.add_argument(
        "--model-pair",
        type=int,
        help="Index of the model pair to compare"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        help="Datasets to use for comparison"
    )
    parser.add_argument(
        "--n-statements",
        type=int,
        default=50,
        help="Number of statements to use from datasets"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save results"
    )
    parser.add_argument(
        "--report-format",
        type=str,
        choices=["text", "html", "json", "all"],
        default="all",
        help="Format for the analysis report"
    )
    
    parser.add_argument(
        "--bc-run",
        type=str,
        default="model_difference_analysis",
        help="Behavioral clustering run configuration to use from config.yaml"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def main():
    args = get_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    run_config_manager = RunConfigurationManager()
    run_settings = run_config_manager.get_configuration(args.bc_run)
    
    if not run_settings:
        logger.error(f"Behavioral clustering run configuration '{args.bc_run}' not found")
        return 1
    
    intervention_integration = InterventionIntegration(args.config, run_settings)
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(project_root, "data", "results", f"intervention_comparison_{timestamp}")
    
    if args.run_name:
        logger.info(f"Running batch comparison for run: {args.run_name}")
        results = intervention_integration.run_intervention_batch(
            intervention_run_name=args.run_name,
            output_dir=output_dir
        )
    elif args.model_pair is not None:
        logger.info(f"Running comparison for model pair: {args.model_pair}")
        results = intervention_integration.run_intervention_comparison(
            model_pair_index=args.model_pair,
            dataset_names=args.dataset,
            n_statements=args.n_statements,
            output_dir=output_dir
        )
    else:
        logger.info("Running comparison for all model pairs")
        results = intervention_integration.run_intervention_comparison(
            dataset_names=args.dataset,
            n_statements=args.n_statements,
            output_dir=output_dir
        )
    
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        summary = {}
        for key, value in results.items():
            if isinstance(value, dict) and "analysis" in value:
                summary[key] = {
                    "original_model": value.get("original_model", ""),
                    "intervened_model": value.get("intervened_model", ""),
                    "difference_summary": value.get("analysis", {}).get("difference_summary", "No summary available")
                }
            else:
                summary[key] = value
        
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("INTERVENTION COMPARISON SUMMARY")
    print("=" * 80)
    
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            print(f"Original Model: {value.get('original_model', '')}")
            print(f"Intervened Model: {value.get('intervened_model', '')}")
            print("\nDifference Summary:")
            print(value.get("difference_summary", "No summary available"))
            print("-" * 40)
    
    print(f"\nDetailed results saved to: {output_dir}")
    print(f"Summary saved to: {summary_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
