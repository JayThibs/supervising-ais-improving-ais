"""
Command-line script for generating Report Cards for models using the PRESS algorithm.

Example usage:
  python compare_models_report_cards.py --model1 anthropic/claude-3-5-sonnet-20240620 --model2 openai/gpt-4o
  python compare_models_report_cards.py --config config/report_cards_config.yaml
  python compare_models_report_cards.py --model1 huggingface/meta-llama/Llama-2-7b-chat-hf --model2 huggingface/LLM-LAT/llama2-7b-chat-lat-unlearn-harry-potter-normal
"""

import argparse
import os
import sys
import yaml
import logging
import json
from pathlib import Path
from termcolor import colored
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from .config.run_configuration_manager import RunConfigurationManager
from .evaluation.report_cards import ReportCardGenerator
from .evaluation.model_evaluation_manager import ModelEvaluationManager
from .utils.data_preparation import DataPreparation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_model_string(model_string):
    """Parse a model string of the form family/name."""
    if '/' in model_string:
        family, name = model_string.split('/', 1)
        return family, name
    else:
        return "local", model_string

def get_args():
    parser = argparse.ArgumentParser(
        description="Generate Report Cards for models using the PRESS algorithm"
    )
    
    parser.add_argument(
        "--model1",
        type=str,
        help="First model to compare (format: family/name)"
    )
    parser.add_argument(
        "--model2",
        type=str,
        help="Second model to compare (format: family/name)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["anthropic-model-written-evals"],
        help="Datasets to use for comparison"
    )
    parser.add_argument(
        "--n-statements",
        type=int,
        default=40,
        help="Number of statements to use from datasets"
    )
    
    parser.add_argument(
        "--progression-set-size",
        type=int,
        default=40,
        help="Size of the progression set for PRESS algorithm"
    )
    parser.add_argument(
        "--progression-batch-size",
        type=int,
        default=8,
        help="Batch size for PRESS algorithm"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of iterations for PRESS algorithm"
    )
    parser.add_argument(
        "--word-limit",
        type=int,
        default=768,
        help="Word limit for Report Cards"
    )
    parser.add_argument(
        "--max-subtopics",
        type=int,
        default=12,
        help="Maximum number of subtopics for Report Cards"
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.3,
        help="Threshold for merging Report Cards"
    )
    
    parser.add_argument(
        "--evaluator-model-family",
        type=str,
        default="anthropic",
        help="Model family for the evaluator model"
    )
    parser.add_argument(
        "--evaluator-model-name",
        type=str,
        default="claude-3-5-sonnet-20240620",
        help="Model name for the evaluator model"
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
        "--run",
        type=str,
        default="report_cards",
        help="Run configuration to use from config.yaml"
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
    run_settings = run_config_manager.get_configuration(args.run)
    
    if not run_settings:
        logger.error(colored(f"Run configuration '{args.run}' not found", "red"))
        return 1
    
    if args.model1 and args.model2:
        model1_family, model1_name = parse_model_string(args.model1)
        model2_family, model2_name = parse_model_string(args.model2)
        
        run_settings.model_settings.models = [
            (model1_family, model1_name),
            (model2_family, model2_name)
        ]
        
    if args.dataset:
        run_settings.data_settings.datasets = args.dataset
        
    if args.n_statements:
        run_settings.data_settings.n_statements = args.n_statements
        
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model1_short = run_settings.model_settings.models[0][1].split('/')[-1]
        model2_short = run_settings.model_settings.models[1][1].split('/')[-1]
        output_dir = Path(project_root) / "data" / "results" / f"report_cards_{model1_short}_vs_{model2_short}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    data_prep = DataPreparation()
    model_evaluation_manager = ModelEvaluationManager(run_settings, run_settings.model_settings.models)
    
    report_card_generator = ReportCardGenerator(run_settings=run_settings)
    
    report_card_generator.set_press_parameters(
        progression_set_size=args.progression_set_size,
        progression_batch_size=args.progression_batch_size,
        iterations=args.iterations,
        word_limit=args.word_limit,
        max_subtopics=args.max_subtopics,
        merge_threshold=args.merge_threshold
    )
    
    report_card_generator.set_evaluator_model(
        evaluator_model_family=args.evaluator_model_family,
        evaluator_model_name=args.evaluator_model_name
    )
    
    logger.info(colored("Loading and preprocessing data...", "cyan"))
    statements = data_prep.load_and_preprocess_data(run_settings.data_settings)
    
    if not statements:
        logger.error(colored("No statements loaded from datasets", "red"))
        return 1
        
    logger.info(colored(f"Loaded {len(statements)} statements", "green"))
    
    logger.info(colored("Generating Report Cards...", "cyan"))
    
    comparison_results = report_card_generator.compare_models(
        model_evaluation_manager=model_evaluation_manager,
        model1_family=run_settings.model_settings.models[0][0],
        model1_name=run_settings.model_settings.models[0][1],
        model2_family=run_settings.model_settings.models[1][0],
        model2_name=run_settings.model_settings.models[1][1],
        statements=statements,
        report_progress=True
    )
    
    logger.info(colored("Generating reports...", "cyan"))
    
    result_formats = [args.report_format] if args.report_format != "all" else ["text", "html", "json"]
    
    for format_type in result_formats:
        if format_type == "text":
            text_path = output_dir / "report.md"
            with open(text_path, "w") as f:
                f.write("# Model Comparison Report Cards\n\n")
                f.write(f"## Model 1: {comparison_results['model1']['model']}\n\n")
                f.write(comparison_results['model1']['report_card'])
                f.write("\n\n")
                f.write(f"## Model 2: {comparison_results['model2']['model']}\n\n")
                f.write(comparison_results['model2']['report_card'])
                f.write("\n\n")
                f.write("## Comparison Summary\n\n")
                f.write(comparison_results['comparison_summary'])
            logger.info(f"Text report saved to: {text_path}")
                
        elif format_type == "html":
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Comparison Report Cards</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .summary {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                    .report-card {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                    .model1 {{ background-color: #e6f7ff; }}
                    .model2 {{ background-color: #fff7e6; }}
                    .comparison {{ background-color: #e6ffe6; }}
                </style>
            </head>
            <body>
                <h1>Model Comparison Report Cards</h1>
                
                <h2>Model 1: {comparison_results['model1']['model']}</h2>
                <div class="report-card model1">
                    <pre>{comparison_results['model1']['report_card']}</pre>
                </div>
                
                <h2>Model 2: {comparison_results['model2']['model']}</h2>
                <div class="report-card model2">
                    <pre>{comparison_results['model2']['report_card']}</pre>
                </div>
                
                <h2>Comparison Summary</h2>
                <div class="report-card comparison">
                    <pre>{comparison_results['comparison_summary']}</pre>
                </div>
            </body>
            </html>
            """
            
            html_path = output_dir / "report.html"
            with open(html_path, "w") as f:
                f.write(html_content)
            logger.info(f"HTML report saved to: {html_path}")
                
        elif format_type == "json":
            json_path = output_dir / "report.json"
            with open(json_path, "w") as f:
                json.dump(comparison_results, f, indent=2)
            logger.info(f"JSON report saved to: {json_path}")

    print("\n" + "=" * 80)
    print("MODEL COMPARISON REPORT CARDS")
    print("=" * 80)
    print("\nSummary:")
    print(comparison_results.get("comparison_summary", "No summary available"))
    print("\nDetailed reports saved to:")
    for format_type in result_formats:
        save_path = output_dir / f"report.{format_type}"
        print(f"- {save_path}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
