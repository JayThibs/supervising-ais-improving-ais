"""
Command-line script for comparing behavioral differences between models.

Example usage:
  python compare_models.py --model1 anthropic/claude-3-5-sonnet-20240620 --model2 openai/gpt-4o-2024-05-13
  python compare_models.py --config config/comparison_config.yaml
  python compare_models.py --model1 huggingface/meta-llama/Llama-2-7b-chat-hf --model2 huggingface/LLM-LAT/llama2-7b-chat-lat-unlearn-harry-potter-normal
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

from behavioural_clustering.config.run_configuration_manager import RunConfigurationManager
from behavioural_clustering.evaluation.model_difference_analyzer import ModelDifferenceAnalyzer
from behavioural_clustering.evaluation.model_evaluation_manager import ModelEvaluationManager
from behavioural_clustering.utils.data_preparation import DataPreparation

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
        description="Compare behavioral differences between models"
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
        default=50,
        help="Number of statements to use from datasets"
    )
    
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum number of iterations for the analysis"
    )
    parser.add_argument(
        "--prompts-per-iteration",
        type=int,
        default=20,
        help="Number of prompts to generate per iteration"
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
        default="model_difference_analysis",
        help="Run configuration to use from config.yaml"
    )
    
    parser.add_argument(
        "--analyze-behavior",
        type=str,
        help="Analyze a specific behavior (provide description)"
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
        
    if args.max_iterations:
        run_settings.iterative_settings.max_iterations = args.max_iterations
        
    if args.prompts_per_iteration:
        run_settings.iterative_settings.prompts_per_iteration = args.prompts_per_iteration
        
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        run_settings.directory_settings.results_dir = output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model1_short = run_settings.model_settings.models[0][1].split('/')[-1]
        model2_short = run_settings.model_settings.models[1][1].split('/')[-1]
        output_dir = Path(project_root) / "data" / "results" / f"comparison_{model1_short}_vs_{model2_short}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_settings.directory_settings.results_dir = output_dir
    
    data_prep = DataPreparation()
    model_evaluation_manager = ModelEvaluationManager(run_settings, run_settings.model_settings.models)
    difference_analyzer = ModelDifferenceAnalyzer(run_settings)
    
    logger.info(colored("Loading and preprocessing data...", "cyan"))
    statements = data_prep.load_and_preprocess_data(run_settings.data_settings)
    
    if not statements:
        logger.error(colored("No statements loaded from datasets", "red"))
        return 1
        
    logger.info(colored(f"Loaded {len(statements)} statements", "green"))
    
    logger.info(colored("Running behavioral difference analysis...", "cyan"))
    
    if args.analyze_behavior:
        logger.info(colored(f"Analyzing specific behavior: {args.analyze_behavior}", "cyan"))
        results = difference_analyzer.analyze_specific_behavior(
            model_evaluation_manager=model_evaluation_manager,
            behavior_description=args.analyze_behavior,
            n_test_prompts=args.n_statements,
            report_progress=True
        )
    else:
        results = difference_analyzer.analyze_model_differences(
            model_evaluation_manager=model_evaluation_manager,
            statements=statements,
            report_progress=True
        )
    
    logger.info(colored("Generating analysis report...", "cyan"))
    
    result_formats = [args.report_format] if args.report_format != "all" else ["text", "html", "json"]
    
    for format_type in result_formats:
        save_path = run_settings.directory_settings.results_dir / f"difference_analysis.{format_type}"
        
        if format_type == "json":
            with open(save_path, "w") as f:
                def json_serializer(obj):
                    try:
                        return obj.__dict__
                    except:
                        return str(obj)
                
                json.dump(results, f, indent=2, default=json_serializer)
                
        elif format_type == "html":
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Behavioral Difference Analysis</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .summary {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; }}
                    .difference {{ margin-bottom: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                    .novel {{ background-color: #e6f7ff; }}
                    .absent {{ background-color: #fff7e6; }}
                    .strengthened {{ background-color: #e6ffe6; }}
                    .weakened {{ background-color: #ffe6e6; }}
                </style>
            </head>
            <body>
                <h1>Behavioral Difference Analysis</h1>
                <h2>Models Compared</h2>
                <ul>
                    <li>Model 1: {run_settings.model_settings.models[0][0]}/{run_settings.model_settings.models[0][1]}</li>
                    <li>Model 2: {run_settings.model_settings.models[1][0]}/{run_settings.model_settings.models[1][1]}</li>
                </ul>
                
                <h2>Summary</h2>
                <div class="summary">
                    <pre>{results.get("difference_summary", "No summary available")}</pre>
                </div>
                
                <h2>Detailed Results</h2>
            """
            
            for diff_type, diffs in results.get("differences_by_type", {}).items():
                html_content += f"""
                <h3>{diff_type.capitalize()} Differences ({len(diffs)})</h3>
                """
                
                for i, diff in enumerate(diffs):
                    html_content += f"""
                    <div class="difference {diff_type}">
                        <h4>Difference {i+1}</h4>
                        <p><strong>Description:</strong> {diff.get("description", "No description")}</p>
                        <p><strong>Score:</strong> {diff.get("difference_score", 0):.3f}</p>
                        <p><strong>Validation:</strong> {diff.get("validation_score", 0):.3f}</p>
                        
                        <details>
                            <summary>Supporting Examples</summary>
                            <ul>
                    """
                    
                    for ex in diff.get("supporting_examples", []):
                        if isinstance(ex, list) and len(ex) >= 2:
                            html_content += f"""
                            <li>
                                <strong>Model 1:</strong> <pre>{ex[0][:200] + "..." if len(ex[0]) > 200 else ex[0]}</pre>
                                <strong>Model 2:</strong> <pre>{ex[1][:200] + "..." if len(ex[1]) > 200 else ex[1]}</pre>
                            </li>
                            """
                        
                    html_content += """
                            </ul>
                        </details>
                    </div>
                    """
            
            if "behavior_analysis" in results:
                behavior = results["behavior_analysis"]
                html_content += f"""
                <h2>Specific Behavior Analysis</h2>
                <div class="behavior-analysis">
                    <h3>Behavior: {behavior.get("description", "")}</h3>
                    <h4>Test Prompts</h4>
                    <ul>
                """
                
                for prompt in behavior.get("test_prompts", []):
                    html_content += f"<li>{prompt}</li>\n"
                
                html_content += f"""
                    </ul>
                    <h4>Analysis</h4>
                    <div class="summary">
                        <pre>{behavior.get("summary", "No analysis available")}</pre>
                    </div>
                </div>
                """
            
            html_content += """
            </body>
            </html>
            """
            
            with open(save_path, "w") as f:
                f.write(html_content)
                
        elif format_type == "text":
            with open(save_path, "w") as f:
                f.write("Behavioral Difference Analysis\n")
                f.write("==============================\n\n")
                
                f.write("Models Compared\n")
                f.write("--------------\n")
                f.write(f"Model 1: {run_settings.model_settings.models[0][0]}/{run_settings.model_settings.models[0][1]}\n")
                f.write(f"Model 2: {run_settings.model_settings.models[1][0]}/{run_settings.model_settings.models[1][1]}\n\n")
                
                f.write("Summary\n")
                f.write("-------\n")
                f.write(f"{results.get('difference_summary', 'No summary available')}\n\n")
                
                f.write("Detailed Results\n")
                f.write("----------------\n")
                
                for diff_type, diffs in results.get("differences_by_type", {}).items():
                    f.write(f"\n{diff_type.upper()} DIFFERENCES ({len(diffs)})\n")
                    f.write("=" * (len(diff_type) + 13 + len(str(len(diffs)))) + "\n\n")
                    
                    for i, diff in enumerate(diffs):
                        f.write(f"Difference {i+1}\n")
                        f.write(f"Description: {diff.get('description', 'No description')}\n")
                        f.write(f"Score: {diff.get('difference_score', 0):.3f}\n")
                        f.write(f"Validation: {diff.get('validation_score', 0):.3f}\n")
                        
                        f.write("\nSupporting Examples:\n")
                        for j, ex in enumerate(diff.get("supporting_examples", [])):
                            if isinstance(ex, list) and len(ex) >= 2:
                                f.write(f"\nExample {j+1}:\n")
                                f.write(f"Model 1: {ex[0][:200] + '...' if len(ex[0]) > 200 else ex[0]}\n")
                                f.write(f"Model 2: {ex[1][:200] + '...' if len(ex[1]) > 200 else ex[1]}\n")
                        
                        f.write("\n" + "-" * 40 + "\n")
                
                if "behavior_analysis" in results:
                    behavior = results["behavior_analysis"]
                    f.write("\nSpecific Behavior Analysis\n")
                    f.write("=========================\n\n")
                    f.write(f"Behavior: {behavior.get('description', '')}\n\n")
                    
                    f.write("Test Prompts:\n")
                    for i, prompt in enumerate(behavior.get("test_prompts", [])):
                        f.write(f"{i+1}. {prompt}\n")
                    
                    f.write("\nAnalysis:\n")
                    f.write(f"{behavior.get('summary', 'No analysis available')}\n")
    
    print("\n" + "=" * 80)
    print("BEHAVIORAL DIFFERENCE ANALYSIS SUMMARY")
    print("=" * 80)
    print(results.get("difference_summary", "No summary available"))
    print("\nDetailed results saved to:")
    for format_type in result_formats:
        save_path = run_settings.directory_settings.results_dir / f"difference_analysis.{format_type}"
        print(f"- {save_path}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
