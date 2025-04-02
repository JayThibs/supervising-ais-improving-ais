"""
Test script for the Report Cards command-line interface.
"""

import os
import sys
import logging
from pathlib import Path

project_root = os.path.abspath(os.path.dirname(__file__))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from behavioural_clustering.config.run_settings import RunSettings, ReportCardsSettings
from behavioural_clustering.evaluation.report_cards import ReportCardGenerator
from behavioural_clustering.evaluation.model_evaluation_manager import ModelEvaluationManager
from behavioural_clustering.utils.data_preparation import DataPreparation

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_report_cards_cli():
    """Test the Report Cards command-line interface."""
    logger.info("Testing Report Cards CLI...")
    
    run_settings = RunSettings(name="test_report_cards")
    run_settings.report_cards_settings = ReportCardsSettings()
    
    run_settings.model_settings.models = [
        ("anthropic", "claude-3-5-sonnet-20240620"),
        ("openai", "gpt-4o")
    ]
    
    run_settings.data_settings.datasets = ["anthropic-model-written-evals"]
    run_settings.data_settings.n_statements = 5  # Use a small number for testing
    
    report_card_generator = ReportCardGenerator(run_settings=run_settings)
    
    report_card_generator.set_press_parameters(
        progression_set_size=10,
        progression_batch_size=2,
        iterations=2,
        word_limit=500,
        max_subtopics=8,
        merge_threshold=0.25
    )
    
    report_card_generator.set_evaluator_model(
        evaluator_model_family="anthropic",
        evaluator_model_name="claude-3-5-sonnet-20240620"
    )
    
    data_prep = DataPreparation()
    model_evaluation_manager = ModelEvaluationManager(run_settings, run_settings.model_settings.models)
    
    logger.info("Loading and preprocessing data...")
    statements = data_prep.load_and_preprocess_data(run_settings.data_settings)
    
    if not statements:
        logger.error("No statements loaded from datasets")
        return False
        
    logger.info(f"Loaded {len(statements)} statements")
    
    logger.info("Generating Report Cards...")
    
    try:
        comparison_results = report_card_generator.compare_models(
            model_evaluation_manager=model_evaluation_manager,
            model1_family=run_settings.model_settings.models[0][0],
            model1_name=run_settings.model_settings.models[0][1],
            model2_family=run_settings.model_settings.models[1][0],
            model2_name=run_settings.model_settings.models[1][1],
            statements=statements,
            report_progress=True
        )
        
        logger.info("Report Cards generated successfully!")
        logger.info(f"Model 1: {comparison_results['model1']['model']}")
        logger.info(f"Model 2: {comparison_results['model2']['model']}")
        logger.info("Comparison Summary:")
        logger.info(comparison_results['comparison_summary'][:500] + "...")  # Show first 500 chars
        
        return True
    except Exception as e:
        logger.error(f"Error generating Report Cards: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_report_cards_cli()
    sys.exit(0 if success else 1)
