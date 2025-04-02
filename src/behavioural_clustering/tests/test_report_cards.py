"""
Test script to verify the Report Cards implementation.
"""

import os
import sys
import logging
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from behavioural_clustering.config.run_settings import RunSettings, ReportCardsSettings
from behavioural_clustering.evaluation.report_cards import ReportCardGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_report_card_generator():
    """Test the ReportCardGenerator class."""
    logger.info("Testing ReportCardGenerator...")
    
    run_settings = RunSettings(name="test_report_cards")
    run_settings.report_cards_settings = ReportCardsSettings()
    
    report_card_generator = ReportCardGenerator(run_settings=run_settings)
    
    report_card_generator.set_press_parameters(
        progression_set_size=10,
        progression_batch_size=2,
        iterations=2,
        word_limit=500,
        max_subtopics=8,
        merge_threshold=0.25
    )
    
    assert report_card_generator.progression_set_size == 10
    assert report_card_generator.progression_batch_size == 2
    assert report_card_generator.iterations == 2
    assert report_card_generator.word_limit == 500
    assert report_card_generator.max_subtopics == 8
    assert report_card_generator.merge_threshold == 0.25
    
    report_card_generator.set_evaluator_model(
        evaluator_model_family="anthropic",
        evaluator_model_name="claude-3-5-sonnet-20240620"
    )
    
    assert report_card_generator.evaluator_model_family == "anthropic"
    assert report_card_generator.evaluator_model_name == "claude-3-5-sonnet-20240620"
    
    logger.info("ReportCardGenerator test successful!")
    return True

def test_report_cards_config():
    """Test the Report Cards configuration."""
    logger.info("Testing Report Cards configuration...")
    
    run_settings = RunSettings(name="test_report_cards")
    run_settings.report_cards_settings = ReportCardsSettings()
    
    assert hasattr(run_settings, "report_cards_settings"), "run_settings missing report_cards_settings"
    
    run_settings.update_run_sections(["report_cards"])
    assert "report_cards" in run_settings.run_sections, "report_cards not in run_sections"
    
    logger.info("Report Cards configuration test successful!")
    return True

def main():
    """Run all tests."""
    tests = [
        test_report_card_generator,
        test_report_cards_config
    ]
    
    success = True
    for test in tests:
        try:
            if not test():
                success = False
        except Exception as e:
            logger.error(f"Error running test {test.__name__}: {str(e)}")
            success = False
    
    if success:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
