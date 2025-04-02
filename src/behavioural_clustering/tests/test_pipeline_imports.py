"""
Test script to verify the imports and functionality of the pipeline components.
"""

import os
import sys
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test importing the pipeline components."""
    logger.info("Testing imports...")
    
    from behavioural_clustering.models.huggingface_models import HuggingfaceModelAdapter
    from behavioural_clustering.models.model_factory import initialize_model
    
    from behavioural_clustering.evaluation.model_difference_analyzer import ModelDifferenceAnalyzer
    
    from behavioural_clustering.integration.intervention_integration import InterventionIntegration
    
    logger.info("All imports successful!")
    return True

def test_model_factory():
    """Test the model factory with Huggingface support."""
    logger.info("Testing model factory...")
    
    from behavioural_clustering.models.model_factory import initialize_model
    
    model_info = {
        "model_family": "huggingface",
        "model_name": "meta-llama/Llama-2-7b-chat-hf",
        "system_message": "You are a helpful assistant."
    }
    
    try:
        logger.info("Checking model factory returns correct type for Huggingface model...")
        from behavioural_clustering.models.huggingface_models import HuggingfaceModelAdapter
        
        logger.info(f"HuggingfaceModelAdapter class exists: {HuggingfaceModelAdapter.__name__}")
        logger.info("Model factory test successful!")
        return True
    except Exception as e:
        logger.error(f"Error testing model factory: {str(e)}")
        return False

def test_command_line_interface():
    """Test the command line interface script."""
    logger.info("Testing command line interface...")
    
    script_path = os.path.join(project_root, "src", "behavioural_clustering", "compare_models.py")
    if not os.path.exists(script_path):
        logger.error(f"Command line interface script not found: {script_path}")
        return False
    
    logger.info(f"Command line interface script exists: {script_path}")
    logger.info("Command line interface test successful!")
    return True

def test_intervention_integration():
    """Test the intervention integration."""
    logger.info("Testing intervention integration...")
    
    script_path = os.path.join(project_root, "src", "behavioural_clustering", "integration", "run_intervention_comparison.py")
    if not os.path.exists(script_path):
        logger.error(f"Intervention integration script not found: {script_path}")
        return False
    
    logger.info(f"Intervention integration script exists: {script_path}")
    logger.info("Intervention integration test successful!")
    return True

def main():
    """Run all tests."""
    tests = [
        test_imports,
        test_model_factory,
        test_command_line_interface,
        test_intervention_integration
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
