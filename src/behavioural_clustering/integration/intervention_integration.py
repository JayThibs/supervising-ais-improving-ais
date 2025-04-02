"""
Integration module for connecting the behavioral clustering pipeline with the interventions system.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from behavioural_clustering.config.run_settings import RunSettings
from behavioural_clustering.evaluation.model_difference_analyzer import ModelDifferenceAnalyzer
from behavioural_clustering.evaluation.model_evaluation_manager import ModelEvaluationManager
from behavioural_clustering.utils.data_preparation import DataPreparation

logger = logging.getLogger(__name__)

class InterventionIntegration:
    """
    Integration class for connecting the behavioral clustering pipeline with the interventions system.
    """
    
    def __init__(self, intervention_config_path: str, run_settings: RunSettings):
        """
        Initialize the integration with intervention configuration and run settings.
        
        Args:
            intervention_config_path: Path to the intervention configuration file
            run_settings: Run settings for the behavioral clustering pipeline
        """
        self.intervention_config_path = intervention_config_path
        self.run_settings = run_settings
        self.intervention_config = self._load_intervention_config()
        
    def _load_intervention_config(self) -> Dict[str, Any]:
        """
        Load the intervention configuration from a YAML file.
        
        Returns:
            Dict containing the intervention configuration
        """
        try:
            with open(self.intervention_config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading intervention configuration: {str(e)}")
            return {}
            
    def get_model_pairs(self) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """
        Get pairs of original and intervened models for comparison.
        
        Returns:
            List of model pairs, where each pair is ((family1, name1), (family2, name2))
        """
        model_pairs = []
        
        try:
            for model_info in self.intervention_config.get('models', []):
                intervened_model = model_info.get('name', '')
                
                original_model = model_info.get('original', '')
                
                if intervened_model and original_model:
                    intervened_family = self._determine_model_family(intervened_model)
                    original_family = self._determine_model_family(original_model)
                    
                    model_pairs.append(
                        ((intervened_family, intervened_model), (original_family, original_model))
                    )
                    
            return model_pairs
        except Exception as e:
            logger.error(f"Error getting model pairs: {str(e)}")
            return []
            
    def _determine_model_family(self, model_name: str) -> str:
        """
        Determine the model family based on the model name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model family (e.g., "huggingface", "openai", "anthropic")
        """
        model_name_lower = model_name.lower()
        
        if '/' in model_name:
            return "huggingface"
        elif any(name in model_name_lower for name in ["gpt", "davinci", "curie", "babbage", "ada"]):
            return "openai"
        elif any(name in model_name_lower for name in ["claude", "anthropic"]):
            return "anthropic"
        else:
            return "local"
            
    def run_intervention_comparison(
        self,
        model_pair_index: Optional[int] = None,
        dataset_names: Optional[List[str]] = None,
        n_statements: int = 50,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a comparison between an original model and its intervened version.
        
        Args:
            model_pair_index: Index of the model pair to compare (if None, compare all pairs)
            dataset_names: Names of datasets to use for comparison
            n_statements: Number of statements to use from datasets
            output_dir: Directory to save results
            
        Returns:
            Dict containing comparison results
        """
        model_pairs = self.get_model_pairs()
        
        if not model_pairs:
            logger.error("No model pairs found for comparison")
            return {"error": "No model pairs found for comparison"}
            
        if model_pair_index is not None:
            if model_pair_index < 0 or model_pair_index >= len(model_pairs):
                logger.error(f"Invalid model pair index: {model_pair_index}")
                return {"error": f"Invalid model pair index: {model_pair_index}"}
                
            model_pairs = [model_pairs[model_pair_index]]
            
        if dataset_names:
            self.run_settings.data_settings.datasets = dataset_names
            
        self.run_settings.data_settings.n_statements = n_statements
        
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            self.run_settings.directory_settings.results_dir = output_path
            
        results = {}
        
        for i, ((intervened_family, intervened_name), (original_family, original_name)) in enumerate(model_pairs):
            logger.info(f"Comparing models: {original_name} vs {intervened_name}")
            
            self.run_settings.model_settings.models = [
                (original_family, original_name),
                (intervened_family, intervened_name)
            ]
            
            data_prep = DataPreparation()
            model_evaluation_manager = ModelEvaluationManager(
                self.run_settings,
                self.run_settings.model_settings.models
            )
            difference_analyzer = ModelDifferenceAnalyzer(self.run_settings)
            
            statements = data_prep.load_and_preprocess_data(self.run_settings.data_settings)
            
            if not statements:
                logger.error("No statements loaded from datasets")
                results[f"pair_{i}"] = {"error": "No statements loaded from datasets"}
                continue
                
            pair_results = difference_analyzer.analyze_model_differences(
                model_evaluation_manager=model_evaluation_manager,
                statements=statements,
                report_progress=True
            )
            
            results[f"pair_{i}"] = {
                "original_model": f"{original_family}/{original_name}",
                "intervened_model": f"{intervened_family}/{intervened_name}",
                "analysis": pair_results
            }
            
        return results
        
    def run_intervention_batch(
        self,
        intervention_run_name: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run a batch of intervention comparisons based on the configuration.
        
        Args:
            intervention_run_name: Name of the intervention run configuration to use
            output_dir: Directory to save results
            
        Returns:
            Dict containing batch results
        """
        if intervention_run_name:
            run_config = next(
                (run for run in self.intervention_config.get('evaluation_runs', {}).values()
                 if run.get('name') == intervention_run_name),
                None
            )
            
            if not run_config:
                logger.error(f"Intervention run configuration not found: {intervention_run_name}")
                return {"error": f"Intervention run configuration not found: {intervention_run_name}"}
        else:
            run_configs = list(self.intervention_config.get('evaluation_runs', {}).values())
            
            if not run_configs:
                logger.error("No intervention run configurations found")
                return {"error": "No intervention run configurations found"}
                
            run_config = run_configs[0]
            
        models_to_evaluate = run_config.get('models_to_evaluate', [])
        
        if not models_to_evaluate:
            logger.error("No models to evaluate in the run configuration")
            return {"error": "No models to evaluate in the run configuration"}
            
        datasets = run_config.get('datasets', self.run_settings.data_settings.datasets)
        
        n_statements = run_config.get('n_statements', self.run_settings.data_settings.n_statements)
        
        if output_dir:
            output_path = Path(output_dir)
        else:
            run_name = run_config.get('name', 'default_run')
            output_path = Path(project_root) / "data" / "results" / f"intervention_{run_name}"
            
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for i, model_info in enumerate(models_to_evaluate):
            model_name = model_info.get('name', '')
            
            if not model_name:
                logger.error(f"Invalid model info at index {i}: missing name")
                continue
                
            original_model = next(
                (m.get('original') for m in self.intervention_config.get('models', [])
                 if m.get('name') == model_name),
                None
            )
            
            if not original_model:
                logger.error(f"Original model not found for: {model_name}")
                continue
                
            intervened_family = self._determine_model_family(model_name)
            original_family = self._determine_model_family(original_model)
            
            logger.info(f"Comparing models: {original_model} vs {model_name}")
            
            self.run_settings.model_settings.models = [
                (original_family, original_model),
                (intervened_family, model_name)
            ]
            
            self.run_settings.data_settings.datasets = datasets
            self.run_settings.data_settings.n_statements = n_statements
            
            model_output_path = output_path / f"model_{i}"
            model_output_path.mkdir(parents=True, exist_ok=True)
            self.run_settings.directory_settings.results_dir = model_output_path
            
            data_prep = DataPreparation()
            model_evaluation_manager = ModelEvaluationManager(
                self.run_settings,
                self.run_settings.model_settings.models
            )
            difference_analyzer = ModelDifferenceAnalyzer(self.run_settings)
            
            statements = data_prep.load_and_preprocess_data(self.run_settings.data_settings)
            
            if not statements:
                logger.error("No statements loaded from datasets")
                results[f"model_{i}"] = {"error": "No statements loaded from datasets"}
                continue
                
            model_results = difference_analyzer.analyze_model_differences(
                model_evaluation_manager=model_evaluation_manager,
                statements=statements,
                report_progress=True
            )
            
            results[f"model_{i}"] = {
                "original_model": f"{original_family}/{original_model}",
                "intervened_model": f"{intervened_family}/{model_name}",
                "analysis": model_results
            }
            
        return results
