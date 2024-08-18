import logging
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import numpy as np

from behavioural_clustering.config.run_settings import RunSettings
from behavioural_clustering.evaluation.model_evaluation_manager import ModelEvaluationManager
from behavioural_clustering.evaluation.embeddings import embed_texts

logger = logging.getLogger(__name__)

class ApprovalEvaluationManager:
    def __init__(self, run_settings: RunSettings, model_eval_manager: ModelEvaluationManager):
        self.settings = run_settings
        self.model_eval_manager = model_eval_manager

    def load_or_generate_approvals_data(self, approvals_type: str, text_subset: List[str]) -> Dict[str, List[Any]]:
        """
        Load or generate approval data for the given approval type and text subset.
        
        Args:
            approvals_type (str): The type of approval (e.g., 'personas', 'awareness')
            text_subset (List[str]): The subset of texts to evaluate

        Returns:
            Dict[str, List[Any]]: Processed approval results
        """
        try:
            approval_results_per_model = {}
            for model_family, model in self.model_eval_manager.llms:
                model_approvals = []
                for role_description in self.settings.approval_prompts[approvals_type].values():
                    prompt_template = self.settings.prompt_settings.approval_prompt_template.format(
                        role_description=role_description
                    )
                    list_of_approvals_for_statements = self.model_eval_manager.get_model_approvals(
                        statements=text_subset,
                        prompt_template=prompt_template,
                        model_family=model_family,
                        model=model,
                        system_message=self.settings.prompt_settings.approval_system_message
                    )
                    model_approvals.append(list_of_approvals_for_statements)
                
                approval_results_per_model[model] = model_approvals

            return self.process_approval_results(approval_results_per_model, text_subset, approvals_type)
        except Exception as e:
            logger.error(f"Error in load_or_generate_approvals_data: {str(e)}")
            raise

    def process_approval_results(self, approval_results_per_model: Dict[str, List[List[int]]], text_subset: List[str], approvals_type: str) -> List[Dict[str, Any]]:
        """
        Process the raw approval results into a structured format.

        Args:
            approval_results_per_model (Dict[str, List[List[int]]]): Raw approval results
            text_subset (List[str]): The subset of texts evaluated
            approvals_type (str): The type of approval

        Returns:
            List[Dict[str, Any]]: Processed approval results
        """
        try:
            processed_results = []
            embeddings = embed_texts(text_subset, self.settings.embedding_settings)

            for i, statement in enumerate(text_subset):
                approval_dict = {model: [approvals[i] for approvals in model_approvals] 
                                 for model, model_approvals in approval_results_per_model.items()}
                
                processed_results.append({
                    "approvals": approval_dict,
                    "statement": statement,
                    "embedding": embeddings[i]
                })

            return processed_results
        except Exception as e:
            logger.error(f"Error in process_approval_results: {str(e)}")
            raise

    @lru_cache(maxsize=32)
    def analyze_approvals(self, approvals_data: Tuple[Tuple[Any]], approvals_type: str) -> Dict[str, Any]:
        """
        Analyze the approval data and generate statistics.

        Args:
            approvals_data (Tuple[Tuple[Any]]): Approval data (converted to tuple for caching)
            approvals_type (str): The type of approval

        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            approvals_data = list(map(dict, approvals_data))  # Convert back to list of dicts
            analysis_results = {}
            
            for model in self.model_eval_manager.model_info_list:
                model_name = model['model']
                model_approvals = np.array([data['approvals'][model_name] for data in approvals_data])
                
                analysis_results[model_name] = {
                    "total_approvals": np.sum(model_approvals > 0),
                    "total_disapprovals": np.sum(model_approvals == 0),
                    "total_uncertain": np.sum(model_approvals == -1),
                    "approval_rate": np.mean(model_approvals > 0)
                }

            return analysis_results
        except Exception as e:
            logger.error(f"Error in analyze_approvals: {str(e)}")
            raise

    def get_approval_statistics(self, approvals_data: List[Dict[str, Any]], approvals_type: str) -> Dict[str, Any]:
        """
        Get detailed approval statistics for each role and model.

        Args:
            approvals_data (List[Dict[str, Any]]): Processed approval data
            approvals_type (str): The type of approval

        Returns:
            Dict[str, Any]: Approval statistics
        """
        try:
            statistics = {}
            
            for i, role in enumerate(self.settings.approval_prompts[approvals_type].keys()):
                role_stats = {model['model']: {
                    "approvals": sum(1 for data in approvals_data if data['approvals'][model['model']][i] == 1),
                    "disapprovals": sum(1 for data in approvals_data if data['approvals'][model['model']][i] == 0),
                    "uncertain": sum(1 for data in approvals_data if data['approvals'][model['model']][i] == -1)
                } for model in self.model_eval_manager.model_info_list}
                
                statistics[role] = role_stats

            return statistics
        except Exception as e:
            logger.error(f"Error in get_approval_statistics: {str(e)}")
            raise

    def compare_model_approvals(self, approvals_data: List[Dict[str, Any]], approvals_type: str) -> Dict[str, Any]:
        """
        Compare approval patterns between different models.

        Args:
            approvals_data (List[Dict[str, Any]]): Processed approval data
            approvals_type (str): The type of approval

        Returns:
            Dict[str, Any]: Comparison results
        """
        try:
            comparison_results = {}
            models = [model['model'] for model in self.model_eval_manager.model_info_list]

            for i, role in enumerate(self.settings.approval_prompts[approvals_type].keys()):
                role_comparison = {}
                for model1 in models:
                    for model2 in models:
                        if model1 != model2:
                            agreement = sum(
                                data['approvals'][model1][i] == data['approvals'][model2][i]
                                for data in approvals_data
                            )
                            agreement_rate = agreement / len(approvals_data)
                            role_comparison[f"{model1}_vs_{model2}"] = agreement_rate

                comparison_results[role] = role_comparison

            return comparison_results
        except Exception as e:
            logger.error(f"Error in compare_model_approvals: {str(e)}")
            raise

    def identify_controversial_statements(self, approvals_data: List[Dict[str, Any]], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Identify statements where models disagree significantly.

        Args:
            approvals_data (List[Dict[str, Any]]): Processed approval data
            threshold (float): Threshold for considering a statement controversial

        Returns:
            List[Dict[str, Any]]: List of controversial statements with their approval rates
        """
        try:
            controversial_statements = []

            for data in approvals_data:
                approval_rates = [np.mean(approvals) for approvals in data['approvals'].values()]
                if max(approval_rates) - min(approval_rates) > threshold:
                    controversial_statements.append({
                        "statement": data['statement'],
                        "approval_rates": {model: np.mean(approvals) 
                                           for model, approvals in data['approvals'].items()}
                    })

            return controversial_statements
        except Exception as e:
            logger.error(f"Error in identify_controversial_statements: {str(e)}")
            raise

    def get_model_agreement_matrix(self, approvals_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Generate a matrix showing the agreement between different models across all statements.

        Args:
            approvals_data (List[Dict[str, Any]]): Processed approval data

        Returns:
            Dict[str, Dict[str, float]]: Agreement matrix
        """
        try:
            models = [model['model'] for model in self.model_eval_manager.model_info_list]
            agreement_matrix = {model1: {model2: 0 for model2 in models} for model1 in models}

            for data in approvals_data:
                for model1 in models:
                    for model2 in models:
                        if model1 != model2:
                            agreement = np.mean(np.array(data['approvals'][model1]) == np.array(data['approvals'][model2]))
                            agreement_matrix[model1][model2] += agreement

            # Normalize the agreement scores
            n_statements = len(approvals_data)
            for model1 in models:
                for model2 in models:
                    if model1 != model2:
                        agreement_matrix[model1][model2] /= n_statements

            return agreement_matrix
        except Exception as e:
            logger.error(f"Error in get_model_agreement_matrix: {str(e)}")
            raise