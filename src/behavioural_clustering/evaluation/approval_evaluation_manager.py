import logging
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm
from functools import lru_cache
import numpy as np
import json
from pathlib import Path

from behavioural_clustering.config.run_settings import RunSettings
from behavioural_clustering.evaluation.model_evaluation_manager import ModelEvaluationManager
from behavioural_clustering.evaluation.embeddings import embed_texts

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # This will output to console
        # Uncomment the next line to also log to a file
        # logging.FileHandler("my_log_file.log")
    ]
)
logger = logging.getLogger(__name__)

class ApprovalEvaluationManager:
    def __init__(self, run_settings: RunSettings, model_eval_manager: ModelEvaluationManager):
        self.run_settings = run_settings
        self.model_eval_manager = model_eval_manager
        self.model_info_list = model_eval_manager.model_info_list
        self.approval_prompts = self.load_approval_prompts()

    def load_approval_prompts(self):
        prompts_file = Path(self.run_settings.directory_settings.data_dir) / "prompts" / "approval_prompts.json"
        with open(prompts_file, 'r') as f:
            return json.load(f)

    def load_or_generate_approvals_data(self, approvals_type: str, text_subset: List[str]) -> List[Dict[str, Any]]:
        """
        Load or generate approval data for the given approval type and text subset.
        
        Args:
            approvals_type (str): The type of approval (e.g., 'personas', 'awareness')
            text_subset (List[str]): The subset of texts to evaluate

        Returns:
            List[Dict[str, Any]]: Processed approval results
        """
        approvals_statements = []
        print("Starting to process text subset for approval data generation...")
        for item in tqdm(text_subset, desc="Processing texts"):
            statement = self.extract_statement(item)
            
            if not statement:
                print(f"Warning: Unable to extract statement from item: {item}")
                continue
            
            approvals = {}
            for model_family, model in self.model_eval_manager.llms:
                model_approvals = {}
                for role, system_message_template in self.approval_prompts[approvals_type].items():
                    # From data/prompts/approval_prompts.json
                    if approvals_type == "awareness":
                        system_message = system_message_template.replace("<X>", self.run_settings.prompt_settings.awareness_task)
                    else:
                        system_message = system_message_template

                    # Use the configurable prompt template
                    prompt_template = self.run_settings.prompt_settings.approval_prompt_template
                    # Example of what the prompt could look like:
                    # prompt_template = "Given the following statement, would you approve of it? Please answer with either 'yes' or 'no'.\n\nStatement: {statement}\n\nApproval (yes / no):"
                    
                    print(f"Generating approval for model: {model}, role: {role}")
                    model_approvals[role] = self.model_eval_manager.get_model_approval(
                        statement,
                        prompt_template,
                        model_family,
                        model,
                        system_message
                    )
                approvals[model] = model_approvals
            approvals_statements.append({"approvals": approvals, "statement": statement})
        print("Finished processing text subset.")
        
        return approvals_statements

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
                model_name = model['model_name']
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
            
            for i, role in enumerate(self.run_settings.approval_prompts[approvals_type].keys()):
                role_stats = {model['model_name']: {
                    "approvals": sum(1 for data in approvals_data if data['approvals'][model['model_name']][i] == 1),
                    "disapprovals": sum(1 for data in approvals_data if data['approvals'][model['model_name']][i] == 0),
                    "uncertain": sum(1 for data in approvals_data if data['approvals'][model['model_name']][i] == -1)
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
            models = [model['model_name'] for model in self.model_eval_manager.model_info_list]

            for i, role in enumerate(self.run_settings.approval_prompts[approvals_type].keys()):
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
            models = [model['model_name'] for model in self.model_eval_manager.model_info_list]
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

    def extract_statement(self, item):
        if isinstance(item, dict):
            return item.get('statement') or item.get('question') or item.get('text')
        elif isinstance(item, str):
            return item
        else:
            print(f"Warning: Unexpected item type: {type(item)}")
            return None