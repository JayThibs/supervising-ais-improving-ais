from typing import List, Tuple
from behavioural_clustering.config.run_settings import RunSettings
from behavioural_clustering.utils.model_utils import query_model_on_statements
from behavioural_clustering.models.model_factory import initialize_model
from behavioural_clustering.utils.embedding_manager import EmbeddingManager

class ModelEvaluationManager:
    def __init__(self, run_settings: RunSettings, llms: List[Tuple[str, str]], embedding_manager: EmbeddingManager):
        self.settings = run_settings
        self.llms = llms
        self.model_info_list = [{"model_family": family, "model_name": model_name} for family, model_name in llms]
        self.embedding_manager = embedding_manager

    def generate_responses(self, text_subset):
        query_results_per_model = []
        for model_family, model_name in self.llms:
            print(f"Generating responses using {model_family} - {model_name}")
            query_results = query_model_on_statements(
                text_subset,
                model_family,
                model_name,
                self.settings.prompt_settings.statements_prompt_template,
                self.settings.prompt_settings.statements_system_message
            )
            query_results_per_model.append(query_results)
        return query_results_per_model

    def get_model_approval(
        self,
        statement: str,
        prompt_template: str,
        model_family: str,
        model: str,
        system_message: str,
        approve_strs: List[str] = ["yes"],
        disapprove_strs: List[str] = ["no"],
    ) -> int:
        prompt = prompt_template.format(statement=statement)
        
        model_info = {
            "model_family": model_family,
            "model_name": model,
            "system_message": system_message
        }
        model_instance = initialize_model(model_info, temperature=self.settings.model_settings.temperature, max_tokens=5)

        r = model_instance.generate(prompt=prompt).lower()

        approve_strs_in_response = sum([s in r for s in approve_strs])
        disapprove_strs_in_response = sum([s in r for s in disapprove_strs])

        if approve_strs_in_response and not disapprove_strs_in_response:
            return 1
        elif not approve_strs_in_response and disapprove_strs_in_response:
            return 0
        else:
            # Uncertain response:
            return -1

    def get_embeddings(self, statements: List[str]):
        return self.embedding_manager.get_or_create_embeddings(statements, self.settings.embedding_settings)