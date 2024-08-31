import torch
from typing import List, Tuple
from behavioural_clustering.config.run_settings import RunSettings
from behavioural_clustering.utils.model_utils import query_model_on_statements
from behavioural_clustering.models.model_factory import initialize_model

class ModelEvaluationManager:
    def __init__(self, run_settings: RunSettings, llms: List[Tuple[str, str]]):
        self.settings = run_settings
        self.llms = llms
        self.models = {}  # Store initialized models
        self.model_info_list = [{"model_family": family, "model_name": name} for family, name in llms]

    def unload_all_models(self):
        self.models.clear()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # For MPS (Apple Silicon), set the high watermark ratio to 0
        if torch.backends.mps.is_available():
            import os
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("All models unloaded and memory cleared.")

    def get_or_initialize_model(self, model_family, model_name):
        model_key = f"{model_family}_{model_name}"
        if model_key not in self.models:
            model_info = {
                "model_family": model_family,
                "model_name": model_name,
                "system_message": self.settings.prompt_settings.statements_system_message
            }
            try:
                self.models[model_key] = initialize_model(
                    model_info, 
                    temperature=self.settings.model_settings.temperature, 
                    max_tokens=self.settings.model_settings.generate_responses_max_tokens,
                    device="auto"
                )
            except Exception as e:
                print(f"Error initializing model {model_name}: {str(e)}")
                return None
        return self.models[model_key]

    def unload_model(self, model_family, model_name):
        model_key = f"{model_family}_{model_name}"
        if model_key in self.models:
            del self.models[model_key]

    def generate_responses(self, text_subset):
        self.unload_all_models()  # Unload any previously loaded models
        query_results_per_model = []
        for model_family, model_name in self.llms:
            print(f"Generating responses using {model_family} - {model_name}")
            model = self.get_or_initialize_model(model_family, model_name)
            if model is None:
                print(f"Skipping {model_name} due to initialization error")
                continue
            query_results = query_model_on_statements(
                text_subset,
                model_family,
                model_name,
                self.settings.prompt_settings.statements_prompt_template,
                self.settings.prompt_settings.statements_system_message,
                model,
                max_tokens=self.settings.model_settings.generate_responses_max_tokens
            )
            query_results_per_model.append(query_results)
        return query_results_per_model

    def get_model_approval(
        self,
        statement: str,
        prompt_template: str,
        model_family: str,
        model_name: str,
        system_message: str,
        approve_strs: List[str] = ["yes"],
        disapprove_strs: List[str] = ["no"],
    ) -> int:
        prompt = prompt_template.format(statement=statement)
        model = self.get_or_initialize_model(model_family, model_name)
        max_tokens = self.settings.model_settings.get_model_approval_max_tokens
        if model_family == "local":
            r = model.generate(prompt=prompt, max_tokens=max_tokens).lower()
        else:
            r = model.generate(prompt=prompt, max_tokens=max_tokens).lower()

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