import os
from typing import Dict, List, Tuple, Optional
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class InterventionModelManager:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.models: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

    @staticmethod
    def load_config(config_path: str) -> Dict:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def load_model_and_tokenizer(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        return model, tokenizer

    def load_models(self, model_names: Optional[List[str]] = None) -> None:
        models_to_load = model_names or [model['name'] for model in self.config['models']]
        for model_info in self.config['models']:
            if model_info['name'] in models_to_load:
                model, tokenizer = self.load_model_and_tokenizer(model_info['name'])
                self.models[model_info['name']] = (model, tokenizer)

                if model_info['original'] != model_info['name']:
                    original_model, original_tokenizer = self.load_model_and_tokenizer(model_info['original'])
                    self.models[model_info['original']] = (original_model, original_tokenizer)

    def get_model(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        if model_name not in self.models:
            self.load_models([model_name])
        return self.models[model_name]

    def get_model_pairs(self) -> List[Tuple[str, str]]:
        return [(model['name'], model['original']) for model in self.config['models']]

    def unload_model(self, model_name: str) -> None:
        if model_name in self.models:
            del self.models[model_name]
            torch.cuda.empty_cache()

    def unload_all_models(self) -> None:
        self.models.clear()
        torch.cuda.empty_cache()