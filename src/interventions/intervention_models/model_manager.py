import os
from typing import Dict, List, Tuple, Optional
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class InterventionModelManager:
    """
    A class for managing intervention models and their original counterparts.
    This manager handles loading, unloading, and accessing models and their tokenizers.
    """

    def __init__(self, config_path: str):
        """
        Initialize the InterventionModelManager.

        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config = self.load_config(config_path)
        self.models: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}

    @staticmethod
    def load_config(config_path: str) -> Dict:
        """
        Load the configuration from a YAML file.

        Args:
            config_path: Path to the configuration YAML file.

        Returns:
            A dictionary containing the configuration.
        """
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    def load_model_and_tokenizer(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Load a model and its tokenizer.

        Args:
            model_name: Name of the model to load.

        Returns:
            A tuple containing the loaded model and tokenizer.
        """
        print(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        return model, tokenizer

    def load_models(self, model_names: Optional[List[str]] = None) -> None:
        """
        Load specified models or all models in the configuration.

        This method loads both the intervened models and their original counterparts if they differ.

        Args:
            model_names: Optional list of model names to load. If None, load all models.
        """
        models_to_load = model_names or [model['name'] for model in self.config['models']]
        for model_info in self.config['models']:
            if model_info['name'] in models_to_load:
                # Load the intervened model
                model, tokenizer = self.load_model_and_tokenizer(model_info['name'])
                self.models[model_info['name']] = (model, tokenizer)

                # Load the original model if it's different from the intervened model
                if model_info['original'] != model_info['name']:
                    original_model, original_tokenizer = self.load_model_and_tokenizer(model_info['original'])
                    self.models[model_info['original']] = (original_model, original_tokenizer)

    def get_model(self, model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Get a model and its tokenizer by name, loading it if necessary.

        Args:
            model_name: Name of the model to get.

        Returns:
            A tuple containing the model and tokenizer.
        """
        if model_name not in self.models:
            self.load_models([model_name])
        return self.models[model_name]

    def get_model_pairs(self) -> List[Tuple[str, str]]:
        """
        Get pairs of intervened and original model names.

        Returns:
            A list of tuples, each containing the intervened model name and its original counterpart.
        """
        return [(model['name'], model['original']) for model in self.config['models']]

    def unload_model(self, model_name: str) -> None:
        """
        Unload a model from memory.

        Args:
            model_name: Name of the model to unload.
        """
        if model_name in self.models:
            del self.models[model_name]
            torch.cuda.empty_cache()

    def unload_all_models(self) -> None:
        """
        Unload all models from memory and clear CUDA cache.
        """
        self.models.clear()
        torch.cuda.empty_cache()