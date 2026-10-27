# src/soft_prompting/config/model_utils.py

from typing import List, Dict, Optional
import yaml
from pathlib import Path

class ModelSelector:
    """Helper class for selecting and comparing models."""
    
    def __init__(self, registry_path: Optional[str] = None):
        if registry_path is None:
            registry_path = Path(__file__).parent / "model_registry.yaml"
        
        with open(registry_path) as f:
            self.registry = yaml.safe_load(f)
        
        self.models = {m['name']: m for m in self.registry['models']}
        self.categories = self.registry['categories']
    
    def get_model_info(self, model_name: str) -> dict:
        """Get information about a specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in registry")
        return self.models[model_name]
    
    def get_models_by_category(self, category: str) -> List[str]:
        """Get all models in a specific category."""
        if category not in self.categories:
            raise ValueError(f"Category {category} not found")
        return self.categories[category]
    
    def get_categories(self) -> List[str]:
        """Get all available categories."""
        return list(self.categories.keys())
    
    def get_model_pairs_for_comparison(
        self,
        category: str,
        compare_to_original: bool = True
    ) -> List[tuple[str, str]]:
        """
        Get pairs of models to compare.
        
        Args:
            category: Category of models to compare
            compare_to_original: If True, pair each model with its original base model
                               If False, create pairs within the category
        """
        models = self.get_models_by_category(category)
        pairs = []
        
        if compare_to_original:
            for model in models:
                original = self.models[model]['original']
                pairs.append((model, original))
        else:
            # Compare each model to every other model in category
            for i, model1 in enumerate(models):
                for model2 in models[i+1:]:
                    pairs.append((model1, model2))
        
        return pairs
    
    def create_experiment_config(
        self,
        model1: str,
        model2: str,
        name: Optional[str] = None,
        **kwargs
    ) -> dict:
        """Create an experiment configuration for comparing two models."""
        if name is None:
            name = f"compare_{Path(model1).stem}_vs_{Path(model2).stem}"
            
        model1_info = self.get_model_info(model1)
        model2_info = self.get_model_info(model2)
        
        return {
            "name": name,
            "model_pair": {
                "model_1_name": model1,
                "model_2_name": model2,
                "description": f"Comparing {model1} against {model2}",
                "model_1_temperature": model1_info.get('default_temperature', 0.7),
                "model_1_top_p": model1_info.get('default_top_p', 1.0),
                "model_1_max_new_tokens": model1_info.get('default_max_new_tokens', 100),
                "model_2_temperature": model2_info.get('default_temperature', 0.7),
                "model_2_top_p": model2_info.get('default_top_p', 1.0),
                "model_2_max_new_tokens": model2_info.get('default_max_new_tokens', 100),
            },
            **kwargs
        }

# Example usage:
if __name__ == "__main__":
    selector = ModelSelector()
    
    # Get all sandbagging models
    sandbagging_models = selector.get_models_by_category("sandbagging")
    print("\nSandbagging models:")
    for model in sandbagging_models:
        print(f"- {model}")
    
    # Create experiment pairs comparing to original models
    pairs = selector.get_model_pairs_for_comparison("sandbagging", compare_to_original=True)
    print("\nModel pairs for comparison:")
    for model1, model2 in pairs:
        print(f"- {model1} vs {model2}")
    
    # Create experiment config
    config = selector.create_experiment_config(
        "FelixHofstaetter/mistral-7b-sandbagging-new",
        "mistralai/Mistral-7B-Instruct-v0.3"
    )
    print("\nExample experiment config:")
    print(yaml.dump(config, default_flow_style=False))