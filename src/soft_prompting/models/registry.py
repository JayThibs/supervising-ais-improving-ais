from typing import Dict, List, Optional, Tuple
import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelRegistry:
    """Registry for managing model pairs and configurations."""
    
    def __init__(self, registry_path: Optional[str] = None):
        if registry_path is None:
            # Update the path to look in the project's config directory
            registry_path = Path(__file__).parents[2] / "config" / "model_registry.yaml"
        self.registry = self._load_registry(registry_path)
        
    def _load_registry(self, path: str) -> Dict:
        """Load model registry from YAML."""
        with open(path) as f:
            return yaml.safe_load(f)
            
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a specific model."""
        for model in self.registry['models']:
            if model['name'] == model_name:
                return model
        raise ValueError(f"Model {model_name} not found in registry")
        
    def get_model_pair(self, model_name: str) -> Tuple[str, str]:
        """Get model and its original/base model."""
        model = self.get_model_info(model_name)
        return model['name'], model['original']
        
    def get_models_by_category(self, category: str) -> List[Tuple[str, str]]:
        """Get all models in a category with their base models."""
        if category not in self.registry['categories']:
            raise ValueError(f"Category {category} not found")
            
        return [self.get_model_pair(model) 
                for model in self.registry['categories'][category]]
                
    def get_all_categories(self) -> List[str]:
        """Get list of all available categories."""
        return list(self.registry['categories'].keys())

