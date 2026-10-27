from typing import List, Tuple
from .registry import ModelRegistry

class ModelPairSelector:
    """Helper for selecting model pairs for comparison."""
    
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        
    def get_comparison_pairs(
        self,
        category: str,
        compare_to_original: bool = True
    ) -> List[Tuple[str, str]]:
        """Get pairs of models to compare."""
        if compare_to_original:
            return self.registry.get_models_by_category(category)
            
        # Compare each model to others in category
        models = self.registry.get_models_by_category(category)
        return [(m1, m2) for i, m1 in enumerate(models) 
                for m2 in models[i+1:]]

