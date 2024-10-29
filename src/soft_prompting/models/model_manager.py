# src/soft_prompting/models/model_manager.py

from typing import Dict, List, Optional, Tuple
import torch
from pathlib import Path
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..config.model_utils import ModelSelector

logger = logging.getLogger(__name__)

class ModelPairManager:
    """Unified model pair management with config integration."""
    
    def __init__(
        self,
        config,  # ExperimentConfig
        device: Optional[str] = None,
        test_mode: bool = False
    ):
        self.config = config
        self.device = device or config.device
        self.test_mode = test_mode
        self.model_selector = ModelSelector()
        
        # Initialize model cache
        self._model_cache = {}
        
    def get_model_pair(self, pair_index: int = 0) -> Tuple[str, str]:
        """Get model pair names from config."""
        if hasattr(self.config, 'model_pairs'):
            pair = self.config.model_pairs[pair_index]
            return pair['model_1'], pair['model_2']
        return self.config.model_1_name, self.config.model_2_name

    def load_model_pair(self, pair_index: int = 0):
        """Load model pair specified in config."""
        try:
            # Get model names (either from config or test models if in test mode)
            if self.test_mode:
                model_1_name = "HuggingFaceTB/SmolLM-135M-Instruct"
                model_2_name = "HuggingFaceTB/SmolLM-135M"
            else:
                model_1_name, model_2_name = self.get_model_pair(pair_index)

            # Get model configs from registry
            model_1_info = self.model_selector.get_model_info(model_1_name)
            model_2_info = self.model_selector.get_model_info(model_2_name)

            # Check cache
            cache_key = f"{model_1_name}_{model_2_name}"
            if cache_key in self._model_cache:
                return self._model_cache[cache_key]

            # Load models with their specific configurations
            model_1 = AutoModelForCausalLM.from_pretrained(
                model_1_name,
                device_map=self.device,
                torch_dtype=self.config.torch_dtype,
                load_in_8bit=self.config.load_in_8bit,
                **model_1_info.get('model_kwargs', {})
            )
            
            model_2 = AutoModelForCausalLM.from_pretrained(
                model_2_name,
                device_map=self.device,
                torch_dtype=self.config.torch_dtype,
                load_in_8bit=self.config.load_in_8bit,
                **model_2_info.get('model_kwargs', {})
            )

            # Load tokenizer (using model_1's tokenizer)
            tokenizer = AutoTokenizer.from_pretrained(model_1_name)

            # Cache the results
            self._model_cache[cache_key] = (model_1, model_2, tokenizer)
            return model_1, model_2, tokenizer

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise RuntimeError(f"Failed to load models: {e}")

    def clear_cache(self):
        """Clear the model cache."""
        self._model_cache.clear()
