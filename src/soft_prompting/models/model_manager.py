# src/soft_prompting/models/model_manager.py

from typing import Dict, List, Optional, Tuple
import torch
from pathlib import Path
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from ..utils.device_utils import get_device

logger = logging.getLogger(__name__)

class ModelPairManager:
    """Manage model pairs for divergence analysis using HuggingFace models."""
    
    def __init__(
        self,
        model_1_name: str,
        model_2_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False,
        use_cache: bool = True
    ):
        self.model_1_name = model_1_name
        self.model_2_name = model_2_name
        self.device = get_device(device)
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        self.use_cache = use_cache
        
        # Initialize model cache
        self._model_cache = {}
        
    def load_model_pair(self):
        """Load both models and tokenizer."""
        try:
            # Check cache first if enabled
            cache_key = f"{self.model_1_name}_{self.model_2_name}"
            if self.use_cache and cache_key in self._model_cache:
                return self._model_cache[cache_key]
            
            # Load models and tokenizer
            model_1 = AutoModelForCausalLM.from_pretrained(
                self.model_1_name,
                device_map=self.device,
                torch_dtype=self.torch_dtype,
                load_in_8bit=self.load_in_8bit
            )
            
            # Load second model
            model_2 = AutoModelForCausalLM.from_pretrained(
                self.model_2_name,
                device_map=self.device,
                torch_dtype=self.torch_dtype,
                load_in_8bit=self.load_in_8bit
            )
            
            # Load tokenizer (using model_1's tokenizer)
            tokenizer = AutoTokenizer.from_pretrained(self.model_1_name)
            
            # Cache results if enabled
            if self.use_cache:
                self._model_cache[cache_key] = (model_1, model_2, tokenizer)
                
            return model_1, model_2, tokenizer
        
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise RuntimeError(f"Failed to load models: {e}")
    
    def clear_cache(self):
        """Clear the model cache."""
        self._model_cache.clear()
