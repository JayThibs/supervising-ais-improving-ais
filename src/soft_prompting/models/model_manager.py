# src/soft_prompting/models/model_manager.py

from typing import Dict, List, Optional, Tuple
import torch
from pathlib import Path
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer
from .registry import ModelRegistry

logger = logging.getLogger(__name__)

class ModelPairManager:
    """Manage model pairs for divergence analysis using HuggingFace models."""
    
    def __init__(
        self,
        registry: ModelRegistry,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        load_in_8bit: bool = False
    ):
        self.registry = registry
        self.device = device
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        
        # Cache for loaded models
        self.model_cache = {}
        
    def load_model_pair(
        self,
        model_name: str,
        use_cache: bool = True
    ) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
        """Load a model and its original/base model for comparison."""
        model_name, base_model_name = self.registry.get_model_pair(model_name)
        
        # Load models (with caching)
        model = self._load_model(model_name, use_cache)
        base_model = self._load_model(base_model_name, use_cache)
        
        # Use tokenizer from base model
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return model, base_model, tokenizer
    
    def _load_model(
        self,
        model_name: str,
        use_cache: bool = True
    ) -> PreTrainedModel:
        """Load a HuggingFace model with caching."""
        if use_cache and model_name in self.model_cache:
            return self.model_cache[model_name]
            
        try:
            # Load model with appropriate settings
            load_kwargs = {
                "device_map": self.device,
                "torch_dtype": self.torch_dtype
            }
            
            if self.load_in_8bit:
                load_kwargs.update({
                    "load_in_8bit": True,
                    "device_map": "auto"
                })
                
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )
            
            # Set to evaluation mode
            model.eval()
            
            if use_cache:
                self.model_cache[model_name] = model
                
            return model
            
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    def unload_model(self, model_name: str):
        """Unload model from cache to free memory."""
        if model_name in self.model_cache:
            del self.model_cache[model_name]
            torch.cuda.empty_cache()
            
    def clear_cache(self):
        """Clear entire model cache."""
        self.model_cache.clear()
        torch.cuda.empty_cache()
