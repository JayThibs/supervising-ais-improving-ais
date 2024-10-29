# src/soft_prompting/models/model_manager.py

from typing import Dict, List, Optional, Tuple
import torch
from pathlib import Path
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..config.model_utils import ModelSelector
import yaml
from ..utils.device_utils import get_device

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
        self.device = get_device(device or config.device)
        self.test_mode = test_mode
        
        # Load registry from YAML
        registry_path = Path(__file__).parents[1] / "config" / "model_registry.yaml"
        with open(registry_path) as f:
            self.registry = yaml.safe_load(f)
        
        # Initialize model cache and current models
        self._model_cache = {}
        self.current_models = None
        
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a specific model from registry."""
        for model in self.registry['models']:
            if model['name'] == model_name:
                return model
        raise ValueError(f"Model {model_name} not found in registry")
        
    def get_model_pair(self, pair_index: int = 0) -> Tuple[str, str]:
        """Get model pair names from config."""
        if self.test_mode:
            # Use the test model directly from registry
            test_model = "HuggingFaceTB/SmolLM-135M-Instruct"
            model_info = self.get_model_info(test_model)
            return test_model, model_info['original']  # This will return 'HuggingFaceTB/SmolLM-135M-Instruct' and 'HuggingFaceTB/SmolLM-135M'
        
        if hasattr(self.config, 'model_pairs'):
            pair = self.config.model_pairs[pair_index]
            return pair['model_1'], pair['model_2']
        return self.config.model_1_name, self.config.model_2_name

    def load_model_pair(self, pair_index: int = 0):
        """Load model pair specified in config."""
        try:
            # Get model names using registry
            model_1_name, model_2_name = self.get_model_pair(pair_index)

            # Get model configs from registry
            model_1_info = self.get_model_info(model_1_name)
            
            try:
                model_2_info = self.get_model_info(model_2_name)
            except ValueError:
                model_2_info = {'name': model_2_name, 'model_kwargs': {}}

            # Check cache
            cache_key = f"{model_1_name}_{model_2_name}"
            if cache_key in self._model_cache:
                self.current_models = self._model_cache[cache_key]
                return self.current_models

            # Set up device-specific configurations
            model_kwargs = {
                'device_map': None,  # Don't use device_map for MPS
                'torch_dtype': torch.float32,  # Use float32 for MPS
            }
            
            if self.device == 'cuda':
                model_kwargs.update({
                    'device_map': 'auto',
                    'torch_dtype': torch.float16,
                    'load_in_8bit': self.config.load_in_8bit
                })
            
            # Load models with their specific configurations
            model_1 = AutoModelForCausalLM.from_pretrained(
                model_1_name,
                **model_kwargs,
                **model_1_info.get('model_kwargs', {})
            )
            
            model_2 = AutoModelForCausalLM.from_pretrained(
                model_2_name,
                **model_kwargs,
                **model_2_info.get('model_kwargs', {})
            )

            # Explicitly move models to device for MPS or CPU
            if self.device in ['mps', 'cpu']:
                model_1 = model_1.to(self.device)
                model_2 = model_2.to(self.device)

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_1_name)

            # Cache the results
            self.current_models = (model_1, model_2, tokenizer)
            self._model_cache[cache_key] = self.current_models
            
            return self.current_models

        except Exception as e:
            logger.error(f"Error loading models: {e}")
            logger.error(f"Device: {self.device}")
            logger.error(f"MPS available: {torch.backends.mps.is_available()}")
            logger.error(f"MPS built: {torch.backends.mps.is_built()}")
            raise RuntimeError(f"Failed to load models: {e}")

    def clear_cache(self):
        """Clear the model cache."""
        self._model_cache.clear()

    def validate_models(self) -> bool:
        """Validate model functionality."""
        # Load models if not already loaded
        if self.current_models is None:
            self.current_models = self.load_model_pair()
        
        model_1, model_2, tokenizer = self.current_models
        print("\nRunning model validation checks...")
        
        try:
            # Check model devices
            print(f"Model 1 device: {next(model_1.parameters()).device}")
            print(f"Model 2 device: {next(model_2.parameters()).device}")
            
            # Check model sizes
            m1_params = sum(p.numel() for p in model_1.parameters())
            m2_params = sum(p.numel() for p in model_2.parameters())
            print(f"Model 1 parameters: {m1_params:,}")
            print(f"Model 2 parameters: {m2_params:,}")
            
            # Check tokenizer
            test_text = "Testing tokenizer functionality."
            tokens = tokenizer(test_text, return_tensors="pt")
            print(f"Tokenizer output keys: {tokens.keys()}")
            
            # Test memory usage based on device type
            if self.device == 'cuda' and torch.cuda.is_available():
                print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            elif self.device == 'mps':
                # MPS doesn't have a direct memory query, so we skip it
                print("Running on Apple Silicon MPS")
            else:
                print("Running on CPU")
            
            return True
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False
