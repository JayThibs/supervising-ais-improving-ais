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
        print(f"Getting info for model: {model_name}")
        
        # First check registry
        for model in self.registry.get('models', []):
            if model['name'] == model_name:
                print(f"Found model in registry: {model}")
                return model
        
        # If not in registry, return default configuration
        default_config = {
            'name': model_name,
            'model_kwargs': {},
        }
        print(f"Model not found in registry, using default config: {default_config}")
        return default_config
        
    def get_model_pair(self, pair_index: int = 0) -> Tuple[str, str]:
        """Get model pair names from config."""
        print(f"Getting model pair with index: {pair_index}")
        
        if self.test_mode:
            # Use small test models
            test_model = "HuggingFaceTB/SmolLM-135M-Instruct"
            base_model = "HuggingFaceTB/SmolLM-135M"
            print(f"Test mode: using models {test_model} and {base_model}")
            return test_model, base_model
        
        # Check for model pairs in config
        if not hasattr(self.config, 'model_pairs'):
            raise ValueError("Config missing model_pairs")
            
        try:
            # Access the specific pair using the index
            pair = self.config.model_pairs[pair_index]
            print(f"Selected model pair: {pair}")
            
            # Extract model names from the pair dict
            model_1 = pair.get('model_1')
            model_2 = pair.get('model_2')
            
            if not model_1 or not model_2:
                raise ValueError(f"Invalid model pair configuration: {pair}")
                
            print(f"Using models: {model_1} and {model_2}")
            return model_1, model_2
            
        except IndexError:
            raise ValueError(f"Invalid model pair index {pair_index}. Config has {len(self.config.model_pairs)} pairs.")
        except Exception as e:
            raise ValueError(f"Error accessing model pair: {str(e)}")

    def load_model_pair(self, pair_index: int = 0):
        """Load model pair specified in config."""
        try:
            # Get model names
            model_1_name, model_2_name = self.get_model_pair(pair_index)
            print(f"Loading models: {model_1_name} and {model_2_name}")

            # Check cache
            cache_key = f"{model_1_name}_{model_2_name}"
            if cache_key in self._model_cache:
                print("Using cached models")
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
                    'load_in_8bit': getattr(self.config, 'load_in_8bit', False)
                })
            
            print(f"Loading model 1: {model_1_name}")
            model_1 = AutoModelForCausalLM.from_pretrained(
                model_1_name,
                **model_kwargs
            )
            
            print(f"Loading model 2: {model_2_name}")
            model_2 = AutoModelForCausalLM.from_pretrained(
                model_2_name,
                **model_kwargs
            )

            # Explicitly move models to device for MPS or CPU
            if self.device in ['mps', 'cpu']:
                model_1 = model_1.to(self.device)
                model_2 = model_2.to(self.device)

            print(f"Loading tokenizer from: {model_1_name}")
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
