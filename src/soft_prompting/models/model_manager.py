# src/soft_prompting/models/model_manager.py

from typing import Dict, List, Optional, Tuple
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from ..config.configs import ExperimentConfig
from ..utils.device_utils import get_device
from ..config.model_utils import ModelSelector
import logging

logger = logging.getLogger(__name__)

class ModelPairManager:
    """Manages loading and caching of model pairs."""
    
    def __init__(
        self,
        config: ExperimentConfig,
        test_mode: bool = False
    ):
        self.config = config
        self.test_mode = test_mode
        
        # Initialize model selector
        self.model_selector = ModelSelector()
        
        # Initialize model cache
        self._model_cache = {}
        
        # Store model pairs and current pair info
        self.model_pairs = config.model_pairs
        self.current_pair_index = 0
        
        # Set current model pair
        if not self.model_pairs:
            raise ValueError("No model pairs specified in config")
            
        self.current_models = self.model_pairs[0]
        
        # Get model info from registry
        try:
            self.model_1_info = self.model_selector.get_model_info(self.current_models["model_1"])
            self.model_2_info = self.model_selector.get_model_info(self.current_models["model_2"])
        except ValueError as e:
            logger.warning(f"Model not found in registry: {e}")
            self.model_1_info = {"name": self.current_models["model_1"]}
            self.model_2_info = {"name": self.current_models["model_2"]}
        
        # Initialize models as None
        self._model_1 = None
        self._model_2 = None
        self._tokenizer = None
        
        # Set device
        self.device = get_device(config.device if config.device != "auto" else None)
        
        # Set dtype
        self.torch_dtype = (
            torch.float16 if config.torch_dtype == "float16" 
            else torch.float32
        )
        
        self.load_in_8bit = config.load_in_8bit

    def set_model_pair(self, pair_index: int) -> None:
        """Switch to a different model pair."""
        if pair_index < 0 or pair_index >= len(self.model_pairs):
            raise ValueError(f"Invalid pair index {pair_index}. Must be between 0 and {len(self.model_pairs)-1}")
            
        self.current_pair_index = pair_index
        self.current_models = self.model_pairs[pair_index]
        
        # Update model info from registry
        try:
            self.model_1_info = self.model_selector.get_model_info(self.current_models["model_1"])
            self.model_2_info = self.model_selector.get_model_info(self.current_models["model_2"])
        except ValueError as e:
            logger.warning(f"Model not found in registry: {e}")
            self.model_1_info = {"name": self.current_models["model_1"]}
            self.model_2_info = {"name": self.current_models["model_2"]}
        
        # Clear cached models
        self._model_1 = None
        self._model_2 = None
        self._tokenizer = None

    @property
    def model_1_name(self) -> str:
        """Get name of first model in current pair."""
        return self.current_models["model_1"]
        
    @property
    def model_2_name(self) -> str:
        """Get name of second model in current pair."""
        return self.current_models["model_2"]

    def load_models(self) -> Tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
        """Load both models and tokenizer."""
        if self._model_1 is None:
            self._model_1 = self._load_model(self.model_1_name)
        if self._model_2 is None:
            self._model_2 = self._load_model(self.model_2_name)
        if self._tokenizer is None:
            self._tokenizer = self._load_tokenizer(self.model_1_name)
            
        return self._model_1, self._model_2, self._tokenizer

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
        self._model_1 = None
        self._model_2 = None
        self._tokenizer = None

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
