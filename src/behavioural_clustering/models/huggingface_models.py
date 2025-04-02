"""
Huggingface model adapter for using transformers models with the behavioral clustering pipeline.
"""

import torch
from typing import Dict, Any, Optional, List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class HuggingfaceModelAdapter:
    """
    Adapter for Huggingface transformers models to work with the behavioral clustering pipeline.
    Extends functionality beyond LocalModel with system message handling and model-specific prompt formatting.
    """
    
    def __init__(
        self,
        model_name: str,
        system_message: str = "",
        device: str = "auto",
        temperature: float = 0.7,
        max_tokens: int = 150,
        top_p: float = 1.0,
        **kwargs
    ):
        """
        Initialize a Huggingface model adapter.
        
        Args:
            model_name: Name of the Huggingface model
            system_message: System message to prepend to prompts
            device: Device to run the model on ("auto", "cpu", "cuda", etc.)
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            **kwargs: Additional arguments for model loading
        """
        self.model_name = model_name
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        self.device = self._get_device(device)
        
        logger.info(f"Loading Huggingface model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if device == "auto":
                device_map = "auto"
            else:
                device_map = device
                
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                **kwargs
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            raise
            
        self.cache = {}
        
    def _get_device(self, device):
        """
        Determine the appropriate device for the model.
        
        Args:
            device: Device specification ("auto", "cpu", "cuda", etc.)
            
        Returns:
            torch.device: The device to use
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
        
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """
        Generate a response to the prompt.
        
        Args:
            prompt: Prompt text
            max_tokens: Maximum number of tokens to generate (overrides instance default)
            
        Returns:
            Generated text
        """
        try:
            formatted_prompt = self._format_prompt(prompt)
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            generation_config = {
                "max_new_tokens": max_tokens or self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "do_sample": self.temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id
            }
            
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generation_config
                    )
            except NotImplementedError as e:
                if "MPS" in str(e):
                    logger.warning("MPS device not supported for this operation. Falling back to CPU.")
                    self.model = self.model.to("cpu")
                    inputs = {k: v.to("cpu") for k, v in inputs.items()}
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            **generation_config
                        )
                    self.model = self.model.to(self.device)
                else:
                    raise e
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            response = response[len(formatted_prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating with Huggingface model: {str(e)}")
            return f"Error: {str(e)}"
            
    def _format_prompt(self, prompt: str) -> str:
        """
        Format a prompt with the system message if needed.
        
        Args:
            prompt: Prompt text
            
        Returns:
            Formatted prompt
        """
        if not self.system_message:
            return prompt
            
        model_name_lower = self.model_name.lower()
        
        if "llama-3" in model_name_lower:
            return f"<|system|>\n{self.system_message}\n<|user|>\n{prompt}\n<|assistant|>"
        elif "llama-2" in model_name_lower or "llama2" in model_name_lower:
            return f"<s>[INST] <<SYS>>\n{self.system_message}\n<</SYS>>\n\n{prompt} [/INST]"
        elif "mistral" in model_name_lower:
            return f"<s>[INST] {self.system_message}\n\n{prompt} [/INST]"
        elif "gemma" in model_name_lower:
            return f"<start_of_turn>user\n{self.system_message}\n\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        elif "phi" in model_name_lower:
            return f"<|system|>\n{self.system_message}\n<|user|>\n{prompt}\n<|assistant|>\n"
        elif "falcon" in model_name_lower:
            return f"System: {self.system_message}\nUser: {prompt}\nAssistant:"
        else:
            return f"{self.system_message}\n\n{prompt}"
            
    def unload(self) -> None:
        """
        Unload the model and free resources.
        """
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache()
            self.model = None
            logger.info(f"Unloaded model: {self.model_name}")
            
    def get_model(self):
        """
        Get the underlying model.
        
        Returns:
            The Huggingface model
        """
        return self.model
        
    def get_tokenizer(self):
        """
        Get the tokenizer.
        
        Returns:
            The Huggingface tokenizer
        """
        return self.tokenizer
        
    def get_memory_footprint(self):
        """
        Get the memory footprint of the model in bytes.
        
        Returns:
            int: Memory footprint in bytes
        """
        return sum(p.numel() * p.element_size() for p in self.model.parameters())
        
    @staticmethod
    def list_available_models(cache_dir: Optional[str] = None) -> List[str]:
        """
        List locally available Huggingface models.
        
        Args:
            cache_dir: Optional custom cache directory for Huggingface models
            
        Returns:
            List of model names
        """
        if cache_dir is None:
            cache_dir = os.path.join(Path.home(), ".cache", "huggingface", "hub")
            
        if not os.path.exists(cache_dir):
            return []
            
        models = []
        for root, dirs, _ in os.walk(cache_dir):
            for d in dirs:
                if os.path.exists(os.path.join(root, d, "config.json")):
                    model_path = os.path.relpath(os.path.join(root, d), cache_dir)
                    models.append(model_path)
                    
        return models
        
class HuggingfaceEmbeddingModel:
    """
    Adapter for Huggingface embedding models to work with the behavioral clustering pipeline.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        **kwargs
    ):
        """
        Initialize a Huggingface embedding model adapter.
        
        Args:
            model_name: Name of the Huggingface embedding model
            device: Device to run the model on ("auto", "cpu", "cuda", etc.)
            **kwargs: Additional arguments for model loading
        """
        self.model_name = model_name
        
        self.device = self._get_device(device)
        
        logger.info(f"Loading Huggingface embedding model: {model_name}")
        try:
            from transformers import AutoModel, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            if device == "auto":
                device_map = "auto"
            else:
                device_map = device
                
            self.model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                **kwargs
            )
            
            logger.info(f"Embedding model loaded successfully on device: {self.device}")
        except Exception as e:
            logger.error(f"Error loading embedding model {model_name}: {str(e)}")
            raise
            
    def _get_device(self, device):
        """
        Determine the appropriate device for the model.
        
        Args:
            device: Device specification ("auto", "cpu", "cuda", etc.)
            
        Returns:
            torch.device: The device to use
        """
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)
        
    def embed(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Generate embeddings for the input texts.
        
        Args:
            texts: Input text or list of texts
            
        Returns:
            torch.Tensor: Embeddings for the input texts
        """
        try:
            if isinstance(texts, str):
                texts = [texts]
                
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            if hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output
            else:
                attention_mask = inputs["attention_mask"]
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
            return embeddings.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error generating embeddings with Huggingface model: {str(e)}")
            raise
            
    def unload(self) -> None:
        """
        Unload the model and free resources.
        """
        if hasattr(self, 'model'):
            del self.model
            torch.cuda.empty_cache()
            self.model = None
            logger.info(f"Unloaded embedding model: {self.model_name}")
